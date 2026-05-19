pub mod quantization;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use hnsw_rs::prelude::*;
use simsimd::SpatialSimilarity;

use quantization::{
    MultiVectorQuantizationConfig, MultiVectorQuantizedIndex, QuantizationConfig, QuantizedIndex,
    valid_product_num_sub_vectors, validate_quantization_config,
};

const MAGIC: &[u8; 4] = b"VDB1";
const VERSION: u16 = 7;
const WAL_MAGIC: &[u8; 4] = b"VWL1";
const TYPE_STRING: u8 = 1;
const TYPE_INTEGER: u8 = 2;
const TYPE_FLOAT: u8 = 3;
const TYPE_BOOLEAN: u8 = 4;
const TYPE_NULL: u8 = 5;
const TYPE_LIST: u8 = 6;
const TYPE_MAP: u8 = 7;
const DEFAULT_NAMESPACE: &str = "";
const DEFAULT_VECTOR_NAME: &str = "";
const ANN_MIN_POINTS: usize = 32;
const ANN_SEARCH_MIN_POINTS: usize = 128;
const ANN_OVERSAMPLE: usize = 8;
const ANN_MIN_CANDIDATES: usize = 64;
const ANN_M: usize = 16;
const ANN_EF_CONSTRUCTION: usize = 200;
/// Threshold above which HNSW construction uses parallel batch insert
/// (Rayon-based). Below this, sequential insert is cheaper because of
/// thread setup overhead.
const ANN_PARALLEL_INSERT_THRESHOLD: usize = 256;
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

pub type Result<T> = std::result::Result<T, VectLiteError>;
pub type Metadata = BTreeMap<String, MetadataValue>;
pub type SparseVector = BTreeMap<String, f32>;
pub type NamedVectors = BTreeMap<String, Vec<f32>>;
/// Multi-vectors: a named space maps to N token-level vectors (e.g. ColBERT embeddings).
pub type MultiVectors = BTreeMap<String, Vec<Vec<f32>>>;
type RecordKey = (String, String);

/// Distance metric used for vector similarity computation.
///
/// Each metric defines how vectors are compared and scored.
/// The metric is persisted in the database file and cannot be changed
/// after creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity: `dot(a,b) / (|a| * |b|)`.
    /// Returns a similarity in \[-1, 1\] (higher is more similar).
    /// This is the default metric and the most common choice for text embeddings.
    Cosine,
    /// Euclidean (L2) distance: `sqrt(sum((a_i - b_i)^2))`.
    /// Returns a distance >= 0 (lower is more similar).
    Euclidean,
    /// Dot product: `sum(a_i * b_i)`.
    /// Returns raw inner product (higher is more similar for normalized vectors).
    /// Use this for pre-normalized embeddings (e.g. OpenAI v3 with `dimensions` param).
    DotProduct,
    /// Manhattan (L1) distance: `sum(|a_i - b_i|)`.
    /// Returns a distance >= 0 (lower is more similar).
    Manhattan,
}

impl DistanceMetric {
    /// Serialization tag for the binary format.
    fn to_tag(self) -> u8 {
        match self {
            DistanceMetric::Cosine => 0,
            DistanceMetric::Euclidean => 1,
            DistanceMetric::DotProduct => 2,
            DistanceMetric::Manhattan => 3,
        }
    }

    /// Deserialize from tag byte.
    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(DistanceMetric::Cosine),
            1 => Ok(DistanceMetric::Euclidean),
            2 => Ok(DistanceMetric::DotProduct),
            3 => Ok(DistanceMetric::Manhattan),
            _ => Err(VectLiteError::InvalidFormat(format!(
                "unknown distance metric tag {tag}"
            ))),
        }
    }

    /// Compute similarity between two vectors using SIMD-accelerated routines.
    ///
    /// For all metrics, the returned score is oriented so that **higher is better**
    /// (more similar / closer). Distance metrics (Euclidean, Manhattan) are negated.
    pub fn score(self, left: &[f32], right: &[f32]) -> f32 {
        match self {
            DistanceMetric::Cosine => simd_cosine_similarity(left, right),
            DistanceMetric::Euclidean => {
                // Negate so higher = more similar
                -simd_euclidean_distance(left, right)
            }
            DistanceMetric::DotProduct => simd_dot_product(left, right),
            DistanceMetric::Manhattan => {
                // Negate so higher = more similar
                -simd_manhattan_distance(left, right)
            }
        }
    }

    /// String name suitable for user-facing display and serialization.
    pub fn name(self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::Euclidean => "euclidean",
            DistanceMetric::DotProduct => "dotproduct",
            DistanceMetric::Manhattan => "manhattan",
        }
    }

    /// Parse a metric name (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "cosine" => Ok(DistanceMetric::Cosine),
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            "dotproduct" | "dot" | "dot_product" | "ip" | "inner_product" => {
                Ok(DistanceMetric::DotProduct)
            }
            "manhattan" | "l1" => Ok(DistanceMetric::Manhattan),
            _ => Err(VectLiteError::InvalidFormat(format!(
                "unknown distance metric '{name}'; valid options: cosine, euclidean, dotproduct, manhattan"
            ))),
        }
    }

    /// Whether this metric behaves as a similarity (higher = better)
    /// or a distance (lower = better) in its raw form before negation.
    pub fn is_similarity(self) -> bool {
        matches!(self, DistanceMetric::Cosine | DistanceMetric::DotProduct)
    }
}

impl Default for DistanceMetric {
    fn default() -> Self {
        DistanceMetric::Cosine
    }
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated distance functions (simsimd with scalar fallback)
// ---------------------------------------------------------------------------

/// Cosine similarity using SIMD, returns value in [-1, 1].
fn simd_cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    // simsimd returns cosine *distance* (1 - cos_sim), so we convert.
    match f32::cosine(left, right) {
        Some(dist) => 1.0 - dist as f32,
        None => scalar_cosine_similarity(left, right),
    }
}

/// Euclidean (L2) distance using SIMD, returns value >= 0.
fn simd_euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    match f32::sqeuclidean(left, right) {
        Some(sq) => (sq as f32).sqrt(),
        None => scalar_euclidean_distance(left, right),
    }
}

/// Dot product using SIMD.
fn simd_dot_product(left: &[f32], right: &[f32]) -> f32 {
    // simsimd::SpatialSimilarity::dot returns the raw inner product.
    match f32::dot(left, right) {
        Some(d) => d as f32,
        None => scalar_dot_product(left, right),
    }
}

/// Manhattan (L1) distance using SIMD, returns value >= 0.
fn simd_manhattan_distance(left: &[f32], right: &[f32]) -> f32 {
    // simsimd does not provide L1; use scalar.
    scalar_manhattan_distance(left, right)
}

// ---------------------------------------------------------------------------
// Scalar fallback implementations
// ---------------------------------------------------------------------------

fn scalar_cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (l, r) in left.iter().zip(right.iter()) {
        dot += l * r;
        left_norm += l * l;
        right_norm += r * r;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn scalar_euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r) * (l - r))
        .sum::<f32>()
        .sqrt()
}

fn scalar_dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(l, r)| l * r).sum()
}

fn scalar_manhattan_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| (l - r).abs())
        .sum()
}

#[derive(Clone, Debug)]
enum WalOp {
    Upsert(Record),
    Delete {
        namespace: String,
        id: String,
    },
    UpdateMetadata {
        namespace: String,
        id: String,
        metadata: Metadata,
    },
    SetTtl {
        namespace: String,
        id: String,
        expires_at: Option<f64>,
    },
}

#[derive(Clone, Debug)]
pub enum WriteOperation {
    Insert(Record),
    Upsert(Record),
    Delete { namespace: String, id: String },
}

#[derive(Default)]
struct SparseIndex {
    postings: BTreeMap<String, Vec<SparsePosting>>,
    doc_lengths: BTreeMap<RecordKey, f32>,
    avg_doc_len: f32,
    doc_count: usize,
}

#[derive(Clone, Debug)]
struct SparsePosting {
    key: RecordKey,
    term_weight: f32,
}

#[derive(Debug)]
pub enum VectLiteError {
    Io(io::Error),
    InvalidFormat(String),
    DimensionMismatch { expected: usize, found: usize },
    DuplicateId { namespace: String, id: String },
    ReadOnly,
    LockContention(String),
}

impl fmt::Display for VectLiteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::InvalidFormat(message) => write!(f, "invalid .vdb format: {message}"),
            Self::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "vector dimension mismatch: expected {expected}, found {found}. \
                     If you changed embedding models, delete the existing .vdb file \
                     or use a different path to create a new database with dimension {found}"
                )
            }
            Self::DuplicateId { namespace, id } => {
                if namespace.is_empty() {
                    write!(f, "a record with id '{id}' already exists")
                } else {
                    write!(
                        f,
                        "a record with id '{id}' already exists in namespace '{namespace}'"
                    )
                }
            }
            Self::ReadOnly => write!(f, "database is opened in read-only mode"),
            Self::LockContention(msg) => write!(f, "lock contention: {msg}"),
        }
    }
}

impl StdError for VectLiteError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidFormat(_)
            | Self::DimensionMismatch { .. }
            | Self::DuplicateId { .. }
            | Self::ReadOnly
            | Self::LockContention(_) => None,
        }
    }
}

impl From<io::Error> for VectLiteError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    List(Vec<MetadataValue>),
    Map(BTreeMap<String, MetadataValue>),
}

impl MetadataValue {
    fn type_tag(&self) -> u8 {
        match self {
            Self::String(_) => TYPE_STRING,
            Self::Integer(_) => TYPE_INTEGER,
            Self::Float(_) => TYPE_FLOAT,
            Self::Boolean(_) => TYPE_BOOLEAN,
            Self::Null => TYPE_NULL,
            Self::List(_) => TYPE_LIST,
            Self::Map(_) => TYPE_MAP,
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            Self::Integer(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            Self::String(_) | Self::Boolean(_) | Self::Null | Self::List(_) | Self::Map(_) => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(value) => write!(f, "{value}"),
            Self::Integer(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
            Self::Boolean(value) => write!(f, "{value}"),
            Self::Null => write!(f, "null"),
            Self::List(values) => {
                write!(f, "[")?;
                for (i, value) in values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{value}")?;
                }
                write!(f, "]")
            }
            Self::Map(entries) => {
                write!(f, "{{")?;
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{key}: {value}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<bool> for MetadataValue {
    fn from(value: bool) -> Self {
        Self::Boolean(value)
    }
}

impl From<i64> for MetadataValue {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<i32> for MetadataValue {
    fn from(value: i32) -> Self {
        Self::Integer(value.into())
    }
}

impl From<f64> for MetadataValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<f32> for MetadataValue {
    fn from(value: f32) -> Self {
        Self::Float(value.into())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub namespace: String,
    pub id: String,
    pub vector: Vec<f32>,
    pub vectors: NamedVectors,
    pub sparse: SparseVector,
    pub metadata: Metadata,
    /// Multi-vectors for late interaction scoring (e.g. ColBERT token embeddings).
    /// Each key is a named vector space, and the value is a list of token-level vectors.
    pub multi_vectors: MultiVectors,
    /// Optional Unix-epoch timestamp (seconds, f64) at which this record expires.
    /// `None` means the record never expires.
    pub expires_at: Option<f64>,
}

impl Record {
    fn vector_for(&self, vector_name: Option<&str>) -> Option<&[f32]> {
        match vector_name {
            Some(vector_name) if !vector_name.is_empty() => {
                self.vectors.get(vector_name).map(Vec::as_slice)
            }
            Some(_) | None => Some(self.vector.as_slice()),
        }
    }

    fn dense_vectors(&self) -> impl Iterator<Item = (&str, &Vec<f32>)> {
        std::iter::once((DEFAULT_VECTOR_NAME, &self.vector)).chain(
            self.vectors
                .iter()
                .map(|(name, vector)| (name.as_str(), vector)),
        )
    }

    /// Returns `true` if the record has an `expires_at` timestamp that is in
    /// the past relative to the given `now` epoch (seconds since UNIX epoch).
    fn is_expired_at(&self, now: f64) -> bool {
        self.expires_at.map_or(false, |ts| ts <= now)
    }
}

/// Returns the current time as seconds since the UNIX epoch.
fn now_epoch_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataFilter {
    Eq {
        key: String,
        value: MetadataValue,
    },
    NotEq {
        key: String,
        value: MetadataValue,
    },
    In {
        key: String,
        values: Vec<MetadataValue>,
    },
    NotIn {
        key: String,
        values: Vec<MetadataValue>,
    },
    Contains {
        key: String,
        needle: String,
    },
    GreaterThan {
        key: String,
        value: f64,
    },
    GreaterThanOrEqual {
        key: String,
        value: f64,
    },
    LessThan {
        key: String,
        value: f64,
    },
    LessThanOrEqual {
        key: String,
        value: f64,
    },
    Exists {
        key: String,
    },
    /// Matches if a list-typed field has at least one element satisfying `filter`.
    ElemMatch {
        key: String,
        filter: Box<MetadataFilter>,
    },
    /// Matches if a list-typed field has exactly `size` elements.
    Size {
        key: String,
        size: usize,
    },
    Not(Box<MetadataFilter>),
    And(Vec<MetadataFilter>),
    Or(Vec<MetadataFilter>),
}

impl MetadataFilter {
    pub fn eq(key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        Self::Eq {
            key: key.into(),
            value: value.into(),
        }
    }

    pub fn ne(key: impl Into<String>, value: impl Into<MetadataValue>) -> Self {
        Self::NotEq {
            key: key.into(),
            value: value.into(),
        }
    }

    pub fn r#in(key: impl Into<String>, values: Vec<MetadataValue>) -> Self {
        Self::In {
            key: key.into(),
            values,
        }
    }

    pub fn nin(key: impl Into<String>, values: Vec<MetadataValue>) -> Self {
        Self::NotIn {
            key: key.into(),
            values,
        }
    }

    pub fn contains(key: impl Into<String>, needle: impl Into<String>) -> Self {
        Self::Contains {
            key: key.into(),
            needle: needle.into(),
        }
    }

    pub fn gt(key: impl Into<String>, value: f64) -> Self {
        Self::GreaterThan {
            key: key.into(),
            value,
        }
    }

    pub fn gte(key: impl Into<String>, value: f64) -> Self {
        Self::GreaterThanOrEqual {
            key: key.into(),
            value,
        }
    }

    pub fn lt(key: impl Into<String>, value: f64) -> Self {
        Self::LessThan {
            key: key.into(),
            value,
        }
    }

    pub fn lte(key: impl Into<String>, value: f64) -> Self {
        Self::LessThanOrEqual {
            key: key.into(),
            value,
        }
    }

    pub fn exists(key: impl Into<String>) -> Self {
        Self::Exists { key: key.into() }
    }

    pub fn elem_match(key: impl Into<String>, filter: MetadataFilter) -> Self {
        Self::ElemMatch {
            key: key.into(),
            filter: Box::new(filter),
        }
    }

    pub fn size(key: impl Into<String>, size: usize) -> Self {
        Self::Size {
            key: key.into(),
            size,
        }
    }

    pub fn not(filter: MetadataFilter) -> Self {
        Self::Not(Box::new(filter))
    }

    pub fn and(filters: Vec<MetadataFilter>) -> Self {
        Self::And(filters)
    }

    pub fn or(filters: Vec<MetadataFilter>) -> Self {
        Self::Or(filters)
    }

    fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Self::Eq { key, value } => resolve_dot_path(metadata, key) == Some(value),
            Self::NotEq { key, value } => {
                resolve_dot_path(metadata, key).is_some_and(|candidate| candidate != value)
            }
            Self::In { key, values } => {
                resolve_dot_path(metadata, key).is_some_and(|candidate| values.contains(candidate))
            }
            Self::NotIn { key, values } => {
                resolve_dot_path(metadata, key).is_some_and(|candidate| !values.contains(candidate))
            }
            Self::Contains { key, needle } => resolve_dot_path(metadata, key)
                .and_then(|value| match value {
                    MetadataValue::String(value) => Some(value.contains(needle)),
                    MetadataValue::Integer(_)
                    | MetadataValue::Float(_)
                    | MetadataValue::Boolean(_)
                    | MetadataValue::Null
                    | MetadataValue::List(_)
                    | MetadataValue::Map(_) => None,
                })
                .unwrap_or(false),
            Self::GreaterThan { key, value } => resolve_dot_path(metadata, key)
                .and_then(MetadataValue::as_number)
                .map(|candidate| candidate > *value)
                .unwrap_or(false),
            Self::GreaterThanOrEqual { key, value } => resolve_dot_path(metadata, key)
                .and_then(MetadataValue::as_number)
                .map(|candidate| candidate >= *value)
                .unwrap_or(false),
            Self::LessThan { key, value } => resolve_dot_path(metadata, key)
                .and_then(MetadataValue::as_number)
                .map(|candidate| candidate < *value)
                .unwrap_or(false),
            Self::LessThanOrEqual { key, value } => resolve_dot_path(metadata, key)
                .and_then(MetadataValue::as_number)
                .map(|candidate| candidate <= *value)
                .unwrap_or(false),
            Self::Exists { key } => resolve_dot_path(metadata, key).is_some(),
            Self::ElemMatch { key, filter } => {
                resolve_dot_path(metadata, key)
                    .and_then(|value| match value {
                        MetadataValue::List(items) => Some(items.iter().any(|item| {
                            // Wrap the single element in a temporary metadata map
                            // so the sub-filter can match against it as a virtual record.
                            let mut elem_meta = Metadata::new();
                            // Flatten: if the item is a Map, use it directly.
                            // Otherwise, create a pseudo-map with the element
                            // as value under every key the filter references.
                            match item {
                                MetadataValue::Map(map) => {
                                    for (k, v) in map {
                                        elem_meta.insert(k.clone(), v.clone());
                                    }
                                }
                                _ => {
                                    // Put the scalar value as "_" so simple
                                    // equality filters like {"$eq": 42} work.
                                    elem_meta.insert("_".to_owned(), item.clone());
                                }
                            }
                            filter.matches(&elem_meta)
                        })),
                        _ => None,
                    })
                    .unwrap_or(false)
            }
            Self::Size { key, size } => resolve_dot_path(metadata, key)
                .and_then(|value| match value {
                    MetadataValue::List(items) => Some(items.len() == *size),
                    _ => None,
                })
                .unwrap_or(false),
            Self::Not(filter) => !filter.matches(metadata),
            Self::And(filters) => filters.iter().all(|filter| filter.matches(metadata)),
            Self::Or(filters) => filters.iter().any(|filter| filter.matches(metadata)),
        }
    }
}

/// Resolves a dot-separated key path against a metadata map.
/// E.g. "extra.nested_key" first looks up "extra" then "nested_key" inside it.
fn resolve_dot_path<'a>(metadata: &'a Metadata, key: &str) -> Option<&'a MetadataValue> {
    // Fast path: no dots → direct lookup (also handles keys that literally contain dots
    // which were stored before this feature).
    if !key.contains('.') || metadata.contains_key(key) {
        return metadata.get(key);
    }

    let mut parts = key.split('.');
    let first = parts.next()?;
    let mut current = metadata.get(first)?;

    for part in parts {
        match current {
            MetadataValue::Map(map) => {
                current = map.get(part)?;
            }
            _ => return None,
        }
    }

    Some(current)
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub top_k: usize,
    pub filter: Option<MetadataFilter>,
    /// Optional prefix dimension for Matryoshka embeddings.
    ///
    /// When set, dense scoring uses only the first `truncate_dim` dimensions
    /// of the stored vectors and query. ANN/quantized candidate selection is
    /// bypassed because those indexes are built over the full database
    /// dimension.
    pub truncate_dim: Option<usize>,
}

/// HNSW tuning parameters. Exposed so callers can trade off recall, latency,
/// memory and build time.
///
/// Defaults mirror VectLite's historical built-in values (`m = 16`,
/// `ef_construction = 200`). `ef_search = None` means VectLite picks an
/// `ef_search` derived from `top_k * ANN_OVERSAMPLE`.
///
/// Reference: Malkov & Yashunin, *Efficient and robust approximate nearest
/// neighbor search using Hierarchical Navigable Small World graphs*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexConfig {
    /// Max number of bidirectional links per node. Higher = better recall,
    /// more memory, slower build. Typical range: 8..64.
    pub m: usize,
    /// Width of the search during graph construction. Higher = better recall,
    /// slower build. Typical range: 64..800.
    pub ef_construction: usize,
    /// Width of the search at query time. None = auto (derived from top_k).
    /// Higher = better recall, slower search.
    pub ef_search: Option<usize>,
    /// Use parallel (Rayon-backed) HNSW insertion when the dataset has at
    /// least this many vectors. Defaults to `ANN_PARALLEL_INSERT_THRESHOLD`.
    /// Set very high to disable parallel insert.
    pub parallel_insert_threshold: usize,
    /// Percentage (0..=100) of tombstoned nodes at which the HNSW graph is
    /// rebuilt during `compact()`. A `delete` doesn't physically remove a
    /// node from HNSW (that operation is not supported by the library); the
    /// node is just marked dead and filtered out at search time. Once enough
    /// nodes are dead, search recall and latency degrade, so we rebuild.
    /// Default `30` (rebuild when ≥30% of the graph is dead). Set to `100`
    /// to disable automatic rebuild.
    pub tombstone_rebuild_pct: u8,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            m: ANN_M,
            ef_construction: ANN_EF_CONSTRUCTION,
            ef_search: None,
            parallel_insert_threshold: ANN_PARALLEL_INSERT_THRESHOLD,
            tombstone_rebuild_pct: 30,
        }
    }
}

impl IndexConfig {
    /// A preset tuned for higher recall at the cost of build/search time.
    /// Useful for benchmark comparisons where recall@10 must approach 1.0.
    pub fn high_recall() -> Self {
        Self {
            m: 32,
            ef_construction: 400,
            ef_search: Some(200),
            parallel_insert_threshold: ANN_PARALLEL_INSERT_THRESHOLD,
            tombstone_rebuild_pct: 30,
        }
    }

    /// A preset tuned for fast build & low latency, lower recall.
    pub fn fast() -> Self {
        Self {
            m: 8,
            ef_construction: 100,
            ef_search: Some(40),
            parallel_insert_threshold: ANN_PARALLEL_INSERT_THRESHOLD,
            tombstone_rebuild_pct: 30,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.m == 0 {
            return Err(VectLiteError::InvalidFormat(
                "IndexConfig.m must be >= 1".to_owned(),
            ));
        }
        if self.ef_construction == 0 {
            return Err(VectLiteError::InvalidFormat(
                "IndexConfig.ef_construction must be >= 1".to_owned(),
            ));
        }
        if let Some(ef) = self.ef_search {
            if ef == 0 {
                return Err(VectLiteError::InvalidFormat(
                    "IndexConfig.ef_search must be >= 1 when set".to_owned(),
                ));
            }
        }
        if self.tombstone_rebuild_pct > 100 {
            return Err(VectLiteError::InvalidFormat(
                "IndexConfig.tombstone_rebuild_pct must be in 0..=100".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Controls when the WAL file is `fsync`'d to disk.
///
/// Per-record durability is the default (`PerOp`) but on macOS APFS — and to
/// a lesser extent on Linux ext4 — `fsync` is the dominant cost of single
/// `insert` calls. Relaxing this knob can multiply ingestion throughput by
/// 5–10× at the cost of losing some recently-acknowledged records on an
/// unclean shutdown.
///
/// The WAL is *always* fully synced on `flush()`, `compact()`, and `close()`.
/// So even with `OnFlush`, any data that survives a clean shutdown is
/// durable. The window of vulnerability is limited to:
/// - `EveryN(n)`: at most the last `n - 1` inserts since the last fsync.
/// - `OnFlush`: every insert since the last `flush()` / `compact()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalSyncMode {
    /// `fsync` after every WAL append. Strongest durability, slowest. This is
    /// the default and matches pre-0.11 behaviour.
    PerOp,
    /// `fsync` once every `n` ops. On a crash, up to the last `n - 1` ops
    /// since the last sync may be lost. A good middle ground when streaming
    /// thousands of small records: pick `n` so the worst-case loss is
    /// tolerable (e.g. `64` ≈ a fraction of a second of data).
    EveryN(usize),
    /// Never `fsync` from the per-op path. Sync only at `flush()` / `compact()`
    /// / `close()`. Maximum throughput, weakest durability — appropriate for
    /// bulk ingestion of data that can be regenerated.
    OnFlush,
}

impl Default for WalSyncMode {
    fn default() -> Self {
        WalSyncMode::PerOp
    }
}

impl WalSyncMode {
    fn validate(self) -> Result<()> {
        if let WalSyncMode::EveryN(n) = self {
            if n == 0 {
                return Err(VectLiteError::InvalidFormat(
                    "WalSyncMode::EveryN must be >= 1".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            filter: None,
            truncate_dim: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HybridSearchOptions {
    pub top_k: usize,
    pub filter: Option<MetadataFilter>,
    pub dense_weight: f32,
    pub sparse_weight: f32,
    pub fetch_k: usize,
    pub mmr_lambda: Option<f32>,
    pub vector_name: Option<String>,
    pub fusion: FusionStrategy,
    /// Optional prefix dimension for Matryoshka embeddings.
    pub truncate_dim: Option<usize>,
    /// Multi-vector search: provide per-vector-name queries and their weights.
    /// When non-empty, the dense score is the weighted sum of cosine
    /// similarities over the given vector spaces, and `vector_name` is ignored.
    pub multi_vector_queries: BTreeMap<String, (Vec<f32>, f32)>,
}

impl Default for HybridSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            filter: None,
            dense_weight: 1.0,
            sparse_weight: 1.0,
            fetch_k: 0,
            mmr_lambda: None,
            vector_name: None,
            fusion: FusionStrategy::Linear,
            truncate_dim: None,
            multi_vector_queries: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FusionStrategy {
    Linear,
    Rrf { rank_constant: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub namespace: String,
    pub id: String,
    pub score: f32,
    pub dense_score: f32,
    pub sparse_score: f32,
    pub vector_name: Option<String>,
    pub matched_terms: Vec<String>,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
    pub metadata: Metadata,
    /// Per-term BM25 contribution for this result (only when explain requested).
    pub bm25_term_scores: BTreeMap<String, f32>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct SearchStats {
    pub used_ann: bool,
    pub ann_candidate_count: usize,
    pub exact_fallback: bool,
    pub considered_count: usize,
    pub fetch_k: usize,
    pub mmr_applied: bool,
    pub sparse_candidate_count: usize,
    pub ann_loaded_from_disk: bool,
    pub wal_entries_replayed: usize,
    pub fusion: String,
    /// Effective dense dimensions used for scoring. This can be lower than the
    /// stored database dimension for Matryoshka/prefix searches.
    pub effective_dimension: usize,
    pub matryoshka_truncated: bool,
    /// Timing breakdown in microseconds.
    pub timings: SearchTimings,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchTimings {
    /// Microseconds spent on dense (ANN or brute-force) scoring.
    pub dense_us: u64,
    /// Microseconds spent on sparse (BM25) scoring.
    pub sparse_us: u64,
    /// Microseconds spent on fusion (combining dense + sparse).
    pub fusion_us: u64,
    /// Total end-to-end microseconds.
    pub total_us: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub stats: SearchStats,
}

/// Options for multi-vector (late interaction / ColBERT-style) search.
#[derive(Clone, Debug)]
pub struct MultiVectorSearchOptions {
    /// Number of results to return.
    pub top_k: usize,
    /// Optional metadata filter.
    pub filter: Option<MetadataFilter>,
    /// Optional namespace.
    pub namespace: Option<String>,
}

impl Default for MultiVectorSearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            filter: None,
            namespace: None,
        }
    }
}

/// A result from multi-vector (MaxSim) search.
#[derive(Clone, Debug)]
pub struct MultiVectorSearchResult {
    pub namespace: String,
    pub id: String,
    pub score: f32,
    pub metadata: Metadata,
}

/// A store manages a directory of independent physical collections, each
/// backed by its own `.vdb` file with its own dimension, WAL, and ANN index.
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Open (or create) a store rooted at the given directory.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }
        Ok(Self { root })
    }

    /// Return the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Create a new collection. Returns an error if it already exists.
    pub fn create_collection(&self, name: &str, dimension: usize) -> Result<Database> {
        let path = self.collection_path(name);
        if path.exists() {
            return Err(VectLiteError::InvalidFormat(format!(
                "collection '{name}' already exists"
            )));
        }
        Database::create(path, dimension)
    }

    /// Open an existing collection.
    pub fn open_collection(&self, name: &str) -> Result<Database> {
        let path = self.collection_path(name);
        if !path.exists() {
            return Err(VectLiteError::InvalidFormat(format!(
                "collection '{name}' does not exist"
            )));
        }
        Database::open(path)
    }

    /// Open an existing collection in read-only mode.
    pub fn open_collection_read_only(&self, name: &str) -> Result<Database> {
        let path = self.collection_path(name);
        if !path.exists() {
            return Err(VectLiteError::InvalidFormat(format!(
                "collection '{name}' does not exist"
            )));
        }
        Database::open_read_only(path)
    }

    /// Open an existing collection or create it with the given dimension.
    pub fn open_or_create_collection(&self, name: &str, dimension: usize) -> Result<Database> {
        Database::open_or_create(self.collection_path(name), dimension)
    }

    /// Drop a collection, deleting all its files.
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        let path = self.collection_path(name);
        if !path.exists() {
            return Ok(false);
        }
        // Remove main file, WAL, ANN sidecar files
        let _ = fs::remove_file(&path);
        let wal = wal_path(&path);
        let _ = fs::remove_file(&wal);
        let manifest = ann_manifest_path(&path);
        let _ = fs::remove_file(&manifest);
        let quant = quantization_params_path(&path);
        let _ = fs::remove_file(&quant);
        // Remove any .hnsw.* sidecar files
        if let Some(parent) = path.parent() {
            if let Some(stem) = path.file_name().and_then(|n| n.to_str()) {
                if let Ok(entries) = fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(fname) = entry.file_name().to_str() {
                            if fname.starts_with(&format!("{stem}.ann.")) {
                                let _ = fs::remove_file(entry.path());
                            }
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    /// List all collection names in this store.
    pub fn collections(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("vdb") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_owned());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    fn collection_path(&self, name: &str) -> PathBuf {
        self.root.join(format!("{name}.vdb"))
    }
}

// ---------------------------------------------------------------------------
// Payload indexes  (keyword + numeric)
// ---------------------------------------------------------------------------

/// The type of payload index on a metadata field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadIndexType {
    /// Inverted index for exact string equality, `$in`, `$nin` lookups.
    Keyword,
    /// Ordered B-tree index for numeric range queries (`$gt`, `$gte`, `$lt`, `$lte`).
    Numeric,
}

impl PayloadIndexType {
    fn tag(&self) -> u8 {
        match self {
            Self::Keyword => 1,
            Self::Numeric => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Keyword),
            2 => Ok(Self::Numeric),
            _ => Err(VectLiteError::InvalidFormat(format!(
                "unknown payload index type tag: {tag}"
            ))),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Keyword => "keyword",
            Self::Numeric => "numeric",
        }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "keyword" | "string" | "text" => Ok(Self::Keyword),
            "numeric" | "number" | "int" | "float" | "integer" => Ok(Self::Numeric),
            _ => Err(VectLiteError::InvalidFormat(format!(
                "unknown payload index type: {name:?}"
            ))),
        }
    }
}

/// A keyword (inverted) index: value → set of record keys.
#[derive(Clone, Debug, Default)]
struct KeywordIndex {
    postings: HashMap<String, HashSet<RecordKey>>,
}

impl KeywordIndex {
    fn insert(&mut self, value: &str, key: RecordKey) {
        self.postings
            .entry(value.to_owned())
            .or_default()
            .insert(key);
    }

    fn remove(&mut self, value: &str, key: &RecordKey) {
        if let Some(set) = self.postings.get_mut(value) {
            set.remove(key);
            if set.is_empty() {
                self.postings.remove(value);
            }
        }
    }

    /// Return keys that match `value` exactly (for `$eq`).
    fn lookup_eq(&self, value: &str) -> Option<&HashSet<RecordKey>> {
        self.postings.get(value)
    }

    /// Return keys matching any of `values` (for `$in`).
    fn lookup_in(&self, values: &[&str]) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for value in values {
            if let Some(set) = self.postings.get(*value) {
                result.extend(set.iter().cloned());
            }
        }
        result
    }

    /// Return all indexed keys (universe for negation).
    #[allow(dead_code)]
    fn all_keys(&self) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for set in self.postings.values() {
            result.extend(set.iter().cloned());
        }
        result
    }
}

/// An ordered-float wrapper for BTreeMap keys.
#[derive(Clone, Copy, Debug)]
struct OrdF64(f64);

impl PartialEq for OrdF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0) == std::cmp::Ordering::Equal
    }
}
impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl std::hash::Hash for OrdF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// A numeric (sorted B-tree) index: ordered value → set of record keys.
#[derive(Clone, Debug, Default)]
struct NumericIndex {
    tree: BTreeMap<OrdF64, HashSet<RecordKey>>,
}

impl NumericIndex {
    fn insert(&mut self, value: f64, key: RecordKey) {
        self.tree.entry(OrdF64(value)).or_default().insert(key);
    }

    fn remove(&mut self, value: f64, key: &RecordKey) {
        if let Some(set) = self.tree.get_mut(&OrdF64(value)) {
            set.remove(key);
            if set.is_empty() {
                self.tree.remove(&OrdF64(value));
            }
        }
    }

    /// Return keys where value > threshold.
    fn range_gt(&self, threshold: f64) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for (_, set) in self.tree.range((
            std::ops::Bound::Excluded(OrdF64(threshold)),
            std::ops::Bound::Unbounded,
        )) {
            result.extend(set.iter().cloned());
        }
        result
    }

    /// Return keys where value >= threshold.
    fn range_gte(&self, threshold: f64) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for (_, set) in self.tree.range(OrdF64(threshold)..) {
            result.extend(set.iter().cloned());
        }
        result
    }

    /// Return keys where value < threshold.
    fn range_lt(&self, threshold: f64) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for (_, set) in self.tree.range(..OrdF64(threshold)) {
            result.extend(set.iter().cloned());
        }
        result
    }

    /// Return keys where value <= threshold.
    fn range_lte(&self, threshold: f64) -> HashSet<RecordKey> {
        let mut result = HashSet::new();
        for (_, set) in self.tree.range(..=OrdF64(threshold)) {
            result.extend(set.iter().cloned());
        }
        result
    }

    /// Return keys where value == target (exact match).
    fn lookup_eq(&self, target: f64) -> Option<&HashSet<RecordKey>> {
        self.tree.get(&OrdF64(target))
    }
}

/// A live payload index with its definition and populated data.
#[derive(Clone, Debug)]
enum PayloadIndexData {
    Keyword(KeywordIndex),
    Numeric(NumericIndex),
}

pub struct Database {
    path: PathBuf,
    wal_path: PathBuf,
    dimension: usize,
    metric: DistanceMetric,
    records: BTreeMap<(String, String), Record>,
    ann: AnnCatalog,
    sparse_index: SparseIndex,
    wal_entries_replayed: usize,
    ann_loaded_from_disk: bool,
    read_only: bool,
    /// Holds the lock file open for the lifetime of the database.
    /// Dropping this releases the advisory lock.
    _lock_file: Option<File>,
    /// Cached WAL writer: avoids paying the open() syscall on every insert.
    /// Reset whenever the WAL is rotated (compact, clear_wal).
    wal_writer: Option<BufWriter<File>>,
    /// Controls when `fsync` is issued against the WAL — see [`WalSyncMode`].
    wal_sync_mode: WalSyncMode,
    /// Number of ops appended to the WAL since the last fsync. Used by the
    /// `EveryN` sync mode to decide when to flush+sync.
    wal_ops_since_sync: usize,
    /// True if the in-memory ANN graph(s) have unsaved changes (incremental
    /// inserts, fresh build, or a full rebuild) that have not been written
    /// out via `persist_ann_to_disk`. Set on every mutation in
    /// `apply_wal_batch` / `bulk_ingest` and cleared by `compact_inner` or
    /// an explicit `persist_ann_to_disk`.
    ann_dirty: bool,
    /// True if the quantized PQ index needs to be rebuilt at the next flush
    /// (because records have been inserted/deleted since the last rebuild).
    /// While dirty, the in-memory `quantized` field is set to `None` so
    /// searches transparently fall back to the HNSW path instead of
    /// returning candidates from a stale codebook.
    quantized_dirty: bool,
    /// Same as `quantized_dirty`, but for multi-vector (ColBERT-style)
    /// quantization spaces. Lazy rebuild happens at flush time.
    multi_vector_quantized_dirty: bool,
    /// Optional quantized index for accelerated search.
    quantized: Option<QuantizedIndex>,
    /// Configuration used to build the quantized index (persisted).
    quantization_config: Option<QuantizationConfig>,
    /// Ordered keys mapping quantized index positions to record keys.
    quantized_keys: Vec<RecordKey>,
    /// Optional quantized index for multi-vector (ColBERT) search.
    multi_vector_quantized: BTreeMap<String, MultiVectorQuantizedIndex>,
    /// Configuration for multi-vector quantization (per space).
    multi_vector_quantization_config: BTreeMap<String, MultiVectorQuantizationConfig>,
    /// Ordered keys mapping multi-vector quantized index doc positions to record keys.
    multi_vector_quantized_keys: BTreeMap<String, Vec<RecordKey>>,
    /// Payload index definitions (field → type), persisted in sidecar file.
    payload_index_defs: BTreeMap<String, PayloadIndexType>,
    /// Live payload indexes, populated from records.
    payload_indexes: BTreeMap<String, PayloadIndexData>,
    /// HNSW tuning parameters. Not persisted to disk: this is a per-session
    /// knob so callers can change recall/latency tradeoffs without migrating
    /// data files. A subsequent `set_index_config` triggers a rebuild.
    index_config: IndexConfig,
    /// Contiguous f32 mirror of the default dense vector for every record.
    /// Used by brute-force / rescoring scans for cache-friendly SIMD.
    /// `None` when the arena hasn't been materialised yet for this session.
    vector_arena: Option<VectorArena>,
    /// When true, `vector_arena` is stale (e.g. a delete happened) and must
    /// be rebuilt before use.
    vector_arena_dirty: bool,
}

/// Contiguous-storage mirror of the default dense vector per record.
///
/// In the original layout each `Record.vector` is a separately-allocated
/// `Vec<f32>` and the records themselves live in `BTreeMap` nodes, so a
/// brute-force or rescoring scan pays two pointer hops per record AND
/// touches one cache line per vector — terrible for SIMD throughput.
///
/// This arena stores every vector in a single flat `buf: Vec<f32>` so a scan
/// is a straight contiguous walk (one cache miss per ~16 vectors, vs ~2 per
/// vector). Lance / Arrow use the same trick — see the v0.11 CHANGELOG note.
///
/// The arena is maintained incrementally on insert; deletes are too
/// expensive to compact in place (would shift O(N) f32s) so they just mark
/// the arena dirty and force a lazy full rebuild on next use.
struct VectorArena {
    buf: Vec<f32>,
    keys: Vec<RecordKey>,
    key_to_index: HashMap<RecordKey, usize>,
    dim: usize,
}

impl VectorArena {
    fn new(dim: usize) -> Self {
        Self {
            buf: Vec::new(),
            keys: Vec::new(),
            key_to_index: HashMap::new(),
            dim,
        }
    }

    fn append(&mut self, key: RecordKey, vector: &[f32]) {
        // Defensive: ignore mismatched dims rather than panicking — this is
        // a perf cache, not the source of truth.
        if vector.len() != self.dim {
            return;
        }
        let idx = self.keys.len();
        self.buf.extend_from_slice(vector);
        self.key_to_index.insert(key.clone(), idx);
        self.keys.push(key);
    }

    /// Rebuild from records in BTreeMap order. Called lazily when the arena
    /// is dirty (i.e. after a delete or a full ANN rebuild).
    fn rebuild_from(records: &BTreeMap<RecordKey, Record>, dim: usize) -> Self {
        let mut arena = Self::new(dim);
        arena.buf.reserve(records.len() * dim);
        arena.keys.reserve(records.len());
        arena.key_to_index.reserve(records.len());
        for (key, record) in records {
            if record.vector.len() == dim {
                arena.append(key.clone(), &record.vector);
            }
        }
        arena
    }

    /// Iterator yielding `(key, vector_slice)` pairs. The slice references
    /// the contiguous `buf`, so consumers get cache-friendly SIMD scans.
    #[allow(dead_code)]
    fn iter(&self) -> impl Iterator<Item = (&RecordKey, &[f32])> {
        let dim = self.dim;
        self.keys.iter().enumerate().map(move |(i, k)| {
            let start = i * dim;
            (k, &self.buf[start..start + dim])
        })
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

#[derive(Default)]
struct AnnCatalog {
    global: BTreeMap<String, AnnIndex>,
    namespaces: BTreeMap<String, BTreeMap<String, AnnIndex>>,
}

enum AnnHnsw {
    Cosine(Hnsw<'static, f32, DistCosine>),
    Euclidean(Hnsw<'static, f32, DistL2>),
    DotProduct(Hnsw<'static, f32, DistDot>),
    Manhattan(Hnsw<'static, f32, DistL1>),
}

impl AnnHnsw {
    fn search(&self, query: &[f32], knbn: usize, ef_search: usize) -> Vec<Neighbour> {
        match self {
            AnnHnsw::Cosine(h) => h.search(query, knbn, ef_search),
            AnnHnsw::Euclidean(h) => h.search(query, knbn, ef_search),
            AnnHnsw::DotProduct(h) => h.search(query, knbn, ef_search),
            AnnHnsw::Manhattan(h) => h.search(query, knbn, ef_search),
        }
    }

    /// Incrementally insert a single vector into an existing HNSW graph.
    /// `origin_id` must be unique within the graph and is used to map back
    /// to the caller's record key array.
    fn insert_one(&mut self, vector: &[f32], origin_id: usize) {
        match self {
            AnnHnsw::Cosine(h) => h.insert((vector, origin_id)),
            AnnHnsw::Euclidean(h) => h.insert((vector, origin_id)),
            AnnHnsw::DotProduct(h) => h.insert((vector, origin_id)),
            AnnHnsw::Manhattan(h) => h.insert((vector, origin_id)),
        }
    }

    /// Bulk-insert a batch of vectors in parallel (Rayon-multithreaded).
    /// Significantly faster than repeated `insert_one` when the batch is
    /// large enough to amortise thread setup.
    fn parallel_insert_batch(&mut self, batch: &[(&Vec<f32>, usize)]) {
        match self {
            AnnHnsw::Cosine(h) => h.parallel_insert(batch),
            AnnHnsw::Euclidean(h) => h.parallel_insert(batch),
            AnnHnsw::DotProduct(h) => h.parallel_insert(batch),
            AnnHnsw::Manhattan(h) => h.parallel_insert(batch),
        }
    }

    /// Toggle the `searching_mode` hint on the underlying HNSW. When `true`
    /// the graph is treated as read-only and lookups skip some bookkeeping;
    /// when `false` further inserts are allowed.
    fn set_searching_mode(&mut self, value: bool) {
        match self {
            AnnHnsw::Cosine(h) => h.set_searching_mode(value),
            AnnHnsw::Euclidean(h) => h.set_searching_mode(value),
            AnnHnsw::DotProduct(h) => h.set_searching_mode(value),
            AnnHnsw::Manhattan(h) => h.set_searching_mode(value),
        }
    }

    fn file_dump(&self, directory: &Path, basename: &str) -> Result<()> {
        let result = match self {
            AnnHnsw::Cosine(h) => h.file_dump(directory, basename),
            AnnHnsw::Euclidean(h) => h.file_dump(directory, basename),
            AnnHnsw::DotProduct(h) => h.file_dump(directory, basename),
            AnnHnsw::Manhattan(h) => h.file_dump(directory, basename),
        };
        result.map(|_| ()).map_err(|err| {
            VectLiteError::InvalidFormat(format!("failed to persist ANN index: {err}"))
        })
    }
}

struct AnnIndex {
    hnsw: AnnHnsw,
    /// `keys[i]` is the record key for HNSW origin_id `i`. Always grows; we
    /// never shrink it (HNSW doesn't support compacted deletion). Tombstoned
    /// slots stay in the vec to keep origin_id ↔ key mapping stable.
    keys: Vec<RecordKey>,
    /// Reverse index: `key → origin_id`. Lets `delete` find a record's HNSW
    /// node in O(1). Built alongside `keys` on every (re)build.
    key_to_origin: HashMap<RecordKey, usize>,
    /// Origin_ids that have been logically deleted but are still part of the
    /// HNSW graph. Search filters these out by lookup; a `compact()` rebuilds
    /// the graph once the ratio exceeds `IndexConfig.tombstone_rebuild_pct`.
    tombstones: HashSet<usize>,
}

impl AnnIndex {
    /// Number of live (non-tombstoned) records in the graph.
    fn live_count(&self) -> usize {
        self.keys.len().saturating_sub(self.tombstones.len())
    }

    /// True when the fraction of dead nodes is at or above the configured
    /// rebuild threshold (`IndexConfig.tombstone_rebuild_pct`). Currently
    /// `compact_inner` rebuilds on *any* tombstones because the persisted
    /// manifest format only tracks live record counts — when we add a
    /// tombstone-aware manifest (planned), this becomes the trigger.
    #[allow(dead_code)]
    fn should_rebuild(&self, threshold_pct: u8) -> bool {
        if self.keys.is_empty() {
            return false;
        }
        let pct = (self.tombstones.len() * 100) / self.keys.len();
        pct >= threshold_pct as usize
    }
}

struct AnnManifestEntry {
    namespace: Option<String>,
    vector_name: String,
    record_count: usize,
    key_signature: u64,
    keys: Vec<RecordKey>,
}

#[derive(Clone)]
struct ScoredRecord<'a> {
    record: &'a Record,
    score: f32,
    dense_score: f32,
    sparse_score: f32,
    vector_name: Option<String>,
    matched_terms: Vec<String>,
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
    bm25_term_scores: BTreeMap<String, f32>,
}

impl Database {
    pub fn create(path: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        Self::create_with_metric(path, dimension, DistanceMetric::Cosine)
    }

    pub fn create_with_metric(
        path: impl AsRef<Path>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<Self> {
        ensure_dimension(dimension)?;
        let lock = acquire_exclusive_lock(path.as_ref())?;

        let mut database = Self {
            path: path.as_ref().to_path_buf(),
            wal_path: wal_path(path.as_ref()),
            dimension,
            metric,
            records: BTreeMap::new(),
            ann: AnnCatalog::default(),
            sparse_index: SparseIndex::default(),
            wal_entries_replayed: 0,
            ann_loaded_from_disk: false,
            read_only: false,
            _lock_file: Some(lock),
            wal_writer: None,
            wal_sync_mode: WalSyncMode::default(),
            wal_ops_since_sync: 0,
            ann_dirty: false,
            quantized_dirty: false,
            multi_vector_quantized_dirty: false,
            quantized: None,
            quantization_config: None,
            quantized_keys: Vec::new(),
            multi_vector_quantized: BTreeMap::new(),
            multi_vector_quantization_config: BTreeMap::new(),
            multi_vector_quantized_keys: BTreeMap::new(),
            payload_index_defs: BTreeMap::new(),
            payload_indexes: BTreeMap::new(),
            index_config: IndexConfig::default(),
            vector_arena: None,
            vector_arena_dirty: false,
        };

        database.flush()?;
        Ok(database)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let lock = acquire_exclusive_lock(&path)?;
        let mut file = File::open(&path)?;
        let mut database = Self::read_from(&path, &mut file)?;
        database._lock_file = Some(lock);
        database.read_only = false;
        database.replay_wal()?;
        database.rebuild_sparse_index();
        database.ann_loaded_from_disk = database.try_load_ann_from_disk();
        if !database.ann_loaded_from_disk {
            database.rebuild_ann();
        }
        database.try_load_quantization();
        database.try_load_multi_vector_quantization();
        database.try_load_payload_index_defs();
        database.rebuild_payload_indexes();
        Ok(database)
    }

    /// Open an existing database with a lock timeout in seconds.
    /// If the lock cannot be acquired within the timeout, returns
    /// `VectLiteError::LockContention`.
    pub fn open_with_timeout(path: impl AsRef<Path>, timeout_secs: f64) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let timeout = timeout_duration(timeout_secs, "lock_timeout")?;
        let lock = acquire_exclusive_lock_with_timeout(&path, Some(timeout))?;
        let mut file = File::open(&path)?;
        let mut database = Self::read_from(&path, &mut file)?;
        database._lock_file = Some(lock);
        database.read_only = false;
        database.replay_wal()?;
        database.rebuild_sparse_index();
        database.ann_loaded_from_disk = database.try_load_ann_from_disk();
        if !database.ann_loaded_from_disk {
            database.rebuild_ann();
        }
        database.try_load_quantization();
        database.try_load_multi_vector_quantization();
        database.try_load_payload_index_defs();
        database.rebuild_payload_indexes();
        Ok(database)
    }

    /// Open an existing database in read-only mode. Acquires a shared lock
    /// so multiple readers can coexist. All write operations will return
    /// `VectLiteError::ReadOnly`.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_read_only_with_timeout(path, None)
    }

    /// Open an existing database in read-only mode, optionally waiting for a
    /// shared lock to become available.
    pub fn open_read_only_with_timeout(
        path: impl AsRef<Path>,
        timeout_secs: Option<f64>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let timeout = timeout_secs
            .map(|seconds| timeout_duration(seconds, "lock_timeout"))
            .transpose()?;
        let lock = acquire_shared_lock_with_timeout(&path, timeout)?;
        let mut file = File::open(&path)?;
        let mut database = Self::read_from(&path, &mut file)?;
        database._lock_file = Some(lock);
        database.read_only = true;
        database.replay_wal()?;
        database.rebuild_sparse_index();
        database.ann_loaded_from_disk = database.try_load_ann_from_disk();
        if !database.ann_loaded_from_disk {
            database.rebuild_ann();
        }
        database.try_load_quantization();
        database.try_load_multi_vector_quantization();
        database.try_load_payload_index_defs();
        database.rebuild_payload_indexes();
        Ok(database)
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Returns true if the database has been closed.
    pub fn is_closed(&self) -> bool {
        self._lock_file.is_none() && self.records.is_empty() && self.dimension == 0
    }

    /// Close the database: flush WAL (if writable), release the file lock,
    /// and clear all in-memory state.  After calling this, any further
    /// operation will return an error.
    pub fn close(&mut self) -> Result<()> {
        if self.is_closed() {
            return Ok(());
        }
        // Flush WAL to main file if writable
        if !self.read_only {
            self.compact_inner()?;
        }
        // Drop the cached WAL writer (also closes the underlying file handle).
        self.wal_writer = None;
        // Release the lock by dropping the file handle
        self._lock_file = None;
        // Clear in-memory state
        self.records.clear();
        self.ann = AnnCatalog::default();
        self.sparse_index = SparseIndex::default();
        self.quantized = None;
        self.quantization_config = None;
        self.quantized_keys.clear();
        self.vector_arena = None;
        self.vector_arena_dirty = false;
        self.dimension = 0;
        Ok(())
    }

    fn check_open(&self) -> Result<()> {
        if self.is_closed() {
            return Err(VectLiteError::InvalidFormat(
                "database is closed".to_owned(),
            ));
        }
        Ok(())
    }

    fn check_writable(&self) -> Result<()> {
        self.check_open()?;
        if self.read_only {
            return Err(VectLiteError::ReadOnly);
        }
        Ok(())
    }

    pub fn open_or_create(path: impl AsRef<Path>, dimension: usize) -> Result<Self> {
        Self::open_or_create_with_metric(path, dimension, DistanceMetric::Cosine)
    }

    pub fn open_or_create_with_metric(
        path: impl AsRef<Path>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Result<Self> {
        if path.as_ref().exists() {
            let database = Self::open(path)?;
            if database.dimension != dimension {
                return Err(VectLiteError::DimensionMismatch {
                    expected: database.dimension,
                    found: dimension,
                });
            }
            Ok(database)
        } else {
            Self::create_with_metric(path, dimension, metric)
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Count records, optionally filtered by namespace and/or metadata filter.
    pub fn count_filtered(
        &self,
        namespace: Option<&str>,
        filter: Option<&MetadataFilter>,
    ) -> usize {
        let now = now_epoch_secs();
        // Try to use payload indexes to narrow down candidates.
        let candidates = filter.and_then(|f| self.payload_index_candidates(f, namespace));

        if let Some(ref cand) = candidates {
            // Iterate only over candidate keys (still verify filter for safety).
            cand.iter()
                .filter_map(|key| self.records.get(key))
                .filter(|record| {
                    !record.is_expired_at(now)
                        && filter.map(|f| f.matches(&record.metadata)).unwrap_or(true)
                })
                .count()
        } else {
            self.records
                .iter()
                .filter(|((ns, _), record)| {
                    if record.is_expired_at(now) {
                        return false;
                    }
                    if let Some(target_ns) = namespace {
                        if ns != target_ns {
                            return false;
                        }
                    }
                    if let Some(filter) = filter {
                        if !filter.matches(&record.metadata) {
                            return false;
                        }
                    }
                    true
                })
                .count()
        }
    }

    /// List records by namespace and/or metadata filter without requiring a
    /// vector query.  Returns records ordered by (namespace, id).
    pub fn list(
        &self,
        namespace: Option<&str>,
        filter: Option<&MetadataFilter>,
        limit: usize,
        offset: usize,
    ) -> Vec<&Record> {
        let now = now_epoch_secs();
        // Try to use payload indexes to narrow down candidates.
        let candidates = filter.and_then(|f| self.payload_index_candidates(f, namespace));

        if let Some(ref cand) = candidates {
            // Collect into a sorted vec to maintain (namespace, id) ordering.
            let mut keys: Vec<&RecordKey> = cand.iter().collect();
            keys.sort();
            keys.iter()
                .filter_map(|key| self.records.get(*key))
                .filter(|record| {
                    !record.is_expired_at(now)
                        && filter.map(|f| f.matches(&record.metadata)).unwrap_or(true)
                })
                .skip(offset)
                .take(if limit == 0 { usize::MAX } else { limit })
                .collect()
        } else {
            self.records
                .iter()
                .filter(|((ns, _), record)| {
                    if record.is_expired_at(now) {
                        return false;
                    }
                    if let Some(target_ns) = namespace {
                        if ns != target_ns {
                            return false;
                        }
                    }
                    if let Some(filter) = filter {
                        if !filter.matches(&record.metadata) {
                            return false;
                        }
                    }
                    true
                })
                .skip(offset)
                .take(if limit == 0 { usize::MAX } else { limit })
                .map(|(_, record)| record)
                .collect()
        }
    }

    /// Cursor-based pagination over records. Returns up to `limit` records
    /// whose key is strictly greater than `after` (if provided), plus an
    /// optional next-page cursor.
    ///
    /// The cursor is an opaque `(namespace, id)` pair serialised as
    /// `"namespace\0id"`.  Callers should treat it as an opaque token.
    pub fn list_cursor(
        &self,
        namespace: Option<&str>,
        filter: Option<&MetadataFilter>,
        limit: usize,
        after: Option<&str>,
    ) -> (Vec<&Record>, Option<String>) {
        let now = now_epoch_secs();
        let limit = if limit == 0 { 100 } else { limit };

        // Decode cursor
        let after_key: Option<RecordKey> = after.map(|cursor| {
            let parts: Vec<&str> = cursor.splitn(2, '\0').collect();
            if parts.len() == 2 {
                (parts[0].to_owned(), parts[1].to_owned())
            } else {
                (String::new(), cursor.to_owned())
            }
        });

        let iter = self.records.iter();
        // If we have a cursor, skip to the first key after it.
        let iter: Box<dyn Iterator<Item = (&RecordKey, &Record)> + '_> = match &after_key {
            Some(key) => {
                // BTreeMap range starting from Excluded(key)
                use std::ops::Bound;
                Box::new(
                    self.records
                        .range((Bound::Excluded(key.clone()), Bound::Unbounded)),
                )
            }
            None => Box::new(iter),
        };

        let mut results = Vec::with_capacity(limit + 1);
        for ((ns, _id), record) in iter {
            if record.is_expired_at(now) {
                continue;
            }
            if let Some(target_ns) = namespace {
                if ns != target_ns {
                    continue;
                }
            }
            if let Some(f) = filter {
                if !f.matches(&record.metadata) {
                    continue;
                }
            }
            results.push(record);
            if results.len() > limit {
                break;
            }
        }

        let next_cursor = if results.len() > limit {
            results.pop(); // remove the extra
            results.last().map(|r| format!("{}\0{}", r.namespace, r.id))
        } else {
            None
        };

        (results, next_cursor)
    }

    /// Delete all records matching a filter, optionally within a namespace.
    /// Returns the number of records deleted.
    pub fn delete_by_filter(
        &mut self,
        namespace: Option<&str>,
        filter: &MetadataFilter,
    ) -> Result<usize> {
        self.check_writable()?;
        let keys_to_delete: Vec<(String, String)> = self
            .records
            .iter()
            .filter(|((ns, _), record)| {
                if let Some(target_ns) = namespace {
                    if ns != target_ns {
                        return false;
                    }
                }
                filter.matches(&record.metadata)
            })
            .map(|(key, _)| key.clone())
            .collect();

        let count = keys_to_delete.len();
        if count == 0 {
            return Ok(0);
        }

        let ops: Vec<WalOp> = keys_to_delete
            .into_iter()
            .map(|(namespace, id)| WalOp::Delete { namespace, id })
            .collect();
        self.apply_wal_batch(ops)?;
        Ok(count)
    }

    /// Merge a metadata patch into an existing record without re-writing the
    /// vector.  Keys present in `metadata` overwrite existing keys; keys not
    /// in the patch are left untouched.
    ///
    /// Returns `true` if the record exists and was updated, `false` if the
    /// record was not found (no error is raised).
    pub fn update_metadata(&mut self, id: impl Into<String>, metadata: Metadata) -> Result<bool> {
        self.update_metadata_in_namespace(DEFAULT_NAMESPACE, id, metadata)
    }

    /// Merge a metadata patch into an existing record in the given namespace.
    /// See [`update_metadata`](Self::update_metadata) for details.
    pub fn update_metadata_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        metadata: Metadata,
    ) -> Result<bool> {
        self.check_writable()?;
        let namespace = namespace.into();
        let id = id.into();
        let key = (namespace.clone(), id.clone());
        if !self.records.contains_key(&key) {
            return Ok(false);
        }
        self.apply_wal_batch(vec![WalOp::UpdateMetadata {
            namespace,
            id,
            metadata,
        }])?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // TTL / Expiry API
    // -----------------------------------------------------------------------

    /// Set a time-to-live on a record. The TTL is expressed as seconds from now.
    /// After `ttl_secs` seconds the record will be excluded from reads and
    /// garbage-collected on the next `compact()`.
    ///
    /// Returns `true` if the record was found, `false` otherwise.
    pub fn set_ttl(&mut self, id: &str, ttl_secs: f64) -> Result<bool> {
        self.set_ttl_in_namespace(DEFAULT_NAMESPACE, id, ttl_secs)
    }

    /// Set a time-to-live on a record in a specific namespace.
    pub fn set_ttl_in_namespace(
        &mut self,
        namespace: &str,
        id: &str,
        ttl_secs: f64,
    ) -> Result<bool> {
        self.check_writable()?;
        if ttl_secs < 0.0 || ttl_secs.is_nan() {
            return Err(VectLiteError::InvalidFormat(
                "ttl_secs must be a non-negative finite number".to_owned(),
            ));
        }
        let key = (namespace.to_owned(), id.to_owned());
        if !self.records.contains_key(&key) {
            return Ok(false);
        }
        let expires_at = Some(now_epoch_secs() + ttl_secs);
        self.apply_wal_batch(vec![WalOp::SetTtl {
            namespace: namespace.to_owned(),
            id: id.to_owned(),
            expires_at,
        }])?;
        Ok(true)
    }

    /// Remove the TTL from a record so it never expires.
    /// Returns `true` if the record was found, `false` otherwise.
    pub fn clear_ttl(&mut self, id: &str) -> Result<bool> {
        self.clear_ttl_in_namespace(DEFAULT_NAMESPACE, id)
    }

    /// Remove the TTL from a record in a specific namespace.
    pub fn clear_ttl_in_namespace(&mut self, namespace: &str, id: &str) -> Result<bool> {
        self.check_writable()?;
        let key = (namespace.to_owned(), id.to_owned());
        if !self.records.contains_key(&key) {
            return Ok(false);
        }
        self.apply_wal_batch(vec![WalOp::SetTtl {
            namespace: namespace.to_owned(),
            id: id.to_owned(),
            expires_at: None,
        }])?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Payload index management API
    // -----------------------------------------------------------------------

    /// Create a payload index on a metadata field.
    ///
    /// - `field` — the top-level metadata key to index (e.g. `"category"`, `"price"`).
    /// - `index_type` — `PayloadIndexType::Keyword` or `PayloadIndexType::Numeric`.
    ///
    /// The index is populated immediately from all existing records. Subsequent
    /// mutations (upsert, delete, update_metadata) maintain the index incrementally.
    ///
    /// Returns `true` if the index was created, `false` if an index already exists
    /// for this field.
    pub fn create_index(
        &mut self,
        field: impl Into<String>,
        index_type: PayloadIndexType,
    ) -> Result<bool> {
        self.check_writable()?;
        let field = field.into();
        if self.payload_index_defs.contains_key(&field) {
            return Ok(false);
        }
        self.payload_index_defs.insert(field.clone(), index_type);

        // Build the index from existing records.
        let data = match index_type {
            PayloadIndexType::Keyword => {
                let mut kw = KeywordIndex::default();
                for (key, record) in &self.records {
                    if let Some(MetadataValue::String(val)) = record.metadata.get(&field) {
                        kw.insert(val, key.clone());
                    }
                }
                PayloadIndexData::Keyword(kw)
            }
            PayloadIndexType::Numeric => {
                let mut num = NumericIndex::default();
                for (key, record) in &self.records {
                    if let Some(val) = record
                        .metadata
                        .get(&field)
                        .and_then(MetadataValue::as_number)
                    {
                        num.insert(val, key.clone());
                    }
                }
                PayloadIndexData::Numeric(num)
            }
        };
        self.payload_indexes.insert(field, data);
        self.persist_payload_index_defs()?;
        Ok(true)
    }

    /// Drop a payload index on a metadata field.
    ///
    /// Returns `true` if the index existed and was removed, `false` if there was
    /// no index on this field.
    pub fn drop_index(&mut self, field: &str) -> Result<bool> {
        self.check_writable()?;
        if self.payload_index_defs.remove(field).is_none() {
            return Ok(false);
        }
        self.payload_indexes.remove(field);
        self.persist_payload_index_defs()?;
        Ok(true)
    }

    /// List all payload indexes as `(field, type_name)` pairs.
    pub fn list_indexes(&self) -> Vec<(String, PayloadIndexType)> {
        self.payload_index_defs
            .iter()
            .map(|(field, index_type)| (field.clone(), *index_type))
            .collect()
    }

    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
    ) -> Result<()> {
        self.insert_with_vectors_in_namespace(
            DEFAULT_NAMESPACE,
            id,
            vector,
            NamedVectors::new(),
            SparseVector::new(),
            metadata,
        )
    }

    pub fn upsert(
        &mut self,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
    ) -> Result<()> {
        self.insert(id, vector, metadata)
    }

    pub fn insert_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
    ) -> Result<()> {
        self.insert_with_vectors_in_namespace(
            namespace,
            id,
            vector,
            NamedVectors::new(),
            SparseVector::new(),
            metadata,
        )
    }

    pub fn insert_with_vectors_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        vectors: NamedVectors,
        sparse: SparseVector,
        metadata: Metadata,
    ) -> Result<()> {
        self.check_writable()?;
        let record = self.record_from_parts(namespace, id, vector, vectors, sparse, metadata)?;
        let key = (record.namespace.clone(), record.id.clone());
        if self.records.contains_key(&key) {
            return Err(VectLiteError::DuplicateId {
                namespace: key.0,
                id: key.1,
            });
        }
        self.apply_wal_batch(vec![WalOp::Upsert(record)])?;
        Ok(())
    }

    pub fn insert_with_sparse_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        sparse: SparseVector,
        metadata: Metadata,
    ) -> Result<()> {
        self.insert_with_vectors_in_namespace(
            namespace,
            id,
            vector,
            NamedVectors::new(),
            sparse,
            metadata,
        )
    }

    pub fn upsert_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
    ) -> Result<()> {
        self.upsert_with_vectors_in_namespace(
            namespace,
            id,
            vector,
            NamedVectors::new(),
            SparseVector::new(),
            metadata,
        )
    }

    pub fn upsert_with_sparse_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        sparse: SparseVector,
        metadata: Metadata,
    ) -> Result<()> {
        self.upsert_with_vectors_in_namespace(
            namespace,
            id,
            vector,
            NamedVectors::new(),
            sparse,
            metadata,
        )
    }

    pub fn upsert_with_vectors_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        vectors: NamedVectors,
        sparse: SparseVector,
        metadata: Metadata,
    ) -> Result<()> {
        self.check_writable()?;
        let record = self.record_from_parts(namespace, id, vector, vectors, sparse, metadata)?;
        self.apply_wal_batch(vec![WalOp::Upsert(record)])?;
        Ok(())
    }

    pub fn insert_many<I>(&mut self, records: I) -> Result<usize>
    where
        I: IntoIterator<Item = Record>,
    {
        self.check_writable()?;
        let records = records
            .into_iter()
            .map(|record| {
                self.validate_record(&record)?;
                let key = (record.namespace.clone(), record.id.clone());
                if self.records.contains_key(&key) {
                    return Err(VectLiteError::DuplicateId {
                        namespace: key.0,
                        id: key.1,
                    });
                }
                Ok(record)
            })
            .collect::<Result<Vec<_>>>()?;

        let count = records.len();
        if count == 0 {
            return Ok(0);
        }

        self.apply_wal_batch_deferred(records.into_iter().map(WalOp::Upsert).collect())?;
        self.rebuild_sparse_index();
        self.rebuild_ann();
        self.ann_loaded_from_disk = false;
        self.persist_ann_to_disk()?;
        self.ann_dirty = false;
        self.vector_arena_dirty = true;
        self.rebuild_quantized_index();
        self.quantized_dirty = false;
        self.rebuild_all_multi_vector_quantized_indexes();
        self.multi_vector_quantized_dirty = false;
        Ok(count)
    }

    pub fn upsert_many<I>(&mut self, records: I) -> Result<usize>
    where
        I: IntoIterator<Item = Record>,
    {
        self.check_writable()?;
        let records = records
            .into_iter()
            .map(|record| {
                self.validate_record(&record)?;
                Ok(record)
            })
            .collect::<Result<Vec<_>>>()?;

        let count = records.len();
        if count == 0 {
            return Ok(0);
        }

        self.apply_wal_batch_deferred(records.into_iter().map(WalOp::Upsert).collect())?;
        self.rebuild_sparse_index();
        self.rebuild_ann();
        self.ann_loaded_from_disk = false;
        self.persist_ann_to_disk()?;
        self.ann_dirty = false;
        self.vector_arena_dirty = true;
        self.rebuild_quantized_index();
        self.quantized_dirty = false;
        self.rebuild_all_multi_vector_quantized_indexes();
        self.multi_vector_quantized_dirty = false;
        Ok(count)
    }

    pub fn get(&self, id: &str) -> Option<&Record> {
        self.get_in_namespace(DEFAULT_NAMESPACE, id)
    }

    pub fn get_in_namespace(&self, namespace: &str, id: &str) -> Option<&Record> {
        let now = now_epoch_secs();
        self.records
            .get(&(namespace.to_owned(), id.to_owned()))
            .filter(|record| !record.is_expired_at(now))
    }

    pub fn delete(&mut self, id: &str) -> Result<bool> {
        self.delete_in_namespace(DEFAULT_NAMESPACE, id)
    }

    pub fn delete_in_namespace(&mut self, namespace: &str, id: &str) -> Result<bool> {
        self.check_writable()?;
        let removed = self
            .records
            .contains_key(&(namespace.to_owned(), id.to_owned()));
        if removed {
            self.apply_wal_batch(vec![WalOp::Delete {
                namespace: namespace.to_owned(),
                id: id.to_owned(),
            }])?;
        }
        Ok(removed)
    }

    pub fn delete_many<I, S>(&mut self, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.delete_many_in_namespace(DEFAULT_NAMESPACE, ids)
    }

    pub fn delete_many_in_namespace<I, S>(&mut self, namespace: &str, ids: I) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.check_writable()?;
        let mut removed = 0;
        let mut ops = Vec::new();

        for id in ids {
            if self
                .records
                .contains_key(&(namespace.to_owned(), id.as_ref().to_owned()))
            {
                removed += 1;
                ops.push(WalOp::Delete {
                    namespace: namespace.to_owned(),
                    id: id.as_ref().to_owned(),
                });
            }
        }

        if removed > 0 {
            self.apply_wal_batch(ops)?;
        }

        Ok(removed)
    }

    pub fn search(&self, query: &[f32], options: SearchOptions) -> Result<Vec<SearchResult>> {
        self.search_in_namespace(DEFAULT_NAMESPACE, query, options)
    }

    pub fn search_in_namespace(
        &self,
        namespace: &str,
        query: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .hybrid_search_in_namespace_with_stats(
                namespace,
                Some(query),
                None,
                HybridSearchOptions {
                    top_k: options.top_k,
                    filter: options.filter,
                    truncate_dim: options.truncate_dim,
                    dense_weight: 1.0,
                    sparse_weight: 0.0,
                    ..HybridSearchOptions::default()
                },
            )?
            .results)
    }

    pub fn search_all_namespaces(
        &self,
        query: &[f32],
        options: SearchOptions,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .hybrid_search_all_namespaces_with_stats(
                Some(query),
                None,
                HybridSearchOptions {
                    top_k: options.top_k,
                    filter: options.filter,
                    truncate_dim: options.truncate_dim,
                    dense_weight: 1.0,
                    sparse_weight: 0.0,
                    ..HybridSearchOptions::default()
                },
            )?
            .results)
    }

    pub fn hybrid_search_in_namespace(
        &self,
        namespace: &str,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: HybridSearchOptions,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .hybrid_search_in_namespace_with_stats(namespace, dense_query, sparse_query, options)?
            .results)
    }

    pub fn hybrid_search_in_namespace_with_stats(
        &self,
        namespace: &str,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: HybridSearchOptions,
    ) -> Result<SearchOutcome> {
        self.hybrid_search_internal(dense_query, sparse_query, options, Some(namespace))
    }

    pub fn hybrid_search_all_namespaces(
        &self,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: HybridSearchOptions,
    ) -> Result<Vec<SearchResult>> {
        Ok(self
            .hybrid_search_all_namespaces_with_stats(dense_query, sparse_query, options)?
            .results)
    }

    pub fn hybrid_search_all_namespaces_with_stats(
        &self,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: HybridSearchOptions,
    ) -> Result<SearchOutcome> {
        self.hybrid_search_internal(dense_query, sparse_query, options, None)
    }

    pub fn namespaces(&self) -> Vec<String> {
        self.records
            .keys()
            .map(|(namespace, _)| namespace.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn wal_path(&self) -> &Path {
        &self.wal_path
    }

    pub fn apply_operations(&mut self, operations: Vec<WriteOperation>) -> Result<()> {
        self.check_writable()?;
        let ops = operations
            .into_iter()
            .map(|operation| match operation {
                WriteOperation::Insert(record) => {
                    self.validate_record(&record)?;
                    let key = (record.namespace.clone(), record.id.clone());
                    if self.records.contains_key(&key) {
                        return Err(VectLiteError::DuplicateId {
                            namespace: key.0,
                            id: key.1,
                        });
                    }
                    Ok(WalOp::Upsert(record))
                }
                WriteOperation::Upsert(record) => {
                    self.validate_record(&record)?;
                    Ok(WalOp::Upsert(record))
                }
                WriteOperation::Delete { namespace, id } => Ok(WalOp::Delete { namespace, id }),
            })
            .collect::<Result<Vec<_>>>()?;
        if ops.is_empty() {
            return Ok(());
        }
        self.apply_wal_batch_deferred(ops)?;
        self.rebuild_sparse_index();
        self.rebuild_ann();
        self.ann_loaded_from_disk = false;
        self.persist_ann_to_disk()?;
        self.ann_dirty = false;
        self.vector_arena_dirty = true;
        self.rebuild_quantized_index();
        self.quantized_dirty = false;
        self.rebuild_all_multi_vector_quantized_indexes();
        self.multi_vector_quantized_dirty = false;
        Ok(())
    }

    fn hybrid_search_internal(
        &self,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: HybridSearchOptions,
        namespace: Option<&str>,
    ) -> Result<SearchOutcome> {
        self.check_open()?;
        let effective_dimension =
            self.resolve_dense_search_dimension(dense_query, options.truncate_dim)?;
        if dense_query.is_none() && sparse_query.is_none() {
            return Err(VectLiteError::InvalidFormat(
                "search requires a dense query, a sparse query, or both".to_owned(),
            ));
        }
        // Reject zero-norm query vectors for metrics where similarity is undefined.
        if let Some(query) = dense_query {
            if self.metric.is_similarity() {
                let norm_sq: f32 = query.iter().map(|v| v * v).sum();
                if norm_sq == 0.0 {
                    return Err(VectLiteError::InvalidFormat(
                        "query vector has zero norm; cosine/dot-product similarity is undefined"
                            .to_owned(),
                    ));
                }
            }
        }
        if let Some(mmr_lambda) = options.mmr_lambda {
            if !(0.0..=1.0).contains(&mmr_lambda) {
                return Err(VectLiteError::InvalidFormat(
                    "mmr_lambda must be between 0.0 and 1.0".to_owned(),
                ));
            }
        }
        if let Some(vector_name) = options.vector_name.as_deref() {
            if vector_name.is_empty() {
                return Err(VectLiteError::InvalidFormat(
                    "vector_name must not be empty".to_owned(),
                ));
            }
        }

        let search_start = Instant::now();

        let top_k = if options.top_k == 0 {
            self.records.len()
        } else {
            options.top_k
        };
        let fetch_k = resolve_fetch_k(
            top_k,
            options.fetch_k,
            self.records.len(),
            options.mmr_lambda,
        );
        let vector_name = options.vector_name.as_deref();
        let matryoshka_truncated = effective_dimension
            .map(|dimension| dimension < self.dimension)
            .unwrap_or(false);

        let dense_start = Instant::now();
        // Use quantized index for candidate selection if available (2-stage pipeline).
        // The quantized index operates on the default vector only and globally (not per-namespace).
        let quantized_candidates = if !matryoshka_truncated
            && (vector_name.is_none() || vector_name == Some(DEFAULT_VECTOR_NAME))
        {
            dense_query.and_then(|query| self.quantized_candidate_keys(query, fetch_k))
        } else {
            None
        };
        let ann_candidates = if quantized_candidates.is_some() {
            // Skip HNSW if quantized index provided candidates
            None
        } else {
            dense_query
                .filter(|_| !matryoshka_truncated)
                .and_then(|query| self.ann_candidate_keys(namespace, vector_name, query, fetch_k))
        };
        let effective_dense_candidates = quantized_candidates.or(ann_candidates);
        let dense_us = dense_start.elapsed().as_micros() as u64;

        let sparse_start = Instant::now();
        let sparse_candidates = sparse_query
            .map(|query| self.sparse_candidate_keys(namespace, query, fetch_k))
            .unwrap_or_default();
        let sparse_us = sparse_start.elapsed().as_micros() as u64;

        let candidate_keys = if dense_query.is_none() {
            Some(sparse_candidates.clone())
        } else if dense_query.is_some() && effective_dense_candidates.is_none() {
            None
        } else {
            merge_candidate_keys(
                effective_dense_candidates.as_deref(),
                Some(sparse_candidates.as_slice()),
            )
        };

        // Use payload indexes to narrow candidates when doing a full scan.
        let payload_candidates = options
            .filter
            .as_ref()
            .and_then(|f| self.payload_index_candidates(f, namespace));
        let candidate_keys = match (candidate_keys, payload_candidates) {
            (Some(ck), Some(pc)) => {
                // Intersect ANN/sparse candidates with payload index candidates.
                Some(
                    ck.into_iter()
                        .filter(|k| pc.contains(k))
                        .collect::<Vec<_>>(),
                )
            }
            (None, Some(pc)) => {
                // No ANN candidates but payload index narrowed the set.
                Some(pc.into_iter().collect::<Vec<_>>())
            }
            (ck, None) => ck,
        };

        let mut stats = SearchStats {
            used_ann: effective_dense_candidates.is_some(),
            ann_candidate_count: effective_dense_candidates.as_ref().map_or(0, Vec::len),
            fetch_k,
            sparse_candidate_count: sparse_candidates.len(),
            ann_loaded_from_disk: self.ann_loaded_from_disk,
            wal_entries_replayed: self.wal_entries_replayed,
            fusion: options.fusion.label().to_owned(),
            effective_dimension: effective_dimension.unwrap_or(0),
            matryoshka_truncated,
            ..SearchStats::default()
        };

        let mut results = self.collect_results(
            dense_query,
            sparse_query,
            &options,
            namespace,
            candidate_keys.as_deref(),
            effective_dimension,
        );
        stats.considered_count = results.len();

        if effective_dense_candidates.is_some() && results.len() < fetch_k {
            stats.exact_fallback = true;
            results = self.collect_results(
                dense_query,
                sparse_query,
                &options,
                namespace,
                None,
                effective_dimension,
            );
            stats.considered_count = results.len();
        }

        let fusion_start = Instant::now();
        apply_rank_metadata(&mut results);
        apply_fusion_strategy(
            &mut results,
            &options.fusion,
            options.dense_weight,
            options.sparse_weight,
        );
        sort_scored_records(&mut results);
        let fusion_us = fusion_start.elapsed().as_micros() as u64;

        let mmr_applied = options.mmr_lambda.is_some() && top_k > 1 && results.len() > 1;
        let results = if let Some(mmr_lambda) = options.mmr_lambda {
            apply_mmr(
                results,
                top_k,
                mmr_lambda,
                options.dense_weight,
                options.sparse_weight,
                vector_name,
                self.metric,
                effective_dimension,
            )
        } else {
            let mut results = results;
            results.truncate(top_k);
            results
        };
        stats.mmr_applied = mmr_applied;

        let total_us = search_start.elapsed().as_micros() as u64;
        stats.timings = SearchTimings {
            dense_us,
            sparse_us,
            fusion_us,
            total_us,
        };

        Ok(SearchOutcome {
            results: results
                .into_iter()
                .map(ScoredRecord::into_search_result)
                .collect(),
            stats,
        })
    }

    pub fn flush(&mut self) -> Result<()> {
        self.check_writable()?;
        self.compact_inner()
    }

    /// Configure WAL durability. See [`WalSyncMode`] for the safety / speed
    /// tradeoffs.
    ///
    /// Switching to a more relaxed mode while there are unsync'd bytes in
    /// the WAL is safe — the bytes simply stay in the BufWriter / OS cache
    /// until the next sync point (`flush()`, `compact()`, `close()`, or the
    /// counter reaching `EveryN(n)`). Switching to a *stricter* mode forces
    /// an immediate sync so there is no surprise loss window.
    pub fn set_wal_sync_mode(&mut self, mode: WalSyncMode) -> Result<()> {
        self.check_writable()?;
        mode.validate()?;
        let previous = self.wal_sync_mode;
        self.wal_sync_mode = mode;
        // If we just tightened durability (e.g. moved from OnFlush back to
        // PerOp) and there are pending ops, sync immediately so the user's
        // mental model — "after this call any acknowledged write is durable"
        // — holds.
        let became_stricter = matches!(
            (previous, mode),
            (
                WalSyncMode::OnFlush,
                WalSyncMode::PerOp | WalSyncMode::EveryN(_)
            ) | (WalSyncMode::EveryN(_), WalSyncMode::PerOp)
        );
        if became_stricter && self.wal_ops_since_sync > 0 {
            self.sync_wal()?;
            self.wal_ops_since_sync = 0;
        }
        Ok(())
    }

    /// Return the current WAL sync mode.
    pub fn wal_sync_mode(&self) -> WalSyncMode {
        self.wal_sync_mode
    }

    /// Materialise the contiguous-vector arena up front.
    ///
    /// The arena mirrors the default dense vector of every record in a
    /// single flat `Vec<f32>` — much more cache- and SIMD-friendly than the
    /// default `BTreeMap<Record>` layout. It's normally built lazily on
    /// first use, but if you know a heavy brute-force or rescoring scan is
    /// coming you can pay the build cost up front by calling this. Cheap
    /// when already fresh.
    pub fn prepare_for_scan(&mut self) {
        let _ = self.ensure_vector_arena();
    }

    /// Number of vectors in the contiguous arena, or `None` if the arena
    /// hasn't been materialised yet for this session. Useful for tests and
    /// observability.
    pub fn vector_arena_len(&self) -> Option<usize> {
        self.vector_arena.as_ref().map(VectorArena::len)
    }

    /// Return (live_count, tombstoned_count) summed across every HNSW graph
    /// (global + per-namespace). Useful for monitoring when a `compact()`
    /// would benefit from rebuilding the graph(s).
    pub fn tombstone_stats(&self) -> (usize, usize) {
        let mut live = 0usize;
        let mut dead = 0usize;
        for idx in self.ann.global.values() {
            live += idx.live_count();
            dead += idx.tombstones.len();
        }
        for indexes in self.ann.namespaces.values() {
            for idx in indexes.values() {
                live += idx.live_count();
                dead += idx.tombstones.len();
            }
        }
        (live, dead)
    }

    /// Bulk-ingest many records efficiently. WAL writes happen in batches of
    /// `batch_size`, but the ANN index and sparse index are only rebuilt once
    /// at the very end, making this much faster than `upsert_many` for large
    /// imports.
    ///
    /// Performance notes:
    /// - The WAL is written without a per-batch `fsync` (each batch goes
    ///   through `BufWriter` and is appended to the open file). A single
    ///   `sync_all` is issued at the end. This avoids the per-batch fsync
    ///   tax that dominates ingestion latency on macOS and modern SSDs.
    /// - The final ANN rebuild uses parallel HNSW insertion (Rayon) when
    ///   the dataset is large enough (see
    ///   `IndexConfig.parallel_insert_threshold`).
    pub fn bulk_ingest<I>(&mut self, records: I, batch_size: usize) -> Result<usize>
    where
        I: IntoIterator<Item = Record>,
    {
        self.bulk_ingest_with_config(records, batch_size, None)
    }

    /// Bulk-ingest with an override for the HNSW index configuration. The
    /// override is applied for the rebuild step at the end, so the resulting
    /// graph uses the requested `m` / `ef_construction`. The new config is
    /// also stored on the database (so subsequent searches use the
    /// corresponding `ef_search`).
    pub fn bulk_ingest_with_config<I>(
        &mut self,
        records: I,
        batch_size: usize,
        config: Option<IndexConfig>,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = Record>,
    {
        self.check_writable()?;
        if let Some(cfg) = config {
            cfg.validate()?;
            self.index_config = cfg;
        }
        let batch_size = batch_size.max(1);
        let mut total = 0_usize;
        let mut batch = Vec::with_capacity(batch_size);

        for record in records {
            self.validate_record(&record)?;
            batch.push(WalOp::Upsert(record));

            if batch.len() >= batch_size {
                total += batch.len();
                // Coalesced WAL writes: append without per-batch fsync.
                self.append_wal_batch_unsynced(&batch)?;
                self.apply_ops_in_memory(batch);
                batch = Vec::with_capacity(batch_size);
            }
        }

        if !batch.is_empty() {
            total += batch.len();
            self.append_wal_batch_unsynced(&batch)?;
            self.apply_ops_in_memory(batch);
        }

        if total > 0 {
            // Single fsync at the very end to make all batches durable in
            // one shot. This is the major ingestion optimisation: instead
            // of paying fsync per batch (every `batch_size` records) we pay
            // it once for the whole bulk_ingest call.
            self.sync_wal()?;
            self.rebuild_sparse_index();
            self.rebuild_ann();
            self.ann_loaded_from_disk = false;
            // Persist the freshly-built ANN so a subsequent reopen can skip
            // the rebuild — bulk_ingest is a "batch" operation and callers
            // expect index state to be on disk afterwards.
            self.persist_ann_to_disk()?;
            self.ann_dirty = false;
            self.vector_arena_dirty = true;
            self.rebuild_quantized_index();
            self.quantized_dirty = false;
            self.rebuild_all_multi_vector_quantized_indexes();
            self.multi_vector_quantized_dirty = false;
        }

        Ok(total)
    }

    /// Replace the HNSW tuning parameters and rebuild the ANN index.
    /// Use this to trade off recall vs latency without re-ingesting data.
    pub fn set_index_config(&mut self, config: IndexConfig) -> Result<()> {
        self.check_writable()?;
        config.validate()?;
        let changed_build_params = self.index_config.m != config.m
            || self.index_config.ef_construction != config.ef_construction;
        self.index_config = config;
        if changed_build_params {
            // m / ef_construction affect graph structure → full rebuild.
            self.rebuild_ann();
            self.ann_loaded_from_disk = false;
            self.persist_ann_to_disk()?;
            self.ann_dirty = false;
        }
        Ok(())
    }

    /// Return the current HNSW tuning parameters.
    pub fn index_config(&self) -> IndexConfig {
        self.index_config
    }

    /// Convenience: update only the query-time `ef_search` without rebuilding
    /// the index. Higher = better recall, slower search.
    pub fn set_ef_search(&mut self, ef_search: Option<usize>) -> Result<()> {
        if let Some(ef) = ef_search {
            if ef == 0 {
                return Err(VectLiteError::InvalidFormat(
                    "ef_search must be >= 1".to_owned(),
                ));
            }
        }
        self.index_config.ef_search = ef_search;
        Ok(())
    }

    pub fn compact(&mut self) -> Result<()> {
        self.check_writable()?;
        self.compact_inner()
    }

    // -----------------------------------------------------------------------
    // Quantization API
    // -----------------------------------------------------------------------

    /// Enable quantization on this database. Trains the quantizer on all current
    /// vectors and persists the configuration. Subsequent searches will use the
    /// quantized index for fast candidate selection followed by exact rescoring.
    pub fn enable_quantization(&mut self, config: QuantizationConfig) -> Result<()> {
        self.check_writable()?;
        if self.records.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "cannot enable quantization on an empty database".to_owned(),
            ));
        }
        validate_quantization_config(&config, self.dimension)?;
        self.quantization_config = Some(config);
        self.rebuild_quantized_index();
        self.quantized_dirty = false;
        self.persist_quantization_params()?;
        Ok(())
    }

    /// Disable quantization and remove persisted parameters.
    pub fn disable_quantization(&mut self) -> Result<()> {
        self.check_writable()?;
        self.quantized = None;
        self.quantization_config = None;
        self.quantized_keys.clear();
        self.quantized_dirty = false;
        // Remove the sidecar file
        let params_path = quantization_params_path(&self.path);
        if params_path.exists() {
            fs::remove_file(&params_path)?;
        }
        Ok(())
    }

    /// Returns true if quantization is enabled.
    pub fn is_quantized(&self) -> bool {
        self.quantized.is_some()
    }

    /// Returns the quantization configuration if enabled.
    pub fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    /// Returns all valid Product Quantization `num_sub_vectors` values for this database.
    pub fn valid_num_sub_vectors(&self) -> Vec<usize> {
        valid_product_num_sub_vectors(self.dimension)
    }

    /// Rebuild the quantized index from current records.
    fn rebuild_quantized_index(&mut self) {
        let config = match &self.quantization_config {
            Some(config) => config.clone(),
            None => return,
        };

        if self.records.is_empty() {
            self.quantized = None;
            self.quantized_keys.clear();
            return;
        }

        let mut keys = Vec::with_capacity(self.records.len());
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(self.records.len());

        for (key, record) in &self.records {
            keys.push(key.clone());
            vectors.push(record.vector.clone());
        }

        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        let index = QuantizedIndex::build(&refs, self.dimension, &config);

        self.quantized = Some(index);
        self.quantized_keys = keys;
    }

    /// Persist quantization parameters to a sidecar file.
    fn persist_quantization_params(&self) -> Result<()> {
        let params_path = quantization_params_path(&self.path);
        if let Some(index) = &self.quantized {
            let mut file = File::create(&params_path)?;
            index.write_params(&mut file).map_err(VectLiteError::Io)?;
            file.sync_all()?;
        } else {
            if params_path.exists() {
                fs::remove_file(&params_path)?;
            }
        }
        Ok(())
    }

    /// Try to load quantization parameters from disk and rebuild codes.
    fn try_load_quantization(&mut self) -> bool {
        let params_path = quantization_params_path(&self.path);
        if !params_path.exists() {
            return false;
        }

        let file = match File::open(&params_path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut reader = BufReader::new(file);
        let mut index = match QuantizedIndex::read_params(&mut reader) {
            Ok(idx) => idx,
            Err(_) => return false,
        };

        // Rebuild codes from current records
        let mut keys = Vec::with_capacity(self.records.len());
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(self.records.len());
        for (key, record) in &self.records {
            keys.push(key.clone());
            vectors.push(record.vector.clone());
        }
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
        index.rebuild_codes(&refs);

        self.quantization_config = Some(index.config());
        self.quantized = Some(index);
        self.quantized_keys = keys;
        true
    }

    /// Use the quantized index to get candidate record keys for rescoring.
    fn quantized_candidate_keys(&self, query: &[f32], top_k: usize) -> Option<Vec<RecordKey>> {
        let index = self.quantized.as_ref()?;
        if index.count() == 0 {
            return None;
        }

        let candidate_indices = index.search_candidates_with_metric(query, top_k, self.metric);
        Some(
            candidate_indices
                .into_iter()
                .filter_map(|idx| self.quantized_keys.get(idx).cloned())
                .collect(),
        )
    }

    // -----------------------------------------------------------------------
    // Multi-vector (ColBERT / late interaction) API
    // -----------------------------------------------------------------------

    /// Upsert a record with multi-vector (token-level) embeddings for late interaction.
    pub fn upsert_multi_vectors(
        &mut self,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
        multi_vectors: MultiVectors,
    ) -> Result<()> {
        self.upsert_multi_vectors_in_namespace(
            DEFAULT_NAMESPACE,
            id,
            vector,
            metadata,
            multi_vectors,
        )
    }

    /// Upsert a record with multi-vectors in a specific namespace.
    pub fn upsert_multi_vectors_in_namespace(
        &mut self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        metadata: Metadata,
        multi_vectors: MultiVectors,
    ) -> Result<()> {
        self.check_writable()?;
        let record = self.record_from_parts_full(
            namespace,
            id,
            vector,
            NamedVectors::new(),
            SparseVector::new(),
            metadata,
            multi_vectors,
        )?;
        self.apply_wal_batch(vec![WalOp::Upsert(record)])?;
        Ok(())
    }

    /// Search using multi-vector late interaction (MaxSim) scoring.
    ///
    /// `query_tokens` are the token-level embeddings from the query encoder
    /// (e.g. ColBERT query encoder output).
    /// `space` identifies which multi-vector space to search in.
    pub fn search_multi_vector(
        &self,
        space: &str,
        query_tokens: &[Vec<f32>],
        options: MultiVectorSearchOptions,
    ) -> Result<Vec<MultiVectorSearchResult>> {
        self.check_open()?;
        if space.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "multi-vector space name must not be empty".to_owned(),
            ));
        }
        if query_tokens.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "query_tokens must not be empty".to_owned(),
            ));
        }

        let top_k = if options.top_k == 0 {
            self.records.len()
        } else {
            options.top_k
        };
        let namespace = options.namespace.as_deref();

        // Try quantized multi-vector search first for candidate selection
        let query_refs: Vec<&[f32]> = query_tokens.iter().map(Vec::as_slice).collect();
        let candidate_keys: Option<Vec<RecordKey>> =
            self.multi_vector_quantized.get(space).and_then(|index| {
                let keys = self.multi_vector_quantized_keys.get(space)?;
                let candidate_indices = index.search(&query_refs, top_k);
                Some(
                    candidate_indices
                        .into_iter()
                        .filter_map(|idx| keys.get(idx).cloned())
                        .collect(),
                )
            });

        // Score all candidates with exact MaxSim
        let now = now_epoch_secs();
        let record_iter: Box<dyn Iterator<Item = &Record> + '_> = match &candidate_keys {
            Some(keys) => Box::new(keys.iter().filter_map(|key| self.records.get(key))),
            None => Box::new(self.records.values()),
        };

        let mut scored: Vec<(f32, &Record)> = record_iter
            .filter(|record| {
                !record.is_expired_at(now)
                    && namespace.map(|ns| record.namespace == ns).unwrap_or(true)
                    && record.multi_vectors.contains_key(space)
                    && options
                        .filter
                        .as_ref()
                        .map(|f| f.matches(&record.metadata))
                        .unwrap_or(true)
            })
            .map(|record| {
                let doc_tokens = &record.multi_vectors[space];
                let score = maxsim_score(&query_refs, doc_tokens, self.metric);
                (score, record)
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(score, record)| MultiVectorSearchResult {
                namespace: record.namespace.clone(),
                id: record.id.clone(),
                score,
                metadata: record.metadata.clone(),
            })
            .collect())
    }

    /// Enable 2-bit quantization for a multi-vector space to accelerate
    /// ColBERT-style MaxSim search. Trains the quantizer on all current
    /// token vectors in the given space.
    pub fn enable_multi_vector_quantization(
        &mut self,
        space: &str,
        config: MultiVectorQuantizationConfig,
    ) -> Result<()> {
        self.check_writable()?;
        if space.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "multi-vector space name must not be empty".to_owned(),
            ));
        }

        self.multi_vector_quantization_config
            .insert(space.to_owned(), config);
        self.rebuild_multi_vector_quantized_index(space);
        self.persist_multi_vector_quantization_params(space)?;
        Ok(())
    }

    /// Disable multi-vector quantization for a space.
    pub fn disable_multi_vector_quantization(&mut self, space: &str) -> Result<()> {
        self.check_writable()?;
        self.multi_vector_quantized.remove(space);
        self.multi_vector_quantization_config.remove(space);
        self.multi_vector_quantized_keys.remove(space);
        let params_path = multi_vector_quantization_params_path(&self.path, space);
        if params_path.exists() {
            fs::remove_file(&params_path)?;
        }
        Ok(())
    }

    /// Returns true if multi-vector quantization is enabled for a given space.
    pub fn is_multi_vector_quantized(&self, space: &str) -> bool {
        self.multi_vector_quantized.contains_key(space)
    }

    fn rebuild_multi_vector_quantized_index(&mut self, space: &str) {
        let config = match self.multi_vector_quantization_config.get(space) {
            Some(config) => config.clone(),
            None => return,
        };

        // Collect per-document token vectors for this space
        let mut keys = Vec::new();
        let mut doc_token_vectors: Vec<&[Vec<f32>]> = Vec::new();
        let mut token_dimension = 0_usize;

        for (key, record) in &self.records {
            if let Some(tokens) = record.multi_vectors.get(space) {
                if !tokens.is_empty() {
                    if token_dimension == 0 {
                        token_dimension = tokens[0].len();
                    }
                    keys.push(key.clone());
                    doc_token_vectors.push(tokens.as_slice());
                }
            }
        }

        if doc_token_vectors.is_empty() || token_dimension == 0 {
            self.multi_vector_quantized.remove(space);
            self.multi_vector_quantized_keys.remove(space);
            return;
        }

        let index = MultiVectorQuantizedIndex::build(&doc_token_vectors, token_dimension, &config);

        self.multi_vector_quantized.insert(space.to_owned(), index);
        self.multi_vector_quantized_keys
            .insert(space.to_owned(), keys);
    }

    fn rebuild_all_multi_vector_quantized_indexes(&mut self) {
        let spaces: Vec<String> = self
            .multi_vector_quantization_config
            .keys()
            .cloned()
            .collect();
        for space in spaces {
            self.rebuild_multi_vector_quantized_index(&space);
        }
    }

    fn persist_multi_vector_quantization_params(&self, space: &str) -> Result<()> {
        let params_path = multi_vector_quantization_params_path(&self.path, space);
        if let Some(index) = self.multi_vector_quantized.get(space) {
            let mut file = File::create(&params_path)?;
            index.write_params(&mut file).map_err(VectLiteError::Io)?;
            file.sync_all()?;
        } else {
            if params_path.exists() {
                fs::remove_file(&params_path)?;
            }
        }
        Ok(())
    }

    fn try_load_multi_vector_quantization(&mut self) {
        // Look for .mvquant.<space> sidecar files
        let Some(parent) = self.path.parent() else {
            return;
        };
        let Some(stem) = self.path.file_name().and_then(|n| n.to_str()) else {
            return;
        };
        let prefix = format!("{stem}.mvquant.");

        let entries = match fs::read_dir(parent) {
            Ok(entries) => entries,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let Some(fname) = entry.file_name().to_str().map(String::from) else {
                continue;
            };
            if !fname.starts_with(&prefix) {
                continue;
            }
            let space = &fname[prefix.len()..];
            if space.is_empty() {
                continue;
            }

            let file = match File::open(entry.path()) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut reader = BufReader::new(file);
            let mut index = match MultiVectorQuantizedIndex::read_params(&mut reader) {
                Ok(idx) => idx,
                Err(_) => continue,
            };

            // Rebuild codes from current records
            let mut keys = Vec::new();
            let mut doc_token_vectors: Vec<&[Vec<f32>]> = Vec::new();
            for (key, record) in &self.records {
                if let Some(tokens) = record.multi_vectors.get(space) {
                    if !tokens.is_empty() {
                        keys.push(key.clone());
                        doc_token_vectors.push(tokens.as_slice());
                    }
                }
            }

            if !doc_token_vectors.is_empty() {
                index.rebuild(&doc_token_vectors);
                let MultiVectorQuantizationConfig::TwoBit(ref cfg) =
                    { MultiVectorQuantizationConfig::TwoBit(index.quantizer.config.clone()) };
                self.multi_vector_quantization_config.insert(
                    space.to_owned(),
                    MultiVectorQuantizationConfig::TwoBit(cfg.clone()),
                );
                self.multi_vector_quantized_keys
                    .insert(space.to_owned(), keys);
                self.multi_vector_quantized.insert(space.to_owned(), index);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Payload index helpers
    // -----------------------------------------------------------------------

    fn payload_index_sidecar_path(&self) -> PathBuf {
        let mut p = self.path.clone();
        let name = p.file_name().unwrap_or_default().to_os_string();
        p.set_file_name(format!("{}.pidx", name.to_string_lossy()));
        p
    }

    /// Persist payload index definitions to the sidecar file.
    fn persist_payload_index_defs(&self) -> Result<()> {
        let sidecar = self.payload_index_sidecar_path();
        if self.payload_index_defs.is_empty() {
            // Remove the sidecar if there are no indexes.
            let _ = fs::remove_file(&sidecar);
            return Ok(());
        }
        let file = File::create(&sidecar)?;
        let mut writer = BufWriter::new(file);
        write_u32(&mut writer, u32_from_usize(self.payload_index_defs.len())?)?;
        for (field, index_type) in &self.payload_index_defs {
            write_string(&mut writer, field)?;
            write_u8(&mut writer, index_type.tag())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Load payload index definitions from the sidecar file (if present).
    fn try_load_payload_index_defs(&mut self) {
        let sidecar = self.payload_index_sidecar_path();
        let file = match File::open(&sidecar) {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut reader = BufReader::new(file);
        let count = match read_u32(&mut reader) {
            Ok(n) => usize_from_u32(n).unwrap_or(0),
            Err(_) => return,
        };
        let mut defs = BTreeMap::new();
        for _ in 0..count {
            let field = match read_string(&mut reader) {
                Ok(f) => f,
                Err(_) => return,
            };
            let tag = match read_u8(&mut reader) {
                Ok(t) => t,
                Err(_) => return,
            };
            let index_type = match PayloadIndexType::from_tag(tag) {
                Ok(t) => t,
                Err(_) => return,
            };
            defs.insert(field, index_type);
        }
        self.payload_index_defs = defs;
    }

    /// Rebuild all payload indexes from scratch, based on current `payload_index_defs`
    /// and all records in memory.
    fn rebuild_payload_indexes(&mut self) {
        let mut indexes = BTreeMap::new();
        for (field, index_type) in &self.payload_index_defs {
            let data = match index_type {
                PayloadIndexType::Keyword => {
                    let mut kw = KeywordIndex::default();
                    for (key, record) in &self.records {
                        if let Some(MetadataValue::String(val)) = record.metadata.get(field) {
                            kw.insert(val, key.clone());
                        }
                    }
                    PayloadIndexData::Keyword(kw)
                }
                PayloadIndexType::Numeric => {
                    let mut num = NumericIndex::default();
                    for (key, record) in &self.records {
                        if let Some(val) = record
                            .metadata
                            .get(field)
                            .and_then(MetadataValue::as_number)
                        {
                            num.insert(val, key.clone());
                        }
                    }
                    PayloadIndexData::Numeric(num)
                }
            };
            indexes.insert(field.clone(), data);
        }
        self.payload_indexes = indexes;
    }

    /// Incrementally update payload indexes for an upserted record.
    /// Call with the old record (if any) first to remove stale entries.
    fn payload_index_remove(&mut self, key: &RecordKey, metadata: &Metadata) {
        for (field, data) in &mut self.payload_indexes {
            match data {
                PayloadIndexData::Keyword(kw) => {
                    if let Some(MetadataValue::String(val)) = metadata.get(field) {
                        kw.remove(val, key);
                    }
                }
                PayloadIndexData::Numeric(num) => {
                    if let Some(val) = metadata.get(field).and_then(MetadataValue::as_number) {
                        num.remove(val, key);
                    }
                }
            }
        }
    }

    fn payload_index_insert(&mut self, key: &RecordKey, metadata: &Metadata) {
        for (field, data) in &mut self.payload_indexes {
            match data {
                PayloadIndexData::Keyword(kw) => {
                    if let Some(MetadataValue::String(val)) = metadata.get(field) {
                        kw.insert(val, key.clone());
                    }
                }
                PayloadIndexData::Numeric(num) => {
                    if let Some(val) = metadata.get(field).and_then(MetadataValue::as_number) {
                        num.insert(val, key.clone());
                    }
                }
            }
        }
    }

    /// Use payload indexes to narrow down candidate keys for a filter.
    /// Returns `None` if no indexes can help with this filter (fallback to scan).
    /// Returns `Some(set)` with the set of record keys that *may* match the filter.
    fn payload_index_candidates(
        &self,
        filter: &MetadataFilter,
        namespace: Option<&str>,
    ) -> Option<HashSet<RecordKey>> {
        if self.payload_indexes.is_empty() {
            return None;
        }
        self.payload_index_candidates_inner(filter, namespace)
    }

    fn payload_index_candidates_inner(
        &self,
        filter: &MetadataFilter,
        namespace: Option<&str>,
    ) -> Option<HashSet<RecordKey>> {
        match filter {
            MetadataFilter::Eq { key, value } => {
                // Try keyword index for string equality
                if let Some(PayloadIndexData::Keyword(kw)) = self.payload_indexes.get(key) {
                    if let MetadataValue::String(s) = value {
                        let set = kw.lookup_eq(s).cloned().unwrap_or_default();
                        return Some(self.filter_by_namespace(set, namespace));
                    }
                }
                // Try numeric index for numeric equality
                if let Some(PayloadIndexData::Numeric(num)) = self.payload_indexes.get(key) {
                    if let Some(v) = value.as_number() {
                        let set = num.lookup_eq(v).cloned().unwrap_or_default();
                        return Some(self.filter_by_namespace(set, namespace));
                    }
                }
                None
            }
            MetadataFilter::In { key, values } => {
                if let Some(PayloadIndexData::Keyword(kw)) = self.payload_indexes.get(key) {
                    let str_values: Vec<&str> = values
                        .iter()
                        .filter_map(|v| match v {
                            MetadataValue::String(s) => Some(s.as_str()),
                            _ => None,
                        })
                        .collect();
                    if str_values.len() == values.len() {
                        let set = kw.lookup_in(&str_values);
                        return Some(self.filter_by_namespace(set, namespace));
                    }
                }
                None
            }
            MetadataFilter::GreaterThan { key, value } => {
                if let Some(PayloadIndexData::Numeric(num)) = self.payload_indexes.get(key) {
                    let set = num.range_gt(*value);
                    return Some(self.filter_by_namespace(set, namespace));
                }
                None
            }
            MetadataFilter::GreaterThanOrEqual { key, value } => {
                if let Some(PayloadIndexData::Numeric(num)) = self.payload_indexes.get(key) {
                    let set = num.range_gte(*value);
                    return Some(self.filter_by_namespace(set, namespace));
                }
                None
            }
            MetadataFilter::LessThan { key, value } => {
                if let Some(PayloadIndexData::Numeric(num)) = self.payload_indexes.get(key) {
                    let set = num.range_lt(*value);
                    return Some(self.filter_by_namespace(set, namespace));
                }
                None
            }
            MetadataFilter::LessThanOrEqual { key, value } => {
                if let Some(PayloadIndexData::Numeric(num)) = self.payload_indexes.get(key) {
                    let set = num.range_lte(*value);
                    return Some(self.filter_by_namespace(set, namespace));
                }
                None
            }
            MetadataFilter::And(filters) => {
                // Intersect candidates from all sub-filters that have index support.
                let mut result: Option<HashSet<RecordKey>> = None;
                for sub in filters {
                    if let Some(sub_set) = self.payload_index_candidates_inner(sub, namespace) {
                        result = Some(match result {
                            Some(existing) => existing.intersection(&sub_set).cloned().collect(),
                            None => sub_set,
                        });
                    }
                }
                result
            }
            MetadataFilter::Or(filters) => {
                // Union candidates, but only if ALL sub-filters have index support.
                let mut result = HashSet::new();
                for sub in filters {
                    match self.payload_index_candidates_inner(sub, namespace) {
                        Some(sub_set) => {
                            result.extend(sub_set);
                        }
                        None => return None, // Can't guarantee completeness
                    }
                }
                Some(result)
            }
            // For other filter types, no index support — fallback to scan.
            _ => None,
        }
    }

    fn filter_by_namespace(
        &self,
        keys: HashSet<RecordKey>,
        namespace: Option<&str>,
    ) -> HashSet<RecordKey> {
        match namespace {
            Some(ns) => keys.into_iter().filter(|(n, _)| n == ns).collect(),
            None => keys,
        }
    }

    fn compact_inner(&mut self) -> Result<()> {
        // GC: remove expired records before writing the snapshot.
        let now = now_epoch_secs();
        let expired_keys: Vec<RecordKey> = self
            .records
            .iter()
            .filter(|(_, record)| record.is_expired_at(now))
            .map(|(key, _)| key.clone())
            .collect();
        let has_payload_indexes = !self.payload_indexes.is_empty();
        for key in &expired_keys {
            if has_payload_indexes {
                if let Some(record) = self.records.get(key) {
                    let meta = record.metadata.clone();
                    self.payload_index_remove(key, &meta);
                }
            }
            self.records.remove(key);
        }

        // If any HNSW graph has tombstones, rebuild it before persisting.
        //
        // Two reasons:
        //   1. Crossing `tombstone_rebuild_pct` means search recall has
        //      degraded enough that the user wants a clean graph.
        //   2. Even below the threshold, the persisted manifest's
        //      `record_count` is derived from `self.records` (live only),
        //      but the in-memory `keys` array includes dead slots — so a
        //      persisted-with-tombstones graph would always fail the
        //      record_count check on reopen and rebuild anyway. Rebuilding
        //      *now* dumps a clean graph that survives reload.
        let threshold = self.index_config.tombstone_rebuild_pct;
        let any_tombstones = self
            .ann
            .global
            .values()
            .any(|idx| !idx.tombstones.is_empty())
            || self
                .ann
                .namespaces
                .values()
                .flat_map(|m| m.values())
                .any(|idx| !idx.tombstones.is_empty());
        // (We track `threshold` even though we currently rebuild on any
        // tombstones, so `should_rebuild` could later replace this when we
        // add tombstone persistence in the manifest.)
        let _ = threshold;
        if any_tombstones {
            self.rebuild_ann();
        }

        // Rebuild any lazy indexes that were marked dirty during the session
        // before we persist. This is the point where we pay back the work
        // we deferred from the per-insert hot path:
        //   - the HNSW graph is already up-to-date (incremental inserts),
        //     we just need to dump it.
        //   - the quantized PQ index was dropped on first insert and is
        //     rebuilt now so search can use it again next session.
        //   - same for multi-vector PQ.
        if self.quantized_dirty {
            self.rebuild_quantized_index();
            self.quantized_dirty = false;
        }
        if self.multi_vector_quantized_dirty {
            self.rebuild_all_multi_vector_quantized_indexes();
            self.multi_vector_quantized_dirty = false;
        }

        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let temp_path = temp_path(&self.path);
        let mut file = File::create(&temp_path)?;
        {
            let mut writer = BufWriter::new(&mut file);
            self.write_to(&mut writer)?;
            writer.flush()?;
        }
        file.sync_all()?;

        if self.path.exists() {
            fs::remove_file(&self.path)?;
        }
        fs::rename(temp_path, &self.path)?;
        self.clear_wal()?;
        self.wal_entries_replayed = 0;
        self.persist_ann_to_disk()?;
        self.ann_dirty = false;

        Ok(())
    }

    /// Create an atomic snapshot of the database at `dest`. The snapshot is a
    /// self-contained `.vdb` file (WAL is folded in). The current database is
    /// not modified. Works in both read-only and read-write mode.
    pub fn snapshot(&self, dest: impl AsRef<Path>) -> Result<()> {
        self.check_open()?;
        let dest = dest.as_ref();
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut file = File::create(dest)?;
        {
            let mut writer = BufWriter::new(&mut file);
            self.write_to(&mut writer)?;
            writer.flush()?;
        }
        file.sync_all()?;
        Ok(())
    }

    /// Back up the database to `dest` directory. Creates a complete copy
    /// including the `.vdb` file and ANN sidecar files. The backup is
    /// compacted (WAL folded in). Works in both read-only and read-write mode.
    pub fn backup(&self, dest: impl AsRef<Path>) -> Result<()> {
        self.check_open()?;
        let dest = dest.as_ref();
        fs::create_dir_all(dest)?;

        let file_name = self.path.file_name().ok_or_else(|| {
            VectLiteError::InvalidFormat("database path has no file name".to_owned())
        })?;
        let dest_vdb = dest.join(file_name);
        self.snapshot(&dest_vdb)?;

        // Copy ANN sidecar files
        if let Some(parent) = self.path.parent() {
            if let Some(stem) = self.path.file_name().and_then(|n| n.to_str()) {
                if let Ok(entries) = fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(fname) = entry.file_name().to_str() {
                            if fname.starts_with(&format!("{stem}.ann.")) {
                                let _ = fs::copy(entry.path(), dest.join(fname));
                            }
                        }
                    }
                }
                // Copy ann manifest
                let manifest = ann_manifest_path(&self.path);
                if manifest.exists() {
                    if let Some(manifest_name) = manifest.file_name() {
                        let _ = fs::copy(&manifest, dest.join(manifest_name));
                    }
                }

                // Copy payload index sidecar
                let pidx = self.payload_index_sidecar_path();
                if pidx.exists() {
                    if let Some(pidx_name) = pidx.file_name() {
                        let _ = fs::copy(&pidx, dest.join(pidx_name));
                    }
                }
            }
        }

        Ok(())
    }

    /// Restore a database from a backup directory. Opens the `.vdb` file
    /// found in `source` and returns a new writable Database.
    pub fn restore(source: impl AsRef<Path>, dest: impl AsRef<Path>) -> Result<Self> {
        let source = source.as_ref();
        let dest = dest.as_ref();

        // Find the .vdb file in the source directory
        let mut vdb_file = None;
        for entry in fs::read_dir(source)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("vdb") {
                vdb_file = Some(path);
                break;
            }
        }
        let source_vdb = vdb_file.ok_or_else(|| {
            VectLiteError::InvalidFormat("no .vdb file found in backup directory".to_owned())
        })?;

        // Copy the vdb file
        if let Some(parent) = dest.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::copy(&source_vdb, dest)?;

        // Copy ANN sidecar files
        if let Some(stem) = source_vdb.file_name().and_then(|n| n.to_str()) {
            for entry in fs::read_dir(source)?.flatten() {
                if let Some(fname) = entry.file_name().to_str() {
                    if fname.starts_with(&format!("{stem}.ann.")) || fname.ends_with(".ann") {
                        if let Some(dest_parent) = dest.parent() {
                            let _ = fs::copy(entry.path(), dest_parent.join(fname));
                        }
                    }
                }
            }
        }

        Self::open(dest)
    }

    fn apply_wal_batch(&mut self, ops: Vec<WalOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        let has_sparse = ops.iter().any(|op| match op {
            WalOp::Upsert(record) => {
                !record.sparse.is_empty()
                    || self
                        .records
                        .get(&(record.namespace.clone(), record.id.clone()))
                        .map_or(false, |r| !r.sparse.is_empty())
            }
            WalOp::Delete { namespace, id } => self
                .records
                .get(&(namespace.clone(), id.clone()))
                .map_or(false, |r| !r.sparse.is_empty()),
            WalOp::UpdateMetadata { .. } | WalOp::SetTtl { .. } => false,
        });

        let metadata_only = ops
            .iter()
            .all(|op| matches!(op, WalOp::UpdateMetadata { .. } | WalOp::SetTtl { .. }));

        // Categorise each op so we can route to the fastest correct path:
        //   incremental insert (Upsert with new key) → ann_apply_incremental
        //   tombstone delete   (Delete of present key) → ann_apply_tombstones
        //   anything else (upsert of existing key, etc) → full rebuild
        let mut incremental_eligible = !metadata_only;
        let mut tombstone_only = !metadata_only;
        for op in &ops {
            match op {
                WalOp::Upsert(record) => {
                    let exists = self
                        .records
                        .contains_key(&(record.namespace.clone(), record.id.clone()));
                    if exists {
                        incremental_eligible = false;
                        tombstone_only = false;
                    } else {
                        // New upsert — fine for incremental, but not tombstone-only.
                        tombstone_only = false;
                    }
                }
                WalOp::Delete { namespace, id } => {
                    let exists = self.records.contains_key(&(namespace.clone(), id.clone()));
                    if exists {
                        // OK for tombstone path, but not for incremental.
                        incremental_eligible = false;
                    }
                    // (A delete of a non-existent key is a no-op for both
                    // paths, but we still let it through.)
                }
                WalOp::UpdateMetadata { .. } | WalOp::SetTtl { .. } => {
                    incremental_eligible = false;
                    tombstone_only = false;
                }
            }
        }

        // Collect the keys we'll need to feed to the relevant updater
        // before we move `ops` into `apply_ops_in_memory`.
        let new_keys: Vec<RecordKey> = if incremental_eligible {
            ops.iter()
                .filter_map(|op| match op {
                    WalOp::Upsert(record) => Some((record.namespace.clone(), record.id.clone())),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };
        let deleted_keys: Vec<RecordKey> = if tombstone_only {
            ops.iter()
                .filter_map(|op| match op {
                    WalOp::Delete { namespace, id } => Some((namespace.clone(), id.clone())),
                    _ => None,
                })
                .collect()
        } else {
            Vec::new()
        };

        self.append_wal_batch(&ops)?;
        self.apply_ops_in_memory(ops);

        // Metadata-only updates don't change vectors, so skip all index rebuilds.
        if !metadata_only {
            if has_sparse {
                self.rebuild_sparse_index();
            }
            if incremental_eligible {
                // Fast path: just append the new vectors into the existing
                // HNSW graph(s) instead of rebuilding from scratch. Converts
                // single-record ingestion from O(N log N) per insert to
                // amortised O(log N).
                self.ann_apply_incremental(&new_keys);
                // Keep the contiguous arena in sync. If it hasn't been
                // materialised yet, leave it alone — it'll be lazily built
                // on first read.
                if self.vector_arena.is_some() && !self.vector_arena_dirty {
                    self.arena_apply_incremental(&new_keys);
                }
            } else if tombstone_only {
                // Delete-only fast path: tombstone the corresponding
                // `origin_id`s in each affected HNSW graph. No rebuild;
                // search filters out tombstoned candidates. The graph is
                // rebuilt automatically at the next `compact()` once the
                // tombstone ratio crosses `tombstone_rebuild_pct`.
                self.ann_apply_tombstones(&deleted_keys);
                // The arena can't compact in place without shifting O(N)
                // floats; mark dirty so it's lazily rebuilt on next scan.
                self.vector_arena_dirty = true;
            } else {
                // Slow path: a mixed-mode batch or an update-of-existing.
                // Rebuild the whole catalog.
                self.rebuild_ann();
                self.vector_arena_dirty = true;
            }
            // Defer persistence of the HNSW graph to disk: writing the graph
            // files is expensive (full re-dump + fsync) and is only required
            // for crash recovery on reopen. The WAL gives us that durability
            // already — on reopen, if the persisted graph is stale, it's
            // detected via the manifest signature check and rebuilt from
            // records in memory. Persistence happens at `flush` / `compact`.
            self.ann_loaded_from_disk = false;
            self.ann_dirty = true;
            // Lazy-rebuild quantized indexes too. Drop the in-memory
            // structures so callers get correct (HNSW-fallback) results
            // until the next flush, where we rebuild from the new corpus.
            if self.quantization_config.is_some() {
                self.quantized = None;
                self.quantized_keys.clear();
                self.quantized_dirty = true;
            }
            if !self.multi_vector_quantization_config.is_empty() {
                self.multi_vector_quantized.clear();
                self.multi_vector_quantized_keys.clear();
                self.multi_vector_quantized_dirty = true;
            }
        }
        Ok(())
    }

    /// Write ops to WAL and apply in memory, but defer index rebuilds.
    /// The caller is responsible for calling `rebuild_sparse_index()`,
    /// `rebuild_ann()`, and `persist_ann_to_disk()` after all batches are done.
    fn apply_wal_batch_deferred(&mut self, ops: Vec<WalOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }

        self.append_wal_batch(&ops)?;
        self.apply_ops_in_memory(ops);
        Ok(())
    }

    fn apply_ops_in_memory(&mut self, ops: Vec<WalOp>) {
        let has_payload_indexes = !self.payload_indexes.is_empty();
        for op in ops {
            match op {
                WalOp::Upsert(record) => {
                    let key = (record.namespace.clone(), record.id.clone());
                    if has_payload_indexes {
                        // Remove old index entries if the record already exists.
                        let old_meta = self.records.get(&key).map(|r| r.metadata.clone());
                        if let Some(ref meta) = old_meta {
                            self.payload_index_remove(&key, meta);
                        }
                        self.payload_index_insert(&key, &record.metadata);
                    }
                    self.records.insert(key, record);
                }
                WalOp::Delete { namespace, id } => {
                    let key = (namespace, id);
                    if has_payload_indexes {
                        let old_meta = self.records.get(&key).map(|r| r.metadata.clone());
                        if let Some(ref meta) = old_meta {
                            self.payload_index_remove(&key, meta);
                        }
                    }
                    self.records.remove(&key);
                }
                WalOp::UpdateMetadata {
                    namespace,
                    id,
                    metadata,
                } => {
                    let key = (namespace, id);
                    if has_payload_indexes {
                        if let Some(record) = self.records.get(&key) {
                            let old_meta = record.metadata.clone();
                            self.payload_index_remove(&key, &old_meta);
                        }
                    }
                    if let Some(record) = self.records.get_mut(&key) {
                        for (k, v) in metadata {
                            record.metadata.insert(k, v);
                        }
                    }
                    if has_payload_indexes {
                        if let Some(record) = self.records.get(&key) {
                            let new_meta = record.metadata.clone();
                            self.payload_index_insert(&key, &new_meta);
                        }
                    }
                    // If the record doesn't exist, the update is silently ignored
                    // (same semantics as deleting a non-existent record).
                }
                WalOp::SetTtl {
                    namespace,
                    id,
                    expires_at,
                } => {
                    let key = (namespace, id);
                    if let Some(record) = self.records.get_mut(&key) {
                        record.expires_at = expires_at;
                    }
                }
            }
        }
    }

    fn append_wal_batch(&mut self, ops: &[WalOp]) -> Result<()> {
        // Decide whether this batch should trigger an fsync. We use the
        // ops count in the batch (not 1) so `EveryN` semantics scale across
        // both single inserts and `insert_many` calls.
        let n_ops = ops.len();
        let should_sync = match self.wal_sync_mode {
            WalSyncMode::PerOp => true,
            WalSyncMode::EveryN(n) => {
                self.wal_ops_since_sync = self.wal_ops_since_sync.saturating_add(n_ops);
                if self.wal_ops_since_sync >= n {
                    self.wal_ops_since_sync = 0;
                    true
                } else {
                    false
                }
            }
            WalSyncMode::OnFlush => {
                self.wal_ops_since_sync = self.wal_ops_since_sync.saturating_add(n_ops);
                false
            }
        };
        self.append_wal_batch_inner(ops, should_sync)
    }

    /// Append a WAL batch without issuing an fsync. The caller is responsible
    /// for issuing `sync_wal` later (typically once at the end of a bulk
    /// ingest). This is the hot path for `bulk_ingest`.
    fn append_wal_batch_unsynced(&mut self, ops: &[WalOp]) -> Result<()> {
        // Track pending ops so future `sync_wal` / `compact_inner` calls
        // know to flush them.
        self.wal_ops_since_sync = self.wal_ops_since_sync.saturating_add(ops.len());
        self.append_wal_batch_inner(ops, false)
    }

    /// Append a WAL batch. Reuses a cached `BufWriter<File>` across calls so
    /// the WAL file is only opened once per database session — saving the
    /// `open()` syscall on every single `insert` call, which matters when
    /// per-record overhead is the bottleneck.
    fn append_wal_batch_inner(&mut self, ops: &[WalOp], sync: bool) -> Result<()> {
        if let Some(parent) = self.wal_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        // Lazily create the cached BufWriter, writing the WAL_MAGIC header
        // on first use of a brand-new file.
        if self.wal_writer.is_none() {
            let new_file = !self.wal_path.exists();
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.wal_path)?;
            let mut writer = BufWriter::with_capacity(64 * 1024, file);
            if new_file {
                writer.write_all(WAL_MAGIC)?;
            }
            self.wal_writer = Some(writer);
        }

        // Serialise the batch into a temporary buffer first, so that the
        // single `write_all` we issue to the cached writer is one contiguous
        // user-space copy (BufWriter then bunches everything up further).
        let mut buffer = Vec::new();
        write_u32(&mut buffer, u32_from_usize(ops.len())?)?;
        for op in ops {
            write_wal_op(&mut buffer, op)?;
        }

        let writer = self.wal_writer.as_mut().unwrap();
        write_u32(writer, u32_from_usize(buffer.len())?)?;
        writer.write_all(&buffer)?;

        if sync {
            // Flush BufWriter into the OS, then ask the kernel to make the
            // bytes durable. We must `flush()` before `sync_all()` — sync_all
            // only operates on what's already in the kernel's page cache.
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }

    /// Force a durability fence on the WAL file. Flushes any buffered bytes
    /// from the cached writer and asks the kernel to make them durable in a
    /// single `sync_all`. Used by `bulk_ingest`, `flush`, `close`, and as a
    /// manual fence when running in `EveryN` or `OnFlush` mode.
    fn sync_wal(&mut self) -> Result<()> {
        if let Some(writer) = self.wal_writer.as_mut() {
            writer.flush()?;
            writer.get_ref().sync_all()?;
            self.wal_ops_since_sync = 0;
            return Ok(());
        }
        // Fallback: no cached writer (e.g. WAL was opened externally). Open
        // the file briefly just to issue the sync.
        if !self.wal_path.exists() {
            self.wal_ops_since_sync = 0;
            return Ok(());
        }
        let file = OpenOptions::new().append(true).open(&self.wal_path)?;
        file.sync_all()?;
        self.wal_ops_since_sync = 0;
        Ok(())
    }

    fn replay_wal(&mut self) -> Result<()> {
        if !self.wal_path.exists() {
            self.wal_entries_replayed = 0;
            return Ok(());
        }

        let mut reader = BufReader::new(File::open(&self.wal_path)?);
        let mut magic = [0_u8; 4];
        if let Err(err) = reader.read_exact(&mut magic) {
            if err.kind() == ErrorKind::UnexpectedEof {
                self.wal_entries_replayed = 0;
                return Ok(());
            }
            return Err(err.into());
        }
        if &magic != WAL_MAGIC {
            return Err(VectLiteError::InvalidFormat(
                "invalid WAL header".to_owned(),
            ));
        }

        let mut replayed = 0;
        loop {
            let batch_len = match read_u32(&mut reader) {
                Ok(batch_len) => usize_from_u32(batch_len)?,
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            };

            let mut batch = vec![0_u8; batch_len];
            match reader.read_exact(&mut batch) {
                Ok(()) => {
                    let mut batch_reader = &batch[..];
                    let op_count = usize_from_u32(read_u32(&mut batch_reader)?)?;
                    let mut ops = Vec::with_capacity(op_count);
                    for _ in 0..op_count {
                        ops.push(read_wal_op(&mut batch_reader, self.dimension)?);
                    }
                    self.apply_ops_in_memory(ops);
                    replayed += op_count;
                }
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err.into()),
            }
        }

        self.wal_entries_replayed = replayed;
        Ok(())
    }

    fn clear_wal(&mut self) -> Result<()> {
        // Drop the cached writer first: on POSIX the file would survive the
        // unlink because we still hold an open handle, but we'd then keep
        // appending into the now-detached inode and never see those bytes on
        // disk after reopen.
        self.wal_writer = None;
        if self.wal_path.exists() {
            fs::remove_file(&self.wal_path)?;
        }
        Ok(())
    }

    fn read_from(path: &Path, reader: &mut impl Read) -> Result<Self> {
        let mut magic = [0_u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(VectLiteError::InvalidFormat(
                "missing VDB1 magic header".to_owned(),
            ));
        }

        let version = read_u16(reader)?;
        if !(1..=VERSION).contains(&version) {
            return Err(VectLiteError::InvalidFormat(format!(
                "unsupported version {version}"
            )));
        }

        let dimension = usize_from_u32(read_u32(reader)?)?;
        ensure_dimension(dimension)?;

        let metric = if version >= 6 {
            DistanceMetric::from_tag(read_u8(reader)?)?
        } else {
            DistanceMetric::Cosine
        };

        let record_count = usize_from_u64(read_u64(reader)?)?;
        let mut records = BTreeMap::new();

        for _ in 0..record_count {
            let namespace = if version >= 2 {
                read_string(reader)?
            } else {
                DEFAULT_NAMESPACE.to_owned()
            };
            let id = read_string(reader)?;
            let metadata_count = usize_from_u32(read_u32(reader)?)?;
            let mut metadata = Metadata::new();
            for _ in 0..metadata_count {
                let key = read_string(reader)?;
                let value = read_metadata_value(reader)?;
                metadata.insert(key, value);
            }

            let vector_len = usize_from_u32(read_u32(reader)?)?;
            if vector_len != dimension {
                return Err(VectLiteError::InvalidFormat(format!(
                    "record {id} has vector length {vector_len}, expected {dimension}"
                )));
            }

            let mut vector = Vec::with_capacity(vector_len);
            for _ in 0..vector_len {
                vector.push(read_f32(reader)?);
            }

            let vectors = if version >= 4 {
                read_named_vectors(reader, dimension)?
            } else {
                NamedVectors::new()
            };

            let sparse = if version >= 3 {
                read_sparse_vector(reader)?
            } else {
                SparseVector::new()
            };

            let multi_vectors = if version >= 5 {
                read_multi_vectors(reader)?
            } else {
                MultiVectors::new()
            };

            let expires_at = if version >= 7 {
                let ts = read_f64(reader)?;
                if ts == 0.0 { None } else { Some(ts) }
            } else {
                None
            };

            let record = Record {
                namespace: namespace.clone(),
                id: id.clone(),
                vector,
                vectors,
                sparse,
                metadata,
                multi_vectors,
                expires_at,
            };
            records.insert((namespace, id), record);
        }

        Ok(Self {
            path: path.to_path_buf(),
            wal_path: wal_path(path),
            dimension,
            metric,
            records,
            ann: AnnCatalog::default(),
            sparse_index: SparseIndex::default(),
            wal_entries_replayed: 0,
            ann_loaded_from_disk: false,
            read_only: false,
            _lock_file: None,
            wal_writer: None,
            wal_sync_mode: WalSyncMode::default(),
            wal_ops_since_sync: 0,
            ann_dirty: false,
            quantized_dirty: false,
            multi_vector_quantized_dirty: false,
            quantized: None,
            quantization_config: None,
            quantized_keys: Vec::new(),
            multi_vector_quantized: BTreeMap::new(),
            multi_vector_quantization_config: BTreeMap::new(),
            multi_vector_quantized_keys: BTreeMap::new(),
            payload_index_defs: BTreeMap::new(),
            payload_indexes: BTreeMap::new(),
            index_config: IndexConfig::default(),
            vector_arena: None,
            vector_arena_dirty: false,
        })
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(MAGIC)?;
        write_u16(writer, VERSION)?;
        write_u32(writer, u32_from_usize(self.dimension)?)?;
        write_u8(writer, self.metric.to_tag())?;
        write_u64(writer, u64_from_usize(self.records.len())?)?;

        for record in self.records.values() {
            write_string(writer, &record.namespace)?;
            write_string(writer, &record.id)?;
            write_u32(writer, u32_from_usize(record.metadata.len())?)?;
            for (key, value) in &record.metadata {
                write_string(writer, key)?;
                write_metadata_value(writer, value)?;
            }

            write_u32(writer, u32_from_usize(record.vector.len())?)?;
            for value in &record.vector {
                write_f32(writer, *value)?;
            }
            write_named_vectors(writer, &record.vectors)?;
            write_sparse_vector(writer, &record.sparse)?;
            write_multi_vectors(writer, &record.multi_vectors)?;
            write_f64(writer, record.expires_at.unwrap_or(0.0))?;
        }

        Ok(())
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VectLiteError::DimensionMismatch {
                expected: self.dimension,
                found: vector.len(),
            });
        }

        Ok(())
    }

    fn resolve_dense_search_dimension(
        &self,
        dense_query: Option<&[f32]>,
        truncate_dim: Option<usize>,
    ) -> Result<Option<usize>> {
        let Some(query) = dense_query else {
            return Ok(None);
        };
        if query.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "query vector must not be empty".to_owned(),
            ));
        }
        if query.len() > self.dimension {
            return Err(VectLiteError::DimensionMismatch {
                expected: self.dimension,
                found: query.len(),
            });
        }

        let effective = match truncate_dim {
            Some(0) => {
                return Err(VectLiteError::InvalidFormat(
                    "truncate_dim must be greater than zero".to_owned(),
                ));
            }
            Some(dim) if dim > self.dimension => {
                return Err(VectLiteError::DimensionMismatch {
                    expected: self.dimension,
                    found: dim,
                });
            }
            Some(dim) if dim > query.len() => {
                return Err(VectLiteError::InvalidFormat(format!(
                    "truncate_dim ({dim}) cannot exceed query vector length ({})",
                    query.len()
                )));
            }
            Some(dim) => dim,
            None => {
                // Without explicit truncate_dim, require exact dimension match.
                // Users must pass truncate_dim to opt into Matryoshka truncation.
                if query.len() != self.dimension {
                    return Err(VectLiteError::DimensionMismatch {
                        expected: self.dimension,
                        found: query.len(),
                    });
                }
                query.len()
            }
        };

        Ok(Some(effective))
    }

    fn validate_record(&self, record: &Record) -> Result<()> {
        self.validate_vector(&record.vector)?;

        for (vector_name, vector) in &record.vectors {
            if vector_name.is_empty() {
                return Err(VectLiteError::InvalidFormat(
                    "named vectors must not use an empty name".to_owned(),
                ));
            }
            self.validate_vector(vector)?;
        }

        for (space_name, token_vectors) in &record.multi_vectors {
            if space_name.is_empty() {
                return Err(VectLiteError::InvalidFormat(
                    "multi-vector space names must not be empty".to_owned(),
                ));
            }
            if let Some(first) = token_vectors.first() {
                if first.is_empty() {
                    return Err(VectLiteError::InvalidFormat(format!(
                        "multi-vector space '{space_name}' contains an empty token vector"
                    )));
                }
                let expected_dim = first.len();
                for token_vec in &token_vectors[1..] {
                    if token_vec.len() != expected_dim {
                        return Err(VectLiteError::InvalidFormat(format!(
                            "multi-vector space '{space_name}' has inconsistent token dimensions: expected {expected_dim}, found {}",
                            token_vec.len(),
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Incremental ANN update. Appends the given new records into the
    /// existing HNSW graph(s) without rebuilding them from scratch.
    ///
    /// Preconditions:
    /// - `new_keys` are keys that already live in `self.records` (caller
    ///   must have applied the WAL ops to memory first).
    /// - Each key referenced by `new_keys` did NOT previously exist in
    ///   `self.records` (i.e. it's a true insert, not an update).
    ///
    /// Behaviour per (namespace, vector_name) "slot":
    /// - If a graph already exists, the new vectors are appended to it
    ///   via single-element `hnsw.insert` calls (or `parallel_insert` if
    ///   the batch is large enough to amortise thread overhead).
    /// - If no graph exists but the total record count for that slot has
    ///   now crossed `ANN_MIN_POINTS`, a fresh graph is built from all
    ///   matching records.
    /// - Below `ANN_MIN_POINTS`, we skip — searches will brute-force
    ///   without harm.
    fn ann_apply_incremental(&mut self, new_keys: &[RecordKey]) {
        if new_keys.is_empty() {
            return;
        }
        let cfg = self.index_config;

        // Group the new records by (Option<namespace>, vector_name). Each
        // upserted record contributes to exactly one global slot and one
        // namespace-scoped slot per dense vector it owns.
        let mut groups: BTreeMap<(Option<String>, String), Vec<(RecordKey, Vec<f32>)>> =
            BTreeMap::new();
        for key in new_keys {
            let Some(record) = self.records.get(key) else {
                continue;
            };
            for (vector_name, vector) in record.dense_vectors() {
                let item = (key.clone(), vector.clone());
                groups
                    .entry((None, vector_name.to_owned()))
                    .or_default()
                    .push(item.clone());
                groups
                    .entry((Some(record.namespace.clone()), vector_name.to_owned()))
                    .or_default()
                    .push(item);
            }
        }

        // Two-phase processing to keep the borrow checker happy:
        //   phase 1: classify each slot (needs fresh build vs incremental
        //            append), reading `self.records` only.
        //   phase 2: mutate `self.ann` based on the classifications.
        let mut fresh_builds: Vec<((Option<String>, String), Vec<(RecordKey, Vec<f32>)>)> =
            Vec::new();
        let mut incremental: Vec<((Option<String>, String), Vec<(RecordKey, Vec<f32>)>)> =
            Vec::new();

        for ((opt_ns, vector_name), new_items) in groups {
            let has_existing = match &opt_ns {
                None => self.ann.global.contains_key(&vector_name),
                Some(ns) => self
                    .ann
                    .namespaces
                    .get(ns)
                    .map_or(false, |m| m.contains_key(&vector_name)),
            };

            if has_existing {
                incremental.push(((opt_ns, vector_name), new_items));
                continue;
            }

            // Count matching records (post-insert state) to decide whether
            // we've crossed the build threshold.
            let total = self
                .records
                .iter()
                .filter(|(_, r)| match &opt_ns {
                    Some(ns) => r.namespace == *ns,
                    None => true,
                })
                .filter(|(_, r)| {
                    r.dense_vectors()
                        .any(|(name, _)| name == vector_name.as_str())
                })
                .count();

            if total < ANN_MIN_POINTS {
                continue;
            }

            // Need to build a fresh graph for this slot. Collect ALL matching
            // records (not just the new ones) — owned clones so the build
            // step doesn't borrow `self.records`.
            let mut all_items: Vec<(RecordKey, Vec<f32>)> = Vec::with_capacity(total);
            for (k, r) in &self.records {
                if let Some(ns) = &opt_ns {
                    if r.namespace != *ns {
                        continue;
                    }
                }
                for (name, vec) in r.dense_vectors() {
                    if name == vector_name.as_str() {
                        all_items.push((k.clone(), vec.clone()));
                        break;
                    }
                }
            }
            let _ = new_items; // already folded into `all_items`
            fresh_builds.push(((opt_ns, vector_name), all_items));
        }

        // Phase 2a: build-from-scratch for slots that just crossed the
        // threshold.
        for ((opt_ns, vector_name), all_items) in fresh_builds {
            let records_for_build: Vec<(RecordKey, &Vec<f32>)> =
                all_items.iter().map(|(k, v)| (k.clone(), v)).collect();
            let new_index = build_ann_index(records_for_build, self.metric, &cfg);
            match opt_ns {
                None => {
                    self.ann.global.insert(vector_name, new_index);
                }
                Some(ns) => {
                    self.ann
                        .namespaces
                        .entry(ns)
                        .or_default()
                        .insert(vector_name, new_index);
                }
            }
        }

        // Phase 2b: incremental appends into existing graphs.
        for ((opt_ns, vector_name), new_items) in incremental {
            let idx_opt = match &opt_ns {
                None => self.ann.global.get_mut(&vector_name),
                Some(ns) => self
                    .ann
                    .namespaces
                    .get_mut(ns)
                    .and_then(|m| m.get_mut(&vector_name)),
            };
            let Some(idx) = idx_opt else {
                continue;
            };

            // hnsw_rs marks indexes that have been searched as "searching
            // mode" (a hint that skips some bookkeeping in the data layer).
            // Re-enable mutation mode before we insert — cheap toggle.
            idx.hnsw.set_searching_mode(false);

            if new_items.len() >= cfg.parallel_insert_threshold {
                let start_id = idx.keys.len();
                let batch: Vec<(&Vec<f32>, usize)> = new_items
                    .iter()
                    .enumerate()
                    .map(|(offset, (_, v))| (v, start_id + offset))
                    .collect();
                idx.hnsw.parallel_insert_batch(&batch);
                for (offset, (k, _)) in new_items.into_iter().enumerate() {
                    let origin_id = start_id + offset;
                    idx.key_to_origin.insert(k.clone(), origin_id);
                    idx.keys.push(k);
                }
            } else {
                for (key, vector) in new_items {
                    let origin_id = idx.keys.len();
                    idx.key_to_origin.insert(key.clone(), origin_id);
                    idx.keys.push(key);
                    idx.hnsw.insert_one(vector.as_slice(), origin_id);
                }
            }
        }
    }

    /// Append newly-inserted vectors to the contiguous arena. Caller must
    /// have already inserted the records into `self.records` and confirmed
    /// the arena exists and isn't dirty.
    fn arena_apply_incremental(&mut self, new_keys: &[RecordKey]) {
        let Some(arena) = self.vector_arena.as_mut() else {
            return;
        };
        for key in new_keys {
            if let Some(record) = self.records.get(key) {
                arena.append(key.clone(), &record.vector);
            }
        }
    }

    /// Ensure the contiguous arena is materialised and fresh. Cheap when
    /// already clean; rebuilds from `self.records` (in BTreeMap order) on
    /// first call or after a delete. Allocates `dim * N` f32s.
    fn ensure_vector_arena(&mut self) -> &VectorArena {
        let needs_build = self
            .vector_arena
            .as_ref()
            .map_or(true, |a| self.vector_arena_dirty || a.dim != self.dimension);
        if needs_build {
            self.vector_arena = Some(VectorArena::rebuild_from(&self.records, self.dimension));
            self.vector_arena_dirty = false;
        }
        self.vector_arena.as_ref().unwrap()
    }

    /// Mark the given record keys as deleted in every HNSW graph they live
    /// in. The graph itself is not modified — search filters tombstoned
    /// `origin_id`s. A subsequent `compact()` will rebuild any graph whose
    /// dead ratio exceeds `IndexConfig.tombstone_rebuild_pct`.
    fn ann_apply_tombstones(&mut self, deleted_keys: &[RecordKey]) {
        if deleted_keys.is_empty() {
            return;
        }
        for key in deleted_keys {
            // Global graphs (per vector_name): every graph that contains
            // this key gets the corresponding origin_id tombstoned.
            for (_, idx) in self.ann.global.iter_mut() {
                if let Some(&origin_id) = idx.key_to_origin.get(key) {
                    idx.tombstones.insert(origin_id);
                }
            }
            // Per-namespace graphs: only the namespace this key belongs to
            // has a chance of containing it, but checking all of them is
            // fine — `key_to_origin.get` is O(1) and misses immediately.
            for (_, indexes) in self.ann.namespaces.iter_mut() {
                for (_, idx) in indexes.iter_mut() {
                    if let Some(&origin_id) = idx.key_to_origin.get(key) {
                        idx.tombstones.insert(origin_id);
                    }
                }
            }
        }
    }

    fn rebuild_ann(&mut self) {
        self.ann = AnnCatalog::default();
        let mut global_by_vector: BTreeMap<String, Vec<(RecordKey, &Vec<f32>)>> = BTreeMap::new();
        let mut by_namespace: BTreeMap<String, BTreeMap<String, Vec<(RecordKey, &Vec<f32>)>>> =
            BTreeMap::new();

        for (key, record) in &self.records {
            for (vector_name, vector) in record.dense_vectors() {
                global_by_vector
                    .entry(vector_name.to_owned())
                    .or_default()
                    .push((key.clone(), vector));
                by_namespace
                    .entry(record.namespace.clone())
                    .or_default()
                    .entry(vector_name.to_owned())
                    .or_default()
                    .push((key.clone(), vector));
            }
        }

        let cfg = self.index_config;
        self.ann.global = global_by_vector
            .into_iter()
            .filter_map(|(vector_name, records)| {
                if records.len() < ANN_MIN_POINTS {
                    None
                } else {
                    Some((vector_name, build_ann_index(records, self.metric, &cfg)))
                }
            })
            .collect();

        self.ann.namespaces = by_namespace
            .into_iter()
            .filter_map(|(namespace, indexes)| {
                let indexes = indexes
                    .into_iter()
                    .filter_map(|(vector_name, records)| {
                        if records.len() < ANN_MIN_POINTS {
                            None
                        } else {
                            Some((vector_name, build_ann_index(records, self.metric, &cfg)))
                        }
                    })
                    .collect::<BTreeMap<_, _>>();

                if indexes.is_empty() {
                    None
                } else {
                    Some((namespace, indexes))
                }
            })
            .collect();
    }

    fn try_load_ann_from_disk(&mut self) -> bool {
        let Some(parent) = self.path.parent() else {
            return false;
        };

        let Ok(entries) = read_ann_manifest(&ann_manifest_path(&self.path)) else {
            return false;
        };

        let expected = self.expected_ann_entries();
        if expected.len() != entries.len() {
            return false;
        }

        let mut loaded_global = BTreeMap::new();
        let mut loaded_namespaces: BTreeMap<String, BTreeMap<String, AnnIndex>> = BTreeMap::new();

        for expected_entry in expected {
            let Some(manifest_entry) = entries.iter().find(|entry| {
                entry.namespace == expected_entry.namespace
                    && entry.vector_name == expected_entry.vector_name
            }) else {
                return false;
            };

            if manifest_entry.record_count != expected_entry.record_count
                || manifest_entry.key_signature != expected_entry.key_signature
            {
                return false;
            }

            // For ANN2 manifests, use the persisted keys verbatim — they
            // match the `origin_id`s baked into the HNSW graph file. For
            // ANN1 (no persisted keys), fall back to the recomputed
            // BTreeMap-ordered list, which matches the way ANN1 graphs were
            // always built.
            let keys = if manifest_entry.keys.is_empty() {
                expected_entry.keys.clone()
            } else {
                // Defensive: persisted keys length must agree with the
                // declared record_count and the live record set, else the
                // manifest is inconsistent and we'd rather rebuild than
                // serve wrong neighbours.
                if manifest_entry.keys.len() != manifest_entry.record_count {
                    return false;
                }
                manifest_entry.keys.clone()
            };

            let Some(index) = load_ann_index(
                parent,
                &ann_basename(
                    &self.path,
                    expected_entry.namespace.as_deref(),
                    &expected_entry.vector_name,
                ),
                keys,
                self.metric,
            ) else {
                return false;
            };

            if let Some(namespace) = expected_entry.namespace {
                loaded_namespaces
                    .entry(namespace)
                    .or_default()
                    .insert(expected_entry.vector_name, index);
            } else {
                loaded_global.insert(expected_entry.vector_name, index);
            }
        }

        self.ann = AnnCatalog {
            global: loaded_global,
            namespaces: loaded_namespaces,
        };
        true
    }

    fn persist_ann_to_disk(&self) -> Result<()> {
        let Some(parent) = self.path.parent() else {
            return Ok(());
        };
        if !parent.exists() {
            return Ok(());
        }

        // Use `actual_ann_entries` (NOT `expected_ann_entries`) so the
        // persisted keys array matches the order the HNSW graph stored its
        // `origin_id`s in. After incremental inserts the in-memory keys vec
        // is in insertion order, which usually differs from BTreeMap order.
        let entries = self.actual_ann_entries();
        for entry in &entries {
            let basename = ann_basename(&self.path, entry.namespace.as_deref(), &entry.vector_name);
            let graph_path = parent.join(format!("{basename}.hnsw.graph"));
            let data_path = parent.join(format!("{basename}.hnsw.data"));
            if graph_path.exists() {
                fs::remove_file(&graph_path)?;
            }
            if data_path.exists() {
                fs::remove_file(&data_path)?;
            }

            let index = match &entry.namespace {
                Some(namespace) => self
                    .ann
                    .namespaces
                    .get(namespace)
                    .and_then(|indexes| indexes.get(&entry.vector_name)),
                None => self.ann.global.get(&entry.vector_name),
            };
            if let Some(index) = index {
                index.hnsw.file_dump(parent, &basename)?;
            }
        }

        write_ann_manifest(&ann_manifest_path(&self.path), &entries)
    }

    /// Like `expected_ann_entries`, but populates each entry's `keys` field
    /// from the actual in-memory `AnnIndex.keys` array (insertion order).
    /// This is what gets serialised into the ANN2 manifest, and matches the
    /// `origin_id`s baked into the dumped HNSW graph files.
    fn actual_ann_entries(&self) -> Vec<AnnManifestEntry> {
        let mut entries = Vec::new();
        for (vector_name, index) in &self.ann.global {
            if index.keys.len() < ANN_MIN_POINTS {
                continue;
            }
            entries.push(AnnManifestEntry {
                namespace: None,
                vector_name: vector_name.clone(),
                record_count: index.keys.len(),
                key_signature: record_key_signature(&index.keys),
                keys: index.keys.clone(),
            });
        }
        for (namespace, indexes) in &self.ann.namespaces {
            for (vector_name, index) in indexes {
                if index.keys.len() < ANN_MIN_POINTS {
                    continue;
                }
                entries.push(AnnManifestEntry {
                    namespace: Some(namespace.clone()),
                    vector_name: vector_name.clone(),
                    record_count: index.keys.len(),
                    key_signature: record_key_signature(&index.keys),
                    keys: index.keys.clone(),
                });
            }
        }
        entries
    }

    fn expected_ann_entries(&self) -> Vec<AnnManifestEntry> {
        let mut global: BTreeMap<String, Vec<RecordKey>> = BTreeMap::new();
        let mut by_namespace: BTreeMap<String, BTreeMap<String, Vec<RecordKey>>> = BTreeMap::new();

        for (key, record) in &self.records {
            for (vector_name, _) in record.dense_vectors() {
                global
                    .entry(vector_name.to_owned())
                    .or_default()
                    .push(key.clone());
                by_namespace
                    .entry(record.namespace.clone())
                    .or_default()
                    .entry(vector_name.to_owned())
                    .or_default()
                    .push(key.clone());
            }
        }

        let mut entries = Vec::new();

        for (vector_name, keys) in global {
            if keys.len() < ANN_MIN_POINTS {
                continue;
            }
            entries.push(AnnManifestEntry {
                namespace: None,
                vector_name,
                record_count: keys.len(),
                key_signature: record_key_signature(&keys),
                keys,
            });
        }

        for (namespace, indexes) in by_namespace {
            for (vector_name, keys) in indexes {
                if keys.len() < ANN_MIN_POINTS {
                    continue;
                }
                entries.push(AnnManifestEntry {
                    namespace: Some(namespace.clone()),
                    vector_name,
                    record_count: keys.len(),
                    key_signature: record_key_signature(&keys),
                    keys,
                });
            }
        }

        entries
    }

    fn rebuild_sparse_index(&mut self) {
        self.sparse_index = SparseIndex::default();
        self.sparse_index.doc_count = self.records.len();

        let mut total_doc_len = 0.0_f32;
        for (key, record) in &self.records {
            let doc_len = record
                .sparse
                .values()
                .copied()
                .filter(|weight| *weight > 0.0)
                .sum::<f32>();
            self.sparse_index.doc_lengths.insert(key.clone(), doc_len);
            total_doc_len += doc_len;

            for (term, weight) in &record.sparse {
                if *weight <= 0.0 {
                    continue;
                }
                self.sparse_index
                    .postings
                    .entry(term.clone())
                    .or_default()
                    .push(SparsePosting {
                        key: key.clone(),
                        term_weight: *weight,
                    });
            }
        }

        self.sparse_index.avg_doc_len = if self.sparse_index.doc_count == 0 {
            0.0
        } else {
            total_doc_len / self.sparse_index.doc_count as f32
        };
    }

    fn collect_results(
        &self,
        dense_query: Option<&[f32]>,
        sparse_query: Option<&SparseVector>,
        options: &HybridSearchOptions,
        namespace: Option<&str>,
        candidate_keys: Option<&[RecordKey]>,
        effective_dimension: Option<usize>,
    ) -> Vec<ScoredRecord<'_>> {
        let now = now_epoch_secs();
        let record_iter: Box<dyn Iterator<Item = &Record> + '_> = match candidate_keys {
            Some(keys) => Box::new(keys.iter().filter_map(|key| self.records.get(key))),
            None => Box::new(self.records.values()),
        };

        record_iter
            .filter(|record| {
                !record.is_expired_at(now)
                    && namespace
                        .map(|namespace| record.namespace == namespace)
                        .unwrap_or(true)
                    && (dense_query.is_none()
                        || record.vector_for(options.vector_name.as_deref()).is_some())
                    && options
                        .filter
                        .as_ref()
                        .map(|filter| filter.matches(&record.metadata))
                        .unwrap_or(true)
            })
            .map(|record| {
                let (dense_score, resolved_vector_name) =
                    if !options.multi_vector_queries.is_empty() {
                        // Multi-vector weighted search
                        let mut weighted_sum = 0.0_f32;
                        for (name, (query, weight)) in &options.multi_vector_queries {
                            if let Some(vector) = record.vector_for(Some(name.as_str())) {
                                weighted_sum += weight
                                    * score_dense_prefix(
                                        self.metric,
                                        query,
                                        vector,
                                        effective_dimension,
                                    );
                            }
                        }
                        (weighted_sum, None)
                    } else {
                        let score = dense_query
                            .and_then(|query| {
                                record
                                    .vector_for(options.vector_name.as_deref())
                                    .map(|vector| {
                                        score_dense_prefix(
                                            self.metric,
                                            query,
                                            vector,
                                            effective_dimension,
                                        )
                                    })
                            })
                            .unwrap_or(0.0);
                        (score, options.vector_name.clone())
                    };
                let sparse_score = sparse_query
                    .map(|query| {
                        self.bm25_score((record.namespace.clone(), record.id.clone()), query)
                    })
                    .unwrap_or(0.0);
                let record_key = (record.namespace.clone(), record.id.clone());
                let mut bm25_term_scores = BTreeMap::<String, f32>::new();
                let matched_terms = sparse_query
                    .map(|query| {
                        query
                            .keys()
                            .filter(|term| record.sparse.contains_key(*term))
                            .map(|term| {
                                let score = self.bm25_term_score(
                                    &record_key,
                                    term,
                                    *record.sparse.get(term).unwrap_or(&0.0),
                                );
                                bm25_term_scores.insert(term.clone(), score);
                                term.clone()
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                ScoredRecord {
                    record,
                    score: (options.dense_weight * dense_score)
                        + (options.sparse_weight * sparse_score),
                    dense_score,
                    sparse_score,
                    vector_name: resolved_vector_name,
                    matched_terms,
                    dense_rank: None,
                    sparse_rank: None,
                    bm25_term_scores,
                }
            })
            .collect()
    }

    fn ann_candidate_keys(
        &self,
        namespace: Option<&str>,
        vector_name: Option<&str>,
        query: &[f32],
        top_k: usize,
    ) -> Option<Vec<RecordKey>> {
        let index = match namespace {
            Some(namespace) => self
                .ann
                .namespaces
                .get(namespace)
                .and_then(|indexes| indexes.get(vector_name.unwrap_or(DEFAULT_VECTOR_NAME))),
            None => self
                .ann
                .global
                .get(vector_name.unwrap_or(DEFAULT_VECTOR_NAME)),
        }?;
        // Gate on live (non-tombstoned) record count: if half the graph is
        // dead, treat the live half as if it were the whole corpus.
        let live = index.live_count();
        if live < ANN_SEARCH_MIN_POINTS {
            return None;
        }

        let candidate_count = candidate_count(top_k, live);
        if candidate_count == 0 {
            return None;
        }

        // ef_search controls recall vs latency at query time. When the user
        // explicitly sets `IndexConfig.ef_search`, honour it directly.
        // Otherwise default to max(candidate_count, ef_construction) which is
        // a conservative high-recall heuristic.
        let mut ef_search = match self.index_config.ef_search {
            Some(ef) => ef.max(candidate_count),
            None => candidate_count.max(self.index_config.ef_construction),
        };
        // Over-fetch to compensate for tombstoned candidates we'll drop. Cap
        // at the live count so we don't waste work; we'd never get more
        // distinct results than that anyway.
        if !index.tombstones.is_empty() {
            let dead = index.tombstones.len();
            ef_search = ef_search
                .saturating_add(dead.min(ef_search))
                .min(index.keys.len());
        }
        let fetch_count = candidate_count
            .saturating_add(index.tombstones.len().min(candidate_count))
            .min(index.keys.len());
        let neighbours = index.hnsw.search(query, fetch_count, ef_search);
        Some(
            neighbours
                .into_iter()
                .filter(|n| !index.tombstones.contains(&n.d_id))
                .filter_map(|neighbour| index.keys.get(neighbour.d_id).cloned())
                .take(candidate_count)
                .collect(),
        )
    }

    fn sparse_candidate_keys(
        &self,
        namespace: Option<&str>,
        sparse_query: &SparseVector,
        top_k: usize,
    ) -> Vec<RecordKey> {
        let mut scores = BTreeMap::<RecordKey, f32>::new();
        for (term, query_weight) in sparse_query {
            let Some(postings) = self.sparse_index.postings.get(term) else {
                continue;
            };
            for posting in postings {
                if namespace
                    .map(|namespace| posting.key.0 == namespace)
                    .unwrap_or(true)
                {
                    let key = posting.key.clone();
                    *scores.entry(key.clone()).or_insert(0.0) +=
                        *query_weight * self.bm25_term_score(&key, term, posting.term_weight);
                }
            }
        }

        let mut scored = scores.into_iter().collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.0.cmp(&right.0.0))
                .then_with(|| left.0.1.cmp(&right.0.1))
        });
        let limit = candidate_count(top_k, scored.len());
        scored.into_iter().take(limit).map(|(key, _)| key).collect()
    }

    fn bm25_score(&self, key: RecordKey, sparse_query: &SparseVector) -> f32 {
        sparse_query
            .iter()
            .map(|(term, query_weight)| {
                let doc_weight = self
                    .records
                    .get(&key)
                    .and_then(|record| record.sparse.get(term))
                    .copied()
                    .unwrap_or(0.0);
                if doc_weight <= 0.0 {
                    0.0
                } else {
                    *query_weight * self.bm25_term_score(&key, term, doc_weight)
                }
            })
            .sum()
    }

    fn bm25_term_score(&self, key: &RecordKey, term: &str, doc_weight: f32) -> f32 {
        if doc_weight <= 0.0 || self.sparse_index.doc_count == 0 {
            return 0.0;
        }

        let df = self
            .sparse_index
            .postings
            .get(term)
            .map_or(0, |postings| postings.len()) as f32;
        if df == 0.0 {
            return 0.0;
        }

        let idf = (((self.sparse_index.doc_count as f32 - df + 0.5) / (df + 0.5)) + 1.0).ln();
        let doc_len = self
            .sparse_index
            .doc_lengths
            .get(key)
            .copied()
            .unwrap_or(0.0);
        let norm = if self.sparse_index.avg_doc_len > 0.0 {
            1.0 - BM25_B + BM25_B * (doc_len / self.sparse_index.avg_doc_len)
        } else {
            1.0
        };

        idf * ((doc_weight * (BM25_K1 + 1.0)) / (doc_weight + BM25_K1 * norm))
    }

    #[cfg(test)]
    fn has_ann_index(&self, namespace: Option<&str>, vector_name: Option<&str>) -> bool {
        match namespace {
            Some(namespace) => self.ann.namespaces.get(namespace).is_some_and(|indexes| {
                indexes.contains_key(vector_name.unwrap_or(DEFAULT_VECTOR_NAME))
            }),
            None => self
                .ann
                .global
                .contains_key(vector_name.unwrap_or(DEFAULT_VECTOR_NAME)),
        }
    }

    fn record_from_parts(
        &self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        vectors: NamedVectors,
        sparse: SparseVector,
        metadata: Metadata,
    ) -> Result<Record> {
        self.record_from_parts_full(
            namespace,
            id,
            vector,
            vectors,
            sparse,
            metadata,
            MultiVectors::new(),
        )
    }

    fn record_from_parts_full(
        &self,
        namespace: impl Into<String>,
        id: impl Into<String>,
        vector: impl Into<Vec<f32>>,
        vectors: NamedVectors,
        sparse: SparseVector,
        metadata: Metadata,
        multi_vectors: MultiVectors,
    ) -> Result<Record> {
        let vector = vector.into();
        self.validate_vector(&vector)?;

        for (vector_name, named_vector) in &vectors {
            if vector_name.is_empty() {
                return Err(VectLiteError::InvalidFormat(
                    "named vectors must not use an empty name".to_owned(),
                ));
            }
            self.validate_vector(named_vector)?;
        }

        // Validate multi-vector dimensions
        for (space_name, token_vectors) in &multi_vectors {
            if space_name.is_empty() {
                return Err(VectLiteError::InvalidFormat(
                    "multi-vector space names must not be empty".to_owned(),
                ));
            }
            for token_vec in token_vectors {
                if token_vec.is_empty() {
                    return Err(VectLiteError::InvalidFormat(format!(
                        "multi-vector space '{space_name}' contains an empty token vector"
                    )));
                }
                // Token vectors within a space must all have the same dimension,
                // but that dimension can differ from the database dimension.
                if !token_vectors.is_empty() && token_vec.len() != token_vectors[0].len() {
                    return Err(VectLiteError::InvalidFormat(format!(
                        "multi-vector space '{space_name}' has inconsistent token dimensions: expected {}, found {}",
                        token_vectors[0].len(),
                        token_vec.len(),
                    )));
                }
            }
        }

        Ok(Record {
            namespace: namespace.into(),
            id: id.into(),
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors,
            expires_at: None,
        })
    }
}

impl ScoredRecord<'_> {
    fn into_search_result(self) -> SearchResult {
        SearchResult {
            namespace: self.record.namespace.clone(),
            id: self.record.id.clone(),
            score: self.score,
            dense_score: self.dense_score,
            sparse_score: self.sparse_score,
            vector_name: self.vector_name,
            matched_terms: self.matched_terms,
            dense_rank: self.dense_rank,
            sparse_rank: self.sparse_rank,
            metadata: self.record.metadata.clone(),
            bm25_term_scores: self.bm25_term_scores,
        }
    }
}

impl FusionStrategy {
    fn label(&self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Rrf { .. } => "rrf",
        }
    }
}

fn ensure_dimension(dimension: usize) -> Result<()> {
    if dimension == 0 {
        return Err(VectLiteError::InvalidFormat(
            "dimension must be greater than zero".to_owned(),
        ));
    }

    Ok(())
}

/// MaxSim scoring (ColBERT-style late interaction).
/// For each query token, find the maximum similarity against any document
/// token using the given metric, then sum those maxima across all query tokens.
fn maxsim_score(query_tokens: &[&[f32]], doc_tokens: &[Vec<f32>], metric: DistanceMetric) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for q_token in query_tokens {
        let mut best = f32::NEG_INFINITY;
        for d_token in doc_tokens {
            let sim = metric.score(q_token, d_token);
            if sim > best {
                best = sim;
            }
        }
        total += best;
    }
    total
}

fn sparse_dot_product(left: &SparseVector, right: &SparseVector) -> f32 {
    let (small, large) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };

    small.iter().fold(0.0_f32, |acc, (term, weight)| {
        acc + (*weight * large.get(term).copied().unwrap_or(0.0))
    })
}

fn score_dense_prefix(
    metric: DistanceMetric,
    left: &[f32],
    right: &[f32],
    effective_dimension: Option<usize>,
) -> f32 {
    let dimension = effective_dimension
        .unwrap_or_else(|| left.len().min(right.len()))
        .min(left.len())
        .min(right.len());
    metric.score(&left[..dimension], &right[..dimension])
}

fn build_ann_index(
    records: Vec<(RecordKey, &Vec<f32>)>,
    metric: DistanceMetric,
    config: &IndexConfig,
) -> AnnIndex {
    let max_layer = compute_hnsw_layers(records.len());
    let count = records.len();
    let use_parallel = count >= config.parallel_insert_threshold;

    macro_rules! build_hnsw {
        ($dist_type:ty, $dist_val:expr, $variant:ident) => {{
            let mut hnsw = Hnsw::<f32, $dist_type>::new(
                config.m,
                count.max(1),
                max_layer,
                config.ef_construction,
                $dist_val,
            );
            let mut keys = Vec::with_capacity(count);
            let mut key_to_origin = HashMap::with_capacity(count);
            if use_parallel {
                // hnsw_rs's `parallel_insert` takes `&[(&Vec<T>, usize)]`
                // (the API is built around owned-Vec borrows) and uses Rayon
                // internally so the dominant cost (distance calculations
                // during graph neighbour selection) is multi-threaded.
                let mut batch: Vec<(&Vec<f32>, usize)> = Vec::with_capacity(count);
                for (origin_id, (key, vector)) in records.into_iter().enumerate() {
                    batch.push((vector, origin_id));
                    key_to_origin.insert(key.clone(), origin_id);
                    keys.push(key);
                }
                hnsw.parallel_insert(&batch);
            } else {
                for (origin_id, (key, vector)) in records.into_iter().enumerate() {
                    hnsw.insert((vector.as_slice(), origin_id));
                    key_to_origin.insert(key.clone(), origin_id);
                    keys.push(key);
                }
            }
            hnsw.set_searching_mode(true);
            AnnIndex {
                hnsw: AnnHnsw::$variant(hnsw),
                keys,
                key_to_origin,
                tombstones: HashSet::new(),
            }
        }};
    }

    match metric {
        DistanceMetric::Cosine => build_hnsw!(DistCosine, DistCosine {}, Cosine),
        DistanceMetric::Euclidean => build_hnsw!(DistL2, DistL2 {}, Euclidean),
        DistanceMetric::DotProduct => build_hnsw!(DistDot, DistDot {}, DotProduct),
        DistanceMetric::Manhattan => build_hnsw!(DistL1, DistL1 {}, Manhattan),
    }
}

fn compute_hnsw_layers(record_count: usize) -> usize {
    let _ = record_count;
    16
}

fn candidate_count(top_k: usize, total: usize) -> usize {
    if total <= 256 {
        return total;
    }

    let requested = top_k.max(1);
    requested
        .saturating_mul(ANN_OVERSAMPLE)
        .max(ANN_MIN_CANDIDATES)
        .min(total)
}

fn timeout_duration(timeout_secs: f64, label: &str) -> Result<std::time::Duration> {
    if !timeout_secs.is_finite() || timeout_secs < 0.0 {
        return Err(VectLiteError::InvalidFormat(format!(
            "{label} must be a finite, non-negative number of seconds"
        )));
    }
    Ok(std::time::Duration::from_secs_f64(timeout_secs))
}

fn wal_path(path: &Path) -> PathBuf {
    let mut wal = path.as_os_str().to_os_string();
    wal.push(".wal");
    PathBuf::from(wal)
}

fn lock_path(path: &Path) -> PathBuf {
    let mut lock = path.as_os_str().to_os_string();
    lock.push(".lock");
    PathBuf::from(lock)
}

fn quantization_params_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_os_string();
    p.push(".quant");
    PathBuf::from(p)
}

fn multi_vector_quantization_params_path(path: &Path, space: &str) -> PathBuf {
    let mut p = path.as_os_str().to_os_string();
    p.push(format!(".mvquant.{space}"));
    PathBuf::from(p)
}

fn acquire_exclusive_lock(path: &Path) -> Result<File> {
    acquire_exclusive_lock_with_timeout(path, None)
}

fn acquire_exclusive_lock_with_timeout(
    path: &Path,
    timeout: Option<std::time::Duration>,
) -> Result<File> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path(path))?;

    match timeout {
        None => {
            file.try_lock_exclusive().map_err(|err| {
                VectLiteError::LockContention(format!(
                    "could not acquire exclusive lock on '{}': {err}",
                    path.display()
                ))
            })?;
        }
        Some(duration) => {
            let start = Instant::now();
            let interval = std::time::Duration::from_millis(50);
            loop {
                match file.try_lock_exclusive() {
                    Ok(()) => break,
                    Err(err) => {
                        if start.elapsed() >= duration {
                            return Err(VectLiteError::LockContention(format!(
                                "could not acquire exclusive lock on '{}' after {:.1}s: {err}",
                                path.display(),
                                duration.as_secs_f64()
                            )));
                        }
                        std::thread::sleep(interval);
                    }
                }
            }
        }
    }
    Ok(file)
}

fn acquire_shared_lock_with_timeout(
    path: &Path,
    timeout: Option<std::time::Duration>,
) -> Result<File> {
    let lock_file = lock_path(path);
    if !lock_file.exists() {
        // Lock file may not exist yet for read-only opens on existing dbs
        if let Some(parent) = lock_file.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_file)?;

    match timeout {
        None => {
            file.try_lock_shared().map_err(|err| {
                VectLiteError::LockContention(format!(
                    "could not acquire shared lock on '{}': {err}",
                    path.display()
                ))
            })?;
        }
        Some(duration) => {
            let start = Instant::now();
            let interval = std::time::Duration::from_millis(50);
            loop {
                match file.try_lock_shared() {
                    Ok(()) => break,
                    Err(err) => {
                        if start.elapsed() >= duration {
                            return Err(VectLiteError::LockContention(format!(
                                "could not acquire shared lock on '{}' after {:.1}s: {err}",
                                path.display(),
                                duration.as_secs_f64()
                            )));
                        }
                        std::thread::sleep(interval);
                    }
                }
            }
        }
    }
    Ok(file)
}

fn ann_manifest_path(path: &Path) -> PathBuf {
    let mut manifest = path.as_os_str().to_os_string();
    manifest.push(".ann");
    PathBuf::from(manifest)
}

fn ann_basename(path: &Path, namespace: Option<&str>, vector_name: &str) -> String {
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("vectlite");
    let ns_hex = hex_encode(namespace.unwrap_or(DEFAULT_NAMESPACE).as_bytes());
    let vn_hex = hex_encode(vector_name.as_bytes());
    // Use "_" sentinel for empty components to avoid triple-dot filenames
    // like "c.vdb.ann...hnsw.data".
    let ns_part = if ns_hex.is_empty() { "_" } else { &ns_hex };
    let vn_part = if vn_hex.is_empty() { "_" } else { &vn_hex };
    format!("{stem}.ann.{ns_part}.{vn_part}")
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Order-independent FNV-1a hash over a set of record keys. We sort first so
/// the signature only depends on the SET of keys, not the order they were
/// inserted. Callers can use this to check whether a persisted ANN graph
/// matches the live record set regardless of whether the live `keys` vec is
/// BTreeMap-ordered (full rebuild) or insertion-ordered (incremental
/// updates).
///
/// Historical note: previously the input was always BTreeMap-iterated and
/// therefore already sorted, so the sort step is a no-op for old ANN1
/// manifests — backwards compatible.
fn record_key_signature(keys: &[RecordKey]) -> u64 {
    let mut sorted: Vec<&RecordKey> = keys.iter().collect();
    sorted.sort();
    let mut state = 0xcbf29ce484222325_u64;
    for (namespace, id) in sorted {
        for byte in namespace
            .as_bytes()
            .iter()
            .chain(std::iter::once(&0xff))
            .chain(id.as_bytes().iter())
            .chain(std::iter::once(&0xfe))
        {
            state ^= *byte as u64;
            state = state.wrapping_mul(0x100000001b3);
        }
    }
    state
}

fn load_ann_index(
    directory: &Path,
    basename: &str,
    keys: Vec<RecordKey>,
    metric: DistanceMetric,
) -> Option<AnnIndex> {
    let reloader = Box::leak(Box::new(HnswIo::new(directory, basename)));

    macro_rules! load_with_dist {
        ($dist_val:expr, $variant:ident) => {{
            let mut hnsw = reloader.load_hnsw_with_dist($dist_val).ok()?;
            hnsw.set_searching_mode(true);
            let key_to_origin = keys
                .iter()
                .enumerate()
                .map(|(i, k)| (k.clone(), i))
                .collect();
            Some(AnnIndex {
                hnsw: AnnHnsw::$variant(hnsw),
                keys,
                key_to_origin,
                tombstones: HashSet::new(),
            })
        }};
    }

    match metric {
        DistanceMetric::Cosine => load_with_dist!(DistCosine {}, Cosine),
        DistanceMetric::Euclidean => load_with_dist!(DistL2 {}, Euclidean),
        DistanceMetric::DotProduct => load_with_dist!(DistDot {}, DotProduct),
        DistanceMetric::Manhattan => load_with_dist!(DistL1 {}, Manhattan),
    }
}

/// Write the ANN sidecar manifest. We use format `ANN2`, which (compared to
/// the original `ANN1`) also serialises the actual key array per index in
/// the order the HNSW knows its `origin_id`s. This is required for
/// incremental insertion: without it, a reload would associate the wrong
/// (BTreeMap-ordered) record key with each HNSW origin_id whenever the in
/// memory key array isn't sorted (which happens any time we incrementally
/// append).
fn write_ann_manifest(path: &Path, entries: &[AnnManifestEntry]) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    file.write_all(b"ANN2")?;
    write_u32(&mut file, u32_from_usize(entries.len())?)?;
    for entry in entries {
        write_u8(&mut file, u8::from(entry.namespace.is_some()))?;
        if let Some(namespace) = &entry.namespace {
            write_string(&mut file, namespace)?;
        }
        write_string(&mut file, &entry.vector_name)?;
        write_u64(&mut file, u64_from_usize(entry.record_count)?)?;
        write_u64(&mut file, entry.key_signature)?;
        // ANN2 addition: the full keys array in insertion order.
        write_u64(&mut file, u64_from_usize(entry.keys.len())?)?;
        for (ns, id) in &entry.keys {
            write_string(&mut file, ns)?;
            write_string(&mut file, id)?;
        }
    }
    file.flush()?;
    file.get_ref().sync_all()?;
    Ok(())
}

fn read_ann_manifest(path: &Path) -> Result<Vec<AnnManifestEntry>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)?;
    let version = match &magic {
        b"ANN1" => 1u8,
        b"ANN2" => 2u8,
        _ => {
            return Err(VectLiteError::InvalidFormat(
                "invalid ANN manifest".to_owned(),
            ));
        }
    };

    let count = usize_from_u32(read_u32(&mut file)?)?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let has_namespace = read_u8(&mut file)? != 0;
        let namespace = if has_namespace {
            Some(read_string(&mut file)?)
        } else {
            None
        };
        let vector_name = read_string(&mut file)?;
        let record_count = usize_from_u64(read_u64(&mut file)?)?;
        let key_signature = read_u64(&mut file)?;
        let keys = if version >= 2 {
            let n = usize_from_u64(read_u64(&mut file)?)?;
            let mut keys = Vec::with_capacity(n);
            for _ in 0..n {
                let ns = read_string(&mut file)?;
                let id = read_string(&mut file)?;
                keys.push((ns, id));
            }
            keys
        } else {
            // ANN1 had no persisted keys; caller falls back to recomputing
            // them from `self.records` (which yields BTreeMap-sorted keys,
            // matching the order ANN1 indexes were always built in).
            Vec::new()
        };
        entries.push(AnnManifestEntry {
            namespace,
            vector_name,
            record_count,
            key_signature,
            keys,
        });
    }
    Ok(entries)
}

fn resolve_fetch_k(
    top_k: usize,
    requested_fetch_k: usize,
    total_records: usize,
    mmr_lambda: Option<f32>,
) -> usize {
    if total_records == 0 {
        return 0;
    }

    let default_fetch_k = if mmr_lambda.is_some() {
        top_k.max(1).saturating_mul(4)
    } else {
        top_k.max(1)
    };

    requested_fetch_k
        .max(top_k.max(1))
        .max(default_fetch_k)
        .min(total_records)
}

fn sort_scored_records(results: &mut [ScoredRecord<'_>]) {
    results.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.record.namespace.cmp(&right.record.namespace))
            .then_with(|| left.record.id.cmp(&right.record.id))
    });
}

fn apply_rank_metadata(results: &mut [ScoredRecord<'_>]) {
    let mut dense_order = (0..results.len()).collect::<Vec<_>>();
    dense_order.sort_by(|left, right| {
        results[*right]
            .dense_score
            .total_cmp(&results[*left].dense_score)
            .then_with(|| {
                results[*left]
                    .record
                    .namespace
                    .cmp(&results[*right].record.namespace)
            })
            .then_with(|| results[*left].record.id.cmp(&results[*right].record.id))
    });
    for (rank, index) in dense_order.into_iter().enumerate() {
        if results[index].dense_score > 0.0 {
            results[index].dense_rank = Some(rank + 1);
        }
    }

    let mut sparse_order = (0..results.len()).collect::<Vec<_>>();
    sparse_order.sort_by(|left, right| {
        results[*right]
            .sparse_score
            .total_cmp(&results[*left].sparse_score)
            .then_with(|| {
                results[*left]
                    .record
                    .namespace
                    .cmp(&results[*right].record.namespace)
            })
            .then_with(|| results[*left].record.id.cmp(&results[*right].record.id))
    });
    for (rank, index) in sparse_order.into_iter().enumerate() {
        if results[index].sparse_score > 0.0 {
            results[index].sparse_rank = Some(rank + 1);
        }
    }
}

fn apply_fusion_strategy(
    results: &mut [ScoredRecord<'_>],
    fusion: &FusionStrategy,
    dense_weight: f32,
    sparse_weight: f32,
) {
    match fusion {
        FusionStrategy::Linear => {
            for result in results {
                result.score =
                    (dense_weight * result.dense_score) + (sparse_weight * result.sparse_score);
            }
        }
        FusionStrategy::Rrf { rank_constant } => {
            let rank_constant = (*rank_constant).max(1) as f32;
            for result in results {
                let dense_component = result
                    .dense_rank
                    .map(|rank| dense_weight / (rank_constant + rank as f32))
                    .unwrap_or(0.0);
                let sparse_component = result
                    .sparse_rank
                    .map(|rank| sparse_weight / (rank_constant + rank as f32))
                    .unwrap_or(0.0);
                result.score = dense_component + sparse_component;
            }
        }
    }
}

fn merge_candidate_keys(
    dense_candidates: Option<&[RecordKey]>,
    sparse_candidates: Option<&[RecordKey]>,
) -> Option<Vec<RecordKey>> {
    let mut merged = BTreeSet::new();
    if let Some(dense_candidates) = dense_candidates {
        merged.extend(dense_candidates.iter().cloned());
    }
    if let Some(sparse_candidates) = sparse_candidates {
        merged.extend(sparse_candidates.iter().cloned());
    }

    if merged.is_empty() {
        None
    } else {
        Some(merged.into_iter().collect())
    }
}

fn apply_mmr<'a>(
    candidates: Vec<ScoredRecord<'a>>,
    top_k: usize,
    mmr_lambda: f32,
    dense_weight: f32,
    sparse_weight: f32,
    vector_name: Option<&str>,
    metric: DistanceMetric,
    effective_dimension: Option<usize>,
) -> Vec<ScoredRecord<'a>> {
    let limit = top_k.min(candidates.len());
    if limit <= 1 {
        return candidates.into_iter().take(limit).collect();
    }

    let mut selected: Vec<ScoredRecord<'a>> = Vec::with_capacity(limit);
    let mut used = vec![false; candidates.len()];

    while selected.len() < limit {
        let mut best: Option<(usize, f32)> = None;

        for (index, candidate) in candidates.iter().enumerate() {
            if used[index] {
                continue;
            }

            let diversity_penalty = selected
                .iter()
                .map(|selected_candidate| {
                    record_similarity(
                        candidate.record,
                        selected_candidate.record,
                        dense_weight,
                        sparse_weight,
                        vector_name,
                        metric,
                        effective_dimension,
                    )
                })
                .fold(0.0_f32, f32::max);

            let mmr_score = if selected.is_empty() {
                candidate.score
            } else {
                (mmr_lambda * candidate.score) - ((1.0 - mmr_lambda) * diversity_penalty)
            };

            let replace_best = match best {
                Some((best_index, best_score)) => {
                    mmr_score > best_score
                        || (mmr_score == best_score
                            && candidate.score > candidates[best_index].score)
                        || (mmr_score == best_score
                            && candidate.score == candidates[best_index].score
                            && candidate.record.namespace < candidates[best_index].record.namespace)
                        || (mmr_score == best_score
                            && candidate.score == candidates[best_index].score
                            && candidate.record.namespace
                                == candidates[best_index].record.namespace
                            && candidate.record.id < candidates[best_index].record.id)
                }
                None => true,
            };

            if replace_best {
                best = Some((index, mmr_score));
            }
        }

        let Some((best_index, _)) = best else {
            break;
        };
        used[best_index] = true;
        selected.push(candidates[best_index].clone());
    }

    selected
}

fn record_similarity(
    left: &Record,
    right: &Record,
    dense_weight: f32,
    sparse_weight: f32,
    vector_name: Option<&str>,
    metric: DistanceMetric,
    effective_dimension: Option<usize>,
) -> f32 {
    let dense_score = match (left.vector_for(vector_name), right.vector_for(vector_name)) {
        (Some(left), Some(right)) => score_dense_prefix(metric, left, right, effective_dimension),
        _ => 0.0,
    };

    (dense_weight * dense_score) + (sparse_weight * sparse_dot_product(&left.sparse, &right.sparse))
}

fn temp_path(path: &Path) -> PathBuf {
    let mut temp = path.as_os_str().to_os_string();
    temp.push(".tmp");
    PathBuf::from(temp)
}

fn write_metadata_value(writer: &mut impl Write, value: &MetadataValue) -> Result<()> {
    write_u8(writer, value.type_tag())?;
    match value {
        MetadataValue::String(value) => write_string(writer, value)?,
        MetadataValue::Integer(value) => write_i64(writer, *value)?,
        MetadataValue::Float(value) => write_f64(writer, *value)?,
        MetadataValue::Boolean(value) => write_u8(writer, u8::from(*value))?,
        MetadataValue::Null => {}
        MetadataValue::List(values) => {
            write_u32(writer, u32_from_usize(values.len())?)?;
            for item in values {
                write_metadata_value(writer, item)?;
            }
        }
        MetadataValue::Map(entries) => {
            write_u32(writer, u32_from_usize(entries.len())?)?;
            for (key, val) in entries {
                write_string(writer, key)?;
                write_metadata_value(writer, val)?;
            }
        }
    }
    Ok(())
}

fn read_metadata_value(reader: &mut impl Read) -> Result<MetadataValue> {
    let tag = read_u8(reader)?;
    let value = match tag {
        TYPE_STRING => MetadataValue::String(read_string(reader)?),
        TYPE_INTEGER => MetadataValue::Integer(read_i64(reader)?),
        TYPE_FLOAT => MetadataValue::Float(read_f64(reader)?),
        TYPE_BOOLEAN => MetadataValue::Boolean(read_u8(reader)? != 0),
        TYPE_NULL => MetadataValue::Null,
        TYPE_LIST => {
            let count = usize_from_u32(read_u32(reader)?)?;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_metadata_value(reader)?);
            }
            MetadataValue::List(items)
        }
        TYPE_MAP => {
            let count = usize_from_u32(read_u32(reader)?)?;
            let mut entries = BTreeMap::new();
            for _ in 0..count {
                let key = read_string(reader)?;
                let val = read_metadata_value(reader)?;
                entries.insert(key, val);
            }
            MetadataValue::Map(entries)
        }
        other => {
            return Err(VectLiteError::InvalidFormat(format!(
                "unknown metadata value tag {other}"
            )));
        }
    };
    Ok(value)
}

fn write_sparse_vector(writer: &mut impl Write, sparse: &SparseVector) -> Result<()> {
    write_u32(writer, u32_from_usize(sparse.len())?)?;
    for (term, weight) in sparse {
        write_string(writer, term)?;
        write_f32(writer, *weight)?;
    }
    Ok(())
}

fn read_sparse_vector(reader: &mut impl Read) -> Result<SparseVector> {
    let entry_count = usize_from_u32(read_u32(reader)?)?;
    let mut sparse = SparseVector::new();

    for _ in 0..entry_count {
        let term = read_string(reader)?;
        let weight = read_f32(reader)?;
        sparse.insert(term, weight);
    }

    Ok(sparse)
}

fn write_named_vectors(writer: &mut impl Write, vectors: &NamedVectors) -> Result<()> {
    write_u32(writer, u32_from_usize(vectors.len())?)?;
    for (name, vector) in vectors {
        write_string(writer, name)?;
        write_u32(writer, u32_from_usize(vector.len())?)?;
        for value in vector {
            write_f32(writer, *value)?;
        }
    }
    Ok(())
}

fn read_named_vectors(reader: &mut impl Read, dimension: usize) -> Result<NamedVectors> {
    let vector_count = usize_from_u32(read_u32(reader)?)?;
    let mut vectors = NamedVectors::new();

    for _ in 0..vector_count {
        let name = read_string(reader)?;
        if name.is_empty() {
            return Err(VectLiteError::InvalidFormat(
                "named vectors must not use an empty name".to_owned(),
            ));
        }

        let vector_len = usize_from_u32(read_u32(reader)?)?;
        if vector_len != dimension {
            return Err(VectLiteError::InvalidFormat(format!(
                "named vector {name} has length {vector_len}, expected {dimension}"
            )));
        }

        let mut vector = Vec::with_capacity(vector_len);
        for _ in 0..vector_len {
            vector.push(read_f32(reader)?);
        }

        vectors.insert(name, vector);
    }

    Ok(vectors)
}

fn write_multi_vectors(writer: &mut impl Write, multi_vectors: &MultiVectors) -> Result<()> {
    write_u32(writer, u32_from_usize(multi_vectors.len())?)?;
    for (name, token_vectors) in multi_vectors {
        write_string(writer, name)?;
        // Write the token dimension (0 if empty)
        let token_dim = token_vectors.first().map_or(0, |v| v.len());
        write_u32(writer, u32_from_usize(token_dim)?)?;
        write_u32(writer, u32_from_usize(token_vectors.len())?)?;
        for token_vec in token_vectors {
            for value in token_vec {
                write_f32(writer, *value)?;
            }
        }
    }
    Ok(())
}

fn read_multi_vectors(reader: &mut impl Read) -> Result<MultiVectors> {
    let space_count = usize_from_u32(read_u32(reader)?)?;
    let mut multi_vectors = MultiVectors::new();

    for _ in 0..space_count {
        let name = read_string(reader)?;
        let token_dim = usize_from_u32(read_u32(reader)?)?;
        let token_count = usize_from_u32(read_u32(reader)?)?;
        let mut token_vectors = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let mut vec = Vec::with_capacity(token_dim);
            for _ in 0..token_dim {
                vec.push(read_f32(reader)?);
            }
            token_vectors.push(vec);
        }
        multi_vectors.insert(name, token_vectors);
    }

    Ok(multi_vectors)
}

fn write_wal_op(writer: &mut impl Write, op: &WalOp) -> Result<()> {
    match op {
        WalOp::Upsert(record) => {
            write_u8(writer, 1)?;
            write_string(writer, &record.namespace)?;
            write_string(writer, &record.id)?;
            write_u32(writer, u32_from_usize(record.metadata.len())?)?;
            for (key, value) in &record.metadata {
                write_string(writer, key)?;
                write_metadata_value(writer, value)?;
            }
            write_u32(writer, u32_from_usize(record.vector.len())?)?;
            for value in &record.vector {
                write_f32(writer, *value)?;
            }
            write_named_vectors(writer, &record.vectors)?;
            write_sparse_vector(writer, &record.sparse)?;
            write_multi_vectors(writer, &record.multi_vectors)?;
            write_f64(writer, record.expires_at.unwrap_or(0.0))?;
        }
        WalOp::Delete { namespace, id } => {
            write_u8(writer, 2)?;
            write_string(writer, namespace)?;
            write_string(writer, id)?;
        }
        WalOp::UpdateMetadata {
            namespace,
            id,
            metadata,
        } => {
            write_u8(writer, 3)?;
            write_string(writer, namespace)?;
            write_string(writer, id)?;
            write_u32(writer, u32_from_usize(metadata.len())?)?;
            for (key, value) in metadata {
                write_string(writer, key)?;
                write_metadata_value(writer, value)?;
            }
        }
        WalOp::SetTtl {
            namespace,
            id,
            expires_at,
        } => {
            write_u8(writer, 4)?;
            write_string(writer, namespace)?;
            write_string(writer, id)?;
            write_f64(writer, expires_at.unwrap_or(0.0))?;
        }
    }
    Ok(())
}

fn read_wal_op(reader: &mut impl Read, dimension: usize) -> Result<WalOp> {
    match read_u8(reader)? {
        1 => {
            let namespace = read_string(reader)?;
            let id = read_string(reader)?;
            let metadata_count = usize_from_u32(read_u32(reader)?)?;
            let mut metadata = Metadata::new();
            for _ in 0..metadata_count {
                let key = read_string(reader)?;
                let value = read_metadata_value(reader)?;
                metadata.insert(key, value);
            }
            let vector_len = usize_from_u32(read_u32(reader)?)?;
            if vector_len != dimension {
                return Err(VectLiteError::InvalidFormat(format!(
                    "wal record {id} has vector length {vector_len}, expected {dimension}"
                )));
            }
            let mut vector = Vec::with_capacity(vector_len);
            for _ in 0..vector_len {
                vector.push(read_f32(reader)?);
            }
            let vectors = read_named_vectors(reader, dimension)?;
            let sparse = read_sparse_vector(reader)?;
            let multi_vectors = read_multi_vectors(reader)?;
            let expires_at = {
                let ts = read_f64(reader)?;
                if ts == 0.0 { None } else { Some(ts) }
            };
            Ok(WalOp::Upsert(Record {
                namespace,
                id,
                vector,
                vectors,
                sparse,
                metadata,
                multi_vectors,
                expires_at,
            }))
        }
        2 => Ok(WalOp::Delete {
            namespace: read_string(reader)?,
            id: read_string(reader)?,
        }),
        3 => {
            let namespace = read_string(reader)?;
            let id = read_string(reader)?;
            let metadata_count = usize_from_u32(read_u32(reader)?)?;
            let mut metadata = Metadata::new();
            for _ in 0..metadata_count {
                let key = read_string(reader)?;
                let value = read_metadata_value(reader)?;
                metadata.insert(key, value);
            }
            Ok(WalOp::UpdateMetadata {
                namespace,
                id,
                metadata,
            })
        }
        4 => {
            let namespace = read_string(reader)?;
            let id = read_string(reader)?;
            let ts = read_f64(reader)?;
            let expires_at = if ts == 0.0 { None } else { Some(ts) };
            Ok(WalOp::SetTtl {
                namespace,
                id,
                expires_at,
            })
        }
        other => Err(VectLiteError::InvalidFormat(format!(
            "unknown WAL op tag {other}"
        ))),
    }
}

fn write_string(writer: &mut impl Write, value: &str) -> Result<()> {
    write_u32(writer, u32_from_usize(value.len())?)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let length = usize_from_u32(read_u32(reader)?)?;
    let mut buffer = vec![0_u8; length];
    reader.read_exact(&mut buffer)?;
    String::from_utf8(buffer)
        .map_err(|err| VectLiteError::InvalidFormat(format!("invalid utf-8 string: {err}")))
}

fn write_u8(writer: &mut impl Write, value: u8) -> io::Result<()> {
    writer.write_all(&[value])
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut buffer = [0_u8; 1];
    reader.read_exact(&mut buffer)?;
    Ok(buffer[0])
}

fn write_u16(writer: &mut impl Write, value: u16) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    let mut buffer = [0_u8; 2];
    reader.read_exact(&mut buffer)?;
    Ok(u16::from_le_bytes(buffer))
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut buffer = [0_u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_le_bytes(buffer))
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut buffer = [0_u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(u64::from_le_bytes(buffer))
}

fn write_i64(writer: &mut impl Write, value: i64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    let mut buffer = [0_u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(i64::from_le_bytes(buffer))
}

fn write_f64(writer: &mut impl Write, value: f64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f64(reader: &mut impl Read) -> io::Result<f64> {
    let mut buffer = [0_u8; 8];
    reader.read_exact(&mut buffer)?;
    Ok(f64::from_le_bytes(buffer))
}

fn write_f32(writer: &mut impl Write, value: f32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut buffer = [0_u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(f32::from_le_bytes(buffer))
}

fn u32_from_usize(value: usize) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| VectLiteError::InvalidFormat("value exceeds the u32 storage limit".to_owned()))
}

fn u64_from_usize(value: usize) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| VectLiteError::InvalidFormat("value exceeds the u64 storage limit".to_owned()))
}

fn usize_from_u32(value: u32) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        VectLiteError::InvalidFormat("u32 value cannot fit into usize on this platform".to_owned())
    })
}

fn usize_from_u64(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        VectLiteError::InvalidFormat("u64 value cannot fit into usize on this platform".to_owned())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Database, DistanceMetric, HybridSearchOptions, Metadata, MetadataFilter, MetadataValue,
        MultiVectorSearchOptions, MultiVectors, NamedVectors, PayloadIndexType, Record,
        SearchOptions, SparseVector, VectLiteError,
    };
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn roundtrip_persists_records() {
        let path = temp_file("roundtrip");
        let mut metadata = Metadata::new();
        metadata.insert("source".to_owned(), MetadataValue::from("blog"));

        {
            let mut database = Database::create(&path, 3).expect("create database");
            database
                .insert("doc1", vec![1.0, 0.0, 0.0], metadata.clone())
                .expect("insert record");
        }

        let reopened = Database::open(&path).expect("reopen database");
        assert_eq!(reopened.dimension(), 3);
        assert_eq!(reopened.len(), 1);

        let record = reopened.get("doc1").expect("record exists");
        assert_eq!(record.namespace, "");
        assert_eq!(record.id, "doc1");
        assert_eq!(record.vector, vec![1.0, 0.0, 0.0]);
        assert!(record.sparse.is_empty());
        assert_eq!(record.metadata, metadata);

        cleanup(&path);
    }

    #[test]
    fn search_orders_by_similarity_and_filters_metadata() {
        let path = temp_file("search");
        let mut database = Database::create(&path, 2).expect("create database");

        let mut docs_metadata = Metadata::new();
        docs_metadata.insert("source".to_owned(), MetadataValue::from("notes"));
        docs_metadata.insert("title".to_owned(), MetadataValue::from("auth flow"));

        let mut blog_metadata = Metadata::new();
        blog_metadata.insert("source".to_owned(), MetadataValue::from("blog"));
        blog_metadata.insert("title".to_owned(), MetadataValue::from("shipping"));

        database
            .insert("doc1", vec![1.0, 0.0], docs_metadata)
            .expect("insert doc1");
        database
            .insert("doc2", vec![0.8, 0.2], blog_metadata)
            .expect("insert doc2");
        database
            .insert("doc3", vec![0.0, 1.0], Metadata::new())
            .expect("insert doc3");

        let results = database
            .search(
                &[1.0, 0.0],
                SearchOptions {
                    top_k: 5,
                    filter: Some(MetadataFilter::and(vec![
                        MetadataFilter::eq("source", "notes"),
                        MetadataFilter::contains("title", "auth"),
                    ])),
                    truncate_dim: None,
                },
            )
            .expect("search database");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].score > 0.99);

        cleanup(&path);
    }

    #[test]
    fn delete_removes_records_from_disk() {
        let path = temp_file("delete");
        {
            let mut database = Database::create(&path, 2).expect("create database");
            database
                .insert("doc1", vec![1.0, 0.0], Metadata::new())
                .expect("insert record");
            database.delete("doc1").expect("delete record");
        }

        let reopened = Database::open(&path).expect("reopen database");
        assert!(reopened.get("doc1").is_none());
        assert_eq!(reopened.len(), 0);

        cleanup(&path);
    }

    #[test]
    fn batch_upsert_persists_records_in_one_call() {
        let path = temp_file("batch-upsert");
        {
            let mut database = Database::create(&path, 2).expect("create database");

            let inserted = database
                .upsert_many(vec![
                    Record {
                        namespace: "".to_owned(),
                        id: "doc1".to_owned(),
                        vector: vec![1.0, 0.0],
                        vectors: NamedVectors::new(),
                        sparse: SparseVector::new(),
                        metadata: Metadata::new(),
                        multi_vectors: MultiVectors::new(),
                        expires_at: None,
                    },
                    Record {
                        namespace: "".to_owned(),
                        id: "doc2".to_owned(),
                        vector: vec![0.0, 1.0],
                        vectors: NamedVectors::new(),
                        sparse: SparseVector::new(),
                        metadata: Metadata::new(),
                        multi_vectors: MultiVectors::new(),
                        expires_at: None,
                    },
                ])
                .expect("batch upsert");

            assert_eq!(inserted, 2);
        }

        let reopened = Database::open(&path).expect("reopen database");
        assert_eq!(reopened.len(), 2);

        cleanup(&path);
    }

    #[test]
    fn extended_filters_match_expected_records() {
        let path = temp_file("extended-filters");
        let mut database = Database::create(&path, 2).expect("create database");

        let mut metadata = Metadata::new();
        metadata.insert("source".to_owned(), MetadataValue::from("blog"));
        metadata.insert("priority".to_owned(), MetadataValue::from(10));

        database
            .insert("doc1", vec![1.0, 0.0], metadata)
            .expect("insert record");

        let results = database
            .search(
                &[1.0, 0.0],
                SearchOptions {
                    top_k: 10,
                    filter: Some(MetadataFilter::and(vec![
                        MetadataFilter::ne("source", "notes"),
                        MetadataFilter::gte("priority", 10.0),
                        MetadataFilter::lte("priority", 10.0),
                    ])),
                    truncate_dim: None,
                },
            )
            .expect("search database");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].namespace, "");
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].dense_score > 0.99);
        assert_eq!(results[0].sparse_score, 0.0);

        cleanup(&path);
    }

    #[test]
    fn namespaces_isolate_same_ids() {
        let path = temp_file("namespaces");
        let mut database = Database::create(&path, 2).expect("create database");

        database
            .insert_in_namespace("docs", "doc1", vec![1.0, 0.0], Metadata::new())
            .expect("insert docs");
        database
            .insert_in_namespace("notes", "doc1", vec![0.0, 1.0], Metadata::new())
            .expect("insert notes");

        assert!(database.get_in_namespace("docs", "doc1").is_some());
        assert!(database.get_in_namespace("notes", "doc1").is_some());
        assert!(database.get("doc1").is_none());

        let docs = database
            .search_in_namespace("docs", &[1.0, 0.0], SearchOptions::default())
            .expect("search docs");
        let all = database
            .search_all_namespaces(&[1.0, 0.0], SearchOptions::default())
            .expect("search all");

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].namespace, "docs");
        assert_eq!(all.len(), 2);
        assert_eq!(
            database.namespaces(),
            vec!["docs".to_owned(), "notes".to_owned()]
        );

        cleanup(&path);
    }

    #[test]
    fn hybrid_search_combines_dense_and_sparse_scores() {
        let path = temp_file("hybrid");
        let mut database = Database::create(&path, 2).expect("create database");

        let mut sparse_auth = SparseVector::new();
        sparse_auth.insert("auth".to_owned(), 1.0);
        sparse_auth.insert("sso".to_owned(), 0.5);

        let mut sparse_billing = SparseVector::new();
        sparse_billing.insert("billing".to_owned(), 1.0);

        database
            .upsert_with_sparse_in_namespace(
                "docs",
                "doc1",
                vec![1.0, 0.0],
                sparse_auth,
                Metadata::new(),
            )
            .expect("insert doc1");
        database
            .upsert_with_sparse_in_namespace(
                "docs",
                "doc2",
                vec![1.0, 0.0],
                sparse_billing,
                Metadata::new(),
            )
            .expect("insert doc2");

        let mut query_sparse = SparseVector::new();
        query_sparse.insert("auth".to_owned(), 1.0);

        let results = database
            .hybrid_search_in_namespace(
                "docs",
                Some(&[1.0, 0.0]),
                Some(&query_sparse),
                HybridSearchOptions {
                    top_k: 10,
                    filter: None,
                    dense_weight: 1.0,
                    sparse_weight: 1.0,
                    ..HybridSearchOptions::default()
                },
            )
            .expect("hybrid search");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].sparse_score > results[1].sparse_score);

        cleanup(&path);
    }

    #[test]
    fn upsert_without_sparse_rebuilds_sparse_index() {
        let path = temp_file("sparse-upsert-clear");
        let mut database = Database::create(&path, 2).expect("create database");

        let mut sparse_auth = SparseVector::new();
        sparse_auth.insert("auth".to_owned(), 1.0);

        database
            .upsert_with_sparse_in_namespace(
                "docs",
                "doc1",
                vec![1.0, 0.0],
                sparse_auth,
                Metadata::new(),
            )
            .expect("insert sparse doc");

        let mut query_sparse = SparseVector::new();
        query_sparse.insert("auth".to_owned(), 1.0);

        let initial_outcome = database
            .hybrid_search_in_namespace_with_stats(
                "docs",
                None,
                Some(&query_sparse),
                HybridSearchOptions {
                    top_k: 10,
                    filter: None,
                    dense_weight: 0.0,
                    sparse_weight: 1.0,
                    ..HybridSearchOptions::default()
                },
            )
            .expect("initial sparse search");
        assert_eq!(initial_outcome.stats.sparse_candidate_count, 1);

        database
            .upsert_in_namespace("docs", "doc1", vec![1.0, 0.0], Metadata::new())
            .expect("replace doc without sparse terms");

        let updated_outcome = database
            .hybrid_search_in_namespace_with_stats(
                "docs",
                None,
                Some(&query_sparse),
                HybridSearchOptions {
                    top_k: 10,
                    filter: None,
                    dense_weight: 0.0,
                    sparse_weight: 1.0,
                    ..HybridSearchOptions::default()
                },
            )
            .expect("sparse search after clearing sparse terms");
        assert_eq!(updated_outcome.stats.sparse_candidate_count, 0);
        assert!(updated_outcome.results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn named_vectors_roundtrip_and_search() {
        let path = temp_file("named-vectors");
        {
            let mut database = Database::create(&path, 2).expect("create database");

            let mut doc1_vectors = NamedVectors::new();
            doc1_vectors.insert("title".to_owned(), vec![1.0, 0.0]);
            doc1_vectors.insert("body".to_owned(), vec![0.0, 1.0]);

            let mut doc2_vectors = NamedVectors::new();
            doc2_vectors.insert("title".to_owned(), vec![0.0, 1.0]);
            doc2_vectors.insert("body".to_owned(), vec![1.0, 0.0]);

            database
                .upsert_with_vectors_in_namespace(
                    "docs",
                    "doc1",
                    vec![0.2, 0.8],
                    doc1_vectors,
                    SparseVector::new(),
                    Metadata::new(),
                )
                .expect("insert doc1");
            database
                .upsert_with_vectors_in_namespace(
                    "docs",
                    "doc2",
                    vec![0.8, 0.2],
                    doc2_vectors,
                    SparseVector::new(),
                    Metadata::new(),
                )
                .expect("insert doc2");
        }

        let reopened = Database::open(&path).expect("reopen database");
        let record = reopened
            .get_in_namespace("docs", "doc1")
            .expect("expected stored record");
        assert_eq!(record.vectors.len(), 2);
        assert_eq!(record.vectors.get("title"), Some(&vec![1.0, 0.0]));

        let title_results = reopened
            .hybrid_search_in_namespace(
                "docs",
                Some(&[1.0, 0.0]),
                None,
                HybridSearchOptions {
                    top_k: 2,
                    filter: None,
                    dense_weight: 1.0,
                    sparse_weight: 0.0,
                    vector_name: Some("title".to_owned()),
                    ..HybridSearchOptions::default()
                },
            )
            .expect("search title vector");

        let default_results = reopened
            .search_in_namespace("docs", &[1.0, 0.0], SearchOptions::default())
            .expect("search default vector");

        assert_eq!(
            title_results
                .iter()
                .map(|result| result.id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc1", "doc2"]
        );
        assert_eq!(
            default_results
                .iter()
                .map(|result| result.id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc2", "doc1"]
        );

        cleanup(&path);
    }

    #[test]
    fn ann_index_is_built_for_larger_collections() {
        let path = temp_file("ann");
        let mut database = Database::create(&path, 128).expect("create database");

        for i in 0..128 {
            let mut vector = vec![0.0_f32; 128];
            vector[i] = 1.0;
            database
                .insert_in_namespace("docs", format!("doc{i}"), vector, Metadata::new())
                .expect("insert record");
        }

        assert!(database.has_ann_index(None, None));
        assert!(database.has_ann_index(Some("docs"), None));

        let mut query = vec![0.0_f32; 128];
        query[42] = 1.0;
        let outcome = database
            .hybrid_search_in_namespace_with_stats(
                "docs",
                Some(&query),
                None,
                HybridSearchOptions {
                    top_k: 10,
                    filter: None,
                    dense_weight: 1.0,
                    sparse_weight: 0.0,
                    ..HybridSearchOptions::default()
                },
            )
            .expect("search database");

        assert!(outcome.stats.used_ann);
        assert!(outcome.stats.ann_candidate_count >= 10);
        assert!(!outcome.stats.exact_fallback);
        assert_eq!(outcome.results.len(), 10);

        cleanup(&path);
    }

    #[test]
    fn mmr_diversifies_near_duplicate_results() {
        let path = temp_file("mmr");
        let mut database = Database::create(&path, 2).expect("create database");

        database
            .insert("doc1", vec![1.0, 0.0], Metadata::new())
            .expect("insert doc1");
        database
            .insert("doc2", vec![0.99, 0.01], Metadata::new())
            .expect("insert doc2");
        database
            .insert("doc3", vec![0.7, 0.7], Metadata::new())
            .expect("insert doc3");

        let plain_results = database
            .search(
                &[1.0, 0.0],
                SearchOptions {
                    top_k: 2,
                    filter: None,
                    truncate_dim: None,
                },
            )
            .expect("search database");
        let mmr_outcome = database
            .hybrid_search_in_namespace_with_stats(
                "",
                Some(&[1.0, 0.0]),
                None,
                HybridSearchOptions {
                    top_k: 2,
                    filter: None,
                    dense_weight: 1.0,
                    sparse_weight: 0.0,
                    fetch_k: 3,
                    mmr_lambda: Some(0.3),
                    ..HybridSearchOptions::default()
                },
            )
            .expect("mmr search");

        assert_eq!(
            plain_results
                .iter()
                .map(|result| result.id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc1", "doc2"]
        );
        assert_eq!(
            mmr_outcome
                .results
                .iter()
                .map(|result| result.id.as_str())
                .collect::<Vec<_>>(),
            vec!["doc1", "doc3"]
        );
        assert!(mmr_outcome.stats.mmr_applied);
        assert_eq!(mmr_outcome.stats.fetch_k, 3);
        assert_eq!(mmr_outcome.stats.considered_count, 3);

        cleanup(&path);
    }

    #[test]
    fn closed_database_rejects_result_based_operations() {
        let path = temp_file("closed-db");
        let snapshot = temp_file("closed-db-snapshot");
        let mut database = Database::create(&path, 2).expect("create database");
        database
            .insert("doc1", vec![1.0, 0.0], Metadata::new())
            .expect("insert doc1");
        database.close().expect("close database");

        let search_err = database
            .search(
                &[1.0, 0.0],
                SearchOptions {
                    top_k: 1,
                    filter: None,
                    truncate_dim: None,
                },
            )
            .expect_err("search on closed database should fail");
        assert!(matches!(
            search_err,
            VectLiteError::InvalidFormat(message) if message.contains("database is closed")
        ));

        let snapshot_err = database
            .snapshot(&snapshot)
            .expect_err("snapshot on closed database should fail");
        assert!(matches!(
            snapshot_err,
            VectLiteError::InvalidFormat(message) if message.contains("database is closed")
        ));

        cleanup(&path);
        cleanup(&snapshot);
    }

    #[test]
    fn lock_timeout_must_be_non_negative_and_finite() {
        let path = temp_file("timeout-validation");
        let database = Database::create(&path, 2).expect("create database");

        let negative_err = match Database::open_with_timeout(&path, -1.0) {
            Ok(_) => panic!("negative lock timeout should fail"),
            Err(err) => err,
        };
        assert!(matches!(
            negative_err,
            VectLiteError::InvalidFormat(message) if message.contains("lock_timeout")
        ));

        let nan_err = match Database::open_with_timeout(&path, f64::NAN) {
            Ok(_) => panic!("NaN lock timeout should fail"),
            Err(err) => err,
        };
        assert!(matches!(
            nan_err,
            VectLiteError::InvalidFormat(message) if message.contains("lock_timeout")
        ));

        drop(database);
        cleanup(&path);
    }

    fn temp_file(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "vectlite-{name}-{}-{nanos}.vdb",
            std::process::id()
        ))
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_file(path);
        // Also clean up sidecar files
        let mut quant = path.as_os_str().to_os_string();
        quant.push(".quant");
        let _ = std::fs::remove_file(PathBuf::from(&quant));
        let mut wal = path.as_os_str().to_os_string();
        wal.push(".wal");
        let _ = std::fs::remove_file(PathBuf::from(&wal));
        let mut lock = path.as_os_str().to_os_string();
        lock.push(".lock");
        let _ = std::fs::remove_file(PathBuf::from(&lock));
        // Clean up multi-vector quantization sidecar files (.mvquant.*)
        if let Some(parent) = path.parent() {
            if let Some(stem) = path.file_name().and_then(|n| n.to_str()) {
                let prefix = format!("{stem}.mvquant.");
                if let Ok(entries) = std::fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        if let Some(fname) = entry.file_name().to_str() {
                            if fname.starts_with(&prefix) {
                                let _ = std::fs::remove_file(entry.path());
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Quantization integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn scalar_quantization_enables_search_and_persists() {
        use super::quantization::{QuantizationConfig, ScalarQuantizationConfig};

        let path = temp_file("quant-scalar");
        let dim = 32;

        {
            let mut db = Database::create(&path, dim).expect("create");
            // Insert enough records for meaningful search
            for i in 0..50 {
                let mut v = vec![0.0_f32; dim];
                v[i % dim] = 1.0;
                v[(i + 1) % dim] = 0.5;
                db.upsert(format!("doc{i}"), v, Metadata::new())
                    .expect("upsert");
            }

            // Enable scalar quantization
            db.enable_quantization(QuantizationConfig::Scalar(ScalarQuantizationConfig {
                rescore_multiplier: 5,
            }))
            .expect("enable quant");

            assert!(db.is_quantized());

            // Search should work with quantization
            let query = {
                let mut q = vec![0.0_f32; dim];
                q[0] = 1.0;
                q
            };
            let results = db
                .search(
                    &query,
                    SearchOptions {
                        top_k: 5,
                        filter: None,
                        truncate_dim: None,
                    },
                )
                .expect("search");
            assert!(!results.is_empty());
            // The most similar vector (doc0 has [1,0.5,0,...]) should be first
            assert_eq!(results[0].id, "doc0");
        }

        // Reopen and verify quantization persists
        {
            let db = Database::open(&path).expect("reopen");
            assert!(db.is_quantized());
            assert!(matches!(
                db.quantization_config(),
                Some(QuantizationConfig::Scalar(_))
            ));

            let query = {
                let mut q = vec![0.0_f32; dim];
                q[0] = 1.0;
                q
            };
            let results = db
                .search(
                    &query,
                    SearchOptions {
                        top_k: 5,
                        filter: None,
                        truncate_dim: None,
                    },
                )
                .expect("search after reopen");
            assert!(!results.is_empty());
            assert_eq!(results[0].id, "doc0");
        }

        cleanup(&path);
    }

    #[test]
    fn scalar_quantization_keeps_signed_cosine_neighbor_in_candidate_set() {
        use super::quantization::{QuantizationConfig, ScalarQuantizationConfig};

        let path = temp_file("quant-scalar-signed-recall");
        let dim = 146;

        let mut query = vec![-1.0_f32; dim];
        for value in &mut query[..10] {
            *value = 1.0;
        }

        let mut db = Database::create(&path, dim).expect("create");
        for i in 0..120 {
            db.upsert(format!("high{i:03}"), vec![2.0_f32; dim], Metadata::new())
                .expect("upsert high distractor");
        }

        let mut calibration_low = vec![2.0_f32; dim];
        for value in &mut calibration_low[..10] {
            *value = -1.0;
        }
        db.upsert("calibration-low", calibration_low, Metadata::new())
            .expect("upsert calibration low");
        db.upsert("target", query.clone(), Metadata::new())
            .expect("upsert target");

        db.enable_quantization(QuantizationConfig::Scalar(ScalarQuantizationConfig {
            rescore_multiplier: 1,
        }))
        .expect("enable quant");

        let results = db
            .search(
                &query,
                SearchOptions {
                    top_k: 1,
                    filter: None,
                    truncate_dim: None,
                },
            )
            .expect("search");

        assert_eq!(results[0].id, "target");

        cleanup(&path);
    }

    #[test]
    fn binary_quantization_enables_search() {
        use super::quantization::{BinaryQuantizationConfig, QuantizationConfig};

        let path = temp_file("quant-binary");
        let dim = 64;

        let mut db = Database::create(&path, dim).expect("create");
        for i in 0..100 {
            let mut v = vec![0.0_f32; dim];
            // Set some positive dimensions for the binary representation
            for j in 0..dim {
                v[j] = if (i + j) % 3 == 0 { 1.0 } else { -1.0 };
            }
            db.upsert(format!("doc{i}"), v, Metadata::new())
                .expect("upsert");
        }

        db.enable_quantization(QuantizationConfig::Binary(BinaryQuantizationConfig {
            rescore_multiplier: 10,
        }))
        .expect("enable quant");

        assert!(db.is_quantized());

        // Search: query matches doc0's pattern
        let query: Vec<f32> = (0..dim)
            .map(|j| if j % 3 == 0 { 1.0 } else { -1.0 })
            .collect();
        let results = db
            .search(
                &query,
                SearchOptions {
                    top_k: 5,
                    filter: None,
                    truncate_dim: None,
                },
            )
            .expect("search");
        assert!(!results.is_empty());
        // doc0 should be the best match (identical pattern)
        assert_eq!(results[0].id, "doc0");

        cleanup(&path);
    }

    #[test]
    fn product_quantization_enables_search() {
        use super::quantization::{ProductQuantizationConfig, QuantizationConfig};

        let path = temp_file("quant-pq");
        let dim = 32;

        let mut db = Database::create(&path, dim).expect("create");
        for i in 0..100 {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                .collect();
            db.upsert(format!("doc{i}"), v, Metadata::new())
                .expect("upsert");
        }

        db.enable_quantization(QuantizationConfig::Product(ProductQuantizationConfig {
            num_sub_vectors: 4,
            num_centroids: 16,
            training_iterations: 5,
            rescore_multiplier: 10,
        }))
        .expect("enable quant");

        assert!(db.is_quantized());

        // Search with the same vector as doc0
        let query: Vec<f32> = (0..dim).map(|j| (j * 13 % 100) as f32 / 100.0).collect();
        let results = db
            .search(
                &query,
                SearchOptions {
                    top_k: 5,
                    filter: None,
                    truncate_dim: None,
                },
            )
            .expect("search");
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc0");

        cleanup(&path);
    }

    #[test]
    fn disable_quantization_removes_sidecar() {
        use super::quantization::{QuantizationConfig, ScalarQuantizationConfig};

        let path = temp_file("quant-disable");
        let dim = 8;

        let mut db = Database::create(&path, dim).expect("create");
        for i in 0..10 {
            let v: Vec<f32> = (0..dim).map(|j| (i + j) as f32).collect();
            db.upsert(format!("doc{i}"), v, Metadata::new())
                .expect("upsert");
        }

        db.enable_quantization(QuantizationConfig::Scalar(
            ScalarQuantizationConfig::default(),
        ))
        .expect("enable");
        assert!(db.is_quantized());

        // Verify sidecar exists
        let quant_path = {
            let mut p = path.as_os_str().to_os_string();
            p.push(".quant");
            PathBuf::from(p)
        };
        assert!(quant_path.exists());

        db.disable_quantization().expect("disable");
        assert!(!db.is_quantized());
        assert!(!quant_path.exists());

        cleanup(&path);
    }

    #[test]
    fn quantization_empty_database_returns_error() {
        use super::quantization::{QuantizationConfig, ScalarQuantizationConfig};

        let path = temp_file("quant-empty");
        let mut db = Database::create(&path, 4).expect("create");

        let result = db.enable_quantization(QuantizationConfig::Scalar(
            ScalarQuantizationConfig::default(),
        ));
        assert!(result.is_err());
        assert!(!db.is_quantized());

        cleanup(&path);
    }

    #[test]
    fn product_quantization_invalid_subvector_count_returns_error() {
        use super::quantization::{ProductQuantizationConfig, QuantizationConfig};

        let path = temp_file("quant-pq-invalid-subvectors");
        let mut db = Database::create(&path, 146).expect("create");
        for i in 0..4 {
            db.upsert(
                format!("doc{i}"),
                vec![0.1_f32 + i as f32; 146],
                Metadata::new(),
            )
            .expect("upsert");
        }
        assert_eq!(db.valid_num_sub_vectors(), vec![1, 2, 73, 146]);

        let result =
            db.enable_quantization(QuantizationConfig::Product(ProductQuantizationConfig {
                num_sub_vectors: 16,
                num_centroids: 4,
                training_iterations: 1,
                rescore_multiplier: 1,
            }));

        assert!(matches!(
            result,
            Err(VectLiteError::InvalidFormat(message))
                if message.contains("dimension (146) must be divisible by num_sub_vectors (16)")
        ));
        assert!(!db.is_quantized());

        cleanup(&path);
    }

    // -----------------------------------------------------------------------
    // Multi-vector / ColBERT-style integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn multi_vector_upsert_and_search() {
        let path = temp_file("mv-upsert-search");
        let mut db = Database::create(&path, 3).expect("create");

        // Upsert records with ColBERT-style token vectors
        let mut mv1 = MultiVectors::new();
        mv1.insert(
            "colbert".to_owned(),
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        );
        db.upsert_multi_vectors("doc1", vec![1.0, 0.0, 0.0], Metadata::new(), mv1)
            .expect("upsert doc1");

        let mut mv2 = MultiVectors::new();
        mv2.insert(
            "colbert".to_owned(),
            vec![vec![0.0, 0.0, 1.0], vec![0.0, 1.0, 0.0]],
        );
        db.upsert_multi_vectors("doc2", vec![0.0, 0.0, 1.0], Metadata::new(), mv2)
            .expect("upsert doc2");

        assert_eq!(db.len(), 2);

        // Search with query tokens that strongly match doc1
        let query_tokens = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let results = db
            .search_multi_vector(
                "colbert",
                &query_tokens,
                MultiVectorSearchOptions::default(),
            )
            .expect("search");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1"); // doc1 has perfect MaxSim match

        cleanup(&path);
    }

    #[test]
    fn multi_vector_empty_space_error() {
        let path = temp_file("mv-empty-space");
        let db = Database::create(&path, 3).expect("create");

        let query_tokens = vec![vec![1.0, 0.0, 0.0]];
        let result = db.search_multi_vector("", &query_tokens, MultiVectorSearchOptions::default());
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn multi_vector_empty_query_tokens_error() {
        let path = temp_file("mv-empty-query");
        let db = Database::create(&path, 3).expect("create");

        let query_tokens: Vec<Vec<f32>> = vec![];
        let result = db.search_multi_vector(
            "colbert",
            &query_tokens,
            MultiVectorSearchOptions::default(),
        );
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn multi_vector_search_with_namespace_filter() {
        let path = temp_file("mv-ns-filter");
        let mut db = Database::create(&path, 3).expect("create");

        let mut mv = MultiVectors::new();
        mv.insert("colbert".to_owned(), vec![vec![1.0, 0.0, 0.0]]);
        db.upsert_multi_vectors_in_namespace(
            "ns1",
            "doc1",
            vec![1.0, 0.0, 0.0],
            Metadata::new(),
            mv.clone(),
        )
        .expect("upsert ns1");
        db.upsert_multi_vectors_in_namespace(
            "ns2",
            "doc2",
            vec![0.0, 1.0, 0.0],
            Metadata::new(),
            mv.clone(),
        )
        .expect("upsert ns2");

        let query_tokens = vec![vec![1.0, 0.0, 0.0]];
        let options = MultiVectorSearchOptions {
            top_k: 10,
            filter: None,
            namespace: Some("ns1".to_owned()),
        };
        let results = db
            .search_multi_vector("colbert", &query_tokens, options)
            .expect("search");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[0].namespace, "ns1");

        cleanup(&path);
    }

    #[test]
    fn multi_vector_quantization_enable_disable() {
        use super::quantization::{MultiVectorQuantizationConfig, TwoBitQuantizationConfig};

        let path = temp_file("mv-quant");
        let mut db = Database::create(&path, 3).expect("create");

        // Insert some records with multi-vectors
        for i in 0..10 {
            let mut mv = MultiVectors::new();
            mv.insert(
                "colbert".to_owned(),
                vec![
                    vec![i as f32, 0.0, 0.0],
                    vec![0.0, i as f32, 0.0],
                    vec![0.0, 0.0, i as f32],
                ],
            );
            db.upsert_multi_vectors(
                &format!("doc{i}"),
                vec![i as f32, 0.0, 0.0],
                Metadata::new(),
                mv,
            )
            .expect("upsert");
        }

        assert!(!db.is_multi_vector_quantized("colbert"));

        // Enable quantization
        db.enable_multi_vector_quantization(
            "colbert",
            MultiVectorQuantizationConfig::TwoBit(TwoBitQuantizationConfig {
                rescore_multiplier: 4,
            }),
        )
        .expect("enable");

        assert!(db.is_multi_vector_quantized("colbert"));

        // Search should still work
        let query_tokens = vec![vec![9.0, 0.0, 0.0], vec![0.0, 9.0, 0.0]];
        let results = db
            .search_multi_vector(
                "colbert",
                &query_tokens,
                MultiVectorSearchOptions::default(),
            )
            .expect("search");

        assert!(!results.is_empty());

        // Disable quantization
        db.disable_multi_vector_quantization("colbert")
            .expect("disable");
        assert!(!db.is_multi_vector_quantized("colbert"));

        cleanup(&path);
    }

    #[test]
    fn multi_vector_quantization_persists_across_reopen() {
        use super::quantization::{MultiVectorQuantizationConfig, TwoBitQuantizationConfig};

        let path = temp_file("mv-quant-persist");

        {
            let mut db = Database::create(&path, 3).expect("create");
            for i in 0..10 {
                let mut mv = MultiVectors::new();
                mv.insert(
                    "colbert".to_owned(),
                    vec![
                        vec![i as f32 * 0.1, 0.5, 0.5],
                        vec![0.5, i as f32 * 0.1, 0.5],
                    ],
                );
                db.upsert_multi_vectors(
                    &format!("doc{i}"),
                    vec![1.0, 0.0, 0.0],
                    Metadata::new(),
                    mv,
                )
                .expect("upsert");
            }

            db.enable_multi_vector_quantization(
                "colbert",
                MultiVectorQuantizationConfig::TwoBit(TwoBitQuantizationConfig {
                    rescore_multiplier: 4,
                }),
            )
            .expect("enable");

            assert!(db.is_multi_vector_quantized("colbert"));
        }

        // Reopen and verify quantization was loaded
        let db = Database::open(&path).expect("reopen");
        assert!(db.is_multi_vector_quantized("colbert"));

        // Search should work on reopened database
        let query_tokens = vec![vec![0.9, 0.5, 0.5]];
        let results = db
            .search_multi_vector(
                "colbert",
                &query_tokens,
                MultiVectorSearchOptions::default(),
            )
            .expect("search");
        assert!(!results.is_empty());

        cleanup(&path);
    }

    #[test]
    fn multi_vector_record_persists_across_reopen() {
        let path = temp_file("mv-persist");
        let mut mv = MultiVectors::new();
        mv.insert(
            "colbert".to_owned(),
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
        );

        {
            let mut db = Database::create(&path, 3).expect("create");
            db.upsert_multi_vectors("doc1", vec![1.0, 0.0, 0.0], Metadata::new(), mv.clone())
                .expect("upsert");
        }

        let db = Database::open(&path).expect("reopen");
        let record = db.get("doc1").expect("exists");
        let tokens = record.multi_vectors.get("colbert").expect("colbert space");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(tokens[1], vec![4.0, 5.0, 6.0]);

        cleanup(&path);
    }

    #[test]
    fn multi_vector_maxsim_scoring_correctness() {
        use super::{DistanceMetric, maxsim_score};

        // Two identical sets: MaxSim should be sum of 1.0 per query token
        let query = [&[1.0_f32, 0.0, 0.0][..], &[0.0, 1.0, 0.0]];
        let doc = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let score = maxsim_score(&query, &doc, DistanceMetric::Cosine);
        // cosine(q0, d0) = 1.0, cosine(q0, d1) = 0.0 -> max = 1.0
        // cosine(q1, d0) = 0.0, cosine(q1, d1) = 1.0 -> max = 1.0
        // sum = 2.0
        assert!((score - 2.0).abs() < 1e-6);

        // Orthogonal: each query token has zero max sim
        let query2 = [&[1.0_f32, 0.0, 0.0][..]];
        let doc2 = vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let score2 = maxsim_score(&query2, &doc2, DistanceMetric::Cosine);
        assert!(score2.abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Distance metric tests
    // -----------------------------------------------------------------------

    #[test]
    fn distance_metric_tag_roundtrip() {
        use super::DistanceMetric;
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let tag = metric.to_tag();
            let back = DistanceMetric::from_tag(tag).expect("valid tag");
            assert_eq!(back, metric);
        }
        // Invalid tag
        assert!(DistanceMetric::from_tag(255).is_err());
    }

    #[test]
    fn distance_metric_name_roundtrip() {
        use super::DistanceMetric;
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let name = metric.name();
            let back = DistanceMetric::from_name(name).expect("valid name");
            assert_eq!(back, metric);
        }
    }

    #[test]
    fn distance_metric_name_aliases() {
        use super::DistanceMetric;
        // Euclidean aliases
        assert_eq!(
            DistanceMetric::from_name("l2").unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            DistanceMetric::from_name("L2").unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            DistanceMetric::from_name("EUCLIDEAN").unwrap(),
            DistanceMetric::Euclidean
        );
        // DotProduct aliases
        assert_eq!(
            DistanceMetric::from_name("dot").unwrap(),
            DistanceMetric::DotProduct
        );
        assert_eq!(
            DistanceMetric::from_name("dot_product").unwrap(),
            DistanceMetric::DotProduct
        );
        assert_eq!(
            DistanceMetric::from_name("ip").unwrap(),
            DistanceMetric::DotProduct
        );
        assert_eq!(
            DistanceMetric::from_name("inner_product").unwrap(),
            DistanceMetric::DotProduct
        );
        // Manhattan aliases
        assert_eq!(
            DistanceMetric::from_name("l1").unwrap(),
            DistanceMetric::Manhattan
        );
        assert_eq!(
            DistanceMetric::from_name("L1").unwrap(),
            DistanceMetric::Manhattan
        );
        // Invalid
        assert!(DistanceMetric::from_name("hamming").is_err());
    }

    #[test]
    fn distance_metric_score_cosine() {
        use super::DistanceMetric;
        let a = [1.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0];
        let c = [0.0_f32, 1.0, 0.0];
        // Identical vectors -> similarity ~1.0
        let s1 = DistanceMetric::Cosine.score(&a, &b);
        assert!((s1 - 1.0).abs() < 1e-5, "cosine identical: {s1}");
        // Orthogonal vectors -> similarity ~0.0
        let s2 = DistanceMetric::Cosine.score(&a, &c);
        assert!(s2.abs() < 1e-5, "cosine orthogonal: {s2}");
    }

    #[test]
    fn distance_metric_score_euclidean() {
        use super::DistanceMetric;
        let a = [1.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0];
        let c = [4.0_f32, 0.0, 0.0];
        // Identical -> score = -0.0 (negated distance)
        let s1 = DistanceMetric::Euclidean.score(&a, &b);
        assert!(s1.abs() < 1e-5, "euclidean identical: {s1}");
        // Distance = 3.0, score = -3.0
        let s2 = DistanceMetric::Euclidean.score(&a, &c);
        assert!((s2 - (-3.0)).abs() < 1e-5, "euclidean dist=3: {s2}");
        // Closer is higher score
        let d = [2.0_f32, 0.0, 0.0];
        let s3 = DistanceMetric::Euclidean.score(&a, &d);
        assert!(s3 > s2, "closer should have higher score");
    }

    #[test]
    fn distance_metric_score_dot_product() {
        use super::DistanceMetric;
        // Normalized unit vectors
        let a = [1.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 0.0, 0.0];
        let c = [0.0_f32, 1.0, 0.0];
        // dot(a, b) = 1.0
        let s1 = DistanceMetric::DotProduct.score(&a, &b);
        assert!((s1 - 1.0).abs() < 1e-5, "dot identical: {s1}");
        // dot(a, c) = 0.0
        let s2 = DistanceMetric::DotProduct.score(&a, &c);
        assert!(s2.abs() < 1e-5, "dot orthogonal: {s2}");
        // Higher dot = higher score
        let d = [0.5_f32, 0.5, 0.0];
        let s3 = DistanceMetric::DotProduct.score(&a, &d);
        assert!(s3 > s2, "non-zero dot should be higher than orthogonal");
    }

    #[test]
    fn distance_metric_score_manhattan() {
        use super::DistanceMetric;
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 2.0, 3.0];
        let c = [4.0_f32, 6.0, 3.0];
        // Identical -> score = 0.0 (negated L1 distance)
        let s1 = DistanceMetric::Manhattan.score(&a, &b);
        assert!(s1.abs() < 1e-5, "manhattan identical: {s1}");
        // L1 dist = |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7, score = -7
        let s2 = DistanceMetric::Manhattan.score(&a, &c);
        assert!((s2 - (-7.0)).abs() < 1e-5, "manhattan dist=7: {s2}");
    }

    #[test]
    fn distance_metric_is_similarity() {
        use super::DistanceMetric;
        assert!(DistanceMetric::Cosine.is_similarity());
        assert!(DistanceMetric::DotProduct.is_similarity());
        assert!(!DistanceMetric::Euclidean.is_similarity());
        assert!(!DistanceMetric::Manhattan.is_similarity());
    }

    #[test]
    fn distance_metric_default_is_cosine() {
        use super::DistanceMetric;
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn distance_metric_display() {
        use super::DistanceMetric;
        assert_eq!(format!("{}", DistanceMetric::Cosine), "cosine");
        assert_eq!(format!("{}", DistanceMetric::Euclidean), "euclidean");
        assert_eq!(format!("{}", DistanceMetric::DotProduct), "dotproduct");
        assert_eq!(format!("{}", DistanceMetric::Manhattan), "manhattan");
    }

    #[test]
    fn create_with_metric_persists_metric() {
        use super::DistanceMetric;
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let path = temp_file(&format!("metric-persist-{}", metric.name()));
            {
                let db = Database::create_with_metric(&path, 4, metric).expect("create");
                assert_eq!(db.metric(), metric);
            }
            // Reopen and verify metric
            let db = Database::open(&path).expect("reopen");
            assert_eq!(db.metric(), metric, "metric should persist for {metric}");
            cleanup(&path);
        }
    }

    #[test]
    fn default_create_uses_cosine_metric() {
        use super::DistanceMetric;
        let path = temp_file("metric-default-cosine");
        let db = Database::create(&path, 4).expect("create");
        assert_eq!(db.metric(), DistanceMetric::Cosine);
        drop(db);
        cleanup(&path);
    }

    #[test]
    fn open_or_create_with_metric_creates_new() {
        use super::DistanceMetric;
        let path = temp_file("metric-ooc-new");
        let db = Database::open_or_create_with_metric(&path, 4, DistanceMetric::Euclidean)
            .expect("open_or_create");
        assert_eq!(db.metric(), DistanceMetric::Euclidean);
        drop(db);
        // Reopen with open_or_create again — should keep Euclidean
        let db2 = Database::open_or_create_with_metric(&path, 4, DistanceMetric::Euclidean)
            .expect("reopen");
        assert_eq!(db2.metric(), DistanceMetric::Euclidean);
        drop(db2);
        cleanup(&path);
    }

    #[test]
    fn search_with_euclidean_metric() {
        use super::DistanceMetric;
        let path = temp_file("metric-search-euclidean");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::Euclidean).expect("create");

        // Insert vectors at known distances from query [0, 0, 0]
        db.insert("close", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert close"); // L2 = 1
        db.insert("mid", vec![3.0, 0.0, 0.0], Metadata::new())
            .expect("insert mid"); // L2 = 3
        db.insert("far", vec![5.0, 5.0, 5.0], Metadata::new())
            .expect("insert far"); // L2 = sqrt(75) ≈ 8.66

        let results = db
            .search(
                &[0.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 3,
                    ..Default::default()
                },
            )
            .expect("search");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "close");
        assert_eq!(results[1].id, "mid");
        assert_eq!(results[2].id, "far");
        // Scores should be negative distances
        assert!(results[0].score > results[1].score);
        assert!(results[1].score > results[2].score);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn search_with_dotproduct_metric() {
        use super::DistanceMetric;
        let path = temp_file("metric-search-dot");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::DotProduct).expect("create");

        // Vectors with different dot products with query [1, 0, 0]
        db.insert("high", vec![10.0, 0.0, 0.0], Metadata::new())
            .expect("insert high"); // dot = 10
        db.insert("medium", vec![5.0, 0.0, 0.0], Metadata::new())
            .expect("insert medium"); // dot = 5
        db.insert("low", vec![0.0, 1.0, 0.0], Metadata::new())
            .expect("insert low"); // dot = 0

        let results = db
            .search(
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 3,
                    ..Default::default()
                },
            )
            .expect("search");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "high");
        assert_eq!(results[1].id, "medium");
        assert_eq!(results[2].id, "low");
        assert!(results[0].score > results[1].score);
        assert!(results[1].score > results[2].score);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn search_with_manhattan_metric() {
        use super::DistanceMetric;
        let path = temp_file("metric-search-manhattan");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::Manhattan).expect("create");

        // Vectors at known Manhattan distances from query [0, 0, 0]
        db.insert("close", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert close"); // L1 = 1
        db.insert("mid", vec![2.0, 1.0, 0.0], Metadata::new())
            .expect("insert mid"); // L1 = 3
        db.insert("far", vec![3.0, 3.0, 3.0], Metadata::new())
            .expect("insert far"); // L1 = 9

        let results = db
            .search(
                &[0.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 3,
                    ..Default::default()
                },
            )
            .expect("search");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "close");
        assert_eq!(results[1].id, "mid");
        assert_eq!(results[2].id, "far");
        assert!(results[0].score > results[1].score);
        assert!(results[1].score > results[2].score);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn matryoshka_prefix_search_accepts_short_query() {
        let path = temp_file("matryoshka-short-query");
        let mut db = Database::create(&path, 4).expect("create");
        db.insert("prefix_match", vec![1.0, 0.0, -1.0, -1.0], Metadata::new())
            .expect("insert prefix");
        db.insert("full_match", vec![0.0, 1.0, 1.0, 1.0], Metadata::new())
            .expect("insert full");

        let outcome = db
            .hybrid_search_in_namespace_with_stats(
                "",
                Some(&[1.0, 0.0]),
                None,
                HybridSearchOptions {
                    top_k: 2,
                    truncate_dim: Some(2),
                    ..HybridSearchOptions::default()
                },
            )
            .expect("search");

        assert_eq!(outcome.results[0].id, "prefix_match");
        assert_eq!(outcome.stats.effective_dimension, 2);
        assert!(outcome.stats.matryoshka_truncated);
        assert!(!outcome.stats.used_ann);

        cleanup(&path);
    }

    #[test]
    fn matryoshka_prefix_search_can_truncate_full_query() {
        let path = temp_file("matryoshka-truncate");
        let mut db = Database::create(&path, 4).expect("create");
        db.insert("prefix_match", vec![1.0, 0.0, -1.0, -1.0], Metadata::new())
            .expect("insert prefix");
        db.insert("tail_match", vec![0.0, 1.0, 1.0, 1.0], Metadata::new())
            .expect("insert tail");

        let outcome = db
            .hybrid_search_in_namespace_with_stats(
                "",
                Some(&[1.0, 0.0, 1.0, 1.0]),
                None,
                HybridSearchOptions {
                    top_k: 2,
                    truncate_dim: Some(2),
                    ..HybridSearchOptions::default()
                },
            )
            .expect("search");

        assert_eq!(outcome.results[0].id, "prefix_match");
        assert_eq!(outcome.stats.effective_dimension, 2);
        assert!(outcome.stats.matryoshka_truncated);

        cleanup(&path);
    }

    #[test]
    fn search_with_cosine_metric_explicit() {
        use super::DistanceMetric;
        let path = temp_file("metric-search-cosine-explicit");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::Cosine).expect("create");

        db.insert("aligned", vec![2.0, 0.0, 0.0], Metadata::new())
            .expect("insert aligned"); // cosine = 1.0
        db.insert("diagonal", vec![1.0, 1.0, 0.0], Metadata::new())
            .expect("insert diagonal"); // cosine = 1/sqrt(2) ≈ 0.707
        db.insert("orthogonal", vec![0.0, 0.0, 1.0], Metadata::new())
            .expect("insert orthogonal"); // cosine = 0.0

        let results = db
            .search(
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 3,
                    ..Default::default()
                },
            )
            .expect("search");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "aligned");
        assert_eq!(results[1].id, "diagonal");
        assert_eq!(results[2].id, "orthogonal");
        assert!((results[0].score - 1.0).abs() < 1e-4);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn metric_persists_with_upsert_and_search_cycle() {
        use super::DistanceMetric;
        let path = temp_file("metric-upsert-cycle");
        {
            let mut db =
                Database::create_with_metric(&path, 3, DistanceMetric::Manhattan).expect("create");
            db.upsert("a", vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert a");
            db.upsert("b", vec![0.0, 5.0, 0.0], Metadata::new())
                .expect("upsert b");
        }

        // Reopen and search — metric should still be Manhattan
        let db = Database::open(&path).expect("reopen");
        assert_eq!(db.metric(), DistanceMetric::Manhattan);

        let results = db
            .search(
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 2,
                    ..Default::default()
                },
            )
            .expect("search");
        // "a" is closer (L1=0) vs "b" (L1=6)
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "b");
        assert!(results[0].score > results[1].score);

        cleanup(&path);
    }

    #[test]
    fn simd_cosine_matches_scalar() {
        use super::{scalar_cosine_similarity, simd_cosine_similarity};
        let a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let b = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let simd_val = simd_cosine_similarity(&a, &b);
        let scalar_val = scalar_cosine_similarity(&a, &b);
        assert!(
            (simd_val - scalar_val).abs() < 1e-4,
            "simd={simd_val}, scalar={scalar_val}"
        );
    }

    #[test]
    fn simd_euclidean_matches_scalar() {
        use super::{scalar_euclidean_distance, simd_euclidean_distance};
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let simd_val = simd_euclidean_distance(&a, &b);
        let scalar_val = scalar_euclidean_distance(&a, &b);
        assert!(
            (simd_val - scalar_val).abs() < 1e-3,
            "simd={simd_val}, scalar={scalar_val}"
        );
    }

    #[test]
    fn simd_dot_matches_scalar() {
        use super::{scalar_dot_product, simd_dot_product};
        let a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let b = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let simd_val = simd_dot_product(&a, &b);
        let scalar_val = scalar_dot_product(&a, &b);
        assert!(
            (simd_val - scalar_val).abs() < 1e-4,
            "simd={simd_val}, scalar={scalar_val}"
        );
    }

    // -----------------------------------------------------------------------
    // update_metadata tests
    // -----------------------------------------------------------------------

    #[test]
    fn update_metadata_merges_patch() {
        let path = temp_file("update-metadata-merge");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("source".into(), "blog".into());
        meta.insert("version".into(), MetadataValue::Integer(1));
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        // Patch: update version, add new key
        let mut patch = Metadata::new();
        patch.insert("version".into(), MetadataValue::Integer(2));
        patch.insert("reviewed".into(), MetadataValue::Boolean(true));

        let updated = db.update_metadata("doc1", patch).expect("update");
        assert!(updated);

        let record = db.get("doc1").expect("found");
        assert_eq!(
            record.metadata.get("source"),
            Some(&MetadataValue::String("blog".into()))
        );
        assert_eq!(
            record.metadata.get("version"),
            Some(&MetadataValue::Integer(2))
        );
        assert_eq!(
            record.metadata.get("reviewed"),
            Some(&MetadataValue::Boolean(true))
        );

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn update_metadata_returns_false_for_missing_record() {
        let path = temp_file("update-metadata-missing");
        let mut db = Database::create(&path, 3).expect("create");

        let mut patch = Metadata::new();
        patch.insert("key".into(), "value".into());

        let updated = db.update_metadata("nonexistent", patch).expect("update");
        assert!(!updated);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn update_metadata_does_not_touch_vector() {
        let path = temp_file("update-metadata-vector-intact");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("source".into(), "blog".into());
        db.upsert("doc1", vec![1.0, 2.0, 3.0], meta)
            .expect("upsert");

        let mut patch = Metadata::new();
        patch.insert("source".into(), "updated".into());
        db.update_metadata("doc1", patch).expect("update");

        let record = db.get("doc1").expect("found");
        assert_eq!(record.vector, vec![1.0, 2.0, 3.0]);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn update_metadata_persists_across_reopen() {
        let path = temp_file("update-metadata-persist");
        {
            let mut db = Database::create(&path, 3).expect("create");
            let mut meta = Metadata::new();
            meta.insert("source".into(), "blog".into());
            db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");

            let mut patch = Metadata::new();
            patch.insert("source".into(), "updated".into());
            patch.insert("new_key".into(), MetadataValue::Integer(42));
            db.update_metadata("doc1", patch).expect("update");
        }

        // Reopen and verify
        let db = Database::open(&path).expect("reopen");
        let record = db.get("doc1").expect("found");
        assert_eq!(
            record.metadata.get("source"),
            Some(&MetadataValue::String("updated".into()))
        );
        assert_eq!(
            record.metadata.get("new_key"),
            Some(&MetadataValue::Integer(42))
        );
        assert_eq!(record.vector, vec![1.0, 0.0, 0.0]);

        cleanup(&path);
    }

    #[test]
    fn update_metadata_in_namespace() {
        let path = temp_file("update-metadata-ns");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("key".into(), "original".into());
        db.upsert_in_namespace("ns1", "doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        let mut patch = Metadata::new();
        patch.insert("key".into(), "patched".into());
        let updated = db
            .update_metadata_in_namespace("ns1", "doc1", patch)
            .expect("update");
        assert!(updated);

        let record = db.get_in_namespace("ns1", "doc1").expect("found");
        assert_eq!(
            record.metadata.get("key"),
            Some(&MetadataValue::String("patched".into()))
        );

        // Wrong namespace returns false
        let mut patch2 = Metadata::new();
        patch2.insert("key".into(), "nope".into());
        let updated2 = db
            .update_metadata_in_namespace("ns2", "doc1", patch2)
            .expect("update wrong ns");
        assert!(!updated2);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn update_metadata_searchable_after_patch() {
        let path = temp_file("update-metadata-search");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("status".into(), "draft".into());
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        // Before patch: filter matches
        let count = db.count_filtered(None, Some(&MetadataFilter::eq("status", "draft")));
        assert_eq!(count, 1);

        // Patch to "published"
        let mut patch = Metadata::new();
        patch.insert("status".into(), "published".into());
        db.update_metadata("doc1", patch).expect("update");

        // After patch: old filter misses, new filter matches
        let count_draft = db.count_filtered(None, Some(&MetadataFilter::eq("status", "draft")));
        assert_eq!(count_draft, 0);
        let count_pub = db.count_filtered(None, Some(&MetadataFilter::eq("status", "published")));
        assert_eq!(count_pub, 1);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn update_metadata_read_only_fails() {
        let path = temp_file("update-metadata-ro");
        {
            let mut db = Database::create(&path, 3).expect("create");
            db.upsert("doc1", vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert");
        }

        let mut db = Database::open_read_only(&path).expect("open ro");
        let mut patch = Metadata::new();
        patch.insert("key".into(), "val".into());
        let result = db.update_metadata("doc1", patch);
        assert!(result.is_err());

        cleanup(&path);
    }

    // ── Payload Index tests ──────────────────────────────────────────────

    #[test]
    fn create_keyword_index_returns_true_on_first_call() {
        let path = temp_file("pidx-create-kw");
        let mut db = Database::create(&path, 3).expect("create");

        let created = db
            .create_index("source", PayloadIndexType::Keyword)
            .expect("create_index");
        assert!(created);

        let indexes = db.list_indexes();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].0, "source");
        assert!(matches!(indexes[0].1, PayloadIndexType::Keyword));

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn create_index_returns_false_on_duplicate() {
        let path = temp_file("pidx-dup");
        let mut db = Database::create(&path, 3).expect("create");

        let first = db
            .create_index("source", PayloadIndexType::Keyword)
            .expect("first");
        assert!(first);

        let second = db
            .create_index("source", PayloadIndexType::Keyword)
            .expect("second");
        assert!(!second);

        assert_eq!(db.list_indexes().len(), 1);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn create_numeric_index() {
        let path = temp_file("pidx-numeric");
        let mut db = Database::create(&path, 3).expect("create");

        let created = db
            .create_index("price", PayloadIndexType::Numeric)
            .expect("create");
        assert!(created);

        let indexes = db.list_indexes();
        assert_eq!(indexes.len(), 1);
        assert!(matches!(indexes[0].1, PayloadIndexType::Numeric));

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn drop_index_returns_true_and_removes() {
        let path = temp_file("pidx-drop");
        let mut db = Database::create(&path, 3).expect("create");

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");
        assert_eq!(db.list_indexes().len(), 1);

        let dropped = db.drop_index("source").expect("drop");
        assert!(dropped);
        assert_eq!(db.list_indexes().len(), 0);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn drop_index_returns_false_for_nonexistent() {
        let path = temp_file("pidx-drop-missing");
        let mut db = Database::create(&path, 3).expect("create");

        let dropped = db.drop_index("nope").expect("drop");
        assert!(!dropped);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn list_indexes_empty_by_default() {
        let path = temp_file("pidx-list-empty");
        let db = Database::create(&path, 3).expect("create");
        assert!(db.list_indexes().is_empty());

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn keyword_index_accelerates_eq_count() {
        let path = temp_file("pidx-kw-eq");
        let mut db = Database::create(&path, 3).expect("create");

        // Insert records with different sources
        for i in 0..50 {
            let mut meta = Metadata::new();
            meta.insert("source".into(), format!("cat{}", i % 5).into());
            meta.insert("idx".into(), MetadataValue::Integer(i));
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        // Create keyword index on "source"
        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");

        // count_filtered with $eq should use the index
        let count = db.count_filtered(None, Some(&MetadataFilter::eq("source", "cat0")));
        assert_eq!(count, 10); // 0, 5, 10, 15, 20, 25, 30, 35, 40, 45

        let count2 = db.count_filtered(None, Some(&MetadataFilter::eq("source", "cat3")));
        assert_eq!(count2, 10);

        // Non-matching value
        let count3 = db.count_filtered(None, Some(&MetadataFilter::eq("source", "cat99")));
        assert_eq!(count3, 0);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn keyword_index_accelerates_in_filter() {
        let path = temp_file("pidx-kw-in");
        let mut db = Database::create(&path, 3).expect("create");

        for i in 0..20 {
            let mut meta = Metadata::new();
            meta.insert("tag".into(), format!("t{}", i % 4).into());
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("tag", PayloadIndexType::Keyword)
            .expect("create");

        let filter = MetadataFilter::r#in(
            "tag",
            vec![
                MetadataValue::String("t0".into()),
                MetadataValue::String("t2".into()),
            ],
        );
        let count = db.count_filtered(None, Some(&filter));
        assert_eq!(count, 10); // t0: 5, t2: 5

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn numeric_index_accelerates_range_queries() {
        let path = temp_file("pidx-num-range");
        let mut db = Database::create(&path, 3).expect("create");

        for i in 0..100 {
            let mut meta = Metadata::new();
            meta.insert("score".into(), MetadataValue::Float(i as f64));
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("score", PayloadIndexType::Numeric)
            .expect("create");

        // $gt 90 → 91..99 = 9 records
        let count_gt = db.count_filtered(None, Some(&MetadataFilter::gt("score", 90.0)));
        assert_eq!(count_gt, 9);

        // $gte 90 → 90..99 = 10 records
        let count_gte = db.count_filtered(None, Some(&MetadataFilter::gte("score", 90.0)));
        assert_eq!(count_gte, 10);

        // $lt 10 → 0..9 = 10 records
        let count_lt = db.count_filtered(None, Some(&MetadataFilter::lt("score", 10.0)));
        assert_eq!(count_lt, 10);

        // $lte 10 → 0..10 = 11 records
        let count_lte = db.count_filtered(None, Some(&MetadataFilter::lte("score", 10.0)));
        assert_eq!(count_lte, 11);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn numeric_index_eq_lookup() {
        let path = temp_file("pidx-num-eq");
        let mut db = Database::create(&path, 3).expect("create");

        for i in 0..20 {
            let mut meta = Metadata::new();
            meta.insert("priority".into(), MetadataValue::Float((i % 3) as f64));
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("priority", PayloadIndexType::Numeric)
            .expect("create");

        // $eq on numeric field via the index
        let filter = MetadataFilter::eq("priority", MetadataValue::Float(0.0));
        let count = db.count_filtered(None, Some(&filter));
        // 0 % 3 == 0: i=0,3,6,9,12,15,18 → 7 records
        assert_eq!(count, 7);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_persists_across_reopen() {
        let path = temp_file("pidx-persist");
        {
            let mut db = Database::create(&path, 3).expect("create");

            let mut meta = Metadata::new();
            meta.insert("source".into(), "blog".into());
            db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");

            let mut meta2 = Metadata::new();
            meta2.insert("source".into(), "docs".into());
            db.upsert("doc2", vec![0.0, 1.0, 0.0], meta2)
                .expect("upsert");

            db.create_index("source", PayloadIndexType::Keyword)
                .expect("create");
        }

        // Reopen and verify index survives
        let db = Database::open(&path).expect("reopen");
        let indexes = db.list_indexes();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].0, "source");

        // Index should be functional after reopen
        let count = db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog")));
        assert_eq!(count, 1);

        cleanup(&path);
    }

    #[test]
    fn payload_index_incremental_upsert_adds_to_index() {
        let path = temp_file("pidx-incr-upsert");
        let mut db = Database::create(&path, 3).expect("create");

        // Create index first (empty)
        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");

        // Now upsert records — they should be indexed incrementally
        let mut meta = Metadata::new();
        meta.insert("source".into(), "blog".into());
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        let count = db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog")));
        assert_eq!(count, 1);

        // Upsert another
        let mut meta2 = Metadata::new();
        meta2.insert("source".into(), "blog".into());
        db.upsert("doc2", vec![0.0, 1.0, 0.0], meta2)
            .expect("upsert");

        let count2 = db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog")));
        assert_eq!(count2, 2);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_incremental_delete_removes_from_index() {
        let path = temp_file("pidx-incr-delete");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("source".into(), "blog".into());
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        let mut meta2 = Metadata::new();
        meta2.insert("source".into(), "blog".into());
        db.upsert("doc2", vec![0.0, 1.0, 0.0], meta2)
            .expect("upsert");

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");

        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog"))),
            2
        );

        // Delete one record
        db.delete("doc1").expect("delete");

        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog"))),
            1
        );

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_incremental_upsert_replaces_old_value() {
        let path = temp_file("pidx-incr-replace");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("source".into(), "blog".into());
        db.upsert_in_namespace("", "doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");

        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog"))),
            1
        );

        // Upsert same id with different metadata value (uses upsert_in_namespace for true upsert)
        let mut meta2 = Metadata::new();
        meta2.insert("source".into(), "docs".into());
        db.upsert_in_namespace("", "doc1", vec![1.0, 0.0, 0.0], meta2)
            .expect("upsert replace");

        // Old value gone
        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("source", "blog"))),
            0
        );
        // New value present
        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("source", "docs"))),
            1
        );

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_update_metadata_maintains_index() {
        let path = temp_file("pidx-update-meta");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta = Metadata::new();
        meta.insert("status".into(), "draft".into());
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta)
            .expect("upsert");

        db.create_index("status", PayloadIndexType::Keyword)
            .expect("create");

        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("status", "draft"))),
            1
        );

        // update_metadata changes the indexed field
        let mut patch = Metadata::new();
        patch.insert("status".into(), "published".into());
        db.update_metadata("doc1", patch).expect("update");

        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("status", "draft"))),
            0
        );
        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("status", "published"))),
            1
        );

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_and_filter_intersection() {
        let path = temp_file("pidx-and");
        let mut db = Database::create(&path, 3).expect("create");

        // Insert records with source and priority
        for i in 0..30 {
            let mut meta = Metadata::new();
            meta.insert(
                "source".into(),
                if i % 2 == 0 { "blog" } else { "docs" }.into(),
            );
            meta.insert("priority".into(), MetadataValue::Float((i % 3) as f64));
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create source");
        db.create_index("priority", PayloadIndexType::Numeric)
            .expect("create priority");

        // AND(source == "blog", priority > 1) → source=blog(even i) AND priority=2(i%3==2)
        // Even i where i%3==2: 2,8,14,20,26 → 5
        let filter = MetadataFilter::and(vec![
            MetadataFilter::eq("source", "blog"),
            MetadataFilter::gt("priority", 1.0),
        ]);
        let count = db.count_filtered(None, Some(&filter));
        assert_eq!(count, 5);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_with_namespace_filtering() {
        let path = temp_file("pidx-ns");
        let mut db = Database::create(&path, 3).expect("create");

        let mut meta1 = Metadata::new();
        meta1.insert("tag".into(), "rust".into());
        db.upsert_in_namespace("ns1", "doc1", vec![1.0, 0.0, 0.0], meta1)
            .expect("upsert");

        let mut meta2 = Metadata::new();
        meta2.insert("tag".into(), "rust".into());
        db.upsert_in_namespace("ns2", "doc2", vec![0.0, 1.0, 0.0], meta2)
            .expect("upsert");

        db.create_index("tag", PayloadIndexType::Keyword)
            .expect("create");

        // Without namespace → both
        assert_eq!(
            db.count_filtered(None, Some(&MetadataFilter::eq("tag", "rust"))),
            2
        );

        // With namespace → scoped
        assert_eq!(
            db.count_filtered(Some("ns1"), Some(&MetadataFilter::eq("tag", "rust"))),
            1
        );
        assert_eq!(
            db.count_filtered(Some("ns2"), Some(&MetadataFilter::eq("tag", "rust"))),
            1
        );

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn create_index_read_only_fails() {
        let path = temp_file("pidx-ro");
        {
            let mut db = Database::create(&path, 3).expect("create");
            db.upsert("doc1", vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert");
        }

        let mut db = Database::open_read_only(&path).expect("open ro");
        let result = db.create_index("source", PayloadIndexType::Keyword);
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn drop_index_read_only_fails() {
        let path = temp_file("pidx-drop-ro");
        {
            let mut db = Database::create(&path, 3).expect("create");
            db.create_index("source", PayloadIndexType::Keyword)
                .expect("create");
        }

        let mut db = Database::open_read_only(&path).expect("open ro");
        let result = db.drop_index("source");
        assert!(result.is_err());

        cleanup(&path);
    }

    #[test]
    fn multiple_indexes_independent() {
        let path = temp_file("pidx-multi");
        let mut db = Database::create(&path, 3).expect("create");

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("kw");
        db.create_index("score", PayloadIndexType::Numeric)
            .expect("num");

        assert_eq!(db.list_indexes().len(), 2);

        // Drop one, other remains
        db.drop_index("source").expect("drop source");
        let indexes = db.list_indexes();
        assert_eq!(indexes.len(), 1);
        assert_eq!(indexes[0].0, "score");

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_search_returns_correct_results() {
        let path = temp_file("pidx-search");
        let mut db = Database::create(&path, 3).expect("create");

        // Insert records where only some match the filter
        let mut meta1 = Metadata::new();
        meta1.insert("category".into(), "tech".into());
        db.upsert("doc1", vec![1.0, 0.0, 0.0], meta1)
            .expect("upsert");

        let mut meta2 = Metadata::new();
        meta2.insert("category".into(), "science".into());
        db.upsert("doc2", vec![0.9, 0.1, 0.0], meta2)
            .expect("upsert");

        let mut meta3 = Metadata::new();
        meta3.insert("category".into(), "tech".into());
        db.upsert("doc3", vec![0.8, 0.2, 0.0], meta3)
            .expect("upsert");

        // Create keyword index on category
        db.create_index("category", PayloadIndexType::Keyword)
            .expect("create");

        // Search with filter should only return tech records
        let results = db
            .search(
                &[1.0, 0.0, 0.0],
                SearchOptions {
                    top_k: 10,
                    filter: Some(MetadataFilter::eq("category", "tech")),
                    truncate_dim: None,
                },
            )
            .expect("search");
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc3"));
        assert!(!ids.contains(&"doc2"));

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_list_uses_index() {
        let path = temp_file("pidx-list");
        let mut db = Database::create(&path, 3).expect("create");

        for i in 0..20 {
            let mut meta = Metadata::new();
            meta.insert(
                "type".into(),
                if i % 2 == 0 { "even" } else { "odd" }.into(),
            );
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("type", PayloadIndexType::Keyword)
            .expect("create");

        let records = db.list(None, Some(&MetadataFilter::eq("type", "even")), 0, 0);
        assert_eq!(records.len(), 10);
        for r in &records {
            assert_eq!(
                r.metadata.get("type"),
                Some(&MetadataValue::String("even".into()))
            );
        }

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_rebuild_on_create_with_existing_data() {
        let path = temp_file("pidx-rebuild");
        let mut db = Database::create(&path, 3).expect("create");

        // Insert data BEFORE creating the index
        for i in 0..10 {
            let mut meta = Metadata::new();
            meta.insert("color".into(), if i < 4 { "red" } else { "blue" }.into());
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        // Create index AFTER data exists — should rebuild from existing records
        db.create_index("color", PayloadIndexType::Keyword)
            .expect("create");

        let red_count = db.count_filtered(None, Some(&MetadataFilter::eq("color", "red")));
        assert_eq!(red_count, 4);

        let blue_count = db.count_filtered(None, Some(&MetadataFilter::eq("color", "blue")));
        assert_eq!(blue_count, 6);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn numeric_index_combined_range() {
        let path = temp_file("pidx-num-combined");
        let mut db = Database::create(&path, 3).expect("create");

        for i in 0..50 {
            let mut meta = Metadata::new();
            meta.insert("val".into(), MetadataValue::Float(i as f64));
            db.upsert(format!("doc{}", i), vec![1.0, 0.0, 0.0], meta)
                .expect("upsert");
        }

        db.create_index("val", PayloadIndexType::Numeric)
            .expect("create");

        // AND(val >= 10, val < 20) → 10..19 = 10 records
        let filter = MetadataFilter::and(vec![
            MetadataFilter::gte("val", 10.0),
            MetadataFilter::lt("val", 20.0),
        ]);
        let count = db.count_filtered(None, Some(&filter));
        assert_eq!(count, 10);

        drop(db);
        cleanup(&path);
    }

    #[test]
    fn payload_index_sidecar_cleaned_on_drop_last_index() {
        let path = temp_file("pidx-sidecar-clean");
        let mut db = Database::create(&path, 3).expect("create");

        db.create_index("source", PayloadIndexType::Keyword)
            .expect("create");

        // Sidecar should exist
        let sidecar = path.with_extension("vdb.pidx");
        assert!(sidecar.exists());

        db.drop_index("source").expect("drop");

        // After dropping the last index, sidecar might still exist (with empty content)
        // but on reopen, list_indexes should be empty
        drop(db);

        let db2 = Database::open(&path).expect("reopen");
        assert!(db2.list_indexes().is_empty());

        cleanup(&path);
    }

    // -------------------------------------------------------------------
    // TTL / Expiry tests
    // -------------------------------------------------------------------

    #[test]
    fn set_ttl_hides_record_from_get() {
        let path = temp_file("ttl-get");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert("doc1", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert");

        // Set TTL to 0 seconds — effectively already expired.
        assert!(db.set_ttl("doc1", 0.0).expect("set_ttl"));
        // Tiny sleep to ensure timestamp is strictly past
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(db.get("doc1").is_none());
        cleanup(&path);
    }

    #[test]
    fn clear_ttl_makes_record_visible_again() {
        let path = temp_file("ttl-clear");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert("doc1", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert");

        db.set_ttl("doc1", 0.0).expect("set_ttl");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(db.get("doc1").is_none());

        db.clear_ttl("doc1").expect("clear_ttl");
        assert!(db.get("doc1").is_some());
        cleanup(&path);
    }

    #[test]
    fn expired_records_excluded_from_count_and_list() {
        let path = temp_file("ttl-count-list");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert a");
        db.upsert("b", vec![0.0, 1.0, 0.0], Metadata::new())
            .expect("upsert b");

        assert_eq!(db.count_filtered(None, None), 2);
        assert_eq!(db.list(None, None, 0, 0).len(), 2);

        db.set_ttl("a", 0.0).expect("set_ttl");
        std::thread::sleep(std::time::Duration::from_millis(10));

        assert_eq!(db.count_filtered(None, None), 1);
        assert_eq!(db.list(None, None, 0, 0).len(), 1);
        assert_eq!(db.list(None, None, 0, 0)[0].id, "b");
        cleanup(&path);
    }

    #[test]
    fn expired_records_excluded_from_search() {
        let path = temp_file("ttl-search");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert a");
        db.upsert("b", vec![0.0, 1.0, 0.0], Metadata::new())
            .expect("upsert b");

        db.set_ttl("a", 0.0).expect("set_ttl");
        std::thread::sleep(std::time::Duration::from_millis(10));

        let results = db
            .search(&[1.0, 0.0, 0.0], SearchOptions::default())
            .expect("search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
        cleanup(&path);
    }

    #[test]
    fn ttl_persists_across_reopen() {
        let path = temp_file("ttl-persist");
        {
            let mut db = Database::create(&path, 3).expect("create");
            db.upsert("doc1", vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert");
            db.set_ttl("doc1", 0.0).expect("set_ttl");
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
        let db = Database::open(&path).expect("reopen");
        assert!(db.get("doc1").is_none());
        assert_eq!(db.count_filtered(None, None), 0);
        cleanup(&path);
    }

    #[test]
    fn compact_removes_expired_records() {
        let path = temp_file("ttl-compact");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert a");
        db.upsert("b", vec![0.0, 1.0, 0.0], Metadata::new())
            .expect("upsert b");

        db.set_ttl("a", 0.0).expect("set_ttl");
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.compact().expect("compact");
        // After compact, the expired record should be physically removed.
        assert_eq!(db.len(), 1);
        cleanup(&path);
    }

    #[test]
    fn set_ttl_on_nonexistent_record_returns_false() {
        let path = temp_file("ttl-missing");
        let mut db = Database::create(&path, 3).expect("create");
        assert!(!db.set_ttl("ghost", 60.0).expect("set_ttl"));
        assert!(!db.clear_ttl("ghost").expect("clear_ttl"));
        cleanup(&path);
    }

    #[test]
    fn upsert_with_expires_at() {
        let path = temp_file("ttl-upsert-ea");
        let mut db = Database::create(&path, 3).expect("create");
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Already expired
        let record = Record {
            namespace: String::new(),
            id: "doc1".into(),
            vector: vec![1.0, 0.0, 0.0],
            vectors: NamedVectors::new(),
            sparse: SparseVector::new(),
            metadata: Metadata::new(),
            multi_vectors: MultiVectors::new(),
            expires_at: Some(now - 1.0),
        };
        db.upsert_many(std::iter::once(record)).expect("upsert");
        assert!(db.get("doc1").is_none());

        // Far future — visible
        let record2 = Record {
            namespace: String::new(),
            id: "doc2".into(),
            vector: vec![0.0, 1.0, 0.0],
            vectors: NamedVectors::new(),
            sparse: SparseVector::new(),
            metadata: Metadata::new(),
            multi_vectors: MultiVectors::new(),
            expires_at: Some(now + 86400.0),
        };
        db.upsert_many(std::iter::once(record2)).expect("upsert");
        assert!(db.get("doc2").is_some());
        cleanup(&path);
    }

    #[test]
    fn set_ttl_in_namespace() {
        let path = temp_file("ttl-ns");
        let mut db = Database::create(&path, 3).expect("create");
        db.upsert_in_namespace("ns1", "doc1", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("upsert");

        assert!(db.set_ttl_in_namespace("ns1", "doc1", 0.0).expect("set"));
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(db.get_in_namespace("ns1", "doc1").is_none());

        // Wrong namespace returns false
        assert!(
            !db.set_ttl_in_namespace("ns2", "doc1", 60.0)
                .expect("set wrong ns")
        );

        cleanup(&path);
    }

    // -------------------------------------------------------------------
    // Cursor-based pagination tests
    // -------------------------------------------------------------------

    #[test]
    fn list_cursor_basic() {
        let path = temp_file("cursor-basic");
        let mut db = Database::create(&path, 3).expect("create");
        for i in 0..5 {
            db.upsert(&format!("doc{i}"), vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert");
        }

        // First page of 2
        let (page1, cursor1) = db.list_cursor(None, None, 2, None);
        assert_eq!(page1.len(), 2);
        assert!(cursor1.is_some());

        // Second page of 2
        let (page2, cursor2) = db.list_cursor(None, None, 2, cursor1.as_deref());
        assert_eq!(page2.len(), 2);
        assert!(cursor2.is_some());

        // Third page (only 1 remaining)
        let (page3, cursor3) = db.list_cursor(None, None, 2, cursor2.as_deref());
        assert_eq!(page3.len(), 1);
        assert!(cursor3.is_none());

        // No duplicates across pages
        let mut all_ids: Vec<String> = Vec::new();
        for r in page1.iter().chain(page2.iter()).chain(page3.iter()) {
            all_ids.push(r.id.clone());
        }
        all_ids.sort();
        all_ids.dedup();
        assert_eq!(all_ids.len(), 5);

        cleanup(&path);
    }

    #[test]
    fn list_cursor_with_namespace() {
        let path = temp_file("cursor-ns");
        let mut db = Database::create(&path, 3).expect("create");
        for i in 0..3 {
            db.upsert_in_namespace(
                "ns1",
                &format!("doc{i}"),
                vec![1.0, 0.0, 0.0],
                Metadata::new(),
            )
            .expect("upsert");
        }
        for i in 0..2 {
            db.upsert_in_namespace(
                "ns2",
                &format!("doc{i}"),
                vec![0.0, 1.0, 0.0],
                Metadata::new(),
            )
            .expect("upsert");
        }

        let (page1, cursor1) = db.list_cursor(Some("ns1"), None, 2, None);
        assert_eq!(page1.len(), 2);
        assert!(cursor1.is_some());

        let (page2, cursor2) = db.list_cursor(Some("ns1"), None, 2, cursor1.as_deref());
        assert_eq!(page2.len(), 1);
        assert!(cursor2.is_none());

        cleanup(&path);
    }

    #[test]
    fn list_cursor_excludes_expired() {
        let path = temp_file("cursor-ttl");
        let mut db = Database::create(&path, 3).expect("create");
        for i in 0..5 {
            db.upsert(&format!("doc{i}"), vec![1.0, 0.0, 0.0], Metadata::new())
                .expect("upsert");
        }

        // Expire doc1
        db.set_ttl("doc1", 0.0).expect("set ttl");
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Paginate all — should get 4, not 5
        let mut all = Vec::new();
        let mut cursor = None;
        loop {
            let (page, next) = db.list_cursor(None, None, 10, cursor.as_deref());
            all.extend(page.into_iter().cloned());
            if next.is_none() {
                break;
            }
            cursor = next;
        }
        assert_eq!(all.len(), 4);
        assert!(!all.iter().any(|r| r.id == "doc1"));

        cleanup(&path);
    }

    #[test]
    fn list_cursor_empty_database() {
        let path = temp_file("cursor-empty");
        let db = Database::create(&path, 3).expect("create");
        let (page, cursor) = db.list_cursor(None, None, 10, None);
        assert!(page.is_empty());
        assert!(cursor.is_none());
        cleanup(&path);
    }

    // ---------------------------------------------------------------
    // Bug #14: zero-norm query vector should be rejected for cosine
    // ---------------------------------------------------------------

    #[test]
    fn search_zero_norm_query_cosine_rejected() {
        let path = temp_file("zero-norm-cosine");
        let mut db = Database::create(&path, 3).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        let result = db.search(
            &[0.0, 0.0, 0.0],
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "zero-norm cosine search should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("zero norm"),
            "error should mention zero norm: {err_msg}"
        );
        cleanup(&path);
    }

    #[test]
    fn search_zero_norm_query_dotproduct_rejected() {
        let path = temp_file("zero-norm-dot");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::DotProduct).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        let result = db.search(
            &[0.0, 0.0, 0.0],
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "zero-norm dotproduct search should fail");
        cleanup(&path);
    }

    #[test]
    fn search_zero_norm_query_euclidean_allowed() {
        let path = temp_file("zero-norm-euclidean");
        let mut db =
            Database::create_with_metric(&path, 3, DistanceMetric::Euclidean).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        // Euclidean distance from the origin is well-defined; should succeed.
        let result = db.search(
            &[0.0, 0.0, 0.0],
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        );
        assert!(
            result.is_ok(),
            "zero-norm euclidean search should succeed: {:?}",
            result.err()
        );
        cleanup(&path);
    }

    // ---------------------------------------------------------------
    // Bug #15: dimension mismatch in search query should be rejected
    // ---------------------------------------------------------------

    #[test]
    fn search_undersized_query_rejected() {
        let path = temp_file("dim-under");
        let mut db = Database::create(&path, 4).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        // Query dim=2 on a dim=4 database without truncate_dim.
        let result = db.search(
            &[1.0, 0.0],
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "undersized query should fail");
        match result.unwrap_err() {
            VectLiteError::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 4);
                assert_eq!(found, 2);
            }
            other => panic!("expected DimensionMismatch, got: {other}"),
        }
        cleanup(&path);
    }

    #[test]
    fn search_oversized_query_rejected() {
        let path = temp_file("dim-over");
        let mut db = Database::create(&path, 3).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        let result = db.search(
            &[1.0, 0.0, 0.0, 0.0, 0.0],
            SearchOptions {
                top_k: 5,
                ..Default::default()
            },
        );
        assert!(result.is_err(), "oversized query should fail");
        match result.unwrap_err() {
            VectLiteError::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 3);
                assert_eq!(found, 5);
            }
            other => panic!("expected DimensionMismatch, got: {other}"),
        }
        cleanup(&path);
    }

    #[test]
    fn search_undersized_query_with_truncate_dim_allowed() {
        let path = temp_file("dim-matryoshka");
        let mut db = Database::create(&path, 4).expect("create");
        db.insert("a", vec![1.0, 0.0, 0.0, 0.0], Metadata::new())
            .expect("insert");

        // With explicit truncate_dim, undersized queries are Matryoshka-truncated.
        let result = db.search(
            &[1.0, 0.0],
            SearchOptions {
                top_k: 5,
                truncate_dim: Some(2),
                ..Default::default()
            },
        );
        assert!(
            result.is_ok(),
            "truncate_dim query should succeed: {:?}",
            result.err()
        );
        cleanup(&path);
    }
}
