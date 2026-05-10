pub mod quantization;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as StdError;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use fs2::FileExt;
use hnsw_rs::prelude::*;

use quantization::{QuantizationConfig, QuantizedIndex};

const MAGIC: &[u8; 4] = b"VDB1";
const VERSION: u16 = 4;
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
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

pub type Result<T> = std::result::Result<T, VectLiteError>;
pub type Metadata = BTreeMap<String, MetadataValue>;
pub type SparseVector = BTreeMap<String, f32>;
pub type NamedVectors = BTreeMap<String, Vec<f32>>;
type RecordKey = (String, String);

#[derive(Clone, Debug)]
enum WalOp {
    Upsert(Record),
    Delete { namespace: String, id: String },
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
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            top_k: 10,
            filter: None,
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

pub struct Database {
    path: PathBuf,
    wal_path: PathBuf,
    dimension: usize,
    records: BTreeMap<(String, String), Record>,
    ann: AnnCatalog,
    sparse_index: SparseIndex,
    wal_entries_replayed: usize,
    ann_loaded_from_disk: bool,
    read_only: bool,
    /// Holds the lock file open for the lifetime of the database.
    /// Dropping this releases the advisory lock.
    _lock_file: Option<File>,
    /// Optional quantized index for accelerated search.
    quantized: Option<QuantizedIndex>,
    /// Configuration used to build the quantized index (persisted).
    quantization_config: Option<QuantizationConfig>,
    /// Ordered keys mapping quantized index positions to record keys.
    quantized_keys: Vec<RecordKey>,
}

#[derive(Default)]
struct AnnCatalog {
    global: BTreeMap<String, AnnIndex>,
    namespaces: BTreeMap<String, BTreeMap<String, AnnIndex>>,
}

struct AnnIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    keys: Vec<RecordKey>,
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
        ensure_dimension(dimension)?;
        let lock = acquire_exclusive_lock(path.as_ref())?;

        let mut database = Self {
            path: path.as_ref().to_path_buf(),
            wal_path: wal_path(path.as_ref()),
            dimension,
            records: BTreeMap::new(),
            ann: AnnCatalog::default(),
            sparse_index: SparseIndex::default(),
            wal_entries_replayed: 0,
            ann_loaded_from_disk: false,
            read_only: false,
            _lock_file: Some(lock),
            quantized: None,
            quantization_config: None,
            quantized_keys: Vec::new(),
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
        // Release the lock by dropping the file handle
        self._lock_file = None;
        // Clear in-memory state
        self.records.clear();
        self.ann = AnnCatalog::default();
        self.sparse_index = SparseIndex::default();
        self.quantized = None;
        self.quantization_config = None;
        self.quantized_keys.clear();
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
            Self::create(path, dimension)
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn dimension(&self) -> usize {
        self.dimension
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
        self.records
            .iter()
            .filter(|((ns, _), record)| {
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

    /// List records by namespace and/or metadata filter without requiring a
    /// vector query.  Returns records ordered by (namespace, id).
    pub fn list(
        &self,
        namespace: Option<&str>,
        filter: Option<&MetadataFilter>,
        limit: usize,
        offset: usize,
    ) -> Vec<&Record> {
        self.records
            .iter()
            .filter(|((ns, _), record)| {
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
        self.rebuild_quantized_index();
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
        self.rebuild_quantized_index();
        Ok(count)
    }

    pub fn get(&self, id: &str) -> Option<&Record> {
        self.get_in_namespace(DEFAULT_NAMESPACE, id)
    }

    pub fn get_in_namespace(&self, namespace: &str, id: &str) -> Option<&Record> {
        self.records.get(&(namespace.to_owned(), id.to_owned()))
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
        self.rebuild_quantized_index();
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
        if let Some(query) = dense_query {
            self.validate_vector(query)?;
        }
        if dense_query.is_none() && sparse_query.is_none() {
            return Err(VectLiteError::InvalidFormat(
                "search requires a dense query, a sparse query, or both".to_owned(),
            ));
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

        let dense_start = Instant::now();
        // Use quantized index for candidate selection if available (2-stage pipeline).
        // The quantized index operates on the default vector only and globally (not per-namespace).
        let quantized_candidates =
            if vector_name.is_none() || vector_name == Some(DEFAULT_VECTOR_NAME) {
                dense_query.and_then(|query| self.quantized_candidate_keys(query, fetch_k))
            } else {
                None
            };
        let ann_candidates = if quantized_candidates.is_some() {
            // Skip HNSW if quantized index provided candidates
            None
        } else {
            dense_query
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
        let mut stats = SearchStats {
            used_ann: effective_dense_candidates.is_some(),
            ann_candidate_count: effective_dense_candidates.as_ref().map_or(0, Vec::len),
            fetch_k,
            sparse_candidate_count: sparse_candidates.len(),
            ann_loaded_from_disk: self.ann_loaded_from_disk,
            wal_entries_replayed: self.wal_entries_replayed,
            fusion: options.fusion.label().to_owned(),
            ..SearchStats::default()
        };

        let mut results = self.collect_results(
            dense_query,
            sparse_query,
            &options,
            namespace,
            candidate_keys.as_deref(),
        );
        stats.considered_count = results.len();

        if effective_dense_candidates.is_some() && results.len() < fetch_k {
            stats.exact_fallback = true;
            results = self.collect_results(dense_query, sparse_query, &options, namespace, None);
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

    /// Bulk-ingest many records efficiently. WAL writes happen in batches of
    /// `batch_size`, but the ANN index and sparse index are only rebuilt once
    /// at the very end, making this much faster than `upsert_many` for large
    /// imports.
    pub fn bulk_ingest<I>(&mut self, records: I, batch_size: usize) -> Result<usize>
    where
        I: IntoIterator<Item = Record>,
    {
        self.check_writable()?;
        let batch_size = batch_size.max(1);
        let mut total = 0_usize;
        let mut batch = Vec::with_capacity(batch_size);

        for record in records {
            self.validate_record(&record)?;
            batch.push(WalOp::Upsert(record));

            if batch.len() >= batch_size {
                total += batch.len();
                self.append_wal_batch(&batch)?;
                self.apply_ops_in_memory(batch);
                batch = Vec::with_capacity(batch_size);
            }
        }

        if !batch.is_empty() {
            total += batch.len();
            self.append_wal_batch(&batch)?;
            self.apply_ops_in_memory(batch);
        }

        if total > 0 {
            self.rebuild_sparse_index();
            self.rebuild_ann();
            self.ann_loaded_from_disk = false;
            self.persist_ann_to_disk()?;
            self.rebuild_quantized_index();
        }

        Ok(total)
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
        self.quantization_config = Some(config);
        self.rebuild_quantized_index();
        self.persist_quantization_params()?;
        Ok(())
    }

    /// Disable quantization and remove persisted parameters.
    pub fn disable_quantization(&mut self) -> Result<()> {
        self.check_writable()?;
        self.quantized = None;
        self.quantization_config = None;
        self.quantized_keys.clear();
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

        let candidate_indices = index.search_candidates(query, top_k);
        Some(
            candidate_indices
                .into_iter()
                .filter_map(|idx| self.quantized_keys.get(idx).cloned())
                .collect(),
        )
    }

    fn compact_inner(&mut self) -> Result<()> {
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
        });

        self.append_wal_batch(&ops)?;
        self.apply_ops_in_memory(ops);

        if has_sparse {
            self.rebuild_sparse_index();
        }
        self.rebuild_ann();
        self.ann_loaded_from_disk = false;
        self.persist_ann_to_disk()?;
        self.rebuild_quantized_index();
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
        for op in ops {
            match op {
                WalOp::Upsert(record) => {
                    self.records
                        .insert((record.namespace.clone(), record.id.clone()), record);
                }
                WalOp::Delete { namespace, id } => {
                    self.records.remove(&(namespace, id));
                }
            }
        }
    }

    fn append_wal_batch(&self, ops: &[WalOp]) -> Result<()> {
        if let Some(parent) = self.wal_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }

        let new_file = !self.wal_path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)?;

        if new_file {
            file.write_all(WAL_MAGIC)?;
        }

        let mut buffer = Vec::new();
        write_u32(&mut buffer, u32_from_usize(ops.len())?)?;
        for op in ops {
            write_wal_op(&mut buffer, op)?;
        }

        write_u32(&mut file, u32_from_usize(buffer.len())?)?;
        file.write_all(&buffer)?;
        file.sync_all()?;
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

    fn clear_wal(&self) -> Result<()> {
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

            let record = Record {
                namespace: namespace.clone(),
                id: id.clone(),
                vector,
                vectors,
                sparse,
                metadata,
            };
            records.insert((namespace, id), record);
        }

        Ok(Self {
            path: path.to_path_buf(),
            wal_path: wal_path(path),
            dimension,
            records,
            ann: AnnCatalog::default(),
            sparse_index: SparseIndex::default(),
            wal_entries_replayed: 0,
            ann_loaded_from_disk: false,
            read_only: false,
            _lock_file: None,
            quantized: None,
            quantization_config: None,
            quantized_keys: Vec::new(),
        })
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(MAGIC)?;
        write_u16(writer, VERSION)?;
        write_u32(writer, u32_from_usize(self.dimension)?)?;
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

        Ok(())
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

        self.ann.global = global_by_vector
            .into_iter()
            .filter_map(|(vector_name, records)| {
                if records.len() < ANN_MIN_POINTS {
                    None
                } else {
                    Some((vector_name, build_ann_index(records)))
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
                            Some((vector_name, build_ann_index(records)))
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

            let Some(index) = load_ann_index(
                parent,
                &ann_basename(
                    &self.path,
                    expected_entry.namespace.as_deref(),
                    &expected_entry.vector_name,
                ),
                expected_entry.keys.clone(),
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

        let entries = self.expected_ann_entries();
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
                index.hnsw.file_dump(parent, &basename).map_err(|err| {
                    VectLiteError::InvalidFormat(format!("failed to persist ANN index: {err}"))
                })?;
            }
        }

        write_ann_manifest(&ann_manifest_path(&self.path), &entries)
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
    ) -> Vec<ScoredRecord<'_>> {
        let record_iter: Box<dyn Iterator<Item = &Record> + '_> = match candidate_keys {
            Some(keys) => Box::new(keys.iter().filter_map(|key| self.records.get(key))),
            None => Box::new(self.records.values()),
        };

        record_iter
            .filter(|record| {
                namespace
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
                                weighted_sum += weight * cosine_similarity(query, vector);
                            }
                        }
                        (weighted_sum, None)
                    } else {
                        let score = dense_query
                            .and_then(|query| {
                                record
                                    .vector_for(options.vector_name.as_deref())
                                    .map(|vector| cosine_similarity(query, vector))
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
        if index.keys.len() < ANN_SEARCH_MIN_POINTS {
            return None;
        }

        let candidate_count = candidate_count(top_k, index.keys.len());
        if candidate_count == 0 {
            return None;
        }

        let ef_search = candidate_count.max(ANN_EF_CONSTRUCTION);
        let neighbours = index.hnsw.search(query, candidate_count, ef_search);
        Some(
            neighbours
                .into_iter()
                .filter_map(|neighbour| index.keys.get(neighbour.d_id).cloned())
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

        Ok(Record {
            namespace: namespace.into(),
            id: id.into(),
            vector,
            vectors,
            sparse,
            metadata,
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

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
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

fn build_ann_index(records: Vec<(RecordKey, &Vec<f32>)>) -> AnnIndex {
    let max_layer = compute_hnsw_layers(records.len());
    let mut hnsw = Hnsw::<f32, DistCosine>::new(
        ANN_M,
        records.len(),
        max_layer,
        ANN_EF_CONSTRUCTION,
        DistCosine {},
    );

    let mut keys = Vec::with_capacity(records.len());
    for (origin_id, (key, vector)) in records.into_iter().enumerate() {
        hnsw.insert((vector.as_slice(), origin_id));
        keys.push(key);
    }
    hnsw.set_searching_mode(true);

    AnnIndex { hnsw, keys }
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
    format!(
        "{stem}.ann.{}.{}",
        hex_encode(namespace.unwrap_or(DEFAULT_NAMESPACE).as_bytes()),
        hex_encode(vector_name.as_bytes())
    )
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn record_key_signature(keys: &[RecordKey]) -> u64 {
    let mut state = 0xcbf29ce484222325_u64;
    for (namespace, id) in keys {
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

fn load_ann_index(directory: &Path, basename: &str, keys: Vec<RecordKey>) -> Option<AnnIndex> {
    let reloader = Box::leak(Box::new(HnswIo::new(directory, basename)));
    let mut hnsw = reloader.load_hnsw_with_dist(DistCosine {}).ok()?;
    hnsw.set_searching_mode(true);
    Some(AnnIndex { hnsw, keys })
}

fn write_ann_manifest(path: &Path, entries: &[AnnManifestEntry]) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(b"ANN1")?;
    write_u32(&mut file, u32_from_usize(entries.len())?)?;
    for entry in entries {
        write_u8(&mut file, u8::from(entry.namespace.is_some()))?;
        if let Some(namespace) = &entry.namespace {
            write_string(&mut file, namespace)?;
        }
        write_string(&mut file, &entry.vector_name)?;
        write_u64(&mut file, u64_from_usize(entry.record_count)?)?;
        write_u64(&mut file, entry.key_signature)?;
    }
    file.sync_all()?;
    Ok(())
}

fn read_ann_manifest(path: &Path) -> Result<Vec<AnnManifestEntry>> {
    let mut file = BufReader::new(File::open(path)?);
    let mut magic = [0_u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"ANN1" {
        return Err(VectLiteError::InvalidFormat(
            "invalid ANN manifest".to_owned(),
        ));
    }

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
        entries.push(AnnManifestEntry {
            namespace,
            vector_name,
            record_count,
            key_signature,
            keys: Vec::new(),
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
) -> f32 {
    let dense_score = match (left.vector_for(vector_name), right.vector_for(vector_name)) {
        (Some(left), Some(right)) => cosine_similarity(left, right),
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
        }
        WalOp::Delete { namespace, id } => {
            write_u8(writer, 2)?;
            write_string(writer, namespace)?;
            write_string(writer, id)?;
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
            Ok(WalOp::Upsert(Record {
                namespace,
                id,
                vector,
                vectors,
                sparse,
                metadata,
            }))
        }
        2 => Ok(WalOp::Delete {
            namespace: read_string(reader)?,
            id: read_string(reader)?,
        }),
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
        Database, HybridSearchOptions, Metadata, MetadataFilter, MetadataValue, NamedVectors,
        Record, SearchOptions, SparseVector, VectLiteError,
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
                    },
                    Record {
                        namespace: "".to_owned(),
                        id: "doc2".to_owned(),
                        vector: vec![0.0, 1.0],
                        vectors: NamedVectors::new(),
                        sparse: SparseVector::new(),
                        metadata: Metadata::new(),
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
                    },
                )
                .expect("search after reopen");
            assert!(!results.is_empty());
            assert_eq!(results[0].id, "doc0");
        }

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
}
