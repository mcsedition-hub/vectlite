uniffi::include_scaffolding!("vectlite");

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use serde_json::{json, Map, Number, Value};
use vectlite::quantization::{
    BinaryQuantizationConfig, ProductQuantizationConfig, QuantizationConfig,
    ScalarQuantizationConfig,
};
use vectlite::{
    Database as CoreDatabase, DistanceMetric, FusionStrategy, HybridSearchOptions, Metadata,
    MetadataFilter, MetadataValue, PayloadIndexType, Record, SparseVector, Store as CoreStore,
    WriteOperation,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum VectLiteError {
    #[error("io error: {msg}")]
    Io { msg: String },
    #[error("invalid format: {msg}")]
    InvalidFormat { msg: String },
    #[error("dimension mismatch: {msg}")]
    DimensionMismatch { msg: String },
    #[error("duplicate id: {msg}")]
    DuplicateId { msg: String },
    #[error("database is opened in read-only mode")]
    ReadOnly,
    #[error("lock contention: {msg}")]
    LockContention { msg: String },
    #[error("json error: {msg}")]
    JsonError { msg: String },
}

impl From<vectlite::VectLiteError> for VectLiteError {
    fn from(err: vectlite::VectLiteError) -> Self {
        match err {
            vectlite::VectLiteError::Io(e) => VectLiteError::Io { msg: e.to_string() },
            vectlite::VectLiteError::InvalidFormat(msg) => VectLiteError::InvalidFormat { msg },
            vectlite::VectLiteError::DimensionMismatch { expected, found } => {
                VectLiteError::DimensionMismatch {
                    msg: format!("expected {expected}, found {found}"),
                }
            }
            vectlite::VectLiteError::DuplicateId { namespace, id } => VectLiteError::DuplicateId {
                msg: if namespace.is_empty() {
                    format!("id '{id}' already exists")
                } else {
                    format!("id '{id}' already exists in namespace '{namespace}'")
                },
            },
            vectlite::VectLiteError::ReadOnly => VectLiteError::ReadOnly,
            vectlite::VectLiteError::LockContention(msg) => VectLiteError::LockContention { msg },
        }
    }
}

fn json_err(msg: impl Into<String>) -> VectLiteError {
    VectLiteError::JsonError { msg: msg.into() }
}

// ---------------------------------------------------------------------------
// UDL dictionary types
// ---------------------------------------------------------------------------

pub struct SearchResult {
    pub namespace: String,
    pub id: String,
    pub score: f32,
    pub metadata_json: String,
}

pub struct RecordResult {
    pub namespace: String,
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata_json: String,
    pub expires_at: Option<f64>,
}

pub struct CursorPage {
    pub records: Vec<RecordResult>,
    pub cursor: Option<String>,
}

pub struct SearchStatsResult {
    pub results: Vec<SearchResult>,
    pub stats_json: String,
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

pub struct Database {
    inner: RwLock<CoreDatabase>,
}

impl Database {
    // -- Constructors --

    fn open_or_create(
        path: String,
        dimension: u32,
        metric: Option<String>,
    ) -> Result<Self, VectLiteError> {
        let metric = match metric {
            Some(m) => DistanceMetric::from_name(&m)?,
            None => DistanceMetric::Cosine,
        };
        let db = CoreDatabase::open_or_create_with_metric(&path, dimension as usize, metric)?;
        Ok(Database {
            inner: RwLock::new(db),
        })
    }

    fn open_existing(path: String, lock_timeout: Option<f64>) -> Result<Self, VectLiteError> {
        let db = match lock_timeout {
            Some(t) => CoreDatabase::open_with_timeout(&path, t)?,
            None => CoreDatabase::open(&path)?,
        };
        Ok(Database {
            inner: RwLock::new(db),
        })
    }

    fn open_read_only(path: String, lock_timeout: Option<f64>) -> Result<Self, VectLiteError> {
        let db = match lock_timeout {
            Some(t) => CoreDatabase::open_read_only_with_timeout(&path, Some(t))?,
            None => CoreDatabase::open_read_only(&path)?,
        };
        Ok(Database {
            inner: RwLock::new(db),
        })
    }

    // -- Properties --

    fn path(&self) -> String {
        self.read().path().display().to_string()
    }

    fn dimension(&self) -> u32 {
        self.read().dimension() as u32
    }

    fn metric(&self) -> String {
        self.read().metric().name().to_owned()
    }

    fn is_read_only(&self) -> bool {
        self.read().is_read_only()
    }

    fn is_closed(&self) -> bool {
        self.read().is_closed()
    }

    // -- Write --

    fn upsert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        ttl: Option<f64>,
    ) -> Result<(), VectLiteError> {
        let metadata = parse_metadata_opt(&metadata_json)?;
        let ns = namespace.as_deref().unwrap_or("");
        let expires_at = ttl_to_expires_at(ttl);
        let mut db = self.write();
        db.upsert_in_namespace(ns, &id, vector, metadata)?;
        if let Some(ea) = expires_at {
            db.set_ttl_in_namespace(ns, &id, ea)?;
        }
        Ok(())
    }

    fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        ttl: Option<f64>,
    ) -> Result<(), VectLiteError> {
        let metadata = parse_metadata_opt(&metadata_json)?;
        let ns = namespace.as_deref().unwrap_or("");
        let expires_at = ttl_to_expires_at(ttl);
        let mut db = self.write();
        db.insert_in_namespace(ns, &id, vector, metadata)?;
        if let Some(ea) = expires_at {
            db.set_ttl_in_namespace(ns, &id, ea)?;
        }
        Ok(())
    }

    fn delete(&self, id: String, namespace: Option<String>) -> Result<bool, VectLiteError> {
        let ns = namespace.as_deref().unwrap_or("");
        Ok(self.write().delete_in_namespace(ns, &id)?)
    }

    fn delete_many(
        &self,
        ids: Vec<String>,
        namespace: Option<String>,
    ) -> Result<u32, VectLiteError> {
        let ns = namespace.as_deref().unwrap_or("");
        Ok(self
            .write()
            .delete_many_in_namespace(ns, ids.iter().map(String::as_str))? as u32)
    }

    fn delete_by_filter(
        &self,
        filter_json: String,
        namespace: Option<String>,
    ) -> Result<u32, VectLiteError> {
        let filter = parse_filter(&filter_json)?;
        let ns = namespace.as_deref();
        Ok(self.write().delete_by_filter(ns, &filter)? as u32)
    }

    fn update_metadata(
        &self,
        id: String,
        metadata_json: String,
        namespace: Option<String>,
    ) -> Result<bool, VectLiteError> {
        let metadata = parse_metadata(&metadata_json)?;
        let ns = namespace.as_deref().unwrap_or("");
        Ok(self
            .write()
            .update_metadata_in_namespace(ns, &id, metadata)?)
    }

    fn set_ttl(
        &self,
        id: String,
        ttl_secs: f64,
        namespace: Option<String>,
    ) -> Result<bool, VectLiteError> {
        let ns = namespace.as_deref().unwrap_or("");
        Ok(self.write().set_ttl_in_namespace(ns, &id, ttl_secs)?)
    }

    fn clear_ttl(&self, id: String, namespace: Option<String>) -> Result<bool, VectLiteError> {
        let ns = namespace.as_deref().unwrap_or("");
        Ok(self.write().clear_ttl_in_namespace(ns, &id)?)
    }

    // -- Read --

    fn get(&self, id: String, namespace: Option<String>) -> Option<RecordResult> {
        let ns = namespace.as_deref().unwrap_or("");
        let db = self.read();
        db.get_in_namespace(ns, &id).map(record_to_result)
    }

    fn count(&self, namespace: Option<String>, filter_json: Option<String>) -> u32 {
        let filter = filter_json.as_ref().and_then(|j| parse_filter(j).ok());
        let ns = namespace.as_deref();
        let db = self.read();
        db.count_filtered(ns, filter.as_ref()) as u32
    }

    fn list(
        &self,
        namespace: Option<String>,
        filter_json: Option<String>,
        limit: u32,
        offset: u32,
    ) -> Vec<RecordResult> {
        let filter = filter_json.as_ref().and_then(|j| parse_filter(j).ok());
        let ns = namespace.as_deref();
        let db = self.read();
        let records = db.list(
            ns,
            filter.as_ref(),
            if limit == 0 { 0 } else { limit as usize },
            offset as usize,
        );
        records.into_iter().map(record_to_result).collect()
    }

    fn list_cursor(
        &self,
        namespace: Option<String>,
        filter_json: Option<String>,
        limit: u32,
        cursor: Option<String>,
    ) -> Result<CursorPage, VectLiteError> {
        let filter = filter_json.as_ref().map(|j| parse_filter(j)).transpose()?;
        let ns = namespace.as_deref();
        let db = self.read();
        let (records, next_cursor) = db.list_cursor(
            ns,
            filter.as_ref(),
            if limit == 0 { 100 } else { limit as usize },
            cursor.as_deref(),
        );
        Ok(CursorPage {
            records: records.into_iter().map(|r| record_to_result(r)).collect(),
            cursor: next_cursor,
        })
    }

    fn namespaces(&self) -> Vec<String> {
        self.read().namespaces()
    }

    // -- Search --

    fn search(
        &self,
        query: Vec<f32>,
        k: u32,
        filter_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        fusion: Option<String>,
        dense_weight: Option<f32>,
        sparse_weight: Option<f32>,
        mmr_lambda: Option<f32>,
    ) -> Result<Vec<SearchResult>, VectLiteError> {
        let filter = filter_json.as_ref().map(|j| parse_filter(j)).transpose()?;
        let sparse: Option<SparseVector> = match &sparse_json {
            Some(j) => Some(parse_sparse(j)?),
            None => None,
        };
        let ns = namespace.as_deref().unwrap_or("");

        let db = self.read();

        let fusion_strategy = match fusion.as_deref() {
            Some("rrf") => FusionStrategy::Rrf { rank_constant: 60 },
            _ => FusionStrategy::Linear,
        };

        let options = HybridSearchOptions {
            top_k: k as usize,
            filter,
            dense_weight: dense_weight.unwrap_or(1.0),
            sparse_weight: sparse_weight.unwrap_or(1.0),
            fetch_k: 0,
            mmr_lambda,
            vector_name: None,
            fusion: fusion_strategy,
            truncate_dim: None,
            multi_vector_queries: BTreeMap::new(),
        };

        let results = db.hybrid_search_in_namespace(ns, Some(&query), sparse.as_ref(), options)?;

        Ok(results.into_iter().map(core_result_to_ffi).collect())
    }

    fn search_with_stats(
        &self,
        query: Vec<f32>,
        k: u32,
        filter_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        fusion: Option<String>,
        dense_weight: Option<f32>,
        sparse_weight: Option<f32>,
        mmr_lambda: Option<f32>,
    ) -> Result<SearchStatsResult, VectLiteError> {
        let filter = filter_json.as_ref().map(|j| parse_filter(j)).transpose()?;
        let sparse: Option<SparseVector> = match &sparse_json {
            Some(j) => Some(parse_sparse(j)?),
            None => None,
        };
        let ns = namespace.as_deref().unwrap_or("");

        let db = self.read();

        let fusion_strategy = match fusion.as_deref() {
            Some("rrf") => FusionStrategy::Rrf { rank_constant: 60 },
            _ => FusionStrategy::Linear,
        };

        let options = HybridSearchOptions {
            top_k: k as usize,
            filter,
            dense_weight: dense_weight.unwrap_or(1.0),
            sparse_weight: sparse_weight.unwrap_or(1.0),
            fetch_k: 0,
            mmr_lambda,
            vector_name: None,
            fusion: fusion_strategy,
            truncate_dim: None,
            multi_vector_queries: BTreeMap::new(),
        };

        let outcome =
            db.hybrid_search_in_namespace_with_stats(ns, Some(&query), sparse.as_ref(), options)?;

        let stats_json = serde_json::to_string(&json!({
            "used_ann": outcome.stats.used_ann,
            "ann_candidate_count": outcome.stats.ann_candidate_count,
            "exact_fallback": outcome.stats.exact_fallback,
            "considered_count": outcome.stats.considered_count,
            "fetch_k": outcome.stats.fetch_k,
            "mmr_applied": outcome.stats.mmr_applied,
            "sparse_candidate_count": outcome.stats.sparse_candidate_count,
            "fusion": outcome.stats.fusion,
            "timings": {
                "dense_us": outcome.stats.timings.dense_us,
                "sparse_us": outcome.stats.timings.sparse_us,
                "fusion_us": outcome.stats.timings.fusion_us,
                "total_us": outcome.stats.timings.total_us,
            },
        }))
        .map_err(|e| json_err(e.to_string()))?;

        Ok(SearchStatsResult {
            results: outcome
                .results
                .into_iter()
                .map(core_result_to_ffi)
                .collect(),
            stats_json,
        })
    }

    // -- Index --

    fn create_index(&self, field: String, index_type: String) -> Result<(), VectLiteError> {
        let idx_type = PayloadIndexType::from_name(&index_type)?;
        self.write().create_index(&field, idx_type)?;
        Ok(())
    }

    fn drop_index(&self, field: String) -> Result<bool, VectLiteError> {
        Ok(self.write().drop_index(&field)?)
    }

    fn list_indexes_json(&self) -> String {
        let db = self.read();
        let indexes: Vec<Value> = db
            .list_indexes()
            .into_iter()
            .map(|(field, idx_type)| json!({ "field": field, "type": idx_type.name() }))
            .collect();
        serde_json::to_string(&indexes).unwrap_or_else(|_| "[]".to_owned())
    }

    // -- Quantization --

    fn enable_quantization(
        &self,
        method: String,
        options_json: Option<String>,
    ) -> Result<(), VectLiteError> {
        let config = parse_quantization_config(&method, options_json.as_deref())?;
        self.write().enable_quantization(config)?;
        Ok(())
    }

    fn disable_quantization(&self) -> Result<(), VectLiteError> {
        self.write().disable_quantization()?;
        Ok(())
    }

    fn is_quantized(&self) -> bool {
        self.read().is_quantized()
    }

    fn quantization_method(&self) -> Option<String> {
        self.read().quantization_config().map(|c| match c {
            QuantizationConfig::Scalar(_) => "scalar".to_owned(),
            QuantizationConfig::Binary(_) => "binary".to_owned(),
            QuantizationConfig::Product(_) => "product".to_owned(),
        })
    }

    // -- Bulk --

    fn bulk_ingest(&self, records_json: String, batch_size: u32) -> Result<u32, VectLiteError> {
        let value: Value =
            serde_json::from_str(&records_json).map_err(|e| json_err(e.to_string()))?;
        let arr = value
            .as_array()
            .ok_or_else(|| json_err("bulk_ingest expects a JSON array"))?;
        let mut records = Vec::with_capacity(arr.len());
        for item in arr {
            let obj = item
                .as_object()
                .ok_or_else(|| json_err("each record must be a JSON object"))?;
            records.push(json_to_record(obj)?);
        }
        let count = self
            .write()
            .bulk_ingest(records.into_iter(), batch_size as usize)?;
        Ok(count as u32)
    }

    // -- Maintenance --

    fn compact(&self) -> Result<(), VectLiteError> {
        self.write().compact()?;
        Ok(())
    }

    fn flush(&self) -> Result<(), VectLiteError> {
        self.write().flush()?;
        Ok(())
    }

    fn snapshot(&self, dest: String) -> Result<(), VectLiteError> {
        self.read_core().snapshot(&dest)?;
        Ok(())
    }

    fn backup(&self, dest: String) -> Result<(), VectLiteError> {
        self.read_core().backup(&dest)?;
        Ok(())
    }

    fn close(&self) -> Result<(), VectLiteError> {
        self.write().close()?;
        Ok(())
    }

    // -- Transaction --

    fn transaction_execute(&self, operations_json: String) -> Result<(), VectLiteError> {
        let value: Value =
            serde_json::from_str(&operations_json).map_err(|e| json_err(e.to_string()))?;
        let arr = value
            .as_array()
            .ok_or_else(|| json_err("transaction_execute expects a JSON array"))?;
        let mut ops = Vec::with_capacity(arr.len());
        for item in arr {
            let obj = item
                .as_object()
                .ok_or_else(|| json_err("each operation must be a JSON object"))?;
            let op_type = obj
                .get("op")
                .and_then(Value::as_str)
                .ok_or_else(|| json_err("operation must have an 'op' field"))?;
            match op_type {
                "upsert" => {
                    ops.push(WriteOperation::Upsert(json_to_record(obj)?));
                }
                "insert" => {
                    ops.push(WriteOperation::Insert(json_to_record(obj)?));
                }
                "delete" => {
                    let id = obj
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| json_err("delete operation must have an 'id' field"))?;
                    let namespace = obj.get("namespace").and_then(Value::as_str).unwrap_or("");
                    ops.push(WriteOperation::Delete {
                        namespace: namespace.to_owned(),
                        id: id.to_owned(),
                    });
                }
                other => {
                    return Err(json_err(format!("unknown operation type: {other}")));
                }
            }
        }
        self.write().apply_operations(ops)?;
        Ok(())
    }
}

// Lock helpers
impl Database {
    fn read(&self) -> std::sync::RwLockReadGuard<'_, CoreDatabase> {
        self.inner.read().expect("lock poisoned")
    }

    fn read_core(&self) -> std::sync::RwLockReadGuard<'_, CoreDatabase> {
        self.inner.read().expect("lock poisoned")
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, CoreDatabase> {
        self.inner.write().expect("lock poisoned")
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

pub struct Store {
    inner: CoreStore,
}

impl Store {
    fn new(root: String) -> Result<Self, VectLiteError> {
        let store = CoreStore::open(&root)?;
        Ok(Store { inner: store })
    }

    fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    fn create_collection(
        &self,
        name: String,
        dimension: u32,
    ) -> Result<Arc<Database>, VectLiteError> {
        let db = self.inner.create_collection(&name, dimension as usize)?;
        Ok(Arc::new(Database {
            inner: RwLock::new(db),
        }))
    }

    fn open_collection(&self, name: String) -> Result<Arc<Database>, VectLiteError> {
        let db = self.inner.open_collection(&name)?;
        Ok(Arc::new(Database {
            inner: RwLock::new(db),
        }))
    }

    fn open_or_create_collection(
        &self,
        name: String,
        dimension: u32,
    ) -> Result<Arc<Database>, VectLiteError> {
        let db = self
            .inner
            .open_or_create_collection(&name, dimension as usize)?;
        Ok(Arc::new(Database {
            inner: RwLock::new(db),
        }))
    }

    fn drop_collection(&self, name: String) -> Result<bool, VectLiteError> {
        Ok(self.inner.drop_collection(&name)?)
    }

    fn collections(&self) -> Result<Vec<String>, VectLiteError> {
        Ok(self.inner.collections()?)
    }
}

// ---------------------------------------------------------------------------
// Namespace functions
// ---------------------------------------------------------------------------

fn sparse_terms(text: String) -> String {
    // Simple TF-based sparse terms: tokenize, lowercase, compute term frequency.
    let text_lower = text.to_lowercase();
    let tokens: Vec<&str> = text_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let total = tokens.len();
    if total == 0 {
        return "{}".to_owned();
    }
    let mut counts: BTreeMap<String, f32> = BTreeMap::new();
    for token in &tokens {
        *counts.entry((*token).to_owned()).or_insert(0.0) += 1.0 / total as f32;
    }
    let mut map = Map::new();
    for (term, weight) in &counts {
        if let Some(n) = Number::from_f64(*weight as f64) {
            map.insert(term.clone(), Value::Number(n));
        }
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".to_owned())
}

fn restore(source: String, dest: String) -> Result<Arc<Database>, VectLiteError> {
    let db = CoreDatabase::restore(&source, &dest)?;
    Ok(Arc::new(Database {
        inner: RwLock::new(db),
    }))
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

fn parse_metadata_opt(json: &Option<String>) -> Result<Metadata, VectLiteError> {
    match json {
        None => Ok(Metadata::new()),
        Some(s) => parse_metadata(s),
    }
}

fn parse_metadata(json: &str) -> Result<Metadata, VectLiteError> {
    let value: Value = serde_json::from_str(json).map_err(|e| json_err(e.to_string()))?;
    json_to_metadata(&value)
}

fn parse_filter(json: &str) -> Result<MetadataFilter, VectLiteError> {
    let value: Value = serde_json::from_str(json).map_err(|e| json_err(e.to_string()))?;
    json_to_filter(&value)
}

fn parse_sparse(json: &str) -> Result<SparseVector, VectLiteError> {
    let value: Value = serde_json::from_str(json).map_err(|e| json_err(e.to_string()))?;
    let obj = value
        .as_object()
        .ok_or_else(|| json_err("sparse vector must be a JSON object"))?;
    let mut sparse = SparseVector::new();
    for (term, weight) in obj {
        let w = weight
            .as_f64()
            .ok_or_else(|| json_err("sparse weight must be a number"))?;
        sparse.insert(term.clone(), w as f32);
    }
    Ok(sparse)
}

fn json_to_metadata(value: &Value) -> Result<Metadata, VectLiteError> {
    let object = value
        .as_object()
        .ok_or_else(|| json_err("metadata must be a JSON object"))?;
    let mut metadata = Metadata::new();
    for (key, val) in object {
        metadata.insert(key.clone(), json_to_metadata_value(val)?);
    }
    Ok(metadata)
}

fn json_to_metadata_value(value: &Value) -> Result<MetadataValue, VectLiteError> {
    match value {
        Value::Null => Ok(MetadataValue::Null),
        Value::Bool(b) => Ok(MetadataValue::Boolean(*b)),
        Value::String(s) => Ok(MetadataValue::String(s.clone())),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(MetadataValue::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(MetadataValue::Float(f))
            } else {
                Err(json_err("unsupported number value"))
            }
        }
        Value::Array(arr) => {
            let items: Result<Vec<_>, _> = arr.iter().map(json_to_metadata_value).collect();
            Ok(MetadataValue::List(items?))
        }
        Value::Object(obj) => {
            let mut map = BTreeMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_to_metadata_value(v)?);
            }
            Ok(MetadataValue::Map(map))
        }
    }
}

fn json_to_filter(value: &Value) -> Result<MetadataFilter, VectLiteError> {
    let object = value
        .as_object()
        .ok_or_else(|| json_err("filter must be a JSON object"))?;
    let mut filters = Vec::new();
    for (key, val) in object {
        match key.as_str() {
            "$and" => {
                let items = val
                    .as_array()
                    .ok_or_else(|| json_err("$and must be an array"))?;
                let group: Result<Vec<_>, _> = items.iter().map(json_to_filter).collect();
                filters.push(MetadataFilter::and(group?));
            }
            "$or" => {
                let items = val
                    .as_array()
                    .ok_or_else(|| json_err("$or must be an array"))?;
                let group: Result<Vec<_>, _> = items.iter().map(json_to_filter).collect();
                filters.push(MetadataFilter::or(group?));
            }
            "$not" => {
                filters.push(MetadataFilter::not(json_to_filter(val)?));
            }
            field => {
                filters.push(parse_field_filter(field, val)?);
            }
        }
    }
    collapse_filters(filters)
}

fn parse_field_filter(key: &str, value: &Value) -> Result<MetadataFilter, VectLiteError> {
    if let Some(operators) = value.as_object() {
        let mut filters = Vec::new();
        for (op, operand) in operators {
            match op.as_str() {
                "$eq" => filters.push(MetadataFilter::eq(key, json_to_metadata_value(operand)?)),
                "$ne" => filters.push(MetadataFilter::ne(key, json_to_metadata_value(operand)?)),
                "$in" => {
                    let vals = extract_metadata_values(operand)?;
                    filters.push(MetadataFilter::r#in(key, vals));
                }
                "$nin" => {
                    let vals = extract_metadata_values(operand)?;
                    filters.push(MetadataFilter::nin(key, vals));
                }
                "$contains" => {
                    let s = operand
                        .as_str()
                        .ok_or_else(|| json_err("$contains expects a string"))?;
                    filters.push(MetadataFilter::contains(key, s));
                }
                "$gt" => filters.push(MetadataFilter::gt(key, extract_f64(operand)?)),
                "$gte" => filters.push(MetadataFilter::gte(key, extract_f64(operand)?)),
                "$lt" => filters.push(MetadataFilter::lt(key, extract_f64(operand)?)),
                "$lte" => filters.push(MetadataFilter::lte(key, extract_f64(operand)?)),
                "$exists" => {
                    let exists = operand
                        .as_bool()
                        .ok_or_else(|| json_err("$exists expects a boolean"))?;
                    if exists {
                        filters.push(MetadataFilter::exists(key));
                    } else {
                        filters.push(MetadataFilter::not(MetadataFilter::exists(key)));
                    }
                }
                "$elemMatch" => {
                    let sub = if operand
                        .as_object()
                        .map_or(false, |o| o.keys().all(|k| k.starts_with('$')))
                    {
                        parse_field_filter("_", operand)?
                    } else {
                        json_to_filter(operand)?
                    };
                    filters.push(MetadataFilter::elem_match(key, sub));
                }
                "$size" => {
                    let n = operand
                        .as_u64()
                        .ok_or_else(|| json_err("$size expects a positive integer"))?;
                    filters.push(MetadataFilter::size(key, n as usize));
                }
                "$not" => {
                    filters.push(MetadataFilter::not(parse_field_filter(key, operand)?));
                }
                other => {
                    return Err(json_err(format!("unsupported filter operator: {other}")));
                }
            }
        }
        collapse_filters(filters)
    } else {
        Ok(MetadataFilter::eq(key, json_to_metadata_value(value)?))
    }
}

fn collapse_filters(filters: Vec<MetadataFilter>) -> Result<MetadataFilter, VectLiteError> {
    match filters.len() {
        0 => Err(json_err("filter cannot be empty")),
        1 => Ok(filters.into_iter().next().unwrap()),
        _ => Ok(MetadataFilter::and(filters)),
    }
}

fn extract_metadata_values(value: &Value) -> Result<Vec<MetadataValue>, VectLiteError> {
    let items = value
        .as_array()
        .ok_or_else(|| json_err("$in/$nin expects an array"))?;
    items.iter().map(json_to_metadata_value).collect()
}

fn extract_f64(value: &Value) -> Result<f64, VectLiteError> {
    value
        .as_f64()
        .ok_or_else(|| json_err("numeric operator expects a number"))
}

fn ttl_to_expires_at(ttl: Option<f64>) -> Option<f64> {
    ttl.map(|secs| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            + secs
    })
}

fn json_to_record(obj: &Map<String, Value>) -> Result<Record, VectLiteError> {
    let namespace = obj
        .get("namespace")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let id = obj
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| json_err("record must have an 'id' field"))?
        .to_owned();
    let vector = obj
        .get("vector")
        .ok_or_else(|| json_err("record must have a 'vector' field"))
        .and_then(|v| {
            let arr = v
                .as_array()
                .ok_or_else(|| json_err("vector must be an array"))?;
            arr.iter()
                .map(|x| {
                    x.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| json_err("vector elements must be numbers"))
                })
                .collect::<Result<Vec<f32>, _>>()
        })?;
    let metadata = obj
        .get("metadata")
        .map(json_to_metadata)
        .transpose()?
        .unwrap_or_default();
    let sparse = obj
        .get("sparse")
        .map(|v| -> Result<SparseVector, VectLiteError> {
            let o = v
                .as_object()
                .ok_or_else(|| json_err("sparse must be an object"))?;
            let mut s = SparseVector::new();
            for (term, w) in o {
                s.insert(
                    term.clone(),
                    w.as_f64()
                        .ok_or_else(|| json_err("sparse weight must be a number"))?
                        as f32,
                );
            }
            Ok(s)
        })
        .transpose()?
        .unwrap_or_default();
    let ttl = obj.get("ttl").and_then(|v| v.as_f64());
    let expires_at = ttl_to_expires_at(ttl);

    Ok(Record {
        namespace,
        id,
        vector,
        vectors: BTreeMap::new(),
        sparse,
        metadata,
        multi_vectors: BTreeMap::new(),
        expires_at,
    })
}

fn record_to_result(record: &Record) -> RecordResult {
    RecordResult {
        namespace: record.namespace.clone(),
        id: record.id.clone(),
        vector: record.vector.clone(),
        metadata_json: metadata_to_json_string(&record.metadata),
        expires_at: record.expires_at,
    }
}

fn core_result_to_ffi(result: vectlite::SearchResult) -> SearchResult {
    SearchResult {
        namespace: result.namespace,
        id: result.id,
        score: result.score,
        metadata_json: metadata_to_json_string(&result.metadata),
    }
}

fn metadata_to_json_string(metadata: &Metadata) -> String {
    let value = metadata_to_json_value(metadata);
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_owned())
}

fn metadata_to_json_value(metadata: &Metadata) -> Value {
    let mut map = Map::new();
    for (key, val) in metadata {
        map.insert(key.clone(), metadata_value_to_json(val));
    }
    Value::Object(map)
}

fn metadata_value_to_json(value: &MetadataValue) -> Value {
    match value {
        MetadataValue::String(s) => Value::String(s.clone()),
        MetadataValue::Integer(i) => Value::Number(Number::from(*i)),
        MetadataValue::Float(f) => Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        MetadataValue::Boolean(b) => Value::Bool(*b),
        MetadataValue::Null => Value::Null,
        MetadataValue::List(items) => {
            Value::Array(items.iter().map(metadata_value_to_json).collect())
        }
        MetadataValue::Map(entries) => {
            let mut map = Map::new();
            for (k, v) in entries {
                map.insert(k.clone(), metadata_value_to_json(v));
            }
            Value::Object(map)
        }
    }
}

fn parse_quantization_config(
    method: &str,
    options_json: Option<&str>,
) -> Result<QuantizationConfig, VectLiteError> {
    let opts: Value = match options_json {
        Some(j) => serde_json::from_str(j).map_err(|e| json_err(e.to_string()))?,
        None => Value::Object(Map::new()),
    };
    match method {
        "scalar" => Ok(QuantizationConfig::Scalar(ScalarQuantizationConfig {
            rescore_multiplier: opts
                .get("rescoreMultiplier")
                .or_else(|| opts.get("rescore_multiplier"))
                .and_then(Value::as_u64)
                .unwrap_or(4) as usize,
        })),
        "binary" => Ok(QuantizationConfig::Binary(BinaryQuantizationConfig {
            rescore_multiplier: opts
                .get("rescoreMultiplier")
                .or_else(|| opts.get("rescore_multiplier"))
                .and_then(Value::as_u64)
                .unwrap_or(10) as usize,
        })),
        "product" | "pq" => Ok(QuantizationConfig::Product(ProductQuantizationConfig {
            num_sub_vectors: opts
                .get("numSubVectors")
                .or_else(|| opts.get("num_sub_vectors"))
                .and_then(Value::as_u64)
                .unwrap_or(8) as usize,
            num_centroids: opts
                .get("numCentroids")
                .or_else(|| opts.get("num_centroids"))
                .and_then(Value::as_u64)
                .unwrap_or(256) as usize,
            training_iterations: opts
                .get("trainingIterations")
                .or_else(|| opts.get("training_iterations"))
                .and_then(Value::as_u64)
                .unwrap_or(20) as usize,
            rescore_multiplier: opts
                .get("rescoreMultiplier")
                .or_else(|| opts.get("rescore_multiplier"))
                .and_then(Value::as_u64)
                .unwrap_or(4) as usize,
        })),
        other => Err(json_err(format!(
            "unknown quantization method: {other}; valid: scalar, binary, product (alias: pq)"
        ))),
    }
}
