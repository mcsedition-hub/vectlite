use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use napi::Error as NapiError;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{Map, Number, Value, json};
use vectlite::{
    Database as CoreDatabase, FusionStrategy, HybridSearchOptions, Metadata, MetadataFilter,
    MetadataValue, NamedVectors, Record, SearchOutcome, SearchResult, SparseVector,
    Store as CoreStore, WriteOperation,
};

#[napi(js_name = "NativeDatabase")]
pub struct NativeDatabase {
    inner: Arc<RwLock<CoreDatabase>>,
}

#[napi(js_name = "NativeTransaction")]
pub struct NativeTransaction {
    inner: Arc<RwLock<CoreDatabase>>,
    staged: Mutex<TransactionState>,
}

#[derive(Default)]
struct TransactionState {
    ops: Vec<WriteOperation>,
    closed: bool,
}

#[napi(js_name = "NativeStore")]
pub struct NativeStore {
    inner: CoreStore,
}

struct SearchRequest {
    namespace: String,
    all_namespaces: bool,
    sparse: SparseVector,
    options: HybridSearchOptions,
    explain: bool,
    fusion_name: String,
}

#[napi]
impl NativeStore {
    #[napi(getter)]
    pub fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    #[napi(js_name = "createCollection")]
    pub fn create_collection(&self, name: String, dimension: u32) -> Result<NativeDatabase> {
        let database = self
            .inner
            .create_collection(&name, dimension as usize)
            .map_err(to_napi_error)?;
        Ok(NativeDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    #[napi(js_name = "openCollection")]
    pub fn open_collection(&self, name: String) -> Result<NativeDatabase> {
        let database = self.inner.open_collection(&name).map_err(to_napi_error)?;
        Ok(NativeDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    #[napi(js_name = "openOrCreateCollection")]
    pub fn open_or_create_collection(
        &self,
        name: String,
        dimension: u32,
    ) -> Result<NativeDatabase> {
        let database = self
            .inner
            .open_or_create_collection(&name, dimension as usize)
            .map_err(to_napi_error)?;
        Ok(NativeDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    #[napi(js_name = "openCollectionReadOnly")]
    pub fn open_collection_read_only(&self, name: String) -> Result<NativeDatabase> {
        let database = self
            .inner
            .open_collection_read_only(&name)
            .map_err(to_napi_error)?;
        Ok(NativeDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    #[napi(js_name = "dropCollection")]
    pub fn drop_collection(&self, name: String) -> Result<bool> {
        self.inner.drop_collection(&name).map_err(to_napi_error)
    }

    #[napi]
    pub fn collections(&self) -> Result<Vec<String>> {
        self.inner.collections().map_err(to_napi_error)
    }
}

#[napi]
impl NativeDatabase {
    #[napi(getter)]
    pub fn path(&self) -> Result<String> {
        let database = self.read()?;
        Ok(database.path().display().to_string())
    }

    #[napi(getter, js_name = "walPath")]
    pub fn wal_path(&self) -> Result<String> {
        let database = self.read()?;
        Ok(database.wal_path().display().to_string())
    }

    #[napi(getter)]
    pub fn dimension(&self) -> Result<u32> {
        let database = self.read()?;
        Ok(database.dimension() as u32)
    }

    #[napi(getter, js_name = "readOnly")]
    pub fn read_only(&self) -> Result<bool> {
        let database = self.read()?;
        Ok(database.is_read_only())
    }

    #[napi]
    pub fn count(&self, namespace: Option<String>, filter_json: Option<String>) -> Result<u32> {
        let filter = filter_json
            .as_ref()
            .map(|json_str| {
                let value: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| err(format!("invalid filter JSON: {e}")))?;
                json_to_filter(&value)
            })
            .transpose()?;
        if namespace.is_none() && filter.is_none() {
            let database = self.read()?;
            return Ok(database.len() as u32);
        }
        let database = self.read()?;
        Ok(database.count_filtered(namespace.as_deref(), filter.as_ref()) as u32)
    }

    #[napi]
    pub fn namespaces(&self) -> Result<Vec<String>> {
        let database = self.read()?;
        Ok(database.namespaces())
    }

    #[napi]
    pub fn close(&self) -> Result<()> {
        let mut database = self.write()?;
        database.close().map_err(to_napi_error)
    }

    #[napi]
    pub fn list(
        &self,
        namespace: Option<String>,
        filter_json: Option<String>,
        limit: Option<u32>,
        offset: Option<u32>,
    ) -> Result<String> {
        let filter = filter_json
            .as_ref()
            .map(|json_str| {
                let value: serde_json::Value = serde_json::from_str(json_str)
                    .map_err(|e| err(format!("invalid filter JSON: {e}")))?;
                json_to_filter(&value)
            })
            .transpose()?;
        let records = {
            let database = self.read()?;
            database
                .list(
                    namespace.as_deref(),
                    filter.as_ref(),
                    limit.unwrap_or(0) as usize,
                    offset.unwrap_or(0) as usize,
                )
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let json_records: Vec<Value> = records.iter().map(record_to_json).collect();
        stringify_value(Value::Array(json_records))
    }

    #[napi(js_name = "deleteByFilter")]
    pub fn delete_by_filter(&self, filter_json: String, namespace: Option<String>) -> Result<u32> {
        let value: serde_json::Value = serde_json::from_str(&filter_json)
            .map_err(|e| err(format!("invalid filter JSON: {e}")))?;
        let filter = json_to_filter(&value)?;
        let mut database = self.write_open()?;
        database
            .delete_by_filter(namespace.as_deref(), &filter)
            .map(|count| count as u32)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn transaction(&self) -> Result<NativeTransaction> {
        drop(self.read()?);
        Ok(NativeTransaction {
            inner: Arc::clone(&self.inner),
            staged: Mutex::new(TransactionState::default()),
        })
    }

    #[napi]
    pub fn insert(
        &self,
        id: String,
        vector: Vec<f64>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        vectors_json: Option<String>,
    ) -> Result<()> {
        let metadata = parse_metadata_json(metadata_json)?;
        let sparse = parse_sparse_json(sparse_json)?;
        let vectors = parse_named_vectors_json(vectors_json)?;
        let vector = js_vector_to_core(vector, "vector")?;
        let mut database = self.write_open()?;
        database
            .insert_with_vectors_in_namespace(
                namespace.unwrap_or_default(),
                &id,
                vector,
                vectors,
                sparse,
                metadata,
            )
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn upsert(
        &self,
        id: String,
        vector: Vec<f64>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        vectors_json: Option<String>,
    ) -> Result<()> {
        let metadata = parse_metadata_json(metadata_json)?;
        let sparse = parse_sparse_json(sparse_json)?;
        let vectors = parse_named_vectors_json(vectors_json)?;
        let vector = js_vector_to_core(vector, "vector")?;
        let mut database = self.write_open()?;
        database
            .upsert_with_vectors_in_namespace(
                namespace.unwrap_or_default(),
                &id,
                vector,
                vectors,
                sparse,
                metadata,
            )
            .map_err(to_napi_error)
    }

    #[napi(js_name = "insertMany")]
    pub fn insert_many(&self, records_json: String, namespace: Option<String>) -> Result<u32> {
        let records = parse_record_batch_json(&records_json, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database
            .insert_many(records)
            .map(|count| count as u32)
            .map_err(to_napi_error)
    }

    #[napi(js_name = "upsertMany")]
    pub fn upsert_many(&self, records_json: String, namespace: Option<String>) -> Result<u32> {
        let records = parse_record_batch_json(&records_json, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database
            .upsert_many(records)
            .map(|count| count as u32)
            .map_err(to_napi_error)
    }

    #[napi(js_name = "bulkIngest")]
    pub fn bulk_ingest(
        &self,
        records_json: String,
        namespace: Option<String>,
        batch_size: u32,
    ) -> Result<u32> {
        let records = parse_record_batch_json(&records_json, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database
            .bulk_ingest(records, batch_size as usize)
            .map(|count| count as u32)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn get(&self, id: String, namespace: Option<String>) -> Result<Option<String>> {
        let record = {
            let database = self.read()?;
            database
                .get_in_namespace(&namespace.unwrap_or_default(), &id)
                .cloned()
        };

        record
            .map(|record| stringify_value(record_to_json(&record)))
            .transpose()
    }

    #[napi]
    pub fn delete(&self, id: String, namespace: Option<String>) -> Result<bool> {
        let mut database = self.write_open()?;
        database
            .delete_in_namespace(&namespace.unwrap_or_default(), &id)
            .map_err(to_napi_error)
    }

    #[napi(js_name = "deleteMany")]
    pub fn delete_many(&self, ids: Vec<String>, namespace: Option<String>) -> Result<u32> {
        let mut database = self.write_open()?;
        database
            .delete_many_in_namespace(&namespace.unwrap_or_default(), ids)
            .map(|count| count as u32)
            .map_err(to_napi_error)
    }

    #[napi]
    pub fn flush(&self) -> Result<()> {
        let mut database = self.write_open()?;
        database.flush().map_err(to_napi_error)
    }

    #[napi]
    pub fn compact(&self) -> Result<()> {
        let mut database = self.write_open()?;
        database.compact().map_err(to_napi_error)
    }

    #[napi]
    pub fn snapshot(&self, dest: String) -> Result<()> {
        let database = self.read()?;
        database.snapshot(&dest).map_err(to_napi_error)
    }

    #[napi]
    pub fn backup(&self, dest: String) -> Result<()> {
        let database = self.read()?;
        database.backup(&dest).map_err(to_napi_error)
    }

    #[napi]
    pub fn search(&self, query: Option<Vec<f64>>, options_json: Option<String>) -> Result<String> {
        let request = parse_search_request(options_json)?;
        let query = query
            .map(|vector| js_vector_to_core(vector, "query vector"))
            .transpose()?;
        let outcome = self.execute_search(query, &request)?;
        stringify_value(search_results_to_json(
            &outcome.results,
            request.explain,
            &request.fusion_name,
        ))
    }

    #[napi(js_name = "searchWithStats")]
    pub fn search_with_stats(
        &self,
        query: Option<Vec<f64>>,
        options_json: Option<String>,
    ) -> Result<String> {
        let request = parse_search_request(options_json)?;
        let query = query
            .map(|vector| js_vector_to_core(vector, "query vector"))
            .transpose()?;
        let outcome = self.execute_search(query, &request)?;
        stringify_value(search_outcome_to_json(
            &outcome,
            request.explain,
            &request.fusion_name,
        ))
    }
}

#[napi]
impl NativeTransaction {
    #[napi]
    pub fn count(&self) -> Result<u32> {
        let state = self.state()?;
        Ok(state.ops.len() as u32)
    }

    #[napi]
    pub fn insert(
        &self,
        id: String,
        vector: Vec<f64>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        vectors_json: Option<String>,
    ) -> Result<()> {
        let metadata = parse_metadata_json(metadata_json)?;
        let sparse = parse_sparse_json(sparse_json)?;
        let vectors = parse_named_vectors_json(vectors_json)?;
        let vector = js_vector_to_core(vector, "vector")?;
        self.stage(WriteOperation::Insert(Record {
            namespace: namespace.unwrap_or_default(),
            id,
            vector,
            vectors,
            sparse,
            metadata,
        }))
    }

    #[napi]
    pub fn upsert(
        &self,
        id: String,
        vector: Vec<f64>,
        metadata_json: Option<String>,
        namespace: Option<String>,
        sparse_json: Option<String>,
        vectors_json: Option<String>,
    ) -> Result<()> {
        let metadata = parse_metadata_json(metadata_json)?;
        let sparse = parse_sparse_json(sparse_json)?;
        let vectors = parse_named_vectors_json(vectors_json)?;
        let vector = js_vector_to_core(vector, "vector")?;
        self.stage(WriteOperation::Upsert(Record {
            namespace: namespace.unwrap_or_default(),
            id,
            vector,
            vectors,
            sparse,
            metadata,
        }))
    }

    #[napi(js_name = "insertMany")]
    pub fn insert_many(&self, records_json: String, namespace: Option<String>) -> Result<u32> {
        let records = parse_record_batch_json(&records_json, namespace.as_deref())?;
        let count = records.len() as u32;
        for record in records {
            self.stage(WriteOperation::Insert(record))?;
        }
        Ok(count)
    }

    #[napi(js_name = "upsertMany")]
    pub fn upsert_many(&self, records_json: String, namespace: Option<String>) -> Result<u32> {
        let records = parse_record_batch_json(&records_json, namespace.as_deref())?;
        let count = records.len() as u32;
        for record in records {
            self.stage(WriteOperation::Upsert(record))?;
        }
        Ok(count)
    }

    #[napi]
    pub fn delete(&self, id: String, namespace: Option<String>) -> Result<bool> {
        self.stage(WriteOperation::Delete {
            namespace: namespace.unwrap_or_default(),
            id,
        })?;
        Ok(true)
    }

    #[napi(js_name = "deleteMany")]
    pub fn delete_many(&self, ids: Vec<String>, namespace: Option<String>) -> Result<u32> {
        let namespace = namespace.unwrap_or_default();
        let count = ids.len() as u32;
        for id in ids {
            self.stage(WriteOperation::Delete {
                namespace: namespace.clone(),
                id,
            })?;
        }
        Ok(count)
    }

    #[napi]
    pub fn commit(&self) -> Result<()> {
        let ops = {
            let mut state = self.state()?;
            if state.closed {
                return Ok(());
            }
            state.closed = true;
            std::mem::take(&mut state.ops)
        };
        if ops.is_empty() {
            return Ok(());
        }
        let mut database = self.write_db()?;
        database.apply_operations(ops).map_err(to_napi_error)
    }

    #[napi]
    pub fn rollback(&self) -> Result<()> {
        let mut state = self.state()?;
        state.closed = true;
        state.ops.clear();
        Ok(())
    }
}

impl NativeTransaction {
    fn stage(&self, op: WriteOperation) -> Result<()> {
        let mut state = self.state()?;
        if state.closed {
            return Err(err("transaction is already closed"));
        }
        state.ops.push(op);
        Ok(())
    }

    fn state(&self) -> Result<MutexGuard<'_, TransactionState>> {
        self.staged
            .lock()
            .map_err(|_| err("transaction state lock poisoned"))
    }

    fn write_db(&self) -> Result<RwLockWriteGuard<'_, CoreDatabase>> {
        self.inner
            .write()
            .map_err(|_| err("database write lock poisoned"))
    }
}

impl NativeDatabase {
    fn read(&self) -> Result<RwLockReadGuard<'_, CoreDatabase>> {
        let database = self
            .inner
            .read()
            .map_err(|_| err("database read lock poisoned"))?;
        if database.is_closed() {
            return Err(to_napi_error(closed_database_error()));
        }
        Ok(database)
    }

    fn write(&self) -> Result<RwLockWriteGuard<'_, CoreDatabase>> {
        self.inner
            .write()
            .map_err(|_| err("database write lock poisoned"))
    }

    fn write_open(&self) -> Result<RwLockWriteGuard<'_, CoreDatabase>> {
        let database = self.write()?;
        if database.is_closed() {
            return Err(to_napi_error(closed_database_error()));
        }
        Ok(database)
    }

    fn execute_search(
        &self,
        query: Option<Vec<f32>>,
        request: &SearchRequest,
    ) -> Result<SearchOutcome> {
        let sparse_ref = if request.sparse.is_empty() {
            None
        } else {
            Some(&request.sparse)
        };
        let database = self.read()?;
        if request.all_namespaces {
            database
                .hybrid_search_all_namespaces_with_stats(
                    query.as_deref(),
                    sparse_ref,
                    request.options.clone(),
                )
                .map_err(to_napi_error)
        } else {
            database
                .hybrid_search_in_namespace_with_stats(
                    &request.namespace,
                    query.as_deref(),
                    sparse_ref,
                    request.options.clone(),
                )
                .map_err(to_napi_error)
        }
    }
}

#[napi]
pub fn open(
    path: String,
    dimension: Option<u32>,
    read_only: bool,
    lock_timeout: Option<f64>,
) -> Result<NativeDatabase> {
    let database = if read_only {
        if !Path::new(&path).exists() {
            return Err(err("cannot open non-existent database in read-only mode"));
        }
        match lock_timeout {
            Some(timeout) => CoreDatabase::open_read_only_with_timeout(&path, Some(timeout))
                .map_err(to_napi_error)?,
            None => CoreDatabase::open_read_only(&path).map_err(to_napi_error)?,
        }
    } else if Path::new(&path).exists() {
        match (dimension, lock_timeout) {
            (Some(dimension), Some(timeout)) => {
                let db = CoreDatabase::open_with_timeout(&path, timeout).map_err(to_napi_error)?;
                if db.dimension() != dimension as usize {
                    return Err(to_napi_error(vectlite::VectLiteError::DimensionMismatch {
                        expected: db.dimension(),
                        found: dimension as usize,
                    }));
                }
                db
            }
            (Some(dimension), None) => {
                CoreDatabase::open_or_create(&path, dimension as usize).map_err(to_napi_error)?
            }
            (None, Some(timeout)) => {
                CoreDatabase::open_with_timeout(&path, timeout).map_err(to_napi_error)?
            }
            (None, None) => CoreDatabase::open(&path).map_err(to_napi_error)?,
        }
    } else {
        let Some(dimension) = dimension else {
            return Err(err("dimension is required when creating a new database"));
        };
        CoreDatabase::create(&path, dimension as usize).map_err(to_napi_error)?
    };

    Ok(NativeDatabase {
        inner: Arc::new(RwLock::new(database)),
    })
}

#[napi(js_name = "openStore")]
pub fn open_store(root: String) -> Result<NativeStore> {
    let store = CoreStore::open(&root).map_err(to_napi_error)?;
    Ok(NativeStore { inner: store })
}

#[napi]
pub fn restore(source: String, dest: String) -> Result<NativeDatabase> {
    let database = CoreDatabase::restore(&source, &dest).map_err(to_napi_error)?;
    Ok(NativeDatabase {
        inner: Arc::new(RwLock::new(database)),
    })
}

fn parse_search_request(options_json: Option<String>) -> Result<SearchRequest> {
    let value = parse_optional_json(options_json)?;
    let object = expect_optional_object(value.as_ref(), "search options")?;

    let top_k = get_usize(object, "k")?.unwrap_or(10);
    let filter = object
        .and_then(|obj| obj.get("filter"))
        .filter(|value| !value.is_null())
        .map(json_to_filter)
        .transpose()?;
    let namespace = get_string(object, "namespace")?.unwrap_or_default();
    let all_namespaces = get_bool(object, "allNamespaces")?.unwrap_or(false);
    let sparse = object
        .and_then(|obj| obj.get("sparse"))
        .map(json_to_sparse_value)
        .transpose()?
        .unwrap_or_default();
    let dense_weight = get_f32(object, "denseWeight")?.unwrap_or(1.0);
    let sparse_weight = get_f32(object, "sparseWeight")?.unwrap_or(1.0);
    let fetch_k = get_usize(object, "fetchK")?.unwrap_or(0);
    let mmr_lambda = get_optional_f32(object, "mmrLambda")?;
    let vector_name = get_string(object, "vectorName")?;
    let fusion_name = get_string(object, "fusion")?.unwrap_or_else(|| "linear".to_owned());
    let rrf_k = get_usize(object, "rrfK")?.unwrap_or(60);
    let explain = get_bool(object, "explain")?.unwrap_or(false);
    let query_vectors = parse_multi_vector_queries(
        object.and_then(|obj| obj.get("queryVectors")),
        object.and_then(|obj| obj.get("vectorWeights")),
    )?;

    Ok(SearchRequest {
        namespace,
        all_namespaces,
        sparse,
        explain,
        fusion_name: fusion_name.clone(),
        options: HybridSearchOptions {
            top_k,
            filter,
            dense_weight,
            sparse_weight,
            fetch_k,
            mmr_lambda,
            vector_name,
            fusion: parse_fusion(&fusion_name, rrf_k)?,
            multi_vector_queries: query_vectors,
        },
    })
}

fn parse_metadata_json(input: Option<String>) -> Result<Metadata> {
    let value = parse_optional_json(input)?;
    match value {
        None | Some(Value::Null) => Ok(Metadata::new()),
        Some(value) => json_to_metadata(&value),
    }
}

fn parse_sparse_json(input: Option<String>) -> Result<SparseVector> {
    let value = parse_optional_json(input)?;
    match value {
        None | Some(Value::Null) => Ok(SparseVector::new()),
        Some(value) => json_to_sparse_value(&value),
    }
}

fn parse_named_vectors_json(input: Option<String>) -> Result<NamedVectors> {
    let value = parse_optional_json(input)?;
    match value {
        None | Some(Value::Null) => Ok(NamedVectors::new()),
        Some(value) => json_to_named_vectors(&value),
    }
}

fn parse_record_batch_json(input: &str, default_namespace: Option<&str>) -> Result<Vec<Record>> {
    let value: Value = serde_json::from_str(input).map_err(|error| {
        err(format!(
            "records must be valid JSON for batch operations: {error}"
        ))
    })?;
    let items = value
        .as_array()
        .ok_or_else(|| err("records must be a JSON array"))?;
    let mut parsed = Vec::with_capacity(items.len());
    for item in items {
        let object = item
            .as_object()
            .ok_or_else(|| err("each batch record must be a JSON object"))?;
        parsed.push(json_to_record(object, default_namespace)?);
    }
    Ok(parsed)
}

fn parse_optional_json(input: Option<String>) -> Result<Option<Value>> {
    input
        .map(|value| {
            serde_json::from_str(&value)
                .map_err(|error| err(format!("invalid JSON payload: {error}")))
        })
        .transpose()
}

fn json_to_metadata(value: &Value) -> Result<Metadata> {
    let object = value
        .as_object()
        .ok_or_else(|| err("metadata must be a JSON object"))?;
    let mut metadata = Metadata::new();
    for (key, value) in object {
        metadata.insert(key.clone(), json_to_metadata_value(value)?);
    }
    Ok(metadata)
}

fn json_to_metadata_value(value: &Value) -> Result<MetadataValue> {
    match value {
        Value::Null => Ok(MetadataValue::Null),
        Value::Bool(value) => Ok(MetadataValue::Boolean(*value)),
        Value::String(value) => Ok(MetadataValue::String(value.clone())),
        Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                Ok(MetadataValue::Integer(integer))
            } else if let Some(float) = value.as_f64() {
                Ok(MetadataValue::Float(float))
            } else {
                Err(err("metadata number must fit into an integer or float"))
            }
        }
        Value::Array(items) => {
            let mut converted = Vec::with_capacity(items.len());
            for item in items {
                converted.push(json_to_metadata_value(item)?);
            }
            Ok(MetadataValue::List(converted))
        }
        Value::Object(entries) => {
            let mut converted = BTreeMap::new();
            for (key, value) in entries {
                converted.insert(key.clone(), json_to_metadata_value(value)?);
            }
            Ok(MetadataValue::Map(converted))
        }
    }
}

fn json_to_sparse_value(value: &Value) -> Result<SparseVector> {
    let object = value
        .as_object()
        .ok_or_else(|| err("sparse vector must be a JSON object"))?;
    let mut sparse = SparseVector::new();
    for (term, value) in object {
        sparse.insert(term.clone(), value_to_f32(value, "sparse weights")?);
    }
    Ok(sparse)
}

fn json_to_named_vectors(value: &Value) -> Result<NamedVectors> {
    let object = value
        .as_object()
        .ok_or_else(|| err("named vectors must be a JSON object"))?;
    let mut vectors = NamedVectors::new();
    for (name, vector) in object {
        if name.is_empty() {
            return Err(err("named vectors must not use an empty name"));
        }
        vectors.insert(name.clone(), value_to_vector(vector, "named vector")?);
    }
    Ok(vectors)
}

fn json_to_filter(value: &Value) -> Result<MetadataFilter> {
    let object = value
        .as_object()
        .ok_or_else(|| err("filter must be a JSON object"))?;
    let mut filters = Vec::new();
    for (key, value) in object {
        match key.as_str() {
            "$and" => filters.push(MetadataFilter::and(parse_filter_group(value)?)),
            "$or" => filters.push(MetadataFilter::or(parse_filter_group(value)?)),
            "$not" => filters.push(MetadataFilter::not(json_to_filter(value)?)),
            field => filters.push(parse_field_filter(field, value)?),
        }
    }
    collapse_filters(filters, "filter")
}

fn parse_filter_group(value: &Value) -> Result<Vec<MetadataFilter>> {
    let items = value
        .as_array()
        .ok_or_else(|| err("logical filter groups must be JSON arrays"))?;
    let mut filters = Vec::with_capacity(items.len());
    for item in items {
        filters.push(json_to_filter(item)?);
    }
    Ok(filters)
}

fn parse_field_filter(key: &str, value: &Value) -> Result<MetadataFilter> {
    if let Some(operators) = value.as_object() {
        let mut filters = Vec::new();
        for (operator, operand) in operators {
            match operator.as_str() {
                "$eq" => filters.push(MetadataFilter::eq(key, json_to_metadata_value(operand)?)),
                "$ne" => filters.push(MetadataFilter::ne(key, json_to_metadata_value(operand)?)),
                "$in" => filters.push(MetadataFilter::r#in(key, extract_metadata_values(operand)?)),
                "$nin" => filters.push(MetadataFilter::nin(key, extract_metadata_values(operand)?)),
                "$not" => filters.push(MetadataFilter::not(parse_field_filter(key, operand)?)),
                "$contains" => filters.push(MetadataFilter::contains(
                    key,
                    operand
                        .as_str()
                        .ok_or_else(|| err("$contains expects a string"))?,
                )),
                "$gt" => filters.push(MetadataFilter::gt(key, extract_numeric(operand)?)),
                "$gte" => filters.push(MetadataFilter::gte(key, extract_numeric(operand)?)),
                "$lt" => filters.push(MetadataFilter::lt(key, extract_numeric(operand)?)),
                "$lte" => filters.push(MetadataFilter::lte(key, extract_numeric(operand)?)),
                "$exists" => {
                    let exists = operand
                        .as_bool()
                        .ok_or_else(|| err("$exists expects a boolean"))?;
                    if exists {
                        filters.push(MetadataFilter::exists(key));
                    } else {
                        filters.push(MetadataFilter::not(MetadataFilter::exists(key)));
                    }
                }
                "$elemMatch" => {
                    let dict = operand
                        .as_object()
                        .ok_or_else(|| err("$elemMatch expects a JSON object"))?;
                    let all_operators = dict.keys().all(|item| item.starts_with('$'));
                    let sub_filter = if all_operators {
                        parse_field_filter("_", operand)?
                    } else {
                        json_to_filter(operand)?
                    };
                    filters.push(MetadataFilter::elem_match(key, sub_filter));
                }
                "$size" => {
                    filters.push(MetadataFilter::size(key, value_to_usize(operand, "$size")?))
                }
                other => {
                    return Err(err(format!("unsupported filter operator: {other}")));
                }
            }
        }
        collapse_filters(filters, "field filter")
    } else {
        Ok(MetadataFilter::eq(key, json_to_metadata_value(value)?))
    }
}

fn collapse_filters(filters: Vec<MetadataFilter>, context: &str) -> Result<MetadataFilter> {
    match filters.len() {
        0 => Err(err(format!("{context} cannot be empty"))),
        1 => Ok(filters
            .into_iter()
            .next()
            .expect("single filter must exist")),
        _ => Ok(MetadataFilter::and(filters)),
    }
}

fn extract_metadata_values(value: &Value) -> Result<Vec<MetadataValue>> {
    let items = value
        .as_array()
        .ok_or_else(|| err("list filter operands must be JSON arrays"))?;
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        values.push(json_to_metadata_value(item)?);
    }
    Ok(values)
}

fn extract_numeric(value: &Value) -> Result<f64> {
    value
        .as_f64()
        .ok_or_else(|| err("numeric filter operands must be numbers"))
}

fn parse_multi_vector_queries(
    query_vectors: Option<&Value>,
    vector_weights: Option<&Value>,
) -> Result<BTreeMap<String, (Vec<f32>, f32)>> {
    let mut queries = BTreeMap::new();
    let Some(query_vectors) = query_vectors else {
        return Ok(queries);
    };
    let vectors = query_vectors
        .as_object()
        .ok_or_else(|| err("queryVectors must be a JSON object"))?;
    let weights = match vector_weights {
        Some(value) => Some(
            value
                .as_object()
                .ok_or_else(|| err("vectorWeights must be a JSON object"))?,
        ),
        None => None,
    };

    for (name, vector) in vectors {
        let weight = weights
            .and_then(|weights| weights.get(name))
            .map(|value| value_to_f32(value, "vector weights"))
            .transpose()?
            .unwrap_or(1.0);
        queries.insert(
            name.clone(),
            (value_to_vector(vector, "query vector")?, weight),
        );
    }
    Ok(queries)
}

fn parse_fusion(fusion: &str, rrf_k: usize) -> Result<FusionStrategy> {
    match fusion {
        "linear" => Ok(FusionStrategy::Linear),
        "rrf" => Ok(FusionStrategy::Rrf {
            rank_constant: rrf_k.max(1),
        }),
        other => Err(err(format!("unsupported fusion strategy: {other}"))),
    }
}

fn json_to_record(object: &Map<String, Value>, default_namespace: Option<&str>) -> Result<Record> {
    let namespace = object
        .get("namespace")
        .map(value_to_string)
        .transpose()?
        .unwrap_or_else(|| default_namespace.unwrap_or_default().to_owned());
    let id = object
        .get("id")
        .ok_or_else(|| err("batch record is missing 'id'"))?;
    let vector = object
        .get("vector")
        .ok_or_else(|| err("batch record is missing 'vector'"))?;

    let vectors = object
        .get("vectors")
        .map(json_to_named_vectors)
        .transpose()?
        .unwrap_or_default();
    let sparse = object
        .get("sparse")
        .map(json_to_sparse_value)
        .transpose()?
        .unwrap_or_default();
    let metadata = object
        .get("metadata")
        .map(json_to_metadata)
        .transpose()?
        .unwrap_or_default();

    Ok(Record {
        namespace,
        id: value_to_string(id)?,
        vector: value_to_vector(vector, "batch vector")?,
        vectors,
        sparse,
        metadata,
    })
}

fn record_to_json(record: &Record) -> Value {
    json!({
        "namespace": record.namespace,
        "id": record.id,
        "vector": record.vector,
        "vectors": named_vectors_to_json(&record.vectors),
        "sparse": sparse_to_json(&record.sparse),
        "metadata": metadata_to_json(&record.metadata),
    })
}

fn search_results_to_json(results: &[SearchResult], explain: bool, fusion: &str) -> Value {
    Value::Array(
        results
            .iter()
            .map(|result| search_result_to_json(result, explain, fusion))
            .collect(),
    )
}

fn search_outcome_to_json(outcome: &SearchOutcome, explain: bool, fusion: &str) -> Value {
    json!({
        "results": search_results_to_json(&outcome.results, explain, fusion),
        "stats": search_stats_to_json(&outcome.stats),
    })
}

fn search_result_to_json(result: &SearchResult, explain: bool, fusion: &str) -> Value {
    let mut object = Map::new();
    object.insert(
        "namespace".to_owned(),
        Value::String(result.namespace.clone()),
    );
    object.insert("id".to_owned(), Value::String(result.id.clone()));
    object.insert("score".to_owned(), float_value(result.score as f64));
    object.insert(
        "dense_score".to_owned(),
        float_value(result.dense_score as f64),
    );
    object.insert(
        "sparse_score".to_owned(),
        float_value(result.sparse_score as f64),
    );
    object.insert(
        "vector_name".to_owned(),
        result
            .vector_name
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    object.insert(
        "matched_terms".to_owned(),
        Value::Array(
            result
                .matched_terms
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    object.insert(
        "dense_rank".to_owned(),
        result
            .dense_rank
            .map(|rank| Value::Number(Number::from(rank)))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "sparse_rank".to_owned(),
        result
            .sparse_rank
            .map(|rank| Value::Number(Number::from(rank)))
            .unwrap_or(Value::Null),
    );
    object.insert("metadata".to_owned(), metadata_to_json(&result.metadata));

    if explain {
        object.insert(
            "explain".to_owned(),
            json!({
                "fusion": fusion,
                "dense_score": result.dense_score,
                "sparse_score": result.sparse_score,
                "matched_terms": result.matched_terms,
                "vector_name": result.vector_name,
                "dense_rank": result.dense_rank,
                "sparse_rank": result.sparse_rank,
                "bm25_term_scores": result.bm25_term_scores,
            }),
        );
    }

    Value::Object(object)
}

fn search_stats_to_json(stats: &vectlite::SearchStats) -> Value {
    json!({
        "used_ann": stats.used_ann,
        "ann_candidate_count": stats.ann_candidate_count,
        "exact_fallback": stats.exact_fallback,
        "considered_count": stats.considered_count,
        "fetch_k": stats.fetch_k,
        "mmr_applied": stats.mmr_applied,
        "sparse_candidate_count": stats.sparse_candidate_count,
        "ann_loaded_from_disk": stats.ann_loaded_from_disk,
        "wal_entries_replayed": stats.wal_entries_replayed,
        "fusion": stats.fusion,
        "rerank_applied": false,
        "rerank_count": 0,
        "timings": {
            "dense_us": stats.timings.dense_us,
            "sparse_us": stats.timings.sparse_us,
            "fusion_us": stats.timings.fusion_us,
            "total_us": stats.timings.total_us,
        },
    })
}

fn metadata_to_json(metadata: &Metadata) -> Value {
    let mut object = Map::new();
    for (key, value) in metadata {
        object.insert(key.clone(), metadata_value_to_json(value));
    }
    Value::Object(object)
}

fn metadata_value_to_json(value: &MetadataValue) -> Value {
    match value {
        MetadataValue::String(value) => Value::String(value.clone()),
        MetadataValue::Integer(value) => Value::Number(Number::from(*value)),
        MetadataValue::Float(value) => float_value(*value),
        MetadataValue::Boolean(value) => Value::Bool(*value),
        MetadataValue::Null => Value::Null,
        MetadataValue::List(items) => {
            Value::Array(items.iter().map(metadata_value_to_json).collect())
        }
        MetadataValue::Map(entries) => {
            let mut object = Map::new();
            for (key, value) in entries {
                object.insert(key.clone(), metadata_value_to_json(value));
            }
            Value::Object(object)
        }
    }
}

fn sparse_to_json(sparse: &SparseVector) -> Value {
    let mut object = Map::new();
    for (term, weight) in sparse {
        object.insert(term.clone(), float_value(*weight as f64));
    }
    Value::Object(object)
}

fn named_vectors_to_json(vectors: &NamedVectors) -> Value {
    let mut object = Map::new();
    for (name, vector) in vectors {
        object.insert(
            name.clone(),
            Value::Array(
                vector
                    .iter()
                    .map(|value| float_value(*value as f64))
                    .collect(),
            ),
        );
    }
    Value::Object(object)
}

fn stringify_value(value: Value) -> Result<String> {
    serde_json::to_string(&value).map_err(|error| err(format!("failed to serialize JSON: {error}")))
}

fn float_value(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn expect_optional_object<'a>(
    value: Option<&'a Value>,
    label: &str,
) -> Result<Option<&'a Map<String, Value>>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(_) => Err(err(format!("{label} must be a JSON object"))),
    }
}

fn get_string(object: Option<&Map<String, Value>>, key: &str) -> Result<Option<String>> {
    let Some(object) = object else {
        return Ok(None);
    };
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(value_to_string(value)?)),
    }
}

fn get_bool(object: Option<&Map<String, Value>>, key: &str) -> Result<Option<bool>> {
    let Some(object) = object else {
        return Ok(None);
    };
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(err(format!("{key} must be a boolean"))),
    }
}

fn get_usize(object: Option<&Map<String, Value>>, key: &str) -> Result<Option<usize>> {
    let Some(object) = object else {
        return Ok(None);
    };
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(value_to_usize(value, key)?)),
    }
}

fn get_f32(object: Option<&Map<String, Value>>, key: &str) -> Result<Option<f32>> {
    let Some(object) = object else {
        return Ok(None);
    };
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => Ok(Some(value_to_f32(value, key)?)),
    }
}

fn get_optional_f32(object: Option<&Map<String, Value>>, key: &str) -> Result<Option<f32>> {
    get_f32(object, key)
}

fn value_to_string(value: &Value) -> Result<String> {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| err("value must be a string"))
}

fn value_to_vector(value: &Value, label: &str) -> Result<Vec<f32>> {
    let items = value
        .as_array()
        .ok_or_else(|| err(format!("{label} must be a JSON array of numbers")))?;
    let mut vector = Vec::with_capacity(items.len());
    for item in items {
        vector.push(value_to_f32(item, label)?);
    }
    Ok(vector)
}

fn value_to_f32(value: &Value, label: &str) -> Result<f32> {
    value
        .as_f64()
        .map(|value| value as f32)
        .ok_or_else(|| err(format!("{label} must contain numeric values")))
}

fn js_vector_to_core(values: Vec<f64>, label: &str) -> Result<Vec<f32>> {
    let mut vector = Vec::with_capacity(values.len());
    for value in values {
        if !value.is_finite() {
            return Err(err(format!("{label} must contain finite numeric values")));
        }
        vector.push(value as f32);
    }
    Ok(vector)
}

fn value_to_usize(value: &Value, label: &str) -> Result<usize> {
    value
        .as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| err(format!("{label} must be an unsigned integer")))
}

fn err(message: impl Into<String>) -> NapiError {
    NapiError::from_reason(message.into())
}

fn to_napi_error(error: vectlite::VectLiteError) -> NapiError {
    err(error.to_string())
}

fn closed_database_error() -> vectlite::VectLiteError {
    vectlite::VectLiteError::InvalidFormat("database is closed".to_owned())
}
