use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString};
use vectlite::quantization::{
    BinaryQuantizationConfig, MultiVectorQuantizationConfig, ProductQuantizationConfig,
    QuantizationConfig, ScalarQuantizationConfig, TwoBitQuantizationConfig,
    default_product_num_sub_vectors,
};
use vectlite::{
    Database as CoreDatabase, DistanceMetric, FusionStrategy, HybridSearchOptions, Metadata,
    MetadataFilter, MetadataValue, MultiVectorSearchOptions, MultiVectors, NamedVectors,
    PayloadIndexType, Record, SearchOutcome, SearchResult, SparseVector, Store as CoreStore,
    WriteOperation,
};

create_exception!(vectlite, VectLiteError, PyException);
create_exception!(vectlite, VectLiteLockError, VectLiteError);

#[pyclass(module = "vectlite", name = "Database")]
struct PyDatabase {
    inner: Arc<RwLock<CoreDatabase>>,
}

#[pyclass(module = "vectlite", name = "Transaction")]
struct PyTransaction {
    inner: Arc<RwLock<CoreDatabase>>,
    staged: Mutex<TransactionState>,
}

#[derive(Default)]
struct TransactionState {
    ops: Vec<WriteOperation>,
    closed: bool,
}

#[pyclass(module = "vectlite", name = "Store")]
struct PyStore {
    inner: CoreStore,
}

#[pymethods]
impl PyStore {
    #[getter]
    fn root(&self) -> String {
        self.inner.root().display().to_string()
    }

    fn __repr__(&self) -> String {
        format!("Store(root='{}')", self.inner.root().display())
    }

    fn create_collection(&self, name: &str, dimension: usize) -> PyResult<PyDatabase> {
        let database = self
            .inner
            .create_collection(name, dimension)
            .map_err(to_py_error)?;
        Ok(PyDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    fn open_collection(&self, name: &str) -> PyResult<PyDatabase> {
        let database = self.inner.open_collection(name).map_err(to_py_error)?;
        Ok(PyDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    #[pyo3(signature = (name, dimension))]
    fn open_or_create_collection(&self, name: &str, dimension: usize) -> PyResult<PyDatabase> {
        let database = self
            .inner
            .open_or_create_collection(name, dimension)
            .map_err(to_py_error)?;
        Ok(PyDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    fn open_collection_read_only(&self, name: &str) -> PyResult<PyDatabase> {
        let database = self
            .inner
            .open_collection_read_only(name)
            .map_err(to_py_error)?;
        Ok(PyDatabase {
            inner: Arc::new(RwLock::new(database)),
        })
    }

    fn drop_collection(&self, name: &str) -> PyResult<bool> {
        self.inner.drop_collection(name).map_err(to_py_error)
    }

    fn collections(&self) -> PyResult<Vec<String>> {
        self.inner.collections().map_err(to_py_error)
    }

    /// Close the store. This is a no-op (the store holds no open file handles)
    /// but is provided for symmetry with ``Database.close()``.
    fn close(&self) -> PyResult<()> {
        Ok(())
    }
}

#[pyfunction(name = "open_store")]
fn open_store(root: String) -> PyResult<PyStore> {
    let store = CoreStore::open(&root).map_err(to_py_error)?;
    Ok(PyStore { inner: store })
}

#[pymethods]
impl PyDatabase {
    #[getter]
    fn path(&self) -> PyResult<String> {
        let database = self.read()?;
        Ok(database.path().display().to_string())
    }

    #[getter]
    fn wal_path(&self) -> PyResult<String> {
        let database = self.read()?;
        Ok(database.wal_path().display().to_string())
    }

    #[getter]
    fn dimension(&self) -> PyResult<usize> {
        let database = self.read()?;
        Ok(database.dimension())
    }

    #[getter]
    fn metric(&self) -> PyResult<String> {
        let database = self.read()?;
        Ok(database.metric().name().to_owned())
    }

    fn __len__(&self) -> PyResult<usize> {
        let database = self.read()?;
        Ok(database.len())
    }

    fn __repr__(&self) -> PyResult<String> {
        let database = self.read()?;
        Ok(format!(
            "Database(path='{}', dimension={}, size={})",
            database.path().display().to_string(),
            database.dimension(),
            database.len()
        ))
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None, ttl=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyAny>>,
        vectors: Option<Py<PyDict>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let expires_at = ttl_to_expires_at(ttl)?;
        let mut database = self.write_open()?;
        let record = Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors: MultiVectors::new(),
            expires_at,
        };
        database
            .insert_many(std::iter::once(record))
            .map_err(to_py_error)?;
        Ok(())
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None, ttl=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyAny>>,
        vectors: Option<Py<PyDict>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let expires_at = ttl_to_expires_at(ttl)?;
        let mut database = self.write_open()?;
        let record = Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors: MultiVectors::new(),
            expires_at,
        };
        database
            .upsert_many(std::iter::once(record))
            .map_err(to_py_error)?;
        Ok(())
    }

    #[pyo3(signature = (records, namespace=None))]
    fn insert_many(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database.insert_many(records).map_err(to_py_error)
    }

    #[pyo3(signature = (records, namespace=None))]
    fn upsert_many(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database.upsert_many(records).map_err(to_py_error)
    }

    #[pyo3(signature = (records, namespace=None, batch_size=10000))]
    fn bulk_ingest(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
        batch_size: usize,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let mut database = self.write_open()?;
        database
            .bulk_ingest(records, batch_size)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (id, namespace=None))]
    fn get(
        &self,
        py: Python<'_>,
        id: &str,
        namespace: Option<String>,
    ) -> PyResult<Option<Py<PyDict>>> {
        let record = {
            let database = self.read()?;
            database
                .get_in_namespace(&namespace.unwrap_or_default(), id)
                .cloned()
        };

        record
            .map(|record| record_to_pydict(py, &record).map(Into::into))
            .transpose()
    }

    #[pyo3(signature = (id, namespace=None))]
    fn delete(&self, id: &str, namespace: Option<String>) -> PyResult<bool> {
        let mut database = self.write_open()?;
        database
            .delete_in_namespace(&namespace.unwrap_or_default(), id)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (ids, namespace=None))]
    fn delete_many(&self, ids: Vec<String>, namespace: Option<String>) -> PyResult<usize> {
        let mut database = self.write_open()?;
        database
            .delete_many_in_namespace(&namespace.unwrap_or_default(), ids)
            .map_err(to_py_error)
    }

    fn flush(&self) -> PyResult<()> {
        let mut database = self.write_open()?;
        database.flush().map_err(to_py_error)
    }

    fn compact(&self) -> PyResult<()> {
        let mut database = self.write_open()?;
        database.compact().map_err(to_py_error)
    }

    // -------------------------------------------------------------------
    // Quantization
    // -------------------------------------------------------------------

    /// Enable quantization on the database.
    ///
    /// Args:
    ///     method: One of "scalar", "binary", or "product".
    ///     rescore_multiplier: How many candidates to rescore (multiplier of top_k).
    ///     num_sub_vectors: (PQ only) Number of sub-vector partitions.
    ///     num_centroids: (PQ only) Number of centroids per sub-vector (max 256).
    ///     training_iterations: (PQ only) K-means iterations.
    #[pyo3(signature = (method="scalar", rescore_multiplier=None, num_sub_vectors=None, num_centroids=None, training_iterations=None))]
    fn enable_quantization(
        &self,
        method: &str,
        rescore_multiplier: Option<usize>,
        num_sub_vectors: Option<usize>,
        num_centroids: Option<usize>,
        training_iterations: Option<usize>,
    ) -> PyResult<()> {
        let mut database = self.write_open()?;
        let config = parse_quantization_config(
            method,
            rescore_multiplier,
            num_sub_vectors,
            num_centroids,
            training_iterations,
            database.dimension(),
        )?;
        database.enable_quantization(config).map_err(to_py_error)
    }

    /// Disable quantization and remove persisted parameters.
    fn disable_quantization(&self) -> PyResult<()> {
        let mut database = self.write_open()?;
        database.disable_quantization().map_err(to_py_error)
    }

    /// Returns True if quantization is enabled.
    fn is_quantized(&self) -> PyResult<bool> {
        let database = self.read()?;
        Ok(database.is_quantized())
    }

    /// Returns the quantization method name if enabled, else None.
    #[getter]
    fn quantization_method(&self) -> PyResult<Option<String>> {
        let database = self.read()?;
        Ok(database.quantization_config().map(|config| match config {
            QuantizationConfig::Scalar(_) => "scalar".to_owned(),
            QuantizationConfig::Binary(_) => "binary".to_owned(),
            QuantizationConfig::Product(_) => "product".to_owned(),
        }))
    }

    /// Returns valid Product Quantization num_sub_vectors values for this database.
    fn valid_num_sub_vectors(&self) -> PyResult<Vec<usize>> {
        let database = self.read()?;
        Ok(database.valid_num_sub_vectors())
    }

    // ---- Multi-vector / ColBERT-style late interaction ----

    /// Upsert a record with multi-vector token embeddings (ColBERT-style).
    ///
    /// `multi_vectors` is a dict mapping space names to lists of token vectors,
    /// e.g. `{"colbert": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]}`.
    #[pyo3(signature = (id, vector, multi_vectors, metadata=None, namespace=None))]
    fn upsert_multi_vectors(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        multi_vectors: Py<PyDict>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<()> {
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let mv = parse_multi_vectors_dict(py, Some(multi_vectors.bind(py)))?;
        let mut database = self.write_open()?;
        database
            .upsert_multi_vectors_in_namespace(
                namespace.unwrap_or_default(),
                id,
                vector,
                metadata,
                mv,
            )
            .map_err(to_py_error)
    }

    /// Search using multi-vector late interaction (MaxSim) scoring.
    ///
    /// `query_tokens` is a list of token-level query embedding vectors.
    /// `space` identifies which multi-vector space to search in.
    #[pyo3(signature = (space, query_tokens, k=10, filter=None, namespace=None))]
    fn search_multi_vector(
        &self,
        py: Python<'_>,
        space: &str,
        query_tokens: Vec<Vec<f32>>,
        k: usize,
        filter: Option<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<Py<PyList>> {
        let filter = match filter {
            Some(f) => Some(parse_filter_dict(f.bind(py))?),
            None => None,
        };
        let options = MultiVectorSearchOptions {
            top_k: k,
            filter,
            namespace,
        };
        let database = self.read()?;
        let results = database
            .search_multi_vector(space, &query_tokens, options)
            .map_err(to_py_error)?;

        let list = PyList::empty(py);
        for result in results {
            let dict = PyDict::new(py);
            dict.set_item("id", &result.id)?;
            dict.set_item("score", result.score)?;
            dict.set_item("namespace", &result.namespace)?;
            dict.set_item("metadata", metadata_to_pydict(py, &result.metadata)?)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Enable 2-bit quantization for a multi-vector space.
    ///
    /// Currently only the "two_bit" method is supported, which provides ~16x
    /// compression for ColBERT-style token embeddings.
    #[pyo3(signature = (space, method="two_bit", rescore_multiplier=None))]
    fn enable_multi_vector_quantization(
        &self,
        space: &str,
        method: &str,
        rescore_multiplier: Option<usize>,
    ) -> PyResult<()> {
        let config = match method {
            "two_bit" => MultiVectorQuantizationConfig::TwoBit(TwoBitQuantizationConfig {
                rescore_multiplier: rescore_multiplier.unwrap_or(4),
            }),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown multi-vector quantization method: {other}. Supported: two_bit"
                )));
            }
        };
        let mut database = self.write_open()?;
        database
            .enable_multi_vector_quantization(space, config)
            .map_err(to_py_error)
    }

    /// Disable multi-vector quantization for a space.
    fn disable_multi_vector_quantization(&self, space: &str) -> PyResult<()> {
        let mut database = self.write_open()?;
        database
            .disable_multi_vector_quantization(space)
            .map_err(to_py_error)
    }

    /// Returns True if multi-vector quantization is enabled for the given space.
    fn is_multi_vector_quantized(&self, space: &str) -> PyResult<bool> {
        let database = self.read()?;
        Ok(database.is_multi_vector_quantized(space))
    }

    #[getter]
    fn read_only(&self) -> PyResult<bool> {
        let database = self.read()?;
        Ok(database.is_read_only())
    }

    fn snapshot(&self, dest: String) -> PyResult<()> {
        let database = self.read()?;
        database.snapshot(&dest).map_err(to_py_error)
    }

    fn backup(&self, dest: String) -> PyResult<()> {
        let database = self.read()?;
        database.backup(&dest).map_err(to_py_error)
    }

    #[pyo3(signature = (query=None, k=10, filter=None, namespace=None, all_namespaces=false, sparse=None, dense_weight=1.0, sparse_weight=1.0, fetch_k=0, mmr_lambda=None, vector_name=None, fusion="linear", rrf_k=60, truncate_dim=None, explain=false, rerank=None, rerank_k=0, query_vectors=None, vector_weights=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: Option<Vec<f32>>,
        k: usize,
        filter: Option<Py<PyDict>>,
        namespace: Option<String>,
        all_namespaces: bool,
        sparse: Option<Py<PyAny>>,
        dense_weight: f32,
        sparse_weight: f32,
        fetch_k: usize,
        mmr_lambda: Option<f32>,
        vector_name: Option<String>,
        fusion: &str,
        rrf_k: usize,
        truncate_dim: Option<usize>,
        explain: bool,
        rerank: Option<Py<PyAny>>,
        rerank_k: usize,
        query_vectors: Option<Py<PyDict>>,
        vector_weights: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyList>> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let query_payload = build_query_payload(
            py,
            query.as_deref(),
            sparse.as_ref().map(|dict| dict.bind(py)),
            namespace.as_deref(),
            all_namespaces,
            vector_name.as_deref(),
            truncate_dim,
            k,
            fusion,
            explain,
        )?;
        let multi = parse_multi_vector_queries(
            py,
            query_vectors.as_ref().map(|d| d.bind(py)),
            vector_weights.as_ref().map(|d| d.bind(py)),
        )?;
        let outcome = self.execute_search(
            py,
            query,
            k,
            filter,
            namespace,
            all_namespaces,
            sparse,
            dense_weight,
            sparse_weight,
            fetch_k,
            mmr_lambda,
            vector_name,
            parse_fusion(fusion, rrf_k)?,
            truncate_dim,
            multi,
        )?;

        render_search_results(
            py,
            &outcome.results,
            query_payload,
            rerank.as_ref(),
            rerank_k,
            explain,
            fusion,
        )
    }

    #[pyo3(signature = (query=None, k=10, filter=None, namespace=None, all_namespaces=false, sparse=None, dense_weight=1.0, sparse_weight=1.0, fetch_k=0, mmr_lambda=None, vector_name=None, fusion="linear", rrf_k=60, truncate_dim=None, explain=false, rerank=None, rerank_k=0, query_vectors=None, vector_weights=None))]
    fn search_with_stats(
        &self,
        py: Python<'_>,
        query: Option<Vec<f32>>,
        k: usize,
        filter: Option<Py<PyDict>>,
        namespace: Option<String>,
        all_namespaces: bool,
        sparse: Option<Py<PyAny>>,
        dense_weight: f32,
        sparse_weight: f32,
        fetch_k: usize,
        mmr_lambda: Option<f32>,
        vector_name: Option<String>,
        fusion: &str,
        rrf_k: usize,
        truncate_dim: Option<usize>,
        explain: bool,
        rerank: Option<Py<PyAny>>,
        rerank_k: usize,
        query_vectors: Option<Py<PyDict>>,
        vector_weights: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let query_payload = build_query_payload(
            py,
            query.as_deref(),
            sparse.as_ref().map(|dict| dict.bind(py)),
            namespace.as_deref(),
            all_namespaces,
            vector_name.as_deref(),
            truncate_dim,
            k,
            fusion,
            explain,
        )?;
        let multi = parse_multi_vector_queries(
            py,
            query_vectors.as_ref().map(|d| d.bind(py)),
            vector_weights.as_ref().map(|d| d.bind(py)),
        )?;
        let outcome = self.execute_search(
            py,
            query,
            k,
            filter,
            namespace,
            all_namespaces,
            sparse,
            dense_weight,
            sparse_weight,
            fetch_k,
            mmr_lambda,
            vector_name,
            parse_fusion(fusion, rrf_k)?,
            truncate_dim,
            multi,
        )?;

        let (results, rerank_applied, rerank_count) = render_search_result_items(
            py,
            &outcome.results,
            query_payload,
            rerank.as_ref(),
            rerank_k,
            explain,
            fusion,
        )?;
        let response = search_outcome_to_pydict(py, &outcome, results)?;
        let stats = response
            .get_item("stats")?
            .ok_or_else(|| PyValueError::new_err("search outcome is missing stats"))?
            .downcast_into::<PyDict>()?;
        stats.set_item("rerank_applied", rerank_applied)?;
        stats.set_item("rerank_count", rerank_count)?;

        Ok(response.into())
    }

    #[pyo3(signature = (namespace=None, filter=None))]
    fn count(
        &self,
        py: Python<'_>,
        namespace: Option<String>,
        filter: Option<Py<PyDict>>,
    ) -> PyResult<usize> {
        let filter = filter
            .as_ref()
            .map(|f| parse_filter_dict(f.bind(py)))
            .transpose()?;
        let database = self.read()?;
        Ok(database.count_filtered(namespace.as_deref(), filter.as_ref()))
    }

    fn namespaces(&self) -> PyResult<Vec<String>> {
        let database = self.read()?;
        Ok(database.namespaces())
    }

    fn close(&self) -> PyResult<()> {
        let mut database = self.write()?;
        database.close().map_err(to_py_error)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    #[pyo3(signature = (namespace=None, filter=None, limit=0, offset=0))]
    fn list(
        &self,
        py: Python<'_>,
        namespace: Option<String>,
        filter: Option<Py<PyDict>>,
        limit: usize,
        offset: usize,
    ) -> PyResult<Py<PyList>> {
        let filter = filter
            .as_ref()
            .map(|f| parse_filter_dict(f.bind(py)))
            .transpose()?;
        let records = {
            let database = self.read()?;
            database
                .list(namespace.as_deref(), filter.as_ref(), limit, offset)
                .into_iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        let list = PyList::empty(py);
        for record in &records {
            list.append(record_to_pydict(py, record)?)?;
        }
        Ok(list.into())
    }

    #[pyo3(signature = (namespace=None, filter=None, limit=100, cursor=None))]
    fn list_cursor(
        &self,
        py: Python<'_>,
        namespace: Option<String>,
        filter: Option<Py<PyDict>>,
        limit: usize,
        cursor: Option<String>,
    ) -> PyResult<(Py<PyList>, Option<String>)> {
        let filter = filter
            .as_ref()
            .map(|f| parse_filter_dict(f.bind(py)))
            .transpose()?;
        let (records, next_cursor) = {
            let database = self.read()?;
            let (recs, nc) = database.list_cursor(
                namespace.as_deref(),
                filter.as_ref(),
                limit,
                cursor.as_deref(),
            );
            (recs.into_iter().cloned().collect::<Vec<_>>(), nc)
        };
        let list = PyList::empty(py);
        for record in &records {
            list.append(record_to_pydict(py, record)?)?;
        }
        Ok((list.into(), next_cursor))
    }

    #[pyo3(signature = (filter, namespace=None))]
    fn delete_by_filter(
        &self,
        py: Python<'_>,
        filter: Py<PyDict>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let filter = parse_filter_dict(filter.bind(py))?;
        let mut database = self.write_open()?;
        database
            .delete_by_filter(namespace.as_deref(), &filter)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (id, metadata, namespace=None))]
    fn update_metadata(
        &self,
        py: Python<'_>,
        id: &str,
        metadata: Py<PyDict>,
        namespace: Option<String>,
    ) -> PyResult<bool> {
        let patch = parse_metadata_dict(Some(metadata.bind(py)))?;
        let mut database = self.write_open()?;
        database
            .update_metadata_in_namespace(namespace.unwrap_or_default(), id, patch)
            .map_err(to_py_error)
    }

    // -------------------------------------------------------------------
    // TTL / Expiry
    // -------------------------------------------------------------------

    #[pyo3(signature = (id, ttl, namespace=None))]
    fn set_ttl(&self, id: &str, ttl: f64, namespace: Option<String>) -> PyResult<bool> {
        let mut database = self.write_open()?;
        database
            .set_ttl_in_namespace(&namespace.unwrap_or_default(), id, ttl)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (id, namespace=None))]
    fn clear_ttl(&self, id: &str, namespace: Option<String>) -> PyResult<bool> {
        let mut database = self.write_open()?;
        database
            .clear_ttl_in_namespace(&namespace.unwrap_or_default(), id)
            .map_err(to_py_error)
    }

    // -------------------------------------------------------------------
    // Payload Indexes
    // -------------------------------------------------------------------

    #[pyo3(signature = (field, index_type))]
    fn create_index(&self, field: &str, index_type: &str) -> PyResult<bool> {
        let ty = parse_payload_index_type(index_type)?;
        let mut database = self.write_open()?;
        database.create_index(field, ty).map_err(to_py_error)
    }

    #[pyo3(signature = (field,))]
    fn drop_index(&self, field: &str) -> PyResult<bool> {
        let mut database = self.write_open()?;
        database.drop_index(field).map_err(to_py_error)
    }

    fn list_indexes(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let database = self.read()?;
        let indexes = database.list_indexes();
        let list = PyList::empty(py);
        for (field, index_type) in indexes {
            let tuple = (field, index_type.name());
            list.append(tuple.into_pyobject(py)?)?;
        }
        Ok(list.into())
    }

    fn transaction(&self) -> PyResult<PyTransaction> {
        drop(self.read()?);
        Ok(PyTransaction {
            inner: Arc::clone(&self.inner),
            staged: Mutex::new(TransactionState::default()),
        })
    }
}

#[pymethods]
impl PyTransaction {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if exc_type.is_some() {
            self.rollback()?;
        } else {
            self.commit()?;
        }
        Ok(false)
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None, ttl=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyAny>>,
        vectors: Option<Py<PyDict>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let expires_at = ttl_to_expires_at(ttl)?;
        self.stage(WriteOperation::Insert(Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors: MultiVectors::new(),
            expires_at,
        }))
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None, ttl=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyAny>>,
        vectors: Option<Py<PyDict>>,
        ttl: Option<f64>,
    ) -> PyResult<()> {
        let sparse = coerce_sparse_param(py, sparse)?;
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let expires_at = ttl_to_expires_at(ttl)?;
        self.stage(WriteOperation::Upsert(Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors: MultiVectors::new(),
            expires_at,
        }))
    }

    #[pyo3(signature = (records, namespace=None))]
    fn upsert_many(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let count = records.len();
        for record in records {
            self.stage(WriteOperation::Upsert(record))?;
        }
        Ok(count)
    }

    #[pyo3(signature = (records, namespace=None))]
    fn insert_many(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let count = records.len();
        for record in records {
            self.stage(WriteOperation::Insert(record))?;
        }
        Ok(count)
    }

    #[pyo3(signature = (id, namespace=None))]
    fn delete(&self, id: &str, namespace: Option<String>) -> PyResult<bool> {
        self.stage(WriteOperation::Delete {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
        })?;
        Ok(true)
    }

    #[pyo3(signature = (ids, namespace=None))]
    fn delete_many(&self, ids: Vec<String>, namespace: Option<String>) -> PyResult<usize> {
        let namespace = namespace.unwrap_or_default();
        let count = ids.len();
        for id in ids {
            self.stage(WriteOperation::Delete {
                namespace: namespace.clone(),
                id,
            })?;
        }
        Ok(count)
    }

    fn commit(&self) -> PyResult<()> {
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
        database.apply_operations(ops).map_err(to_py_error)
    }

    fn rollback(&self) -> PyResult<()> {
        let mut state = self.state()?;
        state.closed = true;
        state.ops.clear();
        Ok(())
    }

    fn __len__(&self) -> PyResult<usize> {
        let state = self.state()?;
        Ok(state.ops.len())
    }
}

impl PyTransaction {
    fn stage(&self, op: WriteOperation) -> PyResult<()> {
        let mut state = self.state()?;
        if state.closed {
            return Err(VectLiteError::new_err("transaction is already closed"));
        }
        state.ops.push(op);
        Ok(())
    }

    fn state(&self) -> PyResult<MutexGuard<'_, TransactionState>> {
        self.staged
            .lock()
            .map_err(|_| VectLiteError::new_err("transaction state lock poisoned"))
    }

    fn write_db(&self) -> PyResult<RwLockWriteGuard<'_, CoreDatabase>> {
        self.inner
            .write()
            .map_err(|_| VectLiteError::new_err("database write lock poisoned"))
    }
}

impl PyDatabase {
    fn read(&self) -> PyResult<RwLockReadGuard<'_, CoreDatabase>> {
        let database = self
            .inner
            .read()
            .map_err(|_| VectLiteError::new_err("database read lock poisoned"))?;
        if database.is_closed() {
            return Err(to_py_error(closed_database_error()));
        }
        Ok(database)
    }

    fn write(&self) -> PyResult<RwLockWriteGuard<'_, CoreDatabase>> {
        self.inner
            .write()
            .map_err(|_| VectLiteError::new_err("database write lock poisoned"))
    }

    fn write_open(&self) -> PyResult<RwLockWriteGuard<'_, CoreDatabase>> {
        let database = self.write()?;
        if database.is_closed() {
            return Err(to_py_error(closed_database_error()));
        }
        Ok(database)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_search(
        &self,
        py: Python<'_>,
        query: Option<Vec<f32>>,
        k: usize,
        filter: Option<Py<PyDict>>,
        namespace: Option<String>,
        all_namespaces: bool,
        sparse: Option<Py<PyDict>>,
        dense_weight: f32,
        sparse_weight: f32,
        fetch_k: usize,
        mmr_lambda: Option<f32>,
        vector_name: Option<String>,
        fusion: FusionStrategy,
        truncate_dim: Option<usize>,
        multi_vector_queries: std::collections::BTreeMap<String, (Vec<f32>, f32)>,
    ) -> PyResult<SearchOutcome> {
        let filter = filter
            .as_ref()
            .map(|filter| parse_filter_dict(filter.bind(py)))
            .transpose()?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let sparse_ref = if sparse.is_empty() {
            None
        } else {
            Some(&sparse)
        };
        let options = HybridSearchOptions {
            top_k: k,
            filter,
            dense_weight,
            sparse_weight,
            fetch_k,
            mmr_lambda,
            vector_name,
            fusion,
            truncate_dim,
            multi_vector_queries,
        };

        let database = self.read()?;
        if all_namespaces {
            database
                .hybrid_search_all_namespaces_with_stats(query.as_deref(), sparse_ref, options)
                .map_err(to_py_error)
        } else {
            database
                .hybrid_search_in_namespace_with_stats(
                    &namespace.unwrap_or_default(),
                    query.as_deref(),
                    sparse_ref,
                    options,
                )
                .map_err(to_py_error)
        }
    }
}

#[pyfunction(name = "open", signature = (path, dimension=None, read_only=false, lock_timeout=None, metric=None))]
fn open_database(
    path: String,
    dimension: Option<usize>,
    read_only: bool,
    lock_timeout: Option<f64>,
    metric: Option<String>,
) -> PyResult<PyDatabase> {
    let parsed_metric = match metric.as_deref() {
        Some(name) => DistanceMetric::from_name(name).map_err(to_py_error)?,
        None => DistanceMetric::Cosine,
    };

    let database = if read_only {
        if !Path::new(&path).exists() {
            return Err(VectLiteError::new_err(
                "cannot open non-existent database in read-only mode",
            ));
        }
        match lock_timeout {
            Some(timeout) => CoreDatabase::open_read_only_with_timeout(&path, Some(timeout))
                .map_err(to_py_error)?,
            None => CoreDatabase::open_read_only(&path).map_err(to_py_error)?,
        }
    } else if Path::new(&path).exists() {
        match (dimension, lock_timeout) {
            (Some(dimension), Some(timeout)) => {
                // Try open with timeout, check dimension
                let db = CoreDatabase::open_with_timeout(&path, timeout).map_err(to_py_error)?;
                if db.dimension() != dimension {
                    return Err(to_py_error(vectlite::VectLiteError::DimensionMismatch {
                        expected: db.dimension(),
                        found: dimension,
                    })
                    .into());
                }
                db
            }
            (Some(dimension), None) => {
                CoreDatabase::open_or_create_with_metric(&path, dimension, parsed_metric)
                    .map_err(to_py_error)?
            }
            (None, Some(timeout)) => {
                CoreDatabase::open_with_timeout(&path, timeout).map_err(to_py_error)?
            }
            (None, None) => CoreDatabase::open(&path).map_err(to_py_error)?,
        }
    } else {
        let Some(dimension) = dimension else {
            return Err(VectLiteError::new_err(
                "dimension is required when creating a new database",
            ));
        };
        CoreDatabase::create_with_metric(&path, dimension, parsed_metric).map_err(to_py_error)?
    };

    Ok(PyDatabase {
        inner: Arc::new(RwLock::new(database)),
    })
}

#[pyfunction(name = "restore", signature = (source, dest))]
fn restore_database(source: String, dest: String) -> PyResult<PyDatabase> {
    let database = CoreDatabase::restore(&source, &dest).map_err(to_py_error)?;
    Ok(PyDatabase {
        inner: Arc::new(RwLock::new(database)),
    })
}

#[pymodule]
fn _vectlite(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("VectLiteError", m.py().get_type::<VectLiteError>())?;
    m.add("VectLiteLockError", m.py().get_type::<VectLiteLockError>())?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyTransaction>()?;
    m.add_class::<PyStore>()?;
    m.add_function(wrap_pyfunction!(open_database, m)?)?;
    m.add_function(wrap_pyfunction!(open_store, m)?)?;
    m.add_function(wrap_pyfunction!(restore_database, m)?)?;
    Ok(())
}

fn parse_metadata_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<Metadata> {
    let mut metadata = Metadata::new();
    let Some(dict) = dict else {
        return Ok(metadata);
    };

    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        let value = py_to_metadata_value(&value)?;
        metadata.insert(key, value);
    }

    Ok(metadata)
}

/// Validate and coerce the `sparse` parameter. Accepts `None`, a `dict[str, float]`
/// (returned by `sparse_terms()`), or raises a clear error for any other type.
/// Convert an optional TTL (seconds from now) to an absolute `expires_at` timestamp.
fn ttl_to_expires_at(ttl: Option<f64>) -> PyResult<Option<f64>> {
    match ttl {
        None => Ok(None),
        Some(t) if t < 0.0 || t.is_nan() => Err(PyValueError::new_err(
            "ttl must be a non-negative finite number",
        )),
        Some(t) => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            Ok(Some(now + t))
        }
    }
}

fn coerce_sparse_param(py: Python<'_>, sparse: Option<Py<PyAny>>) -> PyResult<Option<Py<PyDict>>> {
    let Some(sparse) = sparse else {
        return Ok(None);
    };
    let obj = sparse.bind(py);
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        return Ok(Some(dict.clone().unbind()));
    }
    if obj.is_instance_of::<PyString>() {
        return Err(PyTypeError::new_err(
            "sparse parameter expects dict[str, float] (the return value of \
             vectlite.sparse_terms()), got str. Use sparse=vectlite.sparse_terms(\"your text\") \
             instead of sparse=\"your text\"",
        ));
    }
    let type_name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    Err(PyTypeError::new_err(format!(
        "sparse parameter expects dict[str, float] (the return value of \
         vectlite.sparse_terms()), got {type_name}",
    )))
}

fn parse_sparse_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<SparseVector> {
    let mut sparse = SparseVector::new();
    let Some(dict) = dict else {
        return Ok(sparse);
    };

    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<f32>()?;
        sparse.insert(key, value);
    }

    Ok(sparse)
}

fn parse_named_vectors_dict(dict: Option<&Bound<'_, PyDict>>) -> PyResult<NamedVectors> {
    let mut vectors = NamedVectors::new();
    let Some(dict) = dict else {
        return Ok(vectors);
    };

    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        if key.is_empty() {
            return Err(PyValueError::new_err(
                "named vectors must not use an empty name",
            ));
        }
        let value = value.extract::<Vec<f32>>()?;
        vectors.insert(key, value);
    }

    Ok(vectors)
}

fn parse_filter_dict(dict: &Bound<'_, PyDict>) -> PyResult<MetadataFilter> {
    let mut filters = Vec::new();

    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        match key.as_str() {
            "$and" => filters.push(MetadataFilter::and(parse_filter_group(&value)?)),
            "$or" => filters.push(MetadataFilter::or(parse_filter_group(&value)?)),
            "$not" => {
                let dict = value.downcast::<PyDict>()?;
                filters.push(MetadataFilter::not(parse_filter_dict(&dict)?));
            }
            field => filters.push(parse_field_filter(field, &value)?),
        }
    }

    collapse_filters(filters, "filter")
}

fn parse_filter_group(value: &Bound<'_, PyAny>) -> PyResult<Vec<MetadataFilter>> {
    let list = value.downcast::<PyList>()?;
    let mut filters = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item.downcast::<PyDict>()?;
        filters.push(parse_filter_dict(&dict)?);
    }
    Ok(filters)
}

fn parse_field_filter(key: &str, value: &Bound<'_, PyAny>) -> PyResult<MetadataFilter> {
    if let Ok(operators) = value.downcast::<PyDict>() {
        let mut filters = Vec::new();
        for (operator, operand) in operators.iter() {
            let operator = operator.extract::<String>()?;
            match operator.as_str() {
                "$eq" => filters.push(MetadataFilter::eq(key, py_to_metadata_value(&operand)?)),
                "$ne" => filters.push(MetadataFilter::ne(key, py_to_metadata_value(&operand)?)),
                "$in" => filters.push(MetadataFilter::r#in(
                    key,
                    extract_metadata_values(&operand)?,
                )),
                "$nin" => {
                    filters.push(MetadataFilter::nin(key, extract_metadata_values(&operand)?))
                }
                "$not" => filters.push(MetadataFilter::not(parse_field_filter(key, &operand)?)),
                "$contains" => {
                    filters.push(MetadataFilter::contains(key, operand.extract::<String>()?))
                }
                "$gt" => filters.push(MetadataFilter::gt(key, extract_numeric(&operand)?)),
                "$gte" => filters.push(MetadataFilter::gte(key, extract_numeric(&operand)?)),
                "$lt" => filters.push(MetadataFilter::lt(key, extract_numeric(&operand)?)),
                "$lte" => filters.push(MetadataFilter::lte(key, extract_numeric(&operand)?)),
                "$exists" => {
                    let exists = operand.extract::<bool>()?;
                    if exists {
                        filters.push(MetadataFilter::exists(key));
                    } else {
                        filters.push(MetadataFilter::not(MetadataFilter::exists(key)));
                    }
                }
                "$elemMatch" => {
                    let dict = operand.downcast::<PyDict>()?;
                    // If all keys in the sub-dict start with '$', treat
                    // as operator conditions against the element itself
                    // (stored under key "_" in the elem_match logic).
                    let all_operators = dict.iter().all(|(k, _)| {
                        k.extract::<String>()
                            .map(|s| s.starts_with('$'))
                            .unwrap_or(false)
                    });
                    let sub_filter = if all_operators {
                        parse_field_filter("_", &dict.as_any())?
                    } else {
                        parse_filter_dict(&dict)?
                    };
                    filters.push(MetadataFilter::elem_match(key, sub_filter));
                }
                "$size" => {
                    let size = operand.extract::<usize>()?;
                    filters.push(MetadataFilter::size(key, size));
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unsupported filter operator: {other}"
                    )));
                }
            }
        }

        collapse_filters(filters, "field filter")
    } else {
        Ok(MetadataFilter::eq(key, py_to_metadata_value(value)?))
    }
}

fn collapse_filters(filters: Vec<MetadataFilter>, context: &str) -> PyResult<MetadataFilter> {
    match filters.len() {
        0 => Err(PyValueError::new_err(format!("{context} cannot be empty"))),
        1 => Ok(filters
            .into_iter()
            .next()
            .expect("single filter must exist")),
        _ => Ok(MetadataFilter::and(filters)),
    }
}

fn py_to_metadata_value(value: &Bound<'_, PyAny>) -> PyResult<MetadataValue> {
    if value.is_none() {
        return Ok(MetadataValue::Null);
    }

    if value.is_instance_of::<PyBool>() {
        return Ok(MetadataValue::Boolean(value.extract::<bool>()?));
    }

    if value.is_instance_of::<PyString>() {
        return Ok(MetadataValue::String(value.extract::<String>()?));
    }

    if value.is_instance_of::<PyInt>() {
        return Ok(MetadataValue::Integer(value.extract::<i64>()?));
    }

    if value.is_instance_of::<PyFloat>() {
        return Ok(MetadataValue::Float(value.extract::<f64>()?));
    }

    if let Ok(list) = value.downcast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_metadata_value(&item)?);
        }
        return Ok(MetadataValue::List(items));
    }

    if let Ok(dict) = value.downcast::<PyDict>() {
        let mut map = std::collections::BTreeMap::new();
        for (k, v) in dict.iter() {
            let key = k.extract::<String>()?;
            let val = py_to_metadata_value(&v)?;
            map.insert(key, val);
        }
        return Ok(MetadataValue::Map(map));
    }

    Err(PyTypeError::new_err(
        "metadata values must be str, int, float, bool, None, list, or dict",
    ))
}

fn extract_numeric(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "boolean values are not valid numeric filter operands",
        ));
    }

    if value.is_instance_of::<PyInt>() {
        return Ok(value.extract::<i64>()? as f64);
    }

    if value.is_instance_of::<PyFloat>() {
        return Ok(value.extract::<f64>()?);
    }

    Err(PyTypeError::new_err(
        "numeric filter operands must be int or float",
    ))
}

fn record_to_pydict<'py>(py: Python<'py>, record: &Record) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("namespace", &record.namespace)?;
    dict.set_item("id", &record.id)?;
    dict.set_item("vector", record.vector.clone())?;
    dict.set_item("vectors", named_vectors_to_pydict(py, &record.vectors)?)?;
    dict.set_item("sparse", sparse_to_pydict(py, &record.sparse)?)?;
    dict.set_item("metadata", metadata_to_pydict(py, &record.metadata)?)?;
    dict.set_item("expires_at", record.expires_at)?;
    Ok(dict)
}

fn parse_record_batch(
    py: Python<'_>,
    records: Vec<Py<PyDict>>,
    default_namespace: Option<&str>,
) -> PyResult<Vec<Record>> {
    let mut parsed = Vec::with_capacity(records.len());

    for record in records {
        let record = record.bind(py);
        let namespace =
            parse_optional_namespace_item(record.get_item("namespace")?, default_namespace)?;
        let id = record
            .get_item("id")?
            .ok_or_else(|| PyValueError::new_err("batch record is missing 'id'"))?
            .extract::<String>()?;
        let vector = record
            .get_item("vector")?
            .ok_or_else(|| PyValueError::new_err("batch record is missing 'vector'"))?
            .extract::<Vec<f32>>()?;
        let vectors = parse_optional_named_vectors_item(record.get_item("vectors")?)?;
        let sparse = parse_optional_sparse_item(record.get_item("sparse")?)?;
        let metadata = parse_optional_metadata_item(record.get_item("metadata")?)?;
        let multi_vectors =
            parse_optional_multi_vectors_item(py, record.get_item("multi_vectors")?)?;
        let ttl = record
            .get_item("ttl")?
            .map(|v| v.extract::<f64>())
            .transpose()?;
        let expires_at = ttl_to_expires_at(ttl)?;

        parsed.push(Record {
            namespace,
            id,
            vector,
            vectors,
            sparse,
            metadata,
            multi_vectors,
            expires_at,
        });
    }

    Ok(parsed)
}

fn parse_optional_metadata_item(item: Option<Bound<'_, PyAny>>) -> PyResult<Metadata> {
    let Some(item) = item else {
        return Ok(Metadata::new());
    };

    if item.is_none() {
        return Ok(Metadata::new());
    }

    let dict = item.downcast::<PyDict>()?;
    parse_metadata_dict(Some(&dict))
}

fn parse_optional_sparse_item(item: Option<Bound<'_, PyAny>>) -> PyResult<SparseVector> {
    let Some(item) = item else {
        return Ok(SparseVector::new());
    };

    if item.is_none() {
        return Ok(SparseVector::new());
    }

    let dict = item.downcast::<PyDict>()?;
    parse_sparse_dict(Some(&dict))
}

fn parse_optional_named_vectors_item(item: Option<Bound<'_, PyAny>>) -> PyResult<NamedVectors> {
    let Some(item) = item else {
        return Ok(NamedVectors::new());
    };

    if item.is_none() {
        return Ok(NamedVectors::new());
    }

    let dict = item.downcast::<PyDict>()?;
    parse_named_vectors_dict(Some(&dict))
}

fn parse_optional_multi_vectors_item(
    py: Python<'_>,
    item: Option<Bound<'_, PyAny>>,
) -> PyResult<MultiVectors> {
    let Some(item) = item else {
        return Ok(MultiVectors::new());
    };
    if item.is_none() {
        return Ok(MultiVectors::new());
    }
    let dict = item.downcast::<PyDict>()?;
    parse_multi_vectors_dict(py, Some(dict))
}

fn parse_multi_vectors_dict(
    _py: Python<'_>,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<MultiVectors> {
    let Some(dict) = dict else {
        return Ok(MultiVectors::new());
    };
    let mut multi_vectors = MultiVectors::new();
    for (key, value) in dict.iter() {
        let space_name: String = key.extract()?;
        let token_list = value.downcast::<PyList>()?;
        let mut token_vectors = Vec::with_capacity(token_list.len());
        for token_item in token_list.iter() {
            let vec: Vec<f32> = token_item.extract()?;
            token_vectors.push(vec);
        }
        multi_vectors.insert(space_name, token_vectors);
    }
    Ok(multi_vectors)
}

fn search_result_to_pydict<'py>(
    py: Python<'py>,
    result: &SearchResult,
    explain: bool,
    fusion: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("namespace", &result.namespace)?;
    dict.set_item("id", &result.id)?;
    dict.set_item("score", result.score)?;
    dict.set_item("dense_score", result.dense_score)?;
    dict.set_item("sparse_score", result.sparse_score)?;
    match &result.vector_name {
        Some(vector_name) => dict.set_item("vector_name", vector_name)?,
        None => dict.set_item("vector_name", py.None())?,
    }
    dict.set_item("matched_terms", &result.matched_terms)?;
    match result.dense_rank {
        Some(rank) => dict.set_item("dense_rank", rank)?,
        None => dict.set_item("dense_rank", py.None())?,
    }
    match result.sparse_rank {
        Some(rank) => dict.set_item("sparse_rank", rank)?,
        None => dict.set_item("sparse_rank", py.None())?,
    }
    dict.set_item("metadata", metadata_to_pydict(py, &result.metadata)?)?;
    if explain {
        let explain_dict = PyDict::new(py);
        explain_dict.set_item("fusion", fusion)?;
        explain_dict.set_item("dense_score", result.dense_score)?;
        explain_dict.set_item("sparse_score", result.sparse_score)?;
        explain_dict.set_item("matched_terms", &result.matched_terms)?;
        match &result.vector_name {
            Some(vector_name) => explain_dict.set_item("vector_name", vector_name)?,
            None => explain_dict.set_item("vector_name", py.None())?,
        }
        match result.dense_rank {
            Some(rank) => explain_dict.set_item("dense_rank", rank)?,
            None => explain_dict.set_item("dense_rank", py.None())?,
        }
        match result.sparse_rank {
            Some(rank) => explain_dict.set_item("sparse_rank", rank)?,
            None => explain_dict.set_item("sparse_rank", py.None())?,
        }
        // Per-term BM25 scores
        let bm25_dict = PyDict::new(py);
        for (term, score) in &result.bm25_term_scores {
            bm25_dict.set_item(term, *score)?;
        }
        explain_dict.set_item("bm25_term_scores", bm25_dict)?;
        dict.set_item("explain", explain_dict)?;
    }
    Ok(dict)
}

fn search_outcome_to_pydict<'py>(
    py: Python<'py>,
    outcome: &SearchOutcome,
    results: Vec<Py<PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let result_list = PyList::empty(py);
    for result in results {
        result_list.append(result.bind(py))?;
    }
    dict.set_item("results", result_list)?;
    dict.set_item("stats", search_stats_to_pydict(py, &outcome.stats)?)?;
    Ok(dict)
}

fn search_stats_to_pydict<'py>(
    py: Python<'py>,
    stats: &vectlite::SearchStats,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("used_ann", stats.used_ann)?;
    dict.set_item("ann_candidate_count", stats.ann_candidate_count)?;
    dict.set_item("exact_fallback", stats.exact_fallback)?;
    dict.set_item("considered_count", stats.considered_count)?;
    dict.set_item("fetch_k", stats.fetch_k)?;
    dict.set_item("mmr_applied", stats.mmr_applied)?;
    dict.set_item("sparse_candidate_count", stats.sparse_candidate_count)?;
    dict.set_item("ann_loaded_from_disk", stats.ann_loaded_from_disk)?;
    dict.set_item("wal_entries_replayed", stats.wal_entries_replayed)?;
    dict.set_item("fusion", &stats.fusion)?;
    dict.set_item("effective_dimension", stats.effective_dimension)?;
    dict.set_item("matryoshka_truncated", stats.matryoshka_truncated)?;
    dict.set_item("rerank_applied", false)?;
    dict.set_item("rerank_count", 0)?;
    let timings = PyDict::new(py);
    timings.set_item("dense_us", stats.timings.dense_us)?;
    timings.set_item("sparse_us", stats.timings.sparse_us)?;
    timings.set_item("fusion_us", stats.timings.fusion_us)?;
    timings.set_item("total_us", stats.timings.total_us)?;
    dict.set_item("timings", timings)?;
    Ok(dict)
}

fn metadata_value_to_py(py: Python<'_>, value: &MetadataValue) -> PyResult<PyObject> {
    match value {
        MetadataValue::String(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        MetadataValue::Integer(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        MetadataValue::Float(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        MetadataValue::Boolean(v) => {
            let obj = (*v).into_pyobject(py)?;
            Ok(obj.to_owned().into_any().unbind())
        }
        MetadataValue::Null => Ok(py.None()),
        MetadataValue::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(metadata_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        MetadataValue::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                dict.set_item(k, metadata_value_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

fn metadata_to_pydict<'py>(py: Python<'py>, metadata: &Metadata) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in metadata {
        dict.set_item(key, metadata_value_to_py(py, value)?)?;
    }
    Ok(dict)
}

fn sparse_to_pydict<'py>(py: Python<'py>, sparse: &SparseVector) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (term, weight) in sparse {
        dict.set_item(term, *weight)?;
    }
    Ok(dict)
}

fn named_vectors_to_pydict<'py>(
    py: Python<'py>,
    vectors: &NamedVectors,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, vector) in vectors {
        dict.set_item(name, vector.clone())?;
    }
    Ok(dict)
}

fn build_query_payload<'py>(
    py: Python<'py>,
    dense_query: Option<&[f32]>,
    sparse_query: Option<&Bound<'py, PyDict>>,
    namespace: Option<&str>,
    all_namespaces: bool,
    vector_name: Option<&str>,
    truncate_dim: Option<usize>,
    k: usize,
    fusion: &str,
    explain: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match dense_query {
        Some(dense_query) => dict.set_item("dense", dense_query.to_vec())?,
        None => dict.set_item("dense", py.None())?,
    }
    match sparse_query {
        Some(sparse_query) => dict.set_item("sparse", sparse_query)?,
        None => dict.set_item("sparse", py.None())?,
    }
    match namespace {
        Some(namespace) => dict.set_item("namespace", namespace)?,
        None => dict.set_item("namespace", py.None())?,
    }
    match vector_name {
        Some(vector_name) => dict.set_item("vector_name", vector_name)?,
        None => dict.set_item("vector_name", py.None())?,
    }
    match truncate_dim {
        Some(truncate_dim) => dict.set_item("truncate_dim", truncate_dim)?,
        None => dict.set_item("truncate_dim", py.None())?,
    }
    dict.set_item("all_namespaces", all_namespaces)?;
    dict.set_item("k", k)?;
    dict.set_item("fusion", fusion)?;
    dict.set_item("explain", explain)?;
    Ok(dict)
}

fn parse_multi_vector_queries(
    _py: Python<'_>,
    query_vectors: Option<&Bound<'_, PyDict>>,
    vector_weights: Option<&Bound<'_, PyDict>>,
) -> PyResult<std::collections::BTreeMap<String, (Vec<f32>, f32)>> {
    let mut result = std::collections::BTreeMap::new();
    let Some(qv) = query_vectors else {
        return Ok(result);
    };
    for (key, value) in qv.iter() {
        let name = key.extract::<String>()?;
        let vector = value.extract::<Vec<f32>>()?;
        let weight = vector_weights
            .and_then(|w| w.get_item(&name).ok().flatten())
            .map(|v| v.extract::<f32>())
            .transpose()?
            .unwrap_or(1.0);
        result.insert(name, (vector, weight));
    }
    Ok(result)
}

fn parse_fusion(fusion: &str, rrf_k: usize) -> PyResult<FusionStrategy> {
    match fusion {
        "linear" => Ok(FusionStrategy::Linear),
        "rrf" => Ok(FusionStrategy::Rrf {
            rank_constant: rrf_k.max(1),
        }),
        other => Err(PyValueError::new_err(format!(
            "unsupported fusion strategy: {other}"
        ))),
    }
}

fn render_search_results(
    py: Python<'_>,
    results: &[SearchResult],
    query_payload: Bound<'_, PyDict>,
    rerank: Option<&Py<PyAny>>,
    rerank_k: usize,
    explain: bool,
    fusion: &str,
) -> PyResult<Py<PyList>> {
    let (results, _, _) = render_search_result_items(
        py,
        results,
        query_payload,
        rerank,
        rerank_k,
        explain,
        fusion,
    )?;
    let list = PyList::empty(py);
    for result in results {
        list.append(result.bind(py))?;
    }
    Ok(list.into())
}

fn render_search_result_items(
    py: Python<'_>,
    results: &[SearchResult],
    query_payload: Bound<'_, PyDict>,
    rerank: Option<&Py<PyAny>>,
    rerank_k: usize,
    explain: bool,
    fusion: &str,
) -> PyResult<(Vec<Py<PyDict>>, bool, usize)> {
    let mut rendered = Vec::with_capacity(results.len());
    for result in results {
        rendered.push(search_result_to_pydict(py, result, explain, fusion)?.unbind());
    }

    apply_rerank_hook(py, rendered, query_payload, rerank, rerank_k)
}

fn apply_rerank_hook(
    py: Python<'_>,
    results: Vec<Py<PyDict>>,
    query_payload: Bound<'_, PyDict>,
    rerank: Option<&Py<PyAny>>,
    rerank_k: usize,
) -> PyResult<(Vec<Py<PyDict>>, bool, usize)> {
    let Some(rerank) = rerank else {
        return Ok((results, false, 0));
    };
    if results.is_empty() {
        return Ok((results, false, 0));
    }

    let limit = if rerank_k == 0 {
        results.len()
    } else {
        rerank_k.min(results.len())
    };
    let candidates = PyList::empty(py);
    for result in results.iter().take(limit) {
        candidates.append(result.bind(py))?;
    }

    let reranked = rerank.bind(py).call1((query_payload, candidates))?;
    let reranked = reranked.downcast_into::<PyList>()?;
    let mut final_results = Vec::with_capacity(results.len());
    let mut seen = BTreeSet::new();

    for item in reranked.iter() {
        let dict = item.downcast_into::<PyDict>()?;
        let key = result_identity(&dict)?;
        if seen.insert(key) {
            final_results.push(dict.unbind());
        }
    }

    for result in results {
        let dict = result.bind(py);
        let key = result_identity(&dict)?;
        if seen.insert(key) {
            final_results.push(result);
        }
    }

    Ok((final_results, true, limit))
}

fn result_identity(result: &Bound<'_, PyDict>) -> PyResult<(String, String)> {
    let id = result
        .get_item("id")?
        .ok_or_else(|| PyValueError::new_err("rerank results must include 'id'"))?
        .extract::<String>()?;
    let namespace = result
        .get_item("namespace")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_default();
    Ok((namespace, id))
}

fn extract_metadata_values(value: &Bound<'_, PyAny>) -> PyResult<Vec<MetadataValue>> {
    let list = value.downcast::<PyList>()?;
    let mut values = Vec::with_capacity(list.len());
    for item in list.iter() {
        values.push(py_to_metadata_value(&item)?);
    }
    Ok(values)
}

fn parse_optional_namespace_item(
    item: Option<Bound<'_, PyAny>>,
    default_namespace: Option<&str>,
) -> PyResult<String> {
    let Some(item) = item else {
        return Ok(default_namespace.unwrap_or_default().to_owned());
    };

    if item.is_none() {
        return Ok(default_namespace.unwrap_or_default().to_owned());
    }

    item.extract::<String>()
}

fn to_py_error(error: vectlite::VectLiteError) -> PyErr {
    match &error {
        vectlite::VectLiteError::LockContention(_) => VectLiteLockError::new_err(error.to_string()),
        _ => VectLiteError::new_err(error.to_string()),
    }
}

fn closed_database_error() -> vectlite::VectLiteError {
    vectlite::VectLiteError::InvalidFormat("database is closed".to_owned())
}

fn parse_quantization_config(
    method: &str,
    rescore_multiplier: Option<usize>,
    num_sub_vectors: Option<usize>,
    num_centroids: Option<usize>,
    training_iterations: Option<usize>,
    dimension: usize,
) -> PyResult<QuantizationConfig> {
    let normalized = method.to_ascii_lowercase();
    match normalized.as_str() {
        "scalar" | "int8" => {
            let default = ScalarQuantizationConfig::default();
            Ok(QuantizationConfig::Scalar(ScalarQuantizationConfig {
                rescore_multiplier: rescore_multiplier.unwrap_or(default.rescore_multiplier),
            }))
        }
        "binary" => {
            let default = BinaryQuantizationConfig::default();
            Ok(QuantizationConfig::Binary(BinaryQuantizationConfig {
                rescore_multiplier: rescore_multiplier.unwrap_or(default.rescore_multiplier),
            }))
        }
        "product" | "pq" => {
            let default = ProductQuantizationConfig::default();
            Ok(QuantizationConfig::Product(ProductQuantizationConfig {
                num_sub_vectors: num_sub_vectors
                    .unwrap_or_else(|| default_product_num_sub_vectors(dimension)),
                num_centroids: num_centroids.unwrap_or(default.num_centroids),
                training_iterations: training_iterations.unwrap_or(default.training_iterations),
                rescore_multiplier: rescore_multiplier.unwrap_or(default.rescore_multiplier),
            }))
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown quantization method '{method}'. Expected: 'scalar', 'binary', or 'pq' (alias: 'product')"
        ))),
    }
}

fn parse_payload_index_type(name: &str) -> PyResult<PayloadIndexType> {
    PayloadIndexType::from_name(name).map_err(to_py_error)
}
