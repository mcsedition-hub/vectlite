use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString};
use vectlite::{
    Database as CoreDatabase, FusionStrategy, HybridSearchOptions, Metadata, MetadataFilter,
    MetadataValue, NamedVectors, Record, SearchOutcome, SearchResult, SparseVector,
    Store as CoreStore, WriteOperation,
};

create_exception!(vectlite, VectLiteError, PyException);

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

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyDict>>,
        vectors: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let mut database = self.write()?;
        database
            .insert_with_vectors_in_namespace(
                namespace.unwrap_or_default(),
                id,
                vector,
                vectors,
                sparse,
                metadata,
            )
            .map_err(to_py_error)
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyDict>>,
        vectors: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        let mut database = self.write()?;
        database
            .upsert_with_vectors_in_namespace(
                namespace.unwrap_or_default(),
                id,
                vector,
                vectors,
                sparse,
                metadata,
            )
            .map_err(to_py_error)
    }

    #[pyo3(signature = (records, namespace=None))]
    fn insert_many(
        &self,
        py: Python<'_>,
        records: Vec<Py<PyDict>>,
        namespace: Option<String>,
    ) -> PyResult<usize> {
        let records = parse_record_batch(py, records, namespace.as_deref())?;
        let mut database = self.write()?;
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
        let mut database = self.write()?;
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
        let mut database = self.write()?;
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
        let mut database = self.write()?;
        database
            .delete_in_namespace(&namespace.unwrap_or_default(), id)
            .map_err(to_py_error)
    }

    #[pyo3(signature = (ids, namespace=None))]
    fn delete_many(&self, ids: Vec<String>, namespace: Option<String>) -> PyResult<usize> {
        let mut database = self.write()?;
        database
            .delete_many_in_namespace(&namespace.unwrap_or_default(), ids)
            .map_err(to_py_error)
    }

    fn flush(&self) -> PyResult<()> {
        let mut database = self.write()?;
        database.flush().map_err(to_py_error)
    }

    fn compact(&self) -> PyResult<()> {
        let mut database = self.write()?;
        database.compact().map_err(to_py_error)
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

    #[pyo3(signature = (query=None, k=10, filter=None, namespace=None, all_namespaces=false, sparse=None, dense_weight=1.0, sparse_weight=1.0, fetch_k=0, mmr_lambda=None, vector_name=None, fusion="linear", rrf_k=60, explain=false, rerank=None, rerank_k=0, query_vectors=None, vector_weights=None))]
    fn search(
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
        fusion: &str,
        rrf_k: usize,
        explain: bool,
        rerank: Option<Py<PyAny>>,
        rerank_k: usize,
        query_vectors: Option<Py<PyDict>>,
        vector_weights: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyList>> {
        let query_payload = build_query_payload(
            py,
            query.as_deref(),
            sparse.as_ref().map(|dict| dict.bind(py)),
            namespace.as_deref(),
            all_namespaces,
            vector_name.as_deref(),
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

    #[pyo3(signature = (query=None, k=10, filter=None, namespace=None, all_namespaces=false, sparse=None, dense_weight=1.0, sparse_weight=1.0, fetch_k=0, mmr_lambda=None, vector_name=None, fusion="linear", rrf_k=60, explain=false, rerank=None, rerank_k=0, query_vectors=None, vector_weights=None))]
    fn search_with_stats(
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
        fusion: &str,
        rrf_k: usize,
        explain: bool,
        rerank: Option<Py<PyAny>>,
        rerank_k: usize,
        query_vectors: Option<Py<PyDict>>,
        vector_weights: Option<Py<PyDict>>,
    ) -> PyResult<Py<PyDict>> {
        let query_payload = build_query_payload(
            py,
            query.as_deref(),
            sparse.as_ref().map(|dict| dict.bind(py)),
            namespace.as_deref(),
            all_namespaces,
            vector_name.as_deref(),
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

    fn count(&self) -> PyResult<usize> {
        self.__len__()
    }

    fn namespaces(&self) -> PyResult<Vec<String>> {
        let database = self.read()?;
        Ok(database.namespaces())
    }

    fn transaction(&self) -> PyResult<PyTransaction> {
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

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyDict>>,
        vectors: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        self.stage(WriteOperation::Insert(Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
        }))
    }

    #[pyo3(signature = (id, vector, metadata=None, namespace=None, sparse=None, vectors=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Py<PyDict>>,
        namespace: Option<String>,
        sparse: Option<Py<PyDict>>,
        vectors: Option<Py<PyDict>>,
    ) -> PyResult<()> {
        let metadata = parse_metadata_dict(metadata.as_ref().map(|dict| dict.bind(py)))?;
        let sparse = parse_sparse_dict(sparse.as_ref().map(|dict| dict.bind(py)))?;
        let vectors = parse_named_vectors_dict(vectors.as_ref().map(|dict| dict.bind(py)))?;
        self.stage(WriteOperation::Upsert(Record {
            namespace: namespace.unwrap_or_default(),
            id: id.to_owned(),
            vector,
            vectors,
            sparse,
            metadata,
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
        self.inner
            .read()
            .map_err(|_| VectLiteError::new_err("database read lock poisoned"))
    }

    fn write(&self) -> PyResult<RwLockWriteGuard<'_, CoreDatabase>> {
        self.inner
            .write()
            .map_err(|_| VectLiteError::new_err("database write lock poisoned"))
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

#[pyfunction(name = "open", signature = (path, dimension=None, read_only=false))]
fn open_database(path: String, dimension: Option<usize>, read_only: bool) -> PyResult<PyDatabase> {
    let database = if read_only {
        if !Path::new(&path).exists() {
            return Err(VectLiteError::new_err(
                "cannot open non-existent database in read-only mode",
            ));
        }
        CoreDatabase::open_read_only(&path).map_err(to_py_error)?
    } else if Path::new(&path).exists() {
        match dimension {
            Some(dimension) => {
                CoreDatabase::open_or_create(&path, dimension).map_err(to_py_error)?
            }
            None => CoreDatabase::open(&path).map_err(to_py_error)?,
        }
    } else {
        let Some(dimension) = dimension else {
            return Err(VectLiteError::new_err(
                "dimension is required when creating a new database",
            ));
        };
        CoreDatabase::create(&path, dimension).map_err(to_py_error)?
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

        parsed.push(Record {
            namespace,
            id,
            vector,
            vectors,
            sparse,
            metadata,
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
    VectLiteError::new_err(error.to_string())
}
