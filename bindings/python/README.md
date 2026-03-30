# Python Binding

The first language target now has a concrete package layout:

- Rust extension crate in `bindings/python/src/lib.rs`
- Python package in `bindings/python/vectlite`
- packaging via `maturin`

## Local Development

```bash
cd bindings/python
maturin develop
pytest
```

## Install From PyPI

```bash
pip install vectlite
```

Release history lives in [`CHANGELOG.md`](https://github.com/mcsedition-hub/vectlite/blob/main/CHANGELOG.md).

## TestPyPI Release

From the repo root:

```bash
./scripts/publish_testpypi.sh
```

Then upload with a TestPyPI token:

```bash
export TEST_PYPI_API_TOKEN="pypi-..."
UPLOAD=1 ./scripts/publish_testpypi.sh
```

The full flow is documented in `docs/testpypi-release.md` in the repository.

## PyPI Release

From the repo root:

```bash
./scripts/publish_pypi.sh
```

Then upload with a PyPI token:

```bash
export PYPI_API_TOKEN="pypi-..."
UPLOAD=1 ./scripts/publish_pypi.sh
```

The full flow is documented in `docs/pypi-release.md` in the repository.

## API

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)
with db.transaction() as tx:
    tx.upsert(
        "doc1",
        embedding,
        {"source": "notes", "priority": 10, "title": "Auth setup"},
        namespace="notes",
        sparse={"auth": 1.0, "sso": 0.5},
        vectors={"title": title_embedding, "body": body_embedding},
    )
    tx.upsert_many(
        [
            {
                "id": "doc2",
                "vector": other_embedding,
                "sparse": {"auth": 0.7},
                "metadata": {"source": "notes", "text": "billing and auth notes"},
            }
        ],
        namespace="notes",
    )

record = db.get("doc1")
results = db.search(
    query,
    k=5,
    filter={"source": {"$ne": "blog"}, "priority": {"$gte": 5, "$lte": 20}},
    namespace="notes",
    sparse={"auth": 1.0},
    vector_name="title",
    dense_weight=1.0,
    sparse_weight=1.0,
    fusion="rrf",
    rrf_k=30,
    fetch_k=20,
    mmr_lambda=0.3,
    explain=True,
    rerank=vectlite.rerankers.compose(
        vectlite.rerankers.text_match(),
        vectlite.rerankers.metadata_boost("source", {"notes": 0.2}),
    ),
)

debug = db.search_with_stats(
    query,
    k=5,
    namespace="notes",
    sparse={"auth": 1.0},
    vector_name="title",
    fusion="rrf",
    fetch_k=20,
    mmr_lambda=0.3,
)

db.compact()
print(db.wal_path)
```

Supported metadata/filter value types are:

- `str`
- `int`
- `float`
- `bool`
- `None`
- `list`
- `dict`

Supported filter operators in the MVP are:

- equality with `{"field": "value"}`
- `{"field": {"$eq": "value"}}`
- `{"field": {"$contains": "auth"}}`
- `{"field": {"$gt": 5}}`
- `{"field": {"$gte": 5}}`
- `{"field": {"$lt": 20}}`
- `{"field": {"$lte": 20}}`
- `{"field": {"$ne": "value"}}`
- `{"field": {"$in": ["a", "b"]}}`
- `{"field": {"$nin": ["a", "b"]}}`
- `{"field": {"$exists": True}}`
- `{"$and": [...]}`
- `{"$or": [...]}`
- `{"$not": {...}}`

Batch helpers available on `Database`:

- `insert_many(records)`
- `upsert_many(records)`
- `delete_many(ids)`

Durability helpers available on `Database`:

- `transaction()` for atomic batched writes
- `wal_path` to inspect the write-ahead log path
- `compact()` / `flush()` to checkpoint the snapshot and clear the WAL

Dense vector helpers available on `Database`:

- `upsert(..., vectors={"title": [...], "body": [...]})`
- `search(..., vector_name="title")`
- `get(id)["vectors"]`

Namespace helpers available on `Database`:

- every CRUD/search method accepts `namespace=...`
- `search(..., all_namespaces=True)`
- `namespaces()`

Text helpers available at package level:

- `vectlite.upsert_text(db, id, text, embed, ...)`
- `vectlite.search_text(db, query, embed, ...)`
- `vectlite.search_text_with_stats(db, query, embed, ...)`
- `vectlite.sparse_terms(text)`

## Retrieval Quality

- sparse retrieval uses a real inverted index with BM25-style scoring
- `fetch_k` controls how many candidates are gathered before truncation
- `mmr_lambda` enables MMR diversification for dense, sparse, or hybrid search
- `fusion="linear"` and `fusion="rrf"` control dense+sparse score fusion
- `rerank(query, results)` can reorder the top candidates from Python
- `rerank_k` limits how many initial candidates are sent into the rerank hook
- `vectlite.rerankers.text_match()` boosts metadata text/title overlap
- `vectlite.rerankers.metadata_boost(field, boosts)` boosts metadata values
- `vectlite.rerankers.compose(...)` chains rerankers sequentially or with RRF
- `db.search_with_stats(...)` returns both results and search diagnostics
- `explain=True` adds per-result debug payloads with ranks, matched terms, and rerank traces
- lower `mmr_lambda` favors diversity more aggressively; higher values stay closer to raw relevance

## ANN Behavior

- dense and hybrid search use HNSW indexes when enough points are present
- named vector spaces get their own dense ANN indexes
- ANN sidecars are persisted on disk and reloaded on open when the manifest still matches the record set
- sparse-only search remains exact but uses the inverted index instead of a full scan
- small collections stay on exact dense search to avoid ANN overhead and low-cardinality edge cases
- writes land in a crash-safe WAL first, then `compact()` checkpoints back into the `.vdb` snapshot
- the `.vdb` snapshot plus `.wal` are the source of truth; ANN sidecars are acceleration artifacts
