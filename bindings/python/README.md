# vectlite

[![PyPI version](https://img.shields.io/pypi/v/vectlite.svg)](https://pypi.org/project/vectlite/)
[![Python versions](https://img.shields.io/pypi/pyversions/vectlite.svg)](https://pypi.org/project/vectlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Embedded vector store for local-first AI applications.**

vectlite is a single-file, zero-dependency vector database written in Rust with Python bindings. It gives you dense + sparse hybrid search, HNSW indexing, metadata filtering, transactions, and crash-safe persistence in a single `.vdb` file -- no server, no Docker, no network calls.

## Installation

```bash
pip install vectlite
```

Requires Python 3.9+. Pre-built wheels are available for macOS (x86_64, arm64), Linux (x86_64, aarch64), and Windows (x86_64).

## Quick Start

```python
import vectlite

# Create or open a database
db = vectlite.open("knowledge.vdb", dimension=384)

# Insert records with vectors, metadata, and sparse terms
db.upsert("doc1", embedding, {"source": "blog", "title": "Auth Guide"})
db.upsert("doc2", embedding2, {"source": "notes", "title": "Billing"})

# Search with filters
results = db.search(embedding_query, k=5, filter={"source": "blog"})

# Clean up
db.compact()
```

## Features

### Core

- **Single-file storage** -- one `.vdb` file per database, portable and easy to back up
- **Dense vectors** -- cosine similarity with automatic HNSW indexing for large collections
- **Sparse vectors** -- BM25-scored inverted index for keyword retrieval
- **Hybrid search** -- dense + sparse fusion with linear or RRF strategies
- **Rich metadata** -- `str`, `int`, `float`, `bool`, `None`, `list`, `dict` values
- **Crash-safe WAL** -- writes land in a write-ahead log first, then checkpoint with `compact()`
- **Transactions** -- atomic batched writes with `db.transaction()`
- **File locking** -- advisory locks prevent corruption from concurrent access

### Search & Retrieval

- **Metadata filters** -- MongoDB-style operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$exists`, `$and`, `$or`, `$not`
- **Nested filters** -- dot-path traversal (`author.name`), `$elemMatch`, `$size` on lists and dicts
- **Named vectors** -- multiple vector spaces per record (`vectors={"title": [...], "body": [...]}`)
- **Multi-vector queries** -- weighted search across vector spaces in a single call
- **MMR diversification** -- `mmr_lambda` controls relevance vs. diversity trade-off
- **Namespaces** -- logical isolation with per-namespace or cross-namespace search
- **Rerankers** -- built-in `text_match()`, `metadata_boost()`, `cross_encoder()`, `bi_encoder()`, composable with `compose()`
- **Observability** -- `search_with_stats()` returns timings, BM25 term scores, ANN stats, and per-result `explain` payloads

### Data Management

- **Physical collections** -- `vectlite.open_store()` manages a directory of independent databases
- **Bulk ingestion** -- `bulk_ingest()` with deferred index rebuilds for fast imports
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy
- **Backup / Restore** -- `db.backup(dir)` and `vectlite.restore(dir, path)` for full roundtrips
- **Read-only mode** -- `vectlite.open(path, read_only=True)` for safe concurrent readers
- **Text analyzers** -- configurable tokenizer pipeline with stopwords, stemming, and n-grams

## Usage

### Hybrid Search with Reranking

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)

# Upsert with dense + sparse vectors
db.upsert(
    "doc1",
    dense_embedding,
    {"source": "docs", "title": "Auth Setup", "text": "How to configure SSO..."},
    sparse=vectlite.sparse_terms("How to configure SSO authentication"),
)

# Hybrid search with reranking
results = db.search(
    query_embedding,
    k=10,
    sparse=vectlite.sparse_terms("SSO authentication"),
    fusion="rrf",
    filter={"source": "docs"},
    explain=True,
    rerank=vectlite.rerankers.compose(
        vectlite.rerankers.text_match(),
        vectlite.rerankers.metadata_boost("source", {"docs": 0.5}),
    ),
)

for result in results:
    print(result["id"], result["score"])
```

### Bulk Ingestion (Recommended for Large Imports)

For ingesting more than a few hundred records, use `bulk_ingest()` instead of calling `upsert()` in a loop. It writes records in WAL batches and rebuilds indexes only once at the end, making it orders of magnitude faster.

```python
records = [
    {
        "id": f"doc{i}",
        "vector": embeddings[i],
        "metadata": {"source": "corpus", "chunk": i},
        "sparse": vectlite.sparse_terms(texts[i]),  # optional
    }
    for i in range(len(texts))
]

count = db.bulk_ingest(records, batch_size=5000)
print(f"Ingested {count} records")
```

The `records` parameter is a `list[dict]` where each dict has keys:
- `id` (str, required) -- unique record identifier
- `vector` (list[float], required) -- dense embedding vector
- `metadata` (dict, optional) -- arbitrary metadata
- `sparse` (dict[str, float], optional) -- sparse terms from `sparse_terms()`
- `vectors` (dict[str, list[float]], optional) -- named vectors
- `namespace` (str, optional) -- namespace override per record

`upsert_many()` and `insert_many()` also accept the same `list[dict]` format and rebuild indexes once, but don't batch WAL writes internally.

### Collections

```python
store = vectlite.open_store("./my_collections")
products = store.create_collection("products", dimension=384)
products.upsert("p1", embedding, {"name": "Widget", "price": 9.99})

logs = store.open_or_create_collection("logs", dimension=128)
print(store.collections())  # ["logs", "products"]
```

### Transactions

```python
with db.transaction() as tx:
    tx.upsert("doc1", emb1, {"source": "a"})
    tx.upsert("doc2", emb2, {"source": "b"})
    tx.delete("old_doc")
# All operations commit atomically or roll back on exception
```

### Text Helpers

```python
# Handles embedding + sparse term generation for you
vectlite.upsert_text(db, "doc1", "Auth setup guide", embed_fn, {"source": "docs"})
results = vectlite.search_text(db, "how to authenticate", embed_fn, k=5)
```

### Analyzers

```python
analyzer = vectlite.analyzers.Analyzer().lowercase().stopwords("en").stemmer("english")
terms = analyzer.sparse_terms("How to authenticate users with SSO")
# Use with upsert: db.upsert("doc1", emb, meta, sparse=terms)
```

### Snapshots & Backup

```python
db.snapshot("/backups/knowledge_2024.vdb")  # Self-contained copy
db.backup("/backups/full/")                 # Full backup with ANN sidecars

restored = vectlite.restore("/backups/full/", "restored.vdb")
```

### Read-Only Mode

```python
ro = vectlite.open("knowledge.vdb", read_only=True)
results = ro.search(query, k=5)  # Reads work
ro.upsert(...)                    # Raises VectLiteError
```

### Search Diagnostics

```python
outcome = db.search_with_stats(query, k=5, sparse=terms, explain=True)

print(outcome["stats"]["timings"])       # {"dense_us": 120, "sparse_us": 45, ...}
print(outcome["stats"]["used_ann"])      # True
print(outcome["results"][0]["explain"])  # Detailed scoring breakdown
```

## Filter Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$eq` | `{"field": {"$eq": "value"}}` | Equal (also `{"field": "value"}`) |
| `$ne` | `{"field": {"$ne": "value"}}` | Not equal |
| `$gt` / `$gte` | `{"field": {"$gt": 5}}` | Greater than (or equal) |
| `$lt` / `$lte` | `{"field": {"$lt": 20}}` | Less than (or equal) |
| `$in` / `$nin` | `{"field": {"$in": ["a", "b"]}}` | In / not in set |
| `$contains` | `{"field": {"$contains": "auth"}}` | Substring match |
| `$exists` | `{"field": {"$exists": True}}` | Field presence |
| `$and` / `$or` | `{"$and": [{...}, {...}]}` | Logical combinators |
| `$not` | `{"$not": {...}}` | Logical negation |
| `$elemMatch` | `{"tags": {"$elemMatch": {"$eq": "rust"}}}` | Match list elements |
| `$size` | `{"tags": {"$size": 3}}` | List length |
| dot-path | `{"author.name": "Alice"}` | Nested field access |

## Database Methods Reference

### Write Methods

| Method | Description |
|--------|-------------|
| `db.upsert(id, vector, metadata, sparse=..., vectors=...)` | Insert or update a single record |
| `db.insert(id, vector, metadata, sparse=..., vectors=...)` | Insert a record (raises on duplicate id) |
| `db.upsert_many(records, namespace=None)` | Upsert a batch of records (single index rebuild) |
| `db.insert_many(records, namespace=None)` | Insert a batch (raises on duplicate ids) |
| `db.bulk_ingest(records, namespace=None, batch_size=10000)` | Fastest bulk import with batched WAL writes |
| `db.delete(id, namespace=None)` | Delete a single record |
| `db.delete_many(ids, namespace=None)` | Delete multiple records by id |

### Read Methods

| Method | Description |
|--------|-------------|
| `db.get(id, namespace=None)` | Get a single record by id |
| `db.search(query, k=10, ...)` | Search and return a list of results |
| `db.search_with_stats(query, k=10, ...)` | Search with detailed performance stats |
| `db.count()` or `len(db)` | Number of records in the database |
| `db.namespaces()` | List all namespaces |
| `db.dimension` | Vector dimension (property) |
| `db.path` | Database file path (property) |
| `db.read_only` | Whether the database is read-only (property) |

### Maintenance Methods

| Method | Description |
|--------|-------------|
| `db.compact()` | Fold WAL into snapshot and persist ANN indexes |
| `db.flush()` | Alias for `compact()` |
| `db.snapshot(dest)` | Create a self-contained `.vdb` copy |
| `db.backup(dest_dir)` | Full backup including ANN sidecar files |
| `db.transaction()` | Begin an atomic transaction (use as context manager) |

## How It Works

- Records are stored in a compact binary `.vdb` snapshot file
- Writes go through a crash-safe WAL (`.wal`) before being applied in memory
- `compact()` folds the WAL into the snapshot and persists HNSW sidecar files
- Dense search uses HNSW indexes (auto-built for collections above ~128 records)
- Sparse search uses an inverted index with BM25 scoring
- Hybrid fusion combines dense + sparse via linear combination or reciprocal rank fusion
- Advisory file locks (`flock`) prevent concurrent write corruption

## Links

- [Official Documentation](https://vectlite.mcsedition.org/)
- [GitHub Repository](https://github.com/mcsedition-hub/vectlite)
- [Issue Tracker](https://github.com/mcsedition-hub/vectlite/issues)
- [Changelog](https://github.com/mcsedition-hub/vectlite/blob/main/CHANGELOG.md)

## License

MIT
