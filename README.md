# vectlite

[![PyPI version](https://img.shields.io/pypi/v/vectlite.svg)](https://pypi.org/project/vectlite/)
[![Python versions](https://img.shields.io/pypi/pyversions/vectlite.svg)](https://pypi.org/project/vectlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-core-orange.svg)](https://www.rust-lang.org/)

**Embedded vector store for local-first AI applications.**

vectlite is a single-file vector database written in Rust with language bindings for Python today and Node from source today. Dense + sparse hybrid search, HNSW indexing, MongoDB-style metadata filters, transactions, crash-safe persistence, and file locking -- all in a portable `.vdb` file. No server, no Docker, no network calls.

## Install

### Python

```bash
pip install vectlite
```

Pre-built wheels for macOS (x86_64, arm64), Linux (x86_64, aarch64), and Windows (x86_64). Requires Python 3.9+.

Install from source:

```bash
pip install git+https://github.com/mcsedition-hub/vectlite.git#subdirectory=bindings/python
```

### Node.js

The Node binding is implemented in-repo and currently builds from source:

```bash
git clone https://github.com/mcsedition-hub/vectlite.git
cd vectlite/bindings/node
npm test
```

This compiles the native addon with `napi-rs` and runs the smoke tests. An npm publication is not live yet.

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
vectlite = { path = "crates/vectlite-core" }
```

## Quick Start (Python)

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)

db.upsert("doc1", embedding, {"source": "blog", "title": "Auth Guide"})
db.upsert("doc2", embedding2, {"source": "notes", "title": "Billing"})

results = db.search(query_embedding, k=5, filter={"source": "blog"})
db.compact()
```

## Features

### Storage & Durability

- **Single-file database** -- one `.vdb` file, portable and easy to back up
- **Crash-safe WAL** -- writes land in a write-ahead log, then checkpoint with `compact()`
- **Transactions** -- atomic batched writes with rollback on exception
- **File locking** -- advisory locks prevent corruption from concurrent access
- **Read-only mode** -- shared locks for safe concurrent readers
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy at any time
- **Backup / Restore** -- full backup with ANN sidecars and restore to a new path
- **Physical collections** -- `open_store()` manages a directory of independent databases
- **Bulk ingestion** -- `bulk_ingest()` with deferred index rebuilds for fast imports

### Search & Retrieval

- **Dense vectors** -- cosine similarity with automatic HNSW indexing
- **Sparse vectors** -- BM25-scored inverted index for keyword retrieval
- **Hybrid search** -- dense + sparse fusion via linear combination or reciprocal rank fusion (RRF)
- **Named vectors** -- multiple vector spaces per record (`"title"`, `"body"`, ...)
- **Multi-vector queries** -- weighted search across vector spaces in a single call
- **MMR diversification** -- tunable relevance vs. diversity trade-off
- **Namespaces** -- logical isolation with per-namespace or cross-namespace search

### Metadata & Filters

- **Rich types** -- `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **MongoDB-style operators** -- `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$exists`
- **Logical combinators** -- `$and`, `$or`, `$not`
- **Nested access** -- dot-path traversal (`author.name`), `$elemMatch`, `$size`

### Reranking & Observability

- **Built-in rerankers** -- `text_match()`, `metadata_boost()`, `cross_encoder()`, `bi_encoder()`
- **Composable** -- chain rerankers sequentially or with RRF via `compose()`
- **Search diagnostics** -- `search_with_stats()` returns timings, BM25 term scores, ANN stats
- **Explain mode** -- per-result scoring breakdown with ranks, matched terms, and rerank traces

### Text Processing

- **Text helpers** -- `upsert_text()` and `search_text()` handle embedding + sparse terms
- **Analyzers** -- configurable tokenizer pipeline with stopwords (en/fr), stemming (Snowball), n-grams, custom filters
- **Weighted fields** -- `sparse_terms_weighted()` for per-field term boosting

## Python API

### Hybrid Search with Reranking

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)

db.upsert(
    "doc1",
    dense_embedding,
    {"source": "docs", "title": "Auth Setup", "text": "How to configure SSO..."},
    sparse=vectlite.sparse_terms("How to configure SSO authentication"),
    vectors={"title": title_embedding, "body": body_embedding},
)

results = db.search(
    query_embedding,
    k=10,
    sparse=vectlite.sparse_terms("SSO authentication"),
    fusion="rrf",
    filter={"source": "docs", "author.level": {"$gte": 3}},
    explain=True,
    rerank=vectlite.rerankers.compose(
        vectlite.rerankers.text_match(),
        vectlite.rerankers.metadata_boost("source", {"docs": 0.5}),
    ),
)
```

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
# Commits atomically; rolls back on exception
```

### Snapshots & Backup

```python
db.snapshot("/backups/knowledge_2024.vdb")
db.backup("/backups/full/")
restored = vectlite.restore("/backups/full/", "restored.vdb")
```

### Read-Only Mode

```python
ro = vectlite.open("knowledge.vdb", read_only=True)
results = ro.search(query, k=5)  # Reads work
ro.upsert(...)                    # Raises VectLiteError
```

### Analyzers

```python
analyzer = vectlite.analyzers.Analyzer().lowercase().stopwords("en").stemmer("english")
terms = analyzer.sparse_terms("How to authenticate users with SSO")
```

### Search Diagnostics

```python
outcome = db.search_with_stats(query, k=5, sparse=terms, explain=True)
print(outcome["stats"]["timings"])       # {"dense_us": 120, "sparse_us": 45, ...}
print(outcome["results"][0]["explain"])  # Detailed scoring breakdown
```

## Rust API

```rust
use vectlite::Database;

fn main() -> vectlite::Result<()> {
    let mut db = Database::open_or_create("knowledge.vdb", 384)?;

    let mut metadata = vectlite::Metadata::new();
    metadata.insert("source".into(), "blog".into());

    db.upsert("doc1", vec![0.9, 0.1, 0.0], metadata)?;

    let results = db.search(
        &[1.0, 0.0, 0.0],
        vectlite::SearchOptions { top_k: 5, filter: None },
    )?;

    for r in results {
        println!("{} -> {:.3}", r.id, r.score);
    }

    // Snapshots and backup work from Rust too
    db.snapshot("backup.vdb")?;
    db.compact()?;

    Ok(())
}
```

## CLI

```bash
cargo run -p vectlite-cli -- init demo.vdb 3
cargo run -p vectlite-cli -- insert demo.vdb doc1 0.9,0.1,0.0 source=blog,title=auth
cargo run -p vectlite-cli -- search demo.vdb 1.0,0.0,0.0 5 title~auth
```

## Repository Layout

```
crates/
  vectlite-core/    # Rust storage engine (the reusable core)
  vectlite-cli/     # CLI for smoke testing and file inspection
bindings/
  python/           # Python package (PyO3 + maturin)
  node/             # Node binding (napi-rs, source build today)
scripts/            # Release and CI scripts
.github/workflows/  # CI: cross-platform wheel builds, tests
```

## Storage Format

Each database consists of:

| File | Purpose |
|------|---------|
| `*.vdb` | Binary snapshot: magic header, version, dimension, records |
| `*.vdb.wal` | Write-ahead log for crash-safe recovery |
| `*.vdb.ann.*` | HNSW sidecar files (acceleration artifacts, regenerated if missing) |
| `*.vdb.lock` | Advisory lock file for concurrency control |

The snapshot + WAL are the source of truth. ANN sidecars are acceleration artifacts. Small collections (<128 records) use exact dense search.

## Language Roadmap

| Language | Status | Package |
|----------|--------|---------|
| Python | Available | [`pip install vectlite`](https://pypi.org/project/vectlite/) |
| Node | Available from source | `bindings/node` with `napi-rs` |
| Swift | Planned | After FFI layer stabilizes |
| Kotlin | Planned | After FFI layer stabilizes |

## Contributing

Found a bug or have a feature request? [Open an issue](https://github.com/mcsedition-hub/vectlite/issues).
Before opening a PR, read [CONTRIBUTING.md](https://github.com/mcsedition-hub/vectlite/blob/main/CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](https://github.com/mcsedition-hub/vectlite/blob/main/CODE_OF_CONDUCT.md).

### Development Setup

```bash
git clone https://github.com/mcsedition-hub/vectlite.git
cd vectlite

# Rust tests
cargo test

# Python development
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop -m bindings/python/Cargo.toml
pytest bindings/python/tests/
```

## Links

- [PyPI Package](https://pypi.org/project/vectlite/)
- [Changelog](https://github.com/mcsedition-hub/vectlite/blob/main/CHANGELOG.md)
- [Issue Tracker](https://github.com/mcsedition-hub/vectlite/issues)
- [Contribution Guide](https://github.com/mcsedition-hub/vectlite/blob/main/CONTRIBUTING.md)

## License

MIT
