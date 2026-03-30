# vectlite

`vectlite` is an embedded vector store for local-first applications. The repository is structured around a reusable Rust core, with the first real binding now targeting Python.

This first cut optimizes for portability and simplicity:

- embedded storage rooted at a single `.vdb` path
- reusable Rust core, no background service
- CRUD for vectors and metadata
- dense+sparse retrieval with metadata filtering
- deterministic binary persistence format

The storage/API boundary is intentionally small so the on-disk format and the search engine can keep evolving without breaking the developer-facing model.

Dense ANN now uses persisted HNSW sidecars, writes flow through a crash-safe WAL before compaction into the snapshot, sparse retrieval uses an inverted index with BM25 scoring, and Python can opt into named-vector search, RRF fusion, MMR diversification, transactions, rerank hooks, built-in rerankers, and search diagnostics.

## Repository Layout

- `crates/vectlite-core`: storage engine and public data model
- `crates/vectlite-cli`: local CLI for smoke testing and file inspection
- `bindings/python`: Python package and `PyO3` extension
- `bindings/node`: second packaging target
- `docs/language-rollout.md`: rollout and binding strategy

## Python API

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)
with db.transaction() as tx:
    tx.upsert(
        "doc1",
        [0.9, 0.1, 0.0],
        {"source": "blog", "priority": 10, "title": "Auth Flow"},
        namespace="docs",
        sparse={"auth": 1.0},
        vectors={"title": [1.0, 0.0, 0.0], "body": [0.0, 1.0, 0.0]},
    )
    tx.upsert_many(
        [
            {
                "id": "doc2",
                "vector": [0.8, 0.2, 0.0],
                "sparse": {"shipping": 1.0},
                "metadata": {"source": "notes", "text": "shipping notes"},
            }
        ],
        namespace="notes",
    )

results = db.search(
    [1.0, 0.0, 0.0],
    k=3,
    filter={"source": {"$ne": "notes"}, "priority": {"$gte": 5}},
    all_namespaces=True,
    sparse={"auth": 1.0},
    vector_name="title",
    fusion="rrf",
    rrf_k=30,
    fetch_k=12,
    mmr_lambda=0.3,
    explain=True,
    rerank=vectlite.rerankers.text_match(),
)

debug = db.search_with_stats(
    [1.0, 0.0, 0.0],
    k=3,
    sparse={"auth": 1.0},
    fusion="rrf",
    fetch_k=12,
    mmr_lambda=0.3,
)

db.compact()
```

Packaging and local dev details live in `bindings/python/README.md`.
The TestPyPI release flow lives in `docs/testpypi-release.md`.

## Rust API

```rust
use vectlite::{Database, Metadata, MetadataFilter, MetadataValue, SearchOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut db = Database::open_or_create("knowledge.vdb", 3)?;

    let mut metadata = Metadata::new();
    metadata.insert("source".into(), MetadataValue::from("blog"));
    metadata.insert("title".into(), MetadataValue::from("auth flow"));

    db.upsert("doc1", vec![0.9, 0.1, 0.0], metadata)?;

    let results = db.search(
        &[1.0, 0.0, 0.0],
        SearchOptions {
            top_k: 3,
            filter: Some(MetadataFilter::contains("title", "auth")),
        },
    )?;

    for result in results {
        println!("{} -> {}", result.id, result.score);
    }

    Ok(())
}
```

## CLI

```bash
cargo run -p vectlite-cli -- init demo.vdb 3
cargo run -p vectlite-cli -- insert demo.vdb doc1 0.9,0.1,0.0 source=blog,title=auth
cargo run -p vectlite-cli -- insert demo.vdb doc2 0.0,1.0,0.0 source=notes,title=shipping
cargo run -p vectlite-cli -- search demo.vdb 1.0,0.0,0.0 5 title~auth
```

## Current Format

Each database now consists of:

- a `.vdb` snapshot with a fixed magic header, version, vector dimension, and records
- a `.wal` write-ahead log for crash-safe write recovery
- `.ann` / `.hnsw.*` sidecars for persisted dense ANN indexes

The snapshot plus WAL are the source of truth. ANN sidecars are acceleration artifacts that can be regenerated. Small collections still fall back to exact dense search.

## Delivery Order

1. Python binding with `PyO3` and `maturin`
2. Node binding with `napi-rs`
3. Framework bindings on top of a stable FFI layer
4. Thin Swift and Kotlin wrappers for mobile-native packaging
