# vectlite

[![PyPI version](https://img.shields.io/pypi/v/vectlite.svg)](https://pypi.org/project/vectlite/)
[![npm version](https://img.shields.io/npm/v/vectlite.svg)](https://www.npmjs.com/package/vectlite)
[![Python versions](https://img.shields.io/pypi/pyversions/vectlite.svg)](https://pypi.org/project/vectlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-core-orange.svg)](https://www.rust-lang.org/)

**Embedded vector store for local-first AI applications.**

vectlite is a single-file vector database written in Rust with language bindings for Python and Node.js, plus experimental UniFFI bindings for Swift and Kotlin. Dense + sparse hybrid search, HNSW indexing, MongoDB-style metadata filters, transactions, crash-safe persistence, and file locking -- all in a portable `.vdb` file. No server, no Docker, no network calls.

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

```bash
npm install vectlite
```

Requires Node.js 18+. Pre-built binaries are available for macOS (x86_64, arm64), Linux (x86_64), and Windows (x86_64). Other platforms fall back to compiling from source (requires Rust/Cargo).

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
vectlite = { path = "crates/vectlite-core" }
```

### Swift (Experimental)

The Swift package lives in `bindings/swift` and uses a UniFFI-generated wrapper plus `VectLiteFFI.xcframework`.

```bash
cd bindings/swift
swift test
```

Rebuild the XCFramework after changing the Rust FFI surface:

```bash
cd bindings/swift
./build-xcframework.sh --release
```

### Kotlin/JVM (Experimental)

The Kotlin package lives in `bindings/kotlin` and compiles the UniFFI-generated source from `bindings/uniffi/generated/kotlin`.

```bash
cd bindings/kotlin
gradle test
```

The Kotlin binding uses JNA to load `libvectlite_uniffi`; the Gradle test task builds the native library and sets `uniffi.component.vectlite.libraryOverride` automatically.

## Quick Start (Python)

```python
import vectlite

with vectlite.open("knowledge.vdb", dimension=384) as db:
    db.upsert("doc1", embedding, {"source": "blog", "title": "Auth Guide"})
    db.upsert("doc2", embedding2, {"source": "notes", "title": "Billing"})

    results = db.search(query_embedding, k=5, filter={"source": "blog"})
    print(db.count(filter={"source": "blog"}))
```

## Quick Start (Node.js)

```js
const vectlite = require('vectlite')

const db = vectlite.open('knowledge.vdb', { dimension: 384 })

db.upsert('doc1', embedding, { source: 'blog', title: 'Auth Guide' })
db.upsert('doc2', embedding2, { source: 'notes', title: 'Billing' })

const results = db.search(queryEmbedding, { k: 5, filter: { source: 'blog' } })
// Equivalent object form:
const sameResults = db.search({ query: queryEmbedding, k: 5, filter: { source: 'blog' } })
console.log(db.count({ filter: { source: 'blog' } }))
db.close()
```

## Quick Start (Swift)

```swift
import VectLite

let db = try Database.openOrCreate(path: "knowledge.vdb", dimension: 384, metric: "cosine")

try db.upsert(
    id: "doc1",
    vector: embedding,
    metadataJson: #"{"source":"blog","title":"Auth Guide"}"#,
    namespace: nil,
    ttl: nil
)

let results = try db.search(
    query: queryEmbedding,
    k: 5,
    filterJson: #"{"source":"blog"}"#,
    namespace: nil,
    sparseJson: nil,
    fusion: nil,
    denseWeight: nil,
    sparseWeight: nil,
    mmrLambda: nil
)

try db.close()
```

## Quick Start (Kotlin)

```kotlin
import uniffi.vectlite.Database

val db = Database.openOrCreate("knowledge.vdb", 384u, "cosine")

db.upsert(
    "doc1",
    embedding,
    """{"source":"blog","title":"Auth Guide"}""",
    null,
    null,
)

val results = db.search(
    queryEmbedding,
    5u,
    """{"source":"blog"}""",
    null,
    null,
    null,
    null,
    null,
    null,
)

db.close()
```

## Features

### Storage & Durability

- **Single-file database** -- one `.vdb` file, portable and easy to back up
- **Crash-safe WAL** -- writes land in a write-ahead log, then checkpoint with `compact()`
- **Transactions** -- atomic batched writes with rollback on exception
- **File locking** -- advisory locks prevent corruption from concurrent access
- **Explicit close** -- release locks deterministically with `db.close()` or Python context managers
- **Lock timeouts** -- bounded retries when opening a locked database
- **Read-only mode** -- shared locks for safe concurrent readers
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy at any time
- **Backup / Restore** -- full backup with ANN sidecars and restore to a new path
- **Physical collections** -- `open_store()` manages a directory of independent databases
- **Bulk ingestion** -- `bulk_ingest()` with deferred index rebuilds for fast imports; `bulk_ingest_array()` for zero-copy NumPy / Float32Array ingest (10--30x faster)
- **TTL / Expiry** -- `set_ttl()` / `clear_ttl()` or `ttl=` on insert/upsert; expired records auto-filtered from reads and GC'd on compact
- **Cursor-based pagination** -- `list_cursor()` for efficient iteration over large collections without offset overhead
- **Schema validation** -- optional typed metadata schemas with sidecar persistence and validated write wrappers

### Search & Retrieval

- **Distance metrics** -- cosine (default), euclidean (L2), dot product (inner product), manhattan (L1) with SIMD acceleration
- **Dense vectors** -- automatic HNSW indexing with metric-aware distance functions and LSM-tree-style segmenting for bounded per-insert cost
- **Sparse vectors** -- BM25-scored inverted index for keyword retrieval
- **Hybrid search** -- dense + sparse fusion via linear combination or reciprocal rank fusion (RRF)
- **Vector quantization** -- scalar (int8, 4x), binary (32x), and product quantization (PQ) with 2-stage rescoring
- **Multi-vector / ColBERT** -- late interaction search with per-token MaxSim scoring and 2-bit quantization (~16x compression)
- **Named vectors** -- multiple vector spaces per record (`"title"`, `"body"`, ...)
- **Multi-vector queries** -- weighted search across vector spaces in a single call
- **MMR diversification** -- tunable relevance vs. diversity trade-off
- **Namespaces** -- logical isolation with per-namespace or cross-namespace search

### Metadata & Filters

- **Rich types** -- `str`, `int`, `float`, `bool`, `None`, `list`, `dict`
- **MongoDB-style operators** -- `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$exists`
- **Logical combinators** -- `$and`, `$or`, `$not`
- **Nested access** -- dot-path traversal (`author.name`), `$elemMatch`, `$size`
- **Filtered counts & listing** -- scan records with `count(...)` and `list(...)` without running a vector search
- **Delete by filter** -- remove whole slices of data with metadata filters plus optional namespace scoping
- **Partial metadata updates** -- `update_metadata()` merges a patch into existing metadata without re-writing the vector or rebuilding indexes
- **Payload indexes** -- keyword and numeric indexes on metadata fields accelerate filtered queries on large collections

### Reranking & Observability

- **Built-in rerankers** -- `text_match()`, `metadata_boost()`, `cross_encoder()`, `bi_encoder()`
- **ONNX cross-encoder** -- local reranking with `onnx_cross_encoder()` (ONNX Runtime, no PyTorch)
- **Composable** -- chain rerankers sequentially or with RRF via `compose()`
- **Search diagnostics** -- `search_with_stats()` returns timings, BM25 term scores, ANN stats
- **Explain mode** -- per-result scoring breakdown with ranks, matched terms, and rerank traces
- **OpenTelemetry** -- optional span-based tracing for search operations; `@opentelemetry/api` (Node) / `opentelemetry-api` (Python) loaded lazily, never a required dependency

### Text Processing

- **Text helpers** -- `upsert_text()` and `search_text()` handle embedding + sparse terms
- **Analyzers** -- configurable tokenizer pipeline with stopwords (en/fr), stemming (Snowball), n-grams, custom filters
- **Weighted fields** -- `sparse_terms_weighted()` for per-field term boosting
- **Embedding providers** -- plug-in factories for OpenAI, Cohere, Voyage, FastEmbed, Sentence Transformers, Ollama
- **CLI** -- `python -m vectlite` or `vectlite` command: stats, list, dump, search, compact, verify, bench, import-jsonl, import-csv

### Integrations

- **LangChain** -- `vectlite.langchain.VectLiteVectorStore` drop-in vector store
- **LlamaIndex** -- `vectlite.llamaindex.VectLiteVectorStore` drop-in vector store

## Python API

### Distance Metrics

```python
# Default is cosine similarity
db = vectlite.open("knowledge.vdb", dimension=384)

# Choose a different metric at creation time
db = vectlite.open("knowledge.vdb", dimension=384, metric="euclidean")  # L2 distance
db = vectlite.open("knowledge.vdb", dimension=384, metric="dotproduct") # inner product
db = vectlite.open("knowledge.vdb", dimension=384, metric="manhattan")  # L1 distance

# Aliases work too: "l2", "dot", "ip", "l1"
db = vectlite.open("knowledge.vdb", dimension=384, metric="l2")

# Check the active metric
print(db.metric)  # "euclidean"
```

The metric is persisted in the database file. Reopening an existing database automatically uses its original metric. Scores are always oriented so that **higher is better** (distance metrics are negated internally).

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

### Bulk Ingestion

For large imports, use `bulk_ingest()` instead of calling `upsert()` in a loop. It batches WAL writes (a single fsync at the end) and builds the HNSW
graph in parallel (Rayon-backed) once at the end.

> Since 0.11 single `insert()` calls also use an **incremental HNSW path** (no
> rebuild per record) and **lazy index persistence**, so streaming workloads
> are 10–50× faster than 0.10. For maximum throughput on very large streaming
> imports you can also relax the WAL durability mode: `on_flush` mode
> coalesces all fsyncs into a single one at `flush()` time, trading a bounded
> amount of recently-acked data on a crash for another 5–10× ingestion
> speedup.

```python
# Python — fastest single-insert streaming path
db.set_wal_sync_mode("on_flush")
for record in stream:
    db.insert(record.id, record.vector, record.metadata)
db.flush()  # one fsync makes the whole batch durable
db.set_wal_sync_mode("per_op")  # back to per-record durability

# Monitor tombstones (deleted records still in the HNSW graph)
live, dead = db.tombstone_stats()
if dead / max(live + dead, 1) > 0.2:
    db.compact()  # rebuilds the graph, clears tombstones
```

```javascript
// Node — same knobs, camelCase
db.setWalSyncMode('on_flush')
for (const r of stream) db.insert(r.id, r.vector, r.metadata)
db.flush()
db.setWalSyncMode('per_op')

const { live, dead } = db.tombstoneStats()
if (dead / Math.max(live + dead, 1) > 0.2) db.compact()
```

```python
records = [
    {
        "id": f"doc{i}",
        "vector": embeddings[i],
        "metadata": {"source": "corpus"},
        "sparse": vectlite.sparse_terms(texts[i]),
    }
    for i in range(len(texts))
]
db.bulk_ingest(records, batch_size=5000)
```

`upsert_many(records)` and `insert_many(records)` accept the same format and also rebuild indexes once.

#### Zero-Copy Bulk Ingest (NumPy)

For the fastest possible ingestion from Python, pass a NumPy array directly.
No per-record dict parsing, no per-element Python-to-Rust crossing, and the
GIL is released for the entire ingest. 10--30x faster than `bulk_ingest(list_of_dicts)`.

```python
import numpy as np

ids = [f"doc{i}" for i in range(len(embeddings))]
vectors = np.array(embeddings, dtype=np.float32)  # shape (N, D), C-contiguous
metadata = [{"source": "corpus"} for _ in ids]    # optional

db.bulk_ingest_array(ids, vectors, metadata=metadata, batch_size=5000)
```

Only writes the default dense vector. For named vectors, sparse terms, or
multi-vectors use `bulk_ingest(list_of_dicts)`.

#### Tuning the HNSW index

`bulk_ingest` accepts optional HNSW knobs so you can trade off recall, latency
and build time per workload:

```python
db.bulk_ingest(
    records,
    batch_size=5000,
    m=32,                        # max links per node (default 16)
    ef_construction=400,         # search width while building (default 200)
    ef_search=200,               # search width at query time (default: auto)
    segment_size_threshold=25000,# vectors per HNSW segment (default 50000)
)
```

You can also retune an existing database:

```python
db.set_index_config(m=32, ef_construction=400)  # rebuilds the ANN index
db.set_ef_search(200)                            # query-time only, no rebuild
db.set_ef_search(None)                           # back to auto

print(db.index_config())
# {'m': 32, 'ef_construction': 400, 'ef_search': 200,
#  'parallel_insert_threshold': 256, 'tombstone_rebuild_pct': 30,
#  'segment_size_threshold': 50000}

# Check how many HNSW segments exist
print(db.ann_segment_count())  # e.g. 3
```

Rule of thumb: raise `m` and `ef_construction` for higher recall at the cost
of build time; raise `ef_search` for higher recall at the cost of latency
without rebuilding the index. Lower `segment_size_threshold` to keep per-insert
cost bounded as the corpus grows (LSM-tree-style segmented HNSW).

### Collections

```python
store = vectlite.open_store("./my_collections")
products = store.create_collection("products", dimension=384)
products.upsert("p1", embedding, {"name": "Widget", "price": 9.99})

logs = store.open_or_create_collection("logs", dimension=128)
print(store.collections())  # ["logs", "products"]

products.close()
logs.close()
store.close()
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
ro = vectlite.open("knowledge.vdb", read_only=True, lock_timeout=5.0)
results = ro.search(query, k=5)  # Reads work
ro.upsert(...)                    # Raises VectLiteError
```

### Listing, Counts, and Lifecycle

```python
db = vectlite.open("knowledge.vdb", dimension=384, lock_timeout=5.0)

recent_docs = db.list(namespace="docs", filter={"stale": False}, limit=20)
doc_count = db.count(namespace="docs", filter={"source": "blog"})
deleted = db.delete_by_filter({"stale": True}, namespace="docs")

# Partial metadata update (merge patch -- only touches specified keys)
db.update_metadata("doc1", {"status": "reviewed", "score": 0.95})

db.close()
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

### Vector Quantization

Reduce in-memory candidate-index usage and accelerate search with quantized vectors. All methods use a 2-stage pipeline: fast quantized candidate selection followed by exact float32 rescoring.

```python
# Scalar quantization (int8) -- smaller in-memory candidate index, minimal recall loss
db.enable_quantization("scalar")

# Binary quantization -- smallest in-memory candidate index, best for normalized embeddings
db.enable_quantization("binary", rescore_multiplier=10)

# Product quantization -- "pq" and "product" are accepted case-insensitively
print(db.valid_num_sub_vectors())  # valid PQ partitions for this dimension
db.enable_quantization("pq", num_sub_vectors=16, num_centroids=256)

# Search works exactly the same -- quantization accelerates it transparently
results = db.search(query_embedding, k=10)

# Check quantization status
print(db.is_quantized())       # True
print(db.quantization_method)  # "scalar", "binary", or "product"

# Disable quantization
db.disable_quantization()
```

`rescore_multiplier` (default **10**) controls the number of quantized candidates rescored with exact float32 scoring: `k * rescore_multiplier`, capped at the collection size. Increase it to trade latency for recall.

For PQ, `num_sub_vectors` must divide the database dimension. If omitted, Vectlite chooses a compatible default; use `db.valid_num_sub_vectors()` to inspect all valid values.

Quantization does not shrink the `.vdb` file on disk. Vectlite keeps the original float32 vectors for exact rescoring and stores quantization parameters in a `.vdb.quant` sidecar file, so total disk footprint can increase slightly. The quantized index auto-rebuilds on inserts and upserts.

### Multi-Vector / ColBERT Search

Store token-level embeddings (ColBERT, ColPali) and search with MaxSim late interaction scoring.

```python
# Upsert a document with per-token embeddings
db.upsert_multi_vectors(
    "doc1",
    dense_vector,                              # standard dense embedding
    {"colbert": [token_vec_1, token_vec_2, ...]},  # token-level vectors
    metadata={"source": "paper"},
)

# MaxSim search: for each query token, find max cosine vs doc tokens, then sum
results = db.search_multi_vector("colbert", query_token_vectors, k=10)

# Enable 2-bit quantization for ColBERT tokens (~16x compression)
db.enable_multi_vector_quantization("colbert")

# Check status
print(db.is_multi_vector_quantized("colbert"))  # True

# Disable
db.disable_multi_vector_quantization("colbert")
```

Multi-vector quantization parameters persist in `.vdb.mvquant.<space>` sidecar files and auto-rebuild on mutation.

### TTL / Expiry

Records can automatically expire after a time-to-live. Expired records are transparently filtered from all reads and permanently removed on `compact()`.

```python
# Set TTL on insert/upsert (seconds)
db.upsert("session1", embedding, {"user": "alice"}, ttl=3600)  # expires in 1 hour

# Set/clear TTL on existing records
db.set_ttl("doc1", 86400)    # expire in 24 hours
db.clear_ttl("doc1")          # remove expiry

# Expired records are invisible to get/list/count/search
record = db.get("session1")   # None after TTL elapses

# compact() garbage-collects expired records from disk
db.compact()
```

### Cursor-Based Pagination

Efficiently iterate over large collections without offset overhead.

```python
# Paginate 100 records at a time
cursor = None
while True:
    page = db.list_cursor(limit=100, cursor=cursor)
    for record in page["records"]:
        process(record)
    cursor = page["cursor"]
    if cursor is None:
        break

# Works with namespace and filter
page = db.list_cursor(namespace="docs", filter={"source": "blog"}, limit=50)
```

### Embedding Providers

Built-in factories for popular embedding APIs. Each provider lazily imports its SDK.

```python
from vectlite import embedders

# OpenAI
embed = embedders.openai(model="text-embedding-3-small", api_key="sk-...")

# Sentence Transformers (local)
embed = embedders.sentence_transformer("all-MiniLM-L6-v2")

# Use with upsert_text / search_text
vectlite.upsert_text(db, "doc1", "Auth setup guide", embed, {"source": "docs"})
results = vectlite.search_text(db, "how to authenticate", embed, k=5)
```

Also available: `embedders.cohere()`, `embedders.voyage()`, `embedders.fastembed()`, `embedders.ollama()`.

### Schema Validation

Define typed metadata schemas for validation before writes.

```python
from vectlite import schema

s = schema.Schema({
    "price": "number",
    "title": "string",
    "tags": "array<string>",
    "author": {"name": "string", "age": "number"},
})

s.validate({"price": 9.99, "title": "Hello"})           # OK
s.validate({"price": "not a number"})                    # raises SchemaError

# Persist alongside the database
s.save(db)                    # writes knowledge.vdb.schema.json
loaded = schema.load(db)     # reads it back

# Validated wrapper auto-checks every write
vdb = schema.validated(db, s)
vdb.upsert("id", vector, {"price": 9.99})               # validates then writes
```

### OpenTelemetry (Python)

Optional tracing for search operations. `opentelemetry-api` is loaded lazily -- not a runtime dependency.

```python
import vectlite

# Auto-detect tracer from opentelemetry.trace if installed
tracer = vectlite.configure_opentelemetry()

# Or supply your own tracer / custom name
vectlite.configure_opentelemetry({"tracer": my_tracer})
vectlite.configure_opentelemetry({"tracer_name": "my-app"})

# Disable
vectlite.configure_opentelemetry(False)
```

### LangChain Integration

```python
from vectlite.langchain import VectLiteVectorStore
from langchain_openai import OpenAIEmbeddings

store = VectLiteVectorStore(
    path="knowledge.vdb",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
)
store.add_texts(["Auth guide", "Billing docs"], metadatas=[{"source": "docs"}] * 2)
results = store.similarity_search("authentication", k=5)
```

### LlamaIndex Integration

```python
from vectlite.llamaindex import VectLiteVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = VectLiteVectorStore(path="knowledge.vdb", dimension=1536)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

### CLI

```bash
# Database stats
vectlite stats knowledge.vdb

# List records (with optional filter)
vectlite list knowledge.vdb --limit 20 --filter '{"source": "blog"}'

# Stream all records via cursor pagination
vectlite dump knowledge.vdb > backup.jsonl

# Search (requires a JSON vector)
vectlite search knowledge.vdb --vector '[0.1, 0.2, ...]' --k 10

# Maintenance
vectlite compact knowledge.vdb
vectlite verify knowledge.vdb

# Benchmarking
vectlite bench knowledge.vdb --queries 1000

# Import data
vectlite import-jsonl knowledge.vdb data.jsonl --dimension 384
vectlite import-csv knowledge.vdb data.csv --dimension 384 --vector-column embedding
```

## Node.js API

### Distance Metrics (Node)

```js
// Default is cosine similarity
const db = vectlite.open('knowledge.vdb', { dimension: 384 })

// Choose a different metric at creation time
const db2 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'euclidean' })
const db3 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'dotproduct' })
const db4 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'manhattan' })

// Check the active metric
console.log(db2.metric) // "euclidean"
```

### Hybrid Search (Node)

```js
const vectlite = require('vectlite')

const db = vectlite.open('knowledge.vdb', { dimension: 384 })

db.upsert(
  'doc1',
  denseEmbedding,
  { source: 'docs', title: 'Auth Setup', text: 'How to configure SSO...' },
  { sparse: vectlite.sparseTerms('How to configure SSO authentication') },
)

const results = db.search(queryEmbedding, {
  k: 10,
  sparse: vectlite.sparseTerms('SSO authentication'),
  fusion: 'rrf',
  filter: { source: 'docs' },
  explain: true,
})
```

### Collections (Node)

```js
const store = vectlite.openStore('./my_collections')
const products = store.createCollection('products', 384)
products.upsert('p1', embedding, { name: 'Widget', price: 9.99 })

const logs = store.openOrCreateCollection('logs', 128)
console.log(store.collections()) // ["logs", "products"]

products.close()
logs.close()
store.close()
```

### Transactions (Node)

```js
const tx = db.transaction()
try {
  tx.upsert('doc1', emb1, { source: 'a' })
  tx.upsert('doc2', emb2, { source: 'b' })
  tx.delete('old_doc')
  tx.commit() // All operations commit atomically
} catch (err) {
  tx.rollback() // Roll back on error
  throw err
}
```

### Text Helpers (Node)

```js
async function run() {
  // embedFn can be sync or async
  await vectlite.upsertText(db, 'doc1', 'Auth setup guide', embedFn, { source: 'docs' })
  const results = await vectlite.searchText(db, 'how to authenticate', embedFn, { k: 5 })
}
```

### Listing, Counts, and Lifecycle (Node)

```js
const db = vectlite.open('knowledge.vdb', { dimension: 384, lockTimeout: 5 })

const docs = db.list({ namespace: 'docs', filter: { stale: false }, limit: 20 })
const count = db.count({ namespace: 'docs', filter: { source: 'blog' } })
const deleted = db.deleteByFilter({ stale: true }, { namespace: 'docs' })

// Partial metadata update (merge patch -- only touches specified keys)
db.updateMetadata('doc1', { status: 'reviewed', score: 0.95 })

db.close()
```

### Search Diagnostics (Node)

```js
const outcome = db.searchWithStats(query, {
  k: 5,
  sparse: terms,
  explain: true,
})

console.log(outcome.stats.timings)      // { dense_us: 120, sparse_us: 45, ... }
console.log(outcome.stats.used_ann)     // true
console.log(outcome.results[0].explain) // Detailed scoring breakdown
```

### Vector Quantization (Node)

```js
// Scalar quantization (int8) -- compact in-memory candidate index
db.enableQuantization('scalar')

// Binary quantization -- smallest in-memory candidate index
db.enableQuantization('binary', { rescoreMultiplier: 10 })

// Product quantization -- "pq" and "product" are accepted case-insensitively
console.log(db.validNumSubVectors()) // valid PQ partitions for this dimension
db.enableQuantization('pq', { numSubVectors: 16, numCentroids: 256 })

// Check status
console.log(db.isQuantized)         // true
console.log(db.quantizationMethod)  // "scalar", "binary", or "product"

// Disable
db.disableQuantization()
```

`rescoreMultiplier` (default **10**) controls the exact-rescore candidate budget (`k * rescoreMultiplier`, capped at collection size). Quantization keeps the original float32 vectors in the `.vdb` file and adds a `.vdb.quant` sidecar, so it is not a disk compression mode.

For PQ, `numSubVectors` must divide the database dimension. If omitted, Vectlite chooses a compatible default; use `db.validNumSubVectors()` to inspect all valid values.

### Multi-Vector / ColBERT Search (Node)

```js
// Upsert with per-token ColBERT embeddings
db.upsertMultiVectors('doc1', denseVector,
  { colbert: [tokenVec1, tokenVec2] },
  { metadata: { source: 'paper' } }
)

// MaxSim search
const results = db.searchMultiVector('colbert', queryTokenVectors)

// Enable 2-bit quantization (~16x compression)
db.enableMultiVectorQuantization('colbert')

// Check and disable
console.log(db.isMultiVectorQuantized('colbert'))  // true
db.disableMultiVectorQuantization('colbert')
```

### TTL / Expiry (Node)

```js
// Set TTL on insert/upsert (seconds)
db.upsert('session1', embedding, { user: 'alice' }, { ttl: 3600 }) // expires in 1 hour

// Set/clear TTL on existing records
db.setTtl('doc1', 86400)    // expire in 24 hours
db.clearTtl('doc1')          // remove expiry

// Expired records are invisible to get/list/count/search
const record = db.get('session1') // null after TTL elapses

// compact() garbage-collects expired records from disk
db.compact()
```

### Cursor-Based Pagination (Node)

```js
// Paginate 100 records at a time
let cursor = null
do {
  const page = db.listCursor({ limit: 100, cursor })
  for (const record of page.records) {
    process(record)
  }
  cursor = page.cursor
} while (cursor !== null)

// Works with namespace and filter
const page = db.listCursor({ namespace: 'docs', filter: { source: 'blog' }, limit: 50 })
```

### Zero-Copy Bulk Ingest (Node)

For the fastest possible ingestion from Node.js, pass a `Float32Array` directly.
The vector data is not JSON-serialised between JS and Rust -- napi-rs gives
Rust a reference into the underlying `ArrayBuffer`.

```js
const ids = embeddings.map((_, i) => `doc${i}`)
const vectorsFlat = new Float32Array(embeddings.flat())
const metadata = ids.map(() => ({ source: 'corpus' })) // optional

db.bulkIngestArray(ids, vectorsFlat, 384, { metadata, batchSize: 5000 })
```

### Async API (Node)

Non-blocking versions of heavy operations that run on the libuv threadpool.

```js
// Async search (returns a Promise)
const results = await db.searchAsync(queryEmbedding, { k: 10, filter: { source: 'blog' } })

// Async search with stats
const outcome = await db.searchWithStatsAsync(queryEmbedding, { k: 10 })

// Async maintenance
await db.flushAsync()
await db.compactAsync()

// Async bulk ingestion
const count = await db.bulkIngestAsync(records, { batchSize: 5000 })
```

### OpenTelemetry (Node)

Optional tracing for search operations. `@opentelemetry/api` is loaded lazily -- not a runtime dependency.

```js
const vectlite = require('vectlite')

// Auto-detect tracer from @opentelemetry/api if installed
const tracer = vectlite.configureOpenTelemetry()

// Or supply your own tracer / custom name
vectlite.configureOpenTelemetry({ tracer: myTracer })
vectlite.configureOpenTelemetry({ tracerName: 'my-app' })

// Disable
vectlite.configureOpenTelemetry(false)
```

## Swift API (Experimental)

The Swift binding wraps the UniFFI layer. All JSON parameters accept Swift string literals.

### Quantization (Swift)

```swift
// Scalar quantization (int8) -- default rescore_multiplier: 10
try db.enableQuantization(method: "scalar", optionsJson: nil)

// Binary quantization with custom rescore multiplier
try db.enableQuantization(method: "binary", optionsJson: #"{"rescoreMultiplier":20}"#)

// Product quantization
try db.enableQuantization(method: "pq", optionsJson: #"{"numSubVectors":16}"#)

// Check status
print(db.isQuantized())          // true
print(db.quantizationMethod())   // Optional("scalar")

// Disable
try db.disableQuantization()
```

### Transactions (Swift)

```swift
let ops = """
[
  {"op":"upsert","id":"doc1","vector":[1,0],"metadata":{"source":"a"}},
  {"op":"delete","id":"old_doc"}
]
"""
try db.transactionExecute(operationsJson: ops)
```

### Available Methods (Swift)

| Method | Description |
|---|---|
| `Database.openOrCreate(path:dimension:metric:)` | Open or create a database |
| `Database.openExisting(path:lockTimeout:)` | Open an existing database |
| `Database.openReadOnly(path:lockTimeout:)` | Open in read-only mode |
| `db.upsert(id:vector:metadataJson:namespace:ttl:)` | Insert or update a record |
| `db.insert(id:vector:metadataJson:namespace:ttl:)` | Insert (throws on duplicate) |
| `db.get(id:namespace:)` | Get a record by id |
| `db.search(query:k:filterJson:namespace:...)` | Vector search |
| `db.searchWithStats(query:k:filterJson:namespace:...)` | Search with stats JSON |
| `db.delete(id:namespace:)` | Delete a record |
| `db.deleteMany(ids:namespace:)` | Delete multiple records |
| `db.deleteByFilter(filterJson:namespace:)` | Delete by filter |
| `db.count(namespace:filterJson:)` | Count records |
| `db.list(namespace:filterJson:limit:offset:)` | List records |
| `db.enableQuantization(method:optionsJson:)` | Enable quantization |
| `db.disableQuantization()` | Disable quantization |
| `db.isQuantized()` | Check quantization status |
| `db.quantizationMethod()` | Active method name |
| `db.bulkIngest(recordsJson:batchSize:)` | Bulk import |
| `db.transactionExecute(operationsJson:)` | Atomic transaction |
| `db.compact()` / `db.flush()` | Persist WAL to snapshot |
| `db.snapshot(dest:)` / `db.backup(dest:)` | Backup |
| `db.close()` | Release resources |

## Kotlin API (Experimental)

The Kotlin binding wraps the same UniFFI layer. JSON parameters are plain strings.

### Quantization (Kotlin)

```kotlin
// Scalar quantization (int8) -- default rescoreMultiplier: 10
db.enableQuantization("scalar", null)

// Binary quantization with custom rescore multiplier
db.enableQuantization("binary", """{"rescoreMultiplier":20}""")

// Product quantization
db.enableQuantization("pq", """{"numSubVectors":16}""")

// Check status
println(db.isQuantized())          // true
println(db.quantizationMethod())   // "scalar"

// Disable
db.disableQuantization()
```

### Transactions (Kotlin)

```kotlin
db.transactionExecute("""[
  {"op":"upsert","id":"doc1","vector":[1,0],"metadata":{"source":"a"}},
  {"op":"delete","id":"old_doc"}
]""")
```

### Available Methods (Kotlin)

| Method | Description |
|---|---|
| `Database.openOrCreate(path, dimension, metric)` | Open or create a database |
| `Database.openExisting(path, lockTimeout?)` | Open an existing database |
| `Database.openReadOnly(path, lockTimeout?)` | Open in read-only mode |
| `db.upsert(id, vector, metadataJson, namespace?, ttl?)` | Insert or update a record |
| `db.insert(id, vector, metadataJson, namespace?, ttl?)` | Insert (throws on duplicate) |
| `db.get(id, namespace?)` | Get a record by id |
| `db.search(query, k, filterJson?, namespace?, ...)` | Vector search |
| `db.searchWithStats(query, k, filterJson?, namespace?, ...)` | Search with stats JSON |
| `db.delete(id, namespace?)` | Delete a record |
| `db.deleteMany(ids, namespace?)` | Delete multiple records |
| `db.deleteByFilter(filterJson, namespace?)` | Delete by filter |
| `db.count(namespace?, filterJson?)` | Count records |
| `db.list(namespace?, filterJson?, limit, offset)` | List records |
| `db.enableQuantization(method, optionsJson?)` | Enable quantization |
| `db.disableQuantization()` | Disable quantization |
| `db.isQuantized()` | Check quantization status |
| `db.quantizationMethod()` | Active method name |
| `db.bulkIngest(recordsJson, batchSize)` | Bulk import |
| `db.transactionExecute(operationsJson)` | Atomic transaction |
| `db.compact()` / `db.flush()` | Persist WAL to snapshot |
| `db.snapshot(dest)` / `db.backup(dest)` | Backup |
| `db.close()` | Release resources |

## Rust API

```rust
use vectlite::{Database, DistanceMetric};

fn main() -> vectlite::Result<()> {
    // Default cosine metric
    let mut db = Database::open_or_create("knowledge.vdb", 384)?;

    // Or choose a specific metric
    let mut db = Database::open_or_create_with_metric("knowledge.vdb", 384, DistanceMetric::Euclidean)?;
    println!("metric: {}", db.metric()); // "euclidean"

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

```text
crates/
  vectlite-core/    # Rust storage engine (the reusable core)
  vectlite-cli/     # CLI for smoke testing and file inspection
bindings/
  python/           # Python package (PyO3 + maturin)
  node/             # Node.js package (napi-rs)
  uniffi/           # Shared UniFFI crate and generated Swift/Kotlin bindings
  swift/            # Swift Package + XCFramework
  kotlin/           # Kotlin/JVM Gradle package
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
| `*.vdb.sparse` | BM25 sparse index sidecar (rebuilt on flush/compact, loaded on open) |
| `*.vdb.col.vector` | Contiguous vector arena sidecar (cache-friendly dense scan, rebuilt on compact) |
| `*.vdb.quant` | Quantization parameters (calibration ranges, PQ codebooks) |
| `*.vdb.mvquant.*` | Multi-vector quantization parameters (2-bit boundaries per space) |
| `*.vdb.pidx` | Payload index definitions (keyword/numeric index metadata) |
| `*.vdb.schema.json` | Optional typed metadata schema (JSON sidecar for schema validation) |
| `*.vdb.lock` | Advisory lock file for concurrency control |

The snapshot + WAL are the source of truth. ANN and quantization sidecars are acceleration artifacts that are regenerated if missing. Small collections (<128 records) use exact dense search.

## Language Roadmap

| Language | Status | Package |
|----------|--------|---------|
| Python | Available | [`pip install vectlite`](https://pypi.org/project/vectlite/) |
| Node.js | Available | [`npm install vectlite`](https://www.npmjs.com/package/vectlite) |
| Swift | Experimental | `bindings/swift` |
| Kotlin/JVM | Experimental | `bindings/kotlin` |

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

# Node.js development
cd bindings/node
npm test

# Swift UniFFI smoke tests
cd bindings/swift
swift test

# Kotlin UniFFI smoke tests
cd bindings/kotlin
gradle test
```

## Links

- [Official Documentation](https://vectlite.mcsedition.org/)
- [PyPI Package](https://pypi.org/project/vectlite/)
- [npm Package](https://www.npmjs.com/package/vectlite)
- [Changelog](https://github.com/mcsedition-hub/vectlite/blob/main/CHANGELOG.md)
- [Issue Tracker](https://github.com/mcsedition-hub/vectlite/issues)
- [Contribution Guide](https://github.com/mcsedition-hub/vectlite/blob/main/CONTRIBUTING.md)

## License

MIT
