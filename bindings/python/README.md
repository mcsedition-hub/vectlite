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

with vectlite.open("knowledge.vdb", dimension=384) as db:
    # Insert records with vectors, metadata, and sparse terms
    db.upsert("doc1", embedding, {"source": "blog", "title": "Auth Guide"})
    db.upsert("doc2", embedding2, {"source": "notes", "title": "Billing"})

    # Search with filters
    results = db.search(embedding_query, k=5, filter={"source": "blog"})

    # Query-free inspection
    print(db.count(filter={"source": "blog"}))
```

## Features

### Core

- **Single-file storage** -- one `.vdb` file per database, portable and easy to back up
- **Distance metrics** -- cosine (default), euclidean (L2), dot product, manhattan (L1) with SIMD acceleration
- **Dense vectors** -- automatic HNSW indexing with metric-aware distance functions
- **Sparse vectors** -- BM25-scored inverted index for keyword retrieval
- **Hybrid search** -- dense + sparse fusion with linear or RRF strategies
- **Vector quantization** -- scalar (int8, 4x), binary (32x), and product quantization (PQ) with 2-stage rescoring
- **Multi-vector / ColBERT** -- late interaction search with per-token MaxSim scoring and 2-bit quantization (~16x compression)
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
- **Payload indexes** -- keyword and numeric indexes on metadata fields accelerate filtered queries on large collections

### Data Management

- **Physical collections** -- `vectlite.open_store()` manages a directory of independent databases
- **Bulk ingestion** -- `bulk_ingest()` with deferred index rebuilds for fast imports
- **Listing & filtered counts** -- `list()` and `count(namespace=..., filter=...)` without a vector query
- **Delete by filter** -- remove matching records across a namespace slice in one call
- **Partial metadata updates** -- `update_metadata()` merges a patch without re-writing the vector or rebuilding indexes
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy
- **Backup / Restore** -- `db.backup(dir)` and `vectlite.restore(dir, path)` for full roundtrips
- **Read-only mode** -- `vectlite.open(path, read_only=True)` for safe concurrent readers
- **Explicit close** -- `db.close()` or `with vectlite.open(...) as db:` to release locks deterministically
- **Lock timeouts** -- `lock_timeout=` retries for bounded lock acquisition waits
- **Text analyzers** -- configurable tokenizer pipeline with stopwords, stemming, and n-grams
- **TTL / Expiry** -- `set_ttl()` / `clear_ttl()` or `ttl=` on insert/upsert; expired records auto-filtered from reads and GC'd on compact
- **Cursor-based pagination** -- `list_cursor()` for efficient iteration over large collections
- **LangChain integration** -- `vectlite.langchain.VectLiteVectorStore` (requires `langchain-core`)
- **LlamaIndex integration** -- `vectlite.llamaindex.VectLiteVectorStore` (requires `llama-index-core`)
- **Built-in embedders** -- `vectlite.embedders.openai()`, `.cohere()`, `.voyage()`, `.fastembed()`, `.sentence_transformer()`, `.ollama()`
- **ONNX reranker** -- `vectlite.rerankers.onnx_cross_encoder()` for zero-PyTorch reranking with onnxruntime
- **CLI** -- `vectlite stats`, `count`, `list`, `dump`, `search`, `compact`, `verify`, `bench`, `import-jsonl`, `import-csv`
- **Schema validation** -- `vectlite.schema.Schema({"price": "number"})` with typed fields, strict mode, and sidecar persistence

## Usage

### Distance Metrics

```python
# Default is cosine similarity
db = vectlite.open("knowledge.vdb", dimension=384)

# Choose a different metric at creation time
db = vectlite.open("knowledge.vdb", dimension=384, metric="euclidean")  # L2 distance
db = vectlite.open("knowledge.vdb", dimension=384, metric="dotproduct") # inner product
db = vectlite.open("knowledge.vdb", dimension=384, metric="manhattan")  # L1 distance

# Aliases: "l2", "dot", "ip", "l1"
print(db.metric)  # "euclidean"
```

The metric is persisted in the database file. Scores are always oriented so that **higher is better**.

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

### Payload Indexes

Create keyword or numeric indexes on metadata fields to accelerate filtered queries on large collections. Indexes are automatically used by `search()`, `count()`, and `list()`.

```python
# Create indexes on frequently-filtered fields
db.create_index("source", "keyword")   # string equality, $in
db.create_index("score", "numeric")    # range queries: $gt, $gte, $lt, $lte

# Filtered queries now use indexes automatically
count = db.count(filter={"source": "blog"})
results = db.search(query, k=10, filter={"score": {"$gte": 0.8}})

# Inspect and manage indexes
print(db.list_indexes())  # [("source", "keyword"), ("score", "numeric")]
db.drop_index("score")
```

### Snapshots & Backup

```python
db.snapshot("/backups/knowledge_2024.vdb")  # Self-contained copy
db.backup("/backups/full/")                 # Full backup with ANN sidecars

restored = vectlite.restore("/backups/full/", "restored.vdb")
```

### Read-Only Mode

```python
ro = vectlite.open("knowledge.vdb", read_only=True, lock_timeout=5.0)
results = ro.search(query, k=5)  # Reads work
ro.upsert(...)                    # Raises VectLiteError
```

### Listing, Counting, and Lifecycle

```python
db = vectlite.open("knowledge.vdb", dimension=384, lock_timeout=5.0)

records = db.list(namespace="docs", filter={"stale": False}, limit=20)
count = db.count(namespace="docs", filter={"source": "blog"})
deleted = db.delete_by_filter({"stale": True}, namespace="docs")

# Partial metadata update (merge patch -- only touches specified keys)
db.update_metadata("doc1", {"status": "reviewed", "score": 0.95})

db.close()
```

### Search Diagnostics

```python
outcome = db.search_with_stats(query, k=5, sparse=terms, explain=True)

print(outcome["stats"]["timings"])       # {"dense_us": 120, "sparse_us": 45, ...}
print(outcome["stats"]["used_ann"])      # True
print(outcome["results"][0]["explain"])  # Detailed scoring breakdown
```

### Vector Quantization

Reduce memory usage and accelerate search with quantized vectors. All methods use a 2-stage pipeline: fast quantized candidate selection followed by exact float32 rescoring.

```python
# Scalar quantization (int8) -- 4x memory reduction, minimal recall loss
db.enable_quantization("scalar")

# Binary quantization -- 32x memory reduction, best for normalized embeddings
db.enable_quantization("binary", rescore_multiplier=10)

# Product quantization -- configurable compression for very large datasets
db.enable_quantization("product", num_sub_vectors=16, num_centroids=256)

# Search is transparently accelerated
results = db.search(query_embedding, k=10)

# Check status
print(db.is_quantized)         # True
print(db.quantization_method)  # "scalar", "binary", or "product"

# Disable
db.disable_quantization()
```

Quantization parameters persist across reopens in a `.vdb.quant` sidecar file.

### Multi-Vector / ColBERT Search

Store token-level embeddings (ColBERT, ColPali) and search with MaxSim late interaction scoring.

```python
# Upsert with per-token ColBERT embeddings
db.upsert_multi_vectors(
    "doc1",
    dense_vector,
    {"colbert": [token_vec_1, token_vec_2, ...]},
    metadata={"source": "paper"},
)

# MaxSim search
results = db.search_multi_vector("colbert", query_token_vectors, k=10)

# Enable 2-bit quantization (~16x compression)
db.enable_multi_vector_quantization("colbert")

# Check and disable
print(db.is_multi_vector_quantized("colbert"))  # True
db.disable_multi_vector_quantization("colbert")
```

### TTL / Expiry

Records can automatically expire after a time-to-live. Expired records are transparently filtered from all reads and permanently removed on `compact()`.

```python
# Set TTL on insert/upsert (seconds)
db.upsert("session1", embedding, {"user": "alice"}, ttl=3600)  # expires in 1 hour

# Set/clear TTL on existing records
db.set_ttl("doc1", 86400)    # expire in 24 hours
db.clear_ttl("doc1")          # remove expiry, record lives forever

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
    page, cursor = db.list_cursor(limit=100, cursor=cursor)
    for record in page:
        process(record)
    if cursor is None:
        break

# Works with namespace and filter
page, cursor = db.list_cursor(namespace="docs", filter={"source": "blog"}, limit=50)
```

### Built-in Embedding Providers

Ready-to-use embedding functions for `upsert_text()` and `search_text()`. Each provider lazy-imports its SDK.

```python
from vectlite import embedders

# OpenAI
embed = embedders.openai("text-embedding-3-small")

# Cohere
embed = embedders.cohere("embed-english-v3.0")

# Voyage AI
embed = embedders.voyage("voyage-3")

# Local with FastEmbed (ONNX, no API calls)
embed = embedders.fastembed("BAAI/bge-small-en-v1.5")

# Local with SentenceTransformers (PyTorch)
embed = embedders.sentence_transformer("sentence-transformers/all-MiniLM-L6-v2")

# Local Ollama server
embed = embedders.ollama("nomic-embed-text")

# Use with text helpers
vectlite.upsert_text(db, "doc1", "Hello world", embed)
results = vectlite.search_text(db, "greeting", embed, k=5)
```

### ONNX Cross-Encoder Reranker

Zero-PyTorch reranking using `onnxruntime`. Same `RerankHook` interface as `cross_encoder()`.

```python
reranker = vectlite.rerankers.onnx_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

results = db.search(query, k=20, rerank=reranker)
```

Requires: `pip install onnxruntime tokenizers huggingface-hub`

### Schema Validation

Define typed schemas for metadata with clear error messages on type mismatch.

```python
from vectlite import schema

# Define a schema
s = schema.Schema({
    "price": "number",
    "title": "string",
    "tags": "array<string>",
    "author": {
        "name": "string",
        "age": "number",
    },
}, strict=True)  # strict=True rejects unknown fields

# Validate manually
s.validate({"price": 9.99, "title": "Hello"})          # OK
s.validate({"price": "free"})                            # raises SchemaError

# Auto-validate on every write
validated_db = schema.validated(db, s)
validated_db.upsert("doc1", vector, {"price": 9.99})    # OK
validated_db.upsert("doc2", vector, {"price": "free"})  # raises SchemaError

# Persist schema alongside the database
s.save(db)                  # writes .vdb.schema.json
loaded = schema.load(db)    # reads it back
```

Supported types: `string`, `number`, `integer`, `boolean`, `null`, `any`, `array`, `array<string>`, `array<number>`, `object`, nested objects.

### LangChain Integration

```python
from vectlite.langchain import VectLiteVectorStore
from langchain_openai import OpenAIEmbeddings

store = VectLiteVectorStore(
    path="my.vdb",
    embedding=OpenAIEmbeddings(),
    dimension=1536,
)

# Add documents
store.add_texts(["Hello world", "How to authenticate"])

# Search
results = store.similarity_search("greeting", k=3)
results_with_scores = store.similarity_search_with_score("greeting", k=3)

# Use with VectorStoreIndex, RetrievalQA, etc.
```

Requires: `pip install langchain-core`

### LlamaIndex Integration

```python
from vectlite.llamaindex import VectLiteVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

store = VectLiteVectorStore(path="my.vdb", dimension=1536)
storage_ctx = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)

query_engine = index.as_query_engine()
response = query_engine.query("How do I authenticate?")
```

Requires: `pip install llama-index-core`

### CLI

Full command-line interface. Install with `pip install vectlite`, then:

```bash
# Database stats
vectlite stats my.vdb

# Count records
vectlite count my.vdb --namespace blog

# List records
vectlite list my.vdb --limit 10 --filter '{"source": "blog"}'

# Dump all records as JSONL
vectlite dump my.vdb > backup.jsonl

# Search
vectlite search my.vdb --query '[1.0, 0.0, 0.5]' --k 5

# Import data
vectlite import-jsonl my.vdb data.jsonl --dimension 384
vectlite import-csv my.vdb data.csv --dimension 384 --vector-col embedding

# Maintenance
vectlite compact my.vdb
vectlite verify my.vdb

# Benchmark
vectlite bench my.vdb --queries 1000 --k 10
```

Also available as `python -m vectlite`.

### OpenTelemetry Integration

vectlite ships with optional OpenTelemetry tracing. When enabled, every
`search_text` and `search_text_with_stats` call is wrapped in a span carrying
semantic DB attributes and search-specific metrics. `opentelemetry-api` is
imported lazily -- it is **not** a runtime dependency.

```python
import vectlite

# Auto-detect: resolves a tracer from opentelemetry.trace if installed
tracer = vectlite.configure_opentelemetry()

# Or supply your own tracer
vectlite.configure_opentelemetry({"tracer": my_tracer})

# Custom tracer name (default: "vectlite")
vectlite.configure_opentelemetry({"tracer_name": "my-app"})

# Disable
vectlite.configure_opentelemetry(False)
```

When a tracer is active, each `search_text` / `search_text_with_stats` call
creates a `vectlite.search` span with these attributes:

| Attribute | Description |
|---|---|
| `db.system` | Always `"vectlite"` |
| `db.operation.name` | Always `"search"` |
| `vectlite.search.k` | Requested result count |
| `vectlite.search.namespace` | Target namespace |
| `vectlite.search.has_dense` | Whether a dense query vector was provided |
| `vectlite.search.has_sparse` | Whether sparse terms were provided |
| `vectlite.search.fusion` | Fusion strategy (`"linear"` or `"rrf"`) |
| `vectlite.search.used_ann` | Whether HNSW was used (set after completion) |
| `vectlite.search.result_count` | Number of results returned (set after completion) |
| `vectlite.search.total_us` | Total search time in microseconds (set after completion) |

If a search raises, the span records the exception and sets an error status
before re-raising.

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
| `db.delete_by_filter(filter, namespace=None)` | Delete all matching records in one filtered pass |
| `db.update_metadata(id, metadata, namespace=None)` | Merge a metadata patch into an existing record (no vector rewrite) |
| `db.set_ttl(id, ttl_secs, namespace=None)` | Set a time-to-live on a record (seconds from now) |
| `db.clear_ttl(id, namespace=None)` | Remove expiry from a record |

### Read Methods

| Method | Description |
|--------|-------------|
| `db.get(id, namespace=None)` | Get a single record by id |
| `db.search(query, k=10, ...)` | Search and return a list of results |
| `db.search_with_stats(query, k=10, ...)` | Search with detailed performance stats |
| `db.count(namespace=None, filter=None)` or `len(db)` | Count records, optionally scoped by namespace/filter |
| `db.list(namespace=None, filter=None, limit=0, offset=0)` | List records without issuing a vector query |
| `db.list_cursor(namespace=None, filter=None, limit=100, cursor=None)` | Cursor-based pagination (returns `(records, next_cursor)`) |
| `db.namespaces()` | List all namespaces |
| `db.dimension` | Vector dimension (property) |
| `db.path` | Database file path (property) |
| `db.metric` | Distance metric name: `"cosine"`, `"euclidean"`, `"dotproduct"`, or `"manhattan"` (property) |
| `db.read_only` | Whether the database is read-only (property) |

### Index Methods

| Method | Description |
|--------|-------------|
| `db.create_index(field, index_type)` | Create a payload index (`"keyword"` or `"numeric"`) on a metadata field |
| `db.drop_index(field)` | Remove an index |
| `db.list_indexes()` | List all active indexes as `[(field, type), ...]` |

### Quantization Methods

| Method | Description |
|--------|-------------|
| `db.enable_quantization(method, ...)` | Enable quantization (`"scalar"`, `"binary"`, or `"product"`) |
| `db.disable_quantization()` | Disable quantization and remove persisted parameters |
| `db.is_quantized` | Whether quantization is enabled (property) |
| `db.quantization_method` | Active method name or `None` (property) |

### Maintenance Methods

| Method | Description |
|--------|-------------|
| `db.compact()` | Fold WAL into snapshot and persist ANN indexes |
| `db.flush()` | Alias for `compact()` |
| `db.snapshot(dest)` | Create a self-contained `.vdb` copy |
| `db.backup(dest_dir)` | Full backup including ANN sidecar files |
| `db.transaction()` | Begin an atomic transaction (use as context manager) |
| `db.close()` | Flush pending state, release the file lock, and invalidate the handle |
| `with vectlite.open(...):` | Python context-manager form of automatic close |

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
