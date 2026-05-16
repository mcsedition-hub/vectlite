# vectlite

[![npm version](https://img.shields.io/npm/v/vectlite.svg)](https://www.npmjs.com/package/vectlite)
[![Node versions](https://img.shields.io/node/v/vectlite.svg)](https://www.npmjs.com/package/vectlite)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Embedded vector store for local-first AI applications.

**vectlite** is a single-file, zero-dependency vector database written in Rust with Node.js bindings. It gives you dense + sparse hybrid search, HNSW indexing, metadata filtering, transactions, and crash-safe persistence in a single `.vdb` file -- no server, no Docker, no network calls.

## Installation

```bash
npm install vectlite
```

Requires Node.js 18+. Pre-built binaries are available for macOS (x86_64, arm64), Linux (x86_64), and Windows (x86_64). Other platforms fall back to compiling from source (requires Rust/Cargo).

## Quick Start

```js
const vectlite = require('vectlite')

// Create or open a database
const db = vectlite.open('knowledge.vdb', { dimension: 384 })

// Insert records with vectors, metadata, and sparse terms
db.upsert('doc1', embedding, { source: 'blog', title: 'Auth Guide' })
db.upsert('doc2', embedding2, { source: 'notes', title: 'Billing' })

// Search with filters
const results = db.search(embeddingQuery, { k: 5, filter: { source: 'blog' } })

// Query-free inspection
console.log(db.count({ filter: { source: 'blog' } }))

// Clean up
db.close()
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
- **Rich metadata** -- string, number, boolean, null, array, and nested object values
- **Crash-safe WAL** -- writes land in a write-ahead log first, then checkpoint with `compact()`
- **Transactions** -- atomic batched writes with `db.transaction()`
- **File locking** -- advisory locks prevent corruption from concurrent access

### Search & Retrieval

- **Metadata filters** -- MongoDB-style operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$exists`, `$and`, `$or`, `$not`
- **Nested filters** -- dot-path traversal (`author.name`), `$elemMatch`, `$size` on arrays and objects
- **Named vectors** -- multiple vector spaces per record (`vectors: { title: [...], body: [...] }`)
- **Multi-vector queries** -- weighted search across vector spaces in a single call
- **MMR diversification** -- `mmrLambda` controls relevance vs. diversity trade-off
- **Namespaces** -- logical isolation with per-namespace or cross-namespace search
- **Observability** -- `searchWithStats()` returns timings, BM25 term scores, ANN stats, and per-result explain payloads
- **Payload indexes** -- keyword and numeric indexes on metadata fields accelerate filtered queries on large collections

### Data Management

- **Physical collections** -- `vectlite.openStore()` manages a directory of independent databases
- **Bulk ingestion** -- `bulkIngest()` with Rayon-parallel HNSW build, coalesced WAL fsync, and tunable `m` / `efConstruction` / `efSearch`
- **Listing & filtered counts** -- `list()` and `count({ namespace, filter })` without a vector query
- **Delete by filter** -- `deleteByFilter()` for bulk deletion by metadata filter
- **Partial metadata updates** -- `updateMetadata()` merges a patch without re-writing the vector or rebuilding indexes
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy
- **Backup / Restore** -- `db.backup(dir)` and `vectlite.restore(dir, path)` for full roundtrips
- **Read-only mode** -- `vectlite.open(path, { readOnly: true })` for safe concurrent readers
- **Explicit close** -- `db.close()` to release locks deterministically
- **Lock timeouts** -- `lockTimeout` for bounded lock acquisition waits
- **TTL / Expiry** -- `setTtl()` / `clearTtl()` or `ttl` option on insert/upsert; expired records auto-filtered from reads and GC'd on compact
- **Cursor-based pagination** -- `listCursor()` for efficient iteration over large collections
- **Async API** -- `searchAsync()`, `compactAsync()`, `flushAsync()`, `bulkIngestAsync()` run on the libuv threadpool

## Usage

### Distance Metrics

```js
// Default is cosine similarity
const db = vectlite.open('knowledge.vdb', { dimension: 384 })

// Choose a different metric at creation time
const db2 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'euclidean' })
const db3 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'dotproduct' })
const db4 = vectlite.open('knowledge.vdb', { dimension: 384, metric: 'manhattan' })

// Aliases: 'l2', 'dot', 'ip', 'l1'
console.log(db2.metric) // "euclidean"
```

The metric is persisted in the database file. Scores are always oriented so that **higher is better**.

### Hybrid Search

```js
const vectlite = require('vectlite')

const db = vectlite.open('knowledge.vdb', { dimension: 384 })

// Upsert with dense + sparse vectors
db.upsert(
  'doc1',
  denseEmbedding,
  { source: 'docs', title: 'Auth Setup', text: 'How to configure SSO...' },
  { sparse: vectlite.sparseTerms('How to configure SSO authentication') },
)

// Hybrid search
const results = db.search(queryEmbedding, {
  k: 10,
  sparse: vectlite.sparseTerms('SSO authentication'),
  fusion: 'rrf',
  filter: { source: 'docs' },
  explain: true,
})

for (const result of results) {
  console.log(result.id, result.score)
}
```

### Collections

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

### Transactions

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

### Text Helpers

```js
async function run() {
  // embedFn can be sync or async
  await vectlite.upsertText(db, 'doc1', 'Auth setup guide', embedFn, { source: 'docs' })
  const results = await vectlite.searchText(db, 'how to authenticate', embedFn, { k: 5 })
}
```

### Snapshots & Backup

```js
db.snapshot('/backups/knowledge_2024.vdb') // Self-contained copy
db.backup('/backups/full/')                // Full backup with ANN sidecars

const restored = vectlite.restore('/backups/full/', 'restored.vdb')
```

### Read-Only Mode

```js
const ro = vectlite.open('knowledge.vdb', { readOnly: true, lockTimeout: 5 })
const results = ro.search(query, { k: 5 }) // Reads work
ro.upsert(...)                              // Throws VectLiteError
```

### Listing, Counting, and Lifecycle

```js
const db = vectlite.open('knowledge.vdb', { dimension: 384, lockTimeout: 5 })

const records = db.list({ namespace: 'docs', filter: { stale: false }, limit: 20 })
const count = db.count({ namespace: 'docs', filter: { source: 'blog' } })
const deleted = db.deleteByFilter({ stale: true }, { namespace: 'docs' })

// Partial metadata update (merge patch -- only touches specified keys)
db.updateMetadata('doc1', { status: 'reviewed', score: 0.95 })

db.close()
```

### Search Diagnostics

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

### Payload Indexes

Create keyword or numeric indexes on metadata fields to accelerate filtered queries on large collections. Indexes are automatically used by `search()`, `count()`, and `list()`.

```js
// Create indexes on frequently-filtered fields
db.createIndex('source', 'keyword')   // string equality, $in
db.createIndex('score', 'numeric')    // range queries: $gt, $gte, $lt, $lte

// Filtered queries now use indexes automatically
const count = db.count({ filter: { source: 'blog' } })
const results = db.search(query, { k: 10, filter: { score: { $gte: 0.8 } } })

// Inspect and manage indexes
console.log(db.listIndexes())  // [{ field: 'source', type: 'keyword' }, ...]
db.dropIndex('score')
```

### Vector Quantization

Reduce in-memory candidate-index usage and accelerate search with quantized vectors. All methods use a 2-stage pipeline: fast quantized candidate selection followed by exact float32 rescoring.

```js
// Scalar quantization (int8) -- smaller in-memory candidate index, minimal recall loss
db.enableQuantization('scalar')

// Binary quantization -- smallest in-memory candidate index, best for normalized embeddings
db.enableQuantization('binary', { rescoreMultiplier: 10 })

// Product quantization -- "pq" and "product" are accepted case-insensitively
console.log(db.validNumSubVectors()) // valid PQ partitions for this dimension
db.enableQuantization('pq', { numSubVectors: 16, numCentroids: 256 })

// Search works exactly the same -- quantization accelerates it transparently
const results = db.search(queryEmbedding, { k: 10 })
const sameResults = db.search({ query: queryEmbedding, k: 10 })

// Check quantization status
console.log(db.isQuantized)         // true
console.log(db.quantizationMethod)  // "scalar", "binary", or "product"

// Disable quantization
db.disableQuantization()
```

`rescoreMultiplier` (default **10**) controls the number of quantized candidates rescored with exact float32 scoring: `k * rescoreMultiplier`, capped at the collection size. Increase it to trade latency for recall.

For PQ, `numSubVectors` must divide the database dimension. If omitted, Vectlite chooses a compatible default; use `db.validNumSubVectors()` to inspect all valid values.

Quantization does not shrink the `.vdb` file on disk. Vectlite keeps the original float32 vectors for exact rescoring and stores quantization parameters in a `.vdb.quant` sidecar file, so total disk footprint can increase slightly. The quantized index auto-rebuilds on inserts and upserts.

### Multi-Vector / ColBERT Search

Store token-level embeddings (ColBERT, ColPali) and search with MaxSim late interaction scoring.

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

### TTL / Expiry

Records can automatically expire after a time-to-live. Expired records are transparently filtered from all reads and permanently removed on `compact()`.

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

### Cursor-Based Pagination

Efficiently iterate over large collections without offset overhead.

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

### Async API

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

### Tuning the HNSW index

`bulkIngest()` and `bulkIngestAsync()` accept optional HNSW parameters that
control the recall/latency trade-off and trigger Rayon-backed parallel graph
construction once the dataset crosses `parallelInsertThreshold` (default 256):

```js
// Higher recall, slightly slower build/search
db.bulkIngest(records, {
  batchSize: 5000,
  m: 32,              // max bidirectional links per node (default 16)
  efConstruction: 400, // build-time search width (default 200)
  efSearch: 200,       // query-time search width (default: auto)
})

// Faster build/search, lower recall
db.bulkIngest(records, { m: 8, efConstruction: 100, efSearch: 40 })
```

The same parameters can be changed at any time without re-ingesting:

```js
db.setIndexConfig({ m: 32, efConstruction: 400 }) // rebuilds the ANN graph
db.setEfSearch(200)                               // query-time only, no rebuild
console.log(db.indexConfig())
// { m: 32, ef_construction: 400, ef_search: 200, parallel_insert_threshold: 256 }
```

Use higher `m` / `efConstruction` / `efSearch` to push Recall@10 toward `1.0`;
use lower values when latency or memory matter more than recall.

### OpenTelemetry Integration

vectlite ships with optional OpenTelemetry tracing. When enabled, every search
call is wrapped in a span carrying semantic DB attributes and search-specific
metrics. `@opentelemetry/api` is loaded lazily -- it is **not** a runtime
dependency.

```js
const vectlite = require('vectlite')

// Auto-detect: resolves a tracer from @opentelemetry/api if installed
const tracer = vectlite.configureOpenTelemetry()

// Or supply your own tracer
vectlite.configureOpenTelemetry({ tracer: myTracer })

// Custom tracer name (default: 'vectlite')
vectlite.configureOpenTelemetry({ tracerName: 'my-app' })

// Disable
vectlite.configureOpenTelemetry(false)
```

When a tracer is active, each `search` / `searchWithStats` / `searchAsync` /
`searchWithStatsAsync` call creates a `vectlite.search` span with these
attributes:

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

If a search throws, the span records the exception and sets an error status
before re-throwing.

## Database Methods Reference

### Write Methods

| Method | Description |
|---|---|
| `db.upsert(id, vector, metadata, options)` | Insert or update a single record |
| `db.insert(id, vector, metadata, options)` | Insert a record (throws on duplicate id) |
| `db.upsertMany(records, { namespace })` | Upsert a batch of records |
| `db.insertMany(records, { namespace })` | Insert a batch |
| `db.bulkIngest(records, { namespace, batchSize, m, efConstruction, efSearch, parallelInsertThreshold })` | Fastest bulk import with coalesced WAL fsync and Rayon-parallel HNSW build |
| `db.setIndexConfig({ m, efConstruction, efSearch, parallelInsertThreshold })` | Update HNSW parameters; rebuilds the ANN graph if `m`/`efConstruction` changed |
| `db.setEfSearch(efSearch)` | Adjust query-time HNSW search width without rebuilding |
| `db.indexConfig()` | Return the current HNSW configuration |
| `db.delete(id, { namespace })` | Delete a single record |
| `db.deleteMany(ids, { namespace })` | Delete multiple records by id |
| `db.deleteByFilter(filter, { namespace })` | Delete all records matching a filter |
| `db.updateMetadata(id, metadata, { namespace })` | Merge a metadata patch into an existing record (no vector rewrite) |
| `db.setTtl(id, seconds, { namespace })` | Set time-to-live on a record (seconds from now) |
| `db.clearTtl(id, { namespace })` | Remove TTL from a record |

### Read Methods

| Method | Description |
|---|---|
| `db.get(id, { namespace })` | Get a single record by id |
| `db.search(query, options)` or `db.search({ query, ...options })` | Search and return a list of results |
| `db.searchWithStats(query, options)` | Search with detailed performance stats |
| `db.count({ namespace, filter })` | Count records, optionally scoped by namespace/filter |
| `db.list({ namespace, filter, limit, offset })` | List records without issuing a vector query |
| `db.listCursor({ namespace, filter, limit, cursor })` | Cursor-based pagination for large collections |
| `db.namespaces()` | List all namespaces |
| `db.dimension` | Vector dimension (property) |
| `db.path` | Database file path (property) |
| `db.metric` | Distance metric name: `"cosine"`, `"euclidean"`, `"dotproduct"`, or `"manhattan"` (property) |
| `db.readOnly` | Whether the database is read-only (property) |

### Index Methods

| Method | Description |
|---|---|
| `db.createIndex(field, indexType)` | Create a payload index (`'keyword'` or `'numeric'`) on a metadata field |
| `db.dropIndex(field)` | Remove an index |
| `db.listIndexes()` | List all active indexes as `[{ field, type }, ...]` |

### Quantization Methods

| Method | Description |
|---|---|
| `db.enableQuantization(method, options)` | Enable quantization (`'scalar'`, `'binary'`, or `'pq'` / `'product'`) |
| `db.disableQuantization()` | Disable quantization and remove persisted parameters |
| `db.isQuantized` | Whether quantization is enabled (property) |
| `db.quantizationMethod` | Active method name or `null` (property) |
| `db.validNumSubVectors()` | Valid PQ `numSubVectors` values for this database dimension |
| `db.enableMultiVectorQuantization(space, options)` | Enable 2-bit quantization for a multi-vector space |
| `db.disableMultiVectorQuantization(space)` | Disable multi-vector quantization for a space |
| `db.isMultiVectorQuantized(space)` | Whether multi-vector quantization is enabled for a space |

### Maintenance Methods

| Method | Description |
|---|---|
| `db.compact()` | Fold WAL into snapshot and persist ANN indexes |
| `db.flush()` | Alias for `compact()` |
| `db.snapshot(dest)` | Create a self-contained `.vdb` copy |
| `db.backup(destDir)` | Full backup including ANN sidecar files |
| `db.transaction()` | Begin an atomic transaction |
| `db.close()` | Flush pending state, release the file lock, and invalidate the handle |

### Async Methods

| Method | Description |
|---|---|
| `db.searchAsync(query, options)` | Non-blocking search (returns Promise) |
| `db.searchWithStatsAsync(query, options)` | Non-blocking search with stats (returns Promise) |
| `db.flushAsync()` | Non-blocking flush/compact (returns Promise) |
| `db.compactAsync()` | Non-blocking compact (returns Promise) |
| `db.bulkIngestAsync(records, options)` | Non-blocking bulk import (returns Promise); accepts the same HNSW tuning options as `bulkIngest` |

## Filter Operators

| Operator | Example | Description |
|---|---|---|
| `$eq` | `{ field: { $eq: 'value' } }` | Equal (also `{ field: 'value' }`) |
| `$ne` | `{ field: { $ne: 'value' } }` | Not equal |
| `$gt` / `$gte` | `{ field: { $gt: 5 } }` | Greater than (or equal) |
| `$lt` / `$lte` | `{ field: { $lt: 20 } }` | Less than (or equal) |
| `$in` / `$nin` | `{ field: { $in: ['a', 'b'] } }` | In / not in set |
| `$contains` | `{ field: { $contains: 'auth' } }` | Substring match |
| `$exists` | `{ field: { $exists: true } }` | Field presence |
| `$and` / `$or` | `{ $and: [{...}, {...}] }` | Logical combinators |
| `$not` | `{ $not: {...} }` | Logical negation |
| `$elemMatch` | `{ tags: { $elemMatch: { $eq: 'rust' } } }` | Match array elements |
| `$size` | `{ tags: { $size: 3 } }` | Array length |
| dot-path | `{ 'author.name': 'Alice' }` | Nested field access |

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
- [PyPI Package](https://pypi.org/project/vectlite/)

## License

MIT
