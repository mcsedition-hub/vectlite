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

// Clean up
db.compact()
```

## Features

### Core

- **Single-file storage** -- one `.vdb` file per database, portable and easy to back up
- **Dense vectors** -- cosine similarity with automatic HNSW indexing for large collections
- **Sparse vectors** -- BM25-scored inverted index for keyword retrieval
- **Hybrid search** -- dense + sparse fusion with linear or RRF strategies
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

### Data Management

- **Physical collections** -- `vectlite.openStore()` manages a directory of independent databases
- **Bulk ingestion** -- `bulkIngest()` with deferred index rebuilds for fast imports
- **Snapshots** -- `db.snapshot(path)` creates a self-contained copy
- **Backup / Restore** -- `db.backup(dir)` and `vectlite.restore(dir, path)` for full roundtrips
- **Read-only mode** -- `vectlite.open(path, { readOnly: true })` for safe concurrent readers

## Usage

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
// Handles embedding + sparse term generation for you
vectlite.upsertText(db, 'doc1', 'Auth setup guide', embedFn, { source: 'docs' })
const results = vectlite.searchText(db, 'how to authenticate', embedFn, { k: 5 })
```

### Snapshots & Backup

```js
db.snapshot('/backups/knowledge_2024.vdb') // Self-contained copy
db.backup('/backups/full/')                // Full backup with ANN sidecars

const restored = vectlite.restore('/backups/full/', 'restored.vdb')
```

### Read-Only Mode

```js
const ro = vectlite.open('knowledge.vdb', { readOnly: true })
const results = ro.search(query, { k: 5 }) // Reads work
ro.upsert(...)                              // Throws VectLiteError
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
