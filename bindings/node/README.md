# Node Binding

The Node binding is published on npm and builds from source on install.

Current state:

- Rust addon implemented with `napi-rs`
- JavaScript wrapper and TypeScript declarations included
- local smoke test available in `bindings/node/tests`
- npm package live as `vectlite`
- installing the npm package requires a working Rust toolchain on the target machine

## Install

```bash
npm install vectlite
```

## Local Build

From the repository root:

```bash
cd bindings/node
npm run build
```

This compiles the Rust addon and writes `bindings/node/vectlite.node`.

## Local Test

```bash
cd bindings/node
npm test
```

## npm Package Model

The npm package is set up as a source-build package:

- `prepack` stages a self-contained native crate plus the core Rust crate
- `install` compiles the addon with Cargo on the target machine
- the published tarball does not ship prebuilt binaries yet

That keeps one source of truth for the Rust core, but it means `npm install vectlite` requires:

- Node 18+
- Rust/Cargo installed
- registry/network access to fetch Rust crates during the build

## Usage

```js
const { open, sparseTerms } = require('./index.js')

const db = open('knowledge.vdb', { dimension: 384 })
db.upsert('doc1', embedding, { source: 'notes', title: 'Auth Guide' })

const results = db.search(queryEmbedding, {
  k: 5,
  sparse: sparseTerms('auth guide'),
  filter: { source: 'notes' },
})
```

## Scope

The initial Node surface covers the core database and store operations:

- `open`, `openStore`, `restore`
- `insert`, `upsert`, `get`, `delete`
- batch writes and bulk ingest
- snapshots, backup, compact, flush
- namespaces and collections
- dense, sparse, and hybrid search
- search stats and text helpers

Not yet included:

- JS callback rerank hooks
- prebuilt binaries
