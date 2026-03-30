# Node Binding

The Node binding is published on npm and uses prebuilt binaries on supported platforms.

Current state:

- Rust addon implemented with `napi-rs`
- JavaScript wrapper and TypeScript declarations included
- local smoke test available in `bindings/node/tests`
- npm package live as `vectlite`
- prebuilt binaries for macOS x64/arm64, Linux x64 (glibc), and Windows x64
- source-build fallback for other platforms

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

The npm package is set up as a hybrid prebuilt + source-build package:

- `prepack` stages a self-contained native crate plus the core Rust crate
- `publish-npm.yml` attaches platform binaries into `prebuilds/`
- `install` uses a matching prebuilt when available
- unsupported platforms fall back to compiling the addon with Cargo

For supported prebuilt targets, `npm install vectlite` only needs Node.

For unsupported targets, installation still requires:

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
- prebuilt binaries for Linux arm64 and musl targets
