# Language Rollout

The rollout order is:

1. Python
2. Node.js
3. Cross-platform frameworks
4. Swift and Kotlin

That order should drive the repository shape.

## Core Rule

`crates/vectlite-core` stays runtime-agnostic:

- no Python-specific types
- no Node-specific types
- no mobile-specific glue
- stable storage and query model

Everything language-facing wraps the same operations:

- `open(path, dimension?)`
- `upsert(id, vector, metadata)`
- `get(id)`
- `delete(id)`
- `search(query, k, filter?)`

## Python First

Python is the first public surface because it is the shortest path to RAG, notebooks, local AI tooling, and developer adoption.

Recommended implementation path:

- binding crate with `PyO3`
- packaging with `maturin`
- Python API should feel like `sqlite3` or `duckdb`: very small, direct, sync by default

Target package shape:

```python
import vectlite

db = vectlite.open("knowledge.vdb", dimension=384)
db.upsert("doc1", embedding, {"source": "blog"})
results = db.search(query, k=5, filter={"source": "blog"})
```

## Node Second

Once Python stabilizes the object model, Node should mirror it closely.

That work is now started: the initial source-built binding lives in `bindings/node` and mirrors the core CRUD, collection, and search flows.

Recommended implementation path:

- wrapper crate with `napi-rs`
- keep API sync where practical for local embedded usage
- match Python naming unless JavaScript conventions require a small adjustment

Target package shape:

```ts
import { open } from "vectlite";

const db = open("knowledge.vdb", { dimension: 384 });
db.upsert("doc1", embedding, { source: "blog" });
const results = db.search(query, { k: 5, filter: { source: "blog" } });
```

## Frameworks Third

Framework wrappers should come after Python and Node because they need a stable lower-level contract.

Likely order:

- Flutter via `dart:ffi`
- React Native via a native bridge or JSI-backed module

At that point, add a thin FFI boundary instead of letting framework code bind directly to Rust internals.

## Swift And Kotlin Last

Native mobile wrappers should land after the FFI contract is proven by framework work:

- Swift Package / XCFramework on iOS
- Android `.so` + Kotlin wrapper on Android

That keeps the mobile surface thin and avoids redesigning the core twice.
