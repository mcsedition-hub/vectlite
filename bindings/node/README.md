# Node Binding Plan

Node comes after Python, once the core object model is stable.

Planned stack:

- Rust wrapper crate depending on `crates/vectlite-core`
- `napi-rs` for the Node module

API target:

```ts
import { open } from "vectlite";

const db = open("knowledge.vdb", { dimension: 384 });
db.upsert("doc1", embedding, { source: "notes" });
const results = db.search(query, { k: 5 });
```
