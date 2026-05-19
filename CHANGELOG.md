# Changelog

All notable changes to `vectlite` will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning while the public API stabilizes.

## [0.11.0] - 2026-05-18

### Performance

This release rewrites the single-record ingestion hot path that bottlenecked
streaming workloads at ~150 vec/s. Expected throughput is now 10–50× higher
out of the box on a typical SSD, and another 5–10× on top of that when the
new WAL `sync_mode` is relaxed.

- **Incremental HNSW insertion.** `insert` / `upsert` no longer rebuild the
  entire HNSW graph(s) on each call. Instead the new vector is appended
  directly into the existing graph via `hnsw_rs::Hnsw::insert` (or
  `parallel_insert` for larger batches). Per-record cost drops from
  `O(N log N)` to `O(log N)`. Falling back to a full rebuild only happens
  when an op is a delete or replaces an existing key — see HNSW tombstoning
  below for the delete fix.
- **Lazy persistence of the ANN graph.** `persist_ann_to_disk` is no longer
  called on every insert. Instead an `ann_dirty` flag is raised and the
  HNSW sidecar files are dumped at `flush` / `compact` / `close` time. The
  WAL still gives per-record durability, so on a crash the ANN is rebuilt
  from records on the next `open()`.
- **Lazy rebuild of quantized indexes.** When quantization is enabled, the
  in-memory PQ / scalar / binary codebook is dropped on the first
  post-build insert and rebuilt at the next flush. Searches between inserts
  and flush transparently fall back to HNSW, never returning stale
  candidates.
- **Cached WAL writer.** A single `BufWriter<File>` is now held for the
  lifetime of the database session. Previous behaviour opened and closed
  the WAL file on every `insert`, paying the `open(2)` syscall per record.
- **WAL `sync_mode` knob (`WalSyncMode`).** Choose `PerOp` (the default,
  fsync per insert), `EveryN(n)` (fsync every N inserts — amortises the
  fsync tax) or `OnFlush` (fsync only at `flush` / `compact` / `close`).
  Configurable via `Database::set_wal_sync_mode`, exposed in the bindings.
  Tradeoff: relaxing the mode trades a bounded amount of recently-acked
  data on crash for a 5–10× throughput multiplier on macOS APFS.
- **HNSW tombstoning.** `delete` no longer triggers a full HNSW rebuild.
  The deleted record's `origin_id` is marked in a per-index tombstone set
  and silently skipped during search. A rebuild happens automatically at
  `compact()` time, or whenever the tombstone ratio crosses
  `IndexConfig.tombstone_rebuild_pct` (default 30).
- **Contiguous dense-vector arena (`VectorArena`).** The default dense
  vector is now mirrored into a single global contiguous `Vec<f32>` arena.
  Per-record vectors are appended at insert time (no extra allocations
  beyond the amortised arena growth), and the arena is lazily rebuilt
  after a delete (which cannot compact in place). Call
  `Database::prepare_for_scan()` to materialise it ahead of a heavy
  brute-force / rescoring workload. Search-path integration is incremental:
  the arena is currently exposed for callers and used as the cache-friendly
  storage layer; wiring it into the default `collect_results` scan is in
  progress in a follow-up to keep this release minimally invasive.
- **ANN manifest format bumped to `ANN2`** to persist the per-graph
  insertion-order keys alongside the HNSW sidecar files. The old `ANN1`
  format is still readable. Required for the incremental-insert path to
  survive a reopen.

### Added

- `WalSyncMode` enum (Rust core) with `PerOp`, `EveryN(usize)`, `OnFlush`.
  Default remains `PerOp` for safety.
- `Database::set_wal_sync_mode`, `Database::wal_sync_mode` (Rust core).
- `IndexConfig.tombstone_rebuild_pct` (default `30`) — once a graph
  accumulates this percentage of tombstones it is rebuilt at the next
  `compact()`.
- `Database::tombstone_stats()` returning the per-graph live / tombstoned
  counts.

### Bindings

The new core APIs are surfaced in both the Python and Node bindings.

**Python (`vectlite`):**
- `db.set_wal_sync_mode(mode, n=None)` — `mode` is `"per_op"`,
  `"every_n"` (pass `n`), or `"on_flush"`.
- `db.wal_sync_mode()` returns `{"mode": "per_op"}` |
  `{"mode": "every_n", "n": 64}` | `{"mode": "on_flush"}`.
- `db.tombstone_stats()` returns `(live, dead)`.
- `db.prepare_for_scan()` materialises the contiguous arena up front.
- `db.vector_arena_len()` returns the arena size or `None`.
- `db.bulk_ingest(..., tombstone_rebuild_pct=N)` and
  `db.set_index_config(..., tombstone_rebuild_pct=N)` accept the new
  HNSW knob. `db.index_config()` includes a `tombstone_rebuild_pct`
  key in the returned dict.

**Node (`vectlite`):**
- `db.setWalSyncMode(mode, n)` — same semantics.
- `db.walSyncMode()` returns `{ mode, n? }`.
- `db.tombstoneStats()` returns `{ live, dead }`.
- `db.prepareForScan()`, `db.vectorArenaLen()`.
- `bulkIngest`, `setIndexConfig`, `indexConfig`, and `bulkIngestAsync`
  accept / return `tombstoneRebuildPct`.
- TypeScript types updated: `WalSyncMode`, `WalSyncModeInfo`,
  `TombstoneStats`.

UniFFI (Swift / Kotlin) plumbing is still pending — open an issue if
you need it for a binding outside Python / Node.

## [0.10.0] - 2026-05-16

### Performance

- `bulk_ingest` is now ~10–20× faster on default settings. Two optimisations:
  - HNSW graph construction now uses `parallel_insert` (Rayon-backed) when
    the dataset is large enough (>= `parallel_insert_threshold`, default 256
    vectors). The dominant cost — distance calculations during graph
    neighbour selection — is now multi-threaded.
  - WAL writes during `bulk_ingest` are coalesced into a single `fsync` at
    the very end instead of `fsync`-per-batch. This removes the
    fsync-per-batch tax that dominates ingestion latency on macOS and
    modern SSDs.
- Synthetic 5k × 384 cosine benchmark on M-class macOS: ingest throughput
  improved from ~47 vec/s (article-run baseline) to ~917 vec/s out of the
  box, and up to ~1782 vec/s with the `fast` preset (lower recall).

### Added

- New `IndexConfig` struct (core Rust) exposing the HNSW tuning knobs
  `m`, `ef_construction`, `ef_search`, and `parallel_insert_threshold`.
  Includes `IndexConfig::high_recall()` and `IndexConfig::fast()` presets.
- `Database::bulk_ingest_with_config(records, batch_size, Option<IndexConfig>)`
  for one-shot tuned ingestion.
- `Database::set_index_config(IndexConfig)` and `Database::set_ef_search(Option<usize>)`
  to retune an existing database. Changing `m` / `ef_construction` triggers a
  full ANN rebuild; changing only `ef_search` is free (query-time only).
- Python: `db.bulk_ingest(records, batch_size=..., m=..., ef_construction=..., ef_search=..., parallel_insert_threshold=...)`.
- Python: `db.set_index_config(...)`, `db.set_ef_search(...)`, `db.index_config()`.
- Node: `db.bulkIngest(records, { m, efConstruction, efSearch, parallelInsertThreshold })`
  and matching `bulkIngestAsync` options.
- Node: `db.indexConfig()`, `db.setEfSearch(...)`, `db.setIndexConfig({ ... })`.
- UniFFI (Swift / Kotlin): new `IndexConfigResult` dictionary plus
  `bulkIngestTuned`, `indexConfig`, `setEfSearch`, `setIndexConfig` on the
  `Database` interface.

## [0.9.3] - 2026-05-13

### Fixed

- UniFFI (Swift/Kotlin) scalar and product quantization `rescore_multiplier` defaults now match the core Rust defaults (10x) instead of being hard-coded to 4x. This fixes the recall regression where Swift scalar quantization scored ~0.66 vs Python's ~0.855. Binary quantization was already correct at 10x.
- UniFFI `enableQuantization()` now accepts `"int8"` as an alias for `"scalar"`, matching Python and Node behaviour.

## [0.9.2] - 2026-05-12

### Fixed

- Node `Database.search*` methods now accept the shorthand `db.search(query, k)` in addition to `db.search(query, options)` and `db.search({ query, ...options })`.
- Kotlin builds no longer force Gradle to locate or provision a JDK 17 toolchain. The binding now compiles with the host JDK while still emitting Java 17-compatible bytecode.

## [0.9.1] - 2026-05-12

### Fixed

- Python quantization introspection now exposes `db.is_quantized()` as a method, matching the rest of the quantization helper API, and the type stubs now include `db.quantization_method`.
- Python documentation now explicitly marks passive database metadata (`db.metric`, `db.dimension`, `db.read_only`, `db.path`, `db.wal_path`) as properties.
- Scalar quantized search now ranks candidates with dequantized metric-aware scores before exact float32 rescoring, avoiding large recall loss from raw `u8` dot-product bias.
- Quantization `rescore_multiplier` now directly controls the rescoring budget (`k * rescore_multiplier`, capped at collection size) instead of being hidden by an internal 100-candidate floor.
- Quantization documentation now clarifies that memory savings apply to the in-memory candidate index, not to `.vdb` disk footprint, because float32 vectors are retained for exact rescoring.
- Node `Database` wrapper now exposes quantization and multi-vector quantization methods that were previously only available on the private `_native` handle.
- Node search now accepts both `db.search(query, options)` and `db.search({ query, ...options })`, with the same support on stats and async variants.
- Product quantization validates invalid `num_sub_vectors` settings as catchable errors instead of panicking, and Python/Node choose a dimension-compatible PQ default when `num_sub_vectors` is omitted.
- Python and Node now expose valid PQ partition helpers (`db.valid_num_sub_vectors()` / `db.validNumSubVectors()`), and quantization method names are parsed case-insensitively with `pq` documented as the primary PQ alias.
- Search with an all-zeros query vector now raises a `VectLiteError` for cosine and dot-product metrics instead of silently returning arbitrary results with score 0. Euclidean and Manhattan metrics are unaffected since distance from the origin is well-defined.
- Search now rejects query vectors whose dimension does not match the database dimension, returning a `DimensionMismatch` error. Previously, undersized queries were silently truncated via Matryoshka logic even without an explicit `truncate_dim` parameter. Users must now pass `truncate_dim` to opt into prefix search.
- `Store.close()` is now available in Python, Node, Swift, and Kotlin bindings for symmetry with `Database.close()`. The method is a no-op (the store holds no open file handles) but prevents `AttributeError` / missing-method surprises.
- HNSW sidecar files no longer use triple-dot filenames (`c.vdb.ann...hnsw.data`). Empty namespace and vector-name components now produce `_` sentinels (e.g. `c.vdb.ann._._.hnsw.data`). Existing triple-dot files are orphaned and the ANN index rebuilds automatically on next use.
- Swift and Kotlin bindings now accept the `"pq"` alias for product quantization in `enableQuantization()`, matching Python and Node behaviour. The error message also lists the alias.
- `build-xcframework.sh` now sets `MACOSX_DEPLOYMENT_TARGET=12.0` and `IPHONEOS_DEPLOYMENT_TARGET=15.0` when building the XCFramework, preventing linker warnings when the framework is consumed on older OS versions.
- Kotlin `build.gradle.kts` now declares `jvmToolchain(17)` so Gradle auto-provisions a compatible JDK, fixing build failures when `JAVA_HOME` points to JDK 25+.

## [0.9.0] - 2026-05-11

### Added

- **Optional OpenTelemetry tracing** for search operations (Python and Node.js).
  - `configure_opentelemetry()` (Python) / `configureOpenTelemetry()` (Node) enables span-based tracing.
  - Each search call creates a `vectlite.search` span with `db.system`, `db.operation.name`, and search-specific attributes (k, namespace, fusion, result counts, timings).
  - `opentelemetry-api` (Python) / `@opentelemetry/api` (Node) is loaded lazily -- never a required runtime dependency.
  - Supports custom tracers, custom tracer names, and explicit disable via `False` / `{ enabled: false }`.
- **Experimental Swift and Kotlin bindings** via a shared UniFFI layer.
  - New `bindings/uniffi` crate with a UDL interface for the core database, store, search, metadata, indexes, TTL, quantization, bulk ingest, backup/restore, and transactions.
  - Swift package in `bindings/swift` with a UniFFI-generated wrapper, `VectLiteFFI.xcframework`, an XCFramework build script, and smoke tests.
  - Kotlin/JVM Gradle package in `bindings/kotlin` compiling the generated UniFFI Kotlin source, loading the native library through JNA, and running smoke tests against the Rust FFI library.

## [0.1.17] - 2026-05-11

### Added

- **TTL / Expiry** -- records can now automatically expire after a time-to-live.
  - `db.set_ttl(id, ttl_secs)` sets a TTL on an existing record; `db.clear_ttl(id)` removes it.
  - `ttl` parameter on `insert()` / `upsert()` (Python, Node) and transaction writes.
  - Expired records are transparently filtered from `get()`, `list()`, `count()`, and `search()` at read time.
  - `compact()` garbage-collects expired records permanently.
  - `expires_at` field returned in record output (epoch seconds, or `null` / `None`).
  - WAL `SetTtl` operation (tag 4) with snapshot persistence.
- **Cursor-based pagination** -- efficient iteration over large collections without offset overhead.
  - Rust core: `Database::list_cursor(namespace, filter, limit, after)` returns `(Vec<Record>, Option<String>)`.
  - Python: `db.list_cursor(namespace, filter, limit, cursor)` returns `(list[Record], str | None)`.
  - Node: `db.listCursor({ namespace, filter, limit, cursor })` returns `{ records, cursor }`.
  - Respects TTL filtering and metadata filters.
- **Async Node API** -- non-blocking versions of heavy operations for Node.js.
  - `db.searchAsync()`, `db.searchWithStatsAsync()`, `db.flushAsync()`, `db.compactAsync()`, `db.bulkIngestAsync()`.
  - Backed by `napi::Task` (runs on libuv threadpool), no tokio dependency.
- **LangChain integration** -- `vectlite.langchain.VectLiteVectorStore` implements the LangChain VectorStore protocol.
  - `add_texts()`, `add_documents()`, `similarity_search()`, `similarity_search_with_score()`, `similarity_search_by_vector()`, `delete()`, `from_texts()`.
  - Requires `langchain-core >= 0.2` (optional dependency).
- **LlamaIndex integration** -- `vectlite.llamaindex.VectLiteVectorStore` implements the LlamaIndex VectorStore protocol.
  - `add()`, `delete()`, `query()` methods compatible with `VectorStoreIndex` and `StorageContext`.
  - Requires `llama-index-core >= 0.10` (optional dependency).
- **Built-in embedding providers** -- `vectlite.embedders` module with ready-to-use factory functions.
  - `embedders.openai()`, `embedders.cohere()`, `embedders.voyage()`, `embedders.fastembed()`, `embedders.sentence_transformer()`, `embedders.ollama()`.
  - Each returns a `Callable[[str], list[float]]` compatible with `upsert_text()` and `search_text()`.
  - All providers lazy-import their SDK (zero hard dependencies).
- **ONNX cross-encoder reranker** -- `rerankers.onnx_cross_encoder()` for zero-PyTorch reranking.
  - Uses `onnxruntime` + `tokenizers` for lightweight cross-encoder inference.
  - Auto-downloads models from HuggingFace Hub; same `RerankHook` interface as `cross_encoder()`.
- **Rich CLI** -- full command-line interface via `vectlite` command or `python -m vectlite`.
  - Subcommands: `stats`, `count`, `list`, `dump`, `search`, `compact`, `verify`, `bench`, `import-jsonl`, `import-csv`.
  - `vectlite stats my.vdb` -- database stats (dimension, metric, record counts, file sizes, indexes).
  - `vectlite bench my.vdb --queries 1000` -- search benchmark with QPS and latency stats.
  - `vectlite dump my.vdb` -- export all records as JSONL via cursor pagination.
  - `vectlite import-jsonl my.vdb data.jsonl` / `vectlite import-csv my.vdb data.csv` -- bulk import.
- **Schema validation** -- optional typed metadata schemas with clear error messages.
  - `schema.Schema({"price": "number", "tags": "array<string>"})` defines field types.
  - Types: `string`, `number`, `integer`, `boolean`, `null`, `any`, `array`, `array<T>`, `object`, nested objects.
  - `schema.validated(db, s)` wraps a database to auto-validate on every write.
  - Schemas persist in `.vdb.schema.json` sidecar files via `save()` / `load()`.
  - `strict=True` rejects unknown fields.

## [0.1.16] - 2026-05-11

### Added

- **Payload indexes** -- create keyword and numeric indexes on metadata fields to accelerate filtered queries 10-100x on large collections.
  - `db.create_index(field, type)` creates an index (`"keyword"` for string equality/`$in`, `"numeric"` for range queries `$gt`/`$gte`/`$lt`/`$lte`).
  - `db.drop_index(field)` removes an index.
  - `db.list_indexes()` returns all active indexes.
  - Indexes are automatically used by `search()`, `count()`, and `list()` to narrow candidates before full filter evaluation.
  - AND filters intersect index results; OR filters union when all sub-filters are indexed.
  - Indexes are incrementally maintained on `upsert()`, `delete()`, and `update_metadata()`.
  - Index definitions persist across close/reopen in a `.vdb.pidx` sidecar file; data is rebuilt from records on open.
  - Sidecar files are included in `backup()` operations.
- Rust core: `Database::create_index()`, `Database::drop_index()`, `Database::list_indexes()`.
- Python binding: `db.create_index(field, index_type)`, `db.drop_index(field)`, `db.list_indexes()`.
- Node binding: `db.createIndex(field, indexType)`, `db.dropIndex(field)`, `db.listIndexes()`.

## [0.1.15] - 2026-05-11

### Added

- **Partial metadata updates** -- new `update_metadata()` method that merges a patch into an existing record's metadata without re-writing the vector or rebuilding indexes.
  - Keys present in the patch overwrite existing keys; keys not in the patch remain untouched.
  - Skips all index rebuilds (ANN, sparse, quantized, multi-vector) when a WAL batch contains only metadata updates.
  - Returns `true` if the record was found and updated, `false` if the id does not exist.
  - Works with namespaces via `update_metadata_in_namespace()` (Rust) or `namespace` parameter (Python/Node).
- Rust core: `Database::update_metadata()`, `Database::update_metadata_in_namespace()`.
- Python binding: `db.update_metadata(id, metadata, namespace=None)`.
- Node binding: `db.updateMetadata(id, metadata, { namespace })`.
- New WAL operation (`UpdateMetadata`, tag 3) with full serialization/deserialization support.

## [0.1.14] - 2026-05-11

### Added

- **Multiple distance metrics** -- databases can now be created with `cosine` (default), `euclidean` (L2), `dotproduct` (inner product), or `manhattan` (L1) distance metrics.
  - The metric is persisted in the database file and automatically loaded on reopen. Older databases (format version <= 5) default to cosine.
  - Aliases are accepted: `l2` for euclidean, `dot` / `ip` / `inner_product` / `dot_product` for dotproduct, `l1` for manhattan.
  - Scores are normalized so that **higher is always better** across all metrics; distance metrics (euclidean, manhattan) are negated.
- **SIMD-accelerated scoring** via the `simsimd` crate for cosine, euclidean (L2), and dot product distance computations, with automatic scalar fallbacks.
  - Manhattan distance uses a scalar implementation (simsimd does not provide L1).
- Rust core: `Database::create_with_metric()`, `Database::open_or_create_with_metric()`, `Database::metric()`, and the `DistanceMetric` enum with `score()`, `from_name()`, `name()`, `is_similarity()`.
- Python binding: `vectlite.open(path, metric="euclidean")` and `db.metric` property.
- Node binding: `vectlite.open(path, { metric: 'euclidean' })` and `db.metric` property.
- HNSW indexes now use metric-specific distance functions (`DistCosine`, `DistL2`, `DistDot`, `DistL1` from hnsw_rs).
- Rust unit tests for `DistanceMetric` enum (tag/name roundtrip, aliases, score correctness, SIMD vs scalar parity).
- Rust integration tests for metric persistence, search ordering with each metric, and create/reopen cycles.
- Python smoke tests for metric creation, aliases, persistence, invalid metric errors, and search ordering with each metric.

### Changed

- Binary format bumped to version 6 to store the distance metric byte after the dimension field.
- All internal cosine similarity calls replaced with `DistanceMetric::score()`, enabling metric-aware scoring throughout search, MMR, MaxSim, and record similarity computations.
- The `simsimd` crate (v6.5) is now a dependency of `vectlite-core`.

## [0.1.13] - 2026-05-11

### Added

- **Multi-vector / late interaction (ColBERT-style)** search with per-document token-level embeddings and MaxSim scoring:
  - Storage of N token vectors per document in named multi-vector spaces (e.g. `"colbert"`, `"colpali"`).
  - MaxSim scoring: for each query token, find the maximum cosine similarity against all document tokens, then sum across query tokens.
  - **2-bit quantization** for ColBERTv2-style token compression (~16x memory reduction), using per-dimension quartile boundaries.
  - Quantized multi-vector search uses a 2-stage pipeline: fast 2-bit approximate MaxSim candidate selection followed by exact float32 rescoring.
- Multi-vector quantization parameters persist in `.vdb.mvquant.<space>` sidecar files and auto-load on database open.
- Quantized multi-vector indexes automatically rebuild on inserts, upserts, and bulk ingestion.
- Rust core: `upsert_multi_vectors()`, `search_multi_vector()`, `enable_multi_vector_quantization()`, `disable_multi_vector_quantization()`, `is_multi_vector_quantized()` on `Database`.
- Python binding: `db.upsert_multi_vectors(id, vector, multi_vectors, ...)`, `db.search_multi_vector(space, query_tokens, ...)`, `db.enable_multi_vector_quantization(space, ...)`, `db.disable_multi_vector_quantization(space)`, `db.is_multi_vector_quantized(space)`.
- Node binding: `db.upsertMultiVectors(id, vector, multiVectorsJson, ...)`, `db.searchMultiVector(space, queryTokensJson, ...)`, `db.enableMultiVectorQuantization(space, ...)`, `db.disableMultiVectorQuantization(space)`, `db.isMultiVectorQuantized(space)`.
- New `TwoBitQuantizer`, `MultiVectorQuantizedIndex` in `quantization.rs` with train, search, serialize/deserialize.
- New `maxsim_score()` function and `MultiVectorSearchOptions` / `MultiVectorSearchResult` types in the Rust core.
- Binary format bumped to version 5 to support `multi_vectors` field on `Record`.
- Rust unit tests for 2-bit quantizer, multi-vector quantized index, and MaxSim scoring correctness.
- Rust integration tests for multi-vector upsert, search, namespace filtering, quantization enable/disable/persist, and record persistence.
- Python smoke tests for multi-vector upsert and search, metadata, namespace filtering, quantization enable/disable/persist, error cases, and record persistence.

### Changed

- The repository README now documents multi-vector / ColBERT features, usage examples, and API for Python and Node.js.
- The storage format table in the repository README now includes the `.vdb.mvquant.*` sidecar files.
- Mutation methods (`insert_many`, `upsert_many`, `apply_operations`, `bulk_ingest`, `apply_wal_batch`) now rebuild multi-vector quantized indexes alongside regular quantized indexes.
- All three `open` methods (`open`, `open_with_timeout`, `open_read_only_with_timeout`) now auto-load multi-vector quantization from sidecar files.

## [0.1.12] - 2026-05-10

### Added

- **Vector quantization** with three strategies for trading in-memory candidate-index size for search speed:
  - **Scalar quantization (int8)** -- compact in-memory candidate index with minimal recall loss.
  - **Binary quantization** -- smallest in-memory candidate index using Hamming distance filtering, best for normalized embeddings.
  - **Product quantization (PQ)** -- configurable compression via k-means sub-vector clustering for very large datasets.
- All quantization methods use a 2-stage pipeline: fast quantized candidate selection followed by exact float32 cosine rescoring.
- Quantization parameters (calibration ranges, PQ codebooks) persist in a `.vdb.quant` sidecar file and auto-load on database open.
- Quantized indexes automatically rebuild on inserts, upserts, and bulk ingestion.
- Rust core: `enable_quantization()`, `disable_quantization()`, `is_quantized()`, `quantization_config()` on `Database`.
- Python binding: `db.enable_quantization(method, ...)`, `db.disable_quantization()`, `db.is_quantized()`, `db.quantization_method`.
- Node binding: `db.enableQuantization(method, optionsJson)`, `db.disableQuantization()`, `db.isQuantized`, `db.quantizationMethod`.
- New `crates/vectlite-core/src/quantization.rs` module with `ScalarQuantizer`, `BinaryQuantizer`, `ProductQuantizer`, and `QuantizedIndex`.
- Rust unit tests for all three quantizers including serialization roundtrips.
- Rust integration tests for enable/disable/persist/error workflows.
- Python smoke tests for scalar, binary, product quantization, disable, and error cases.

### Changed

- The repository README, Python package README, and Node package README now document quantization features, usage examples, and API reference tables.
- `hybrid_search_internal()` now uses quantized candidates as an alternative to HNSW when quantization is enabled.
- The storage format table in the repository README now includes the `.vdb.quant` sidecar file.

## [0.1.11] - 2026-04-01

### Added

- Python and Node bindings now expose explicit database lifecycle controls with `close()` and context-manager-safe close semantics on Python `Database` objects.
- Both bindings now support query-free record scanning with `list(...)`, filtered/scoped record counts, and bulk removal by metadata filter.
- Open calls now support lock wait timeouts across both read-write and read-only entry points (`lock_timeout` in Python, `lockTimeout` in Node).

### Changed

- The repository README plus the Python and Node package READMEs now document the new lifecycle, listing, filtered count, delete-by-filter, and lock-timeout APIs.
- The Node package version is now aligned with the workspace release version again so npm and PyPI releases can be cut from the same source state with matching `0.1.11` tags.

### Fixed

- `close()` now propagates persistence failures instead of silently swallowing WAL compaction errors before releasing the lock.
- Closed databases now fail consistently on public result-bearing operations instead of sometimes behaving like empty databases.
- Invalid lock-timeout inputs such as negative values or `NaN` now raise normal vectlite validation errors instead of risking a panic in Rust.
- The Node wrapper/types and Python stubs are now kept in sync with the runtime surface for `close`, filtered `count`, `list`, `deleteByFilter` / `delete_by_filter`, and lock-timeout options.

## [0.1.10] - 2026-03-31

### Added

- The repository README and Python package README now document `bulk_ingest()`, batch record formats, and a fuller database methods reference including maintenance and diagnostics APIs.

### Changed

- Python sparse-query parameters now raise a clearer `TypeError` when callers pass a string instead of the `dict[str, float]` returned by `vectlite.sparse_terms()`.
- Dimension mismatch errors now explain how to recover after changing embedding models by deleting the existing `.vdb` file or creating a new database path.
- `insert_many()`, `upsert_many()`, and transaction commits now defer index rebuilds until the end of the batch, removing the rebuild-per-operation cost from bulk writes.
- Internal WAL batch application now skips sparse index rebuilds when an operation does not touch sparse terms.
- The PyPI release workflow now reads the workspace version from `[workspace.package]` before validating `py-v*` tags.
- The npm release workflow now falls back to the repository `NPM_TOKEN` secret when present, while still keeping trusted publishing as the default path when no token is configured.

### Fixed

- Upserts that replace a previously sparse record with a record that has no sparse terms now rebuild sparse search state correctly instead of leaving stale sparse candidates behind.
- Sparse-only searches no longer fall back to returning zero-score full-scan results when no sparse candidates match.

## [0.1.8] - 2026-03-30

### Fixed

- Node `0.1.8` keeps the staged Windows prebuilt in place during the prebuilt-loader smoke test, avoiding an `EPERM` cleanup failure on GitHub Actions and allowing npm publication to complete.

## [0.1.7] - 2026-03-30

### Fixed

- Node `0.1.7` is the clean npm release that ships both the async text-embedder support and the Windows prebuilt-loader cleanup fix from the correct tagged commit.

## [0.1.6] - 2026-03-30

### Fixed

- The Node prebuilt-loader smoke test now cleans up safely on Windows, so the cross-platform npm publish workflow can complete instead of failing on `EPERM` during test cleanup.

## [0.1.5] - 2026-03-30

### Fixed

- Node `upsertText()`, `searchText()`, and `searchTextWithStats()` now support async embedding functions that return a `Promise`, matching the documented usage.

## [0.1.4] - 2026-03-30

### Added

- Added a contribution guide, project code of conduct, pull request template, issue templates, and maintainer notes for reviewing community PRs.
- Added an initial Node binding in `bindings/node` with a native `napi-rs` addon, JavaScript wrapper, TypeScript declarations, and smoke tests for CRUD, collections, and text helpers.
- Added a GitHub Actions workflow for npm trusted publishing, with tag-to-package-version validation and package tarball checks before publish.
- Added Node prebuilt-binary support for macOS x64/arm64, Linux x64 (glibc), and Windows x64, with a source-build fallback on unsupported targets.

### Changed

- The repository README now points contributors to the contribution and conduct docs before opening pull requests.
- Contribution and release docs now distinguish local packaging validation from maintainer-only publishing steps, so public contributors are not told to upload releases.
- Local packaging commands in the docs are now explicitly labeled as no-upload validation steps.
- The main CI workflow now runs Node smoke tests on Linux, macOS, and Windows in addition to the Rust and Python checks.
- The repository README now shows the Node binding as available from source instead of just planned.
- The Node package is now structured as a self-contained source-build npm package with prepack/install scripts and a maintainer npm release flow.
- The repository and Node package docs now advertise `npm install vectlite` as the default Node install path.
- Python and Node package releases now use separate tag namespaces (`py-v*` and `node-v*`) so Node-only releases do not trigger PyPI publication.
- Public package metadata and README links now point to the official docs site at `https://vectlite.mcsedition.org/`.
- GitHub Releases can now be created through `scripts/create_github_release.sh`, which prepends links to the official docs, package page, install command, and changelog before auto-generated notes.

## [0.1.3] - 2026-03-30

### Changed

- GitHub Actions workflows now use Node 24 native action versions for checkout, Python setup, and artifact upload/download, instead of forcing Node 24 through a workflow environment flag.
- The GitHub repository README now leads with a fuller product overview, install guidance, quick start, and feature map.
- The Python package README now reflects the broader surface area of the published package, including collections, snapshots, analyzers, rerankers, and diagnostics.

## [0.1.2] - 2026-03-30

### Added

- GitHub Actions CI for Rust formatting and tests plus Python install, test, and packaging validation across Linux, macOS, and Windows.
- Dedicated GitHub Actions release flows for TestPyPI staging and PyPI publishing with repository secrets.
- Project changelog with versioned release notes in the repository root.

### Changed

- Repository and Python package documentation now point directly to the changelog and the published PyPI install path.
- Release documentation now treats changelog updates as part of the standard cut process.
- Release examples now use version placeholders instead of hardcoded historical tags.

## [0.1.1] - 2026-03-30

### Added

- First public PyPI release of `vectlite` as an embedded Python package.
- Cross-platform GitHub Actions workflows for CI, wheel builds, TestPyPI publishing, and PyPI publishing.
- Crash-safe persistence with a snapshot, write-ahead log, and persisted ANN sidecars in the Rust core.
- Dense ANN, sparse BM25-style retrieval, hybrid dense+sparse fusion, MMR diversification, and RRF fusion.
- Python transactions, namespaces, named vectors, rerank hooks, built-in rerankers, and search diagnostics.
- Richer metadata value support across the Rust core and Python binding, including `None`, lists, and dictionaries.

### Changed

- README and package docs now advertise `pip install vectlite` as the default install path.
- Release scripts now support idempotent reruns with `--skip-existing`.
- The release workflows now use a supported macOS Intel runner for private-repo wheel builds.

### Fixed

- Duplicate inserts now raise a dedicated error instead of silently behaving like upserts.
- `open()` in the Python binding now raises `VectLiteError` for vectlite-specific failures.
- The Python package now exposes `__version__`.
- Package metadata now includes project URLs for the repository, issues, and changelog.
