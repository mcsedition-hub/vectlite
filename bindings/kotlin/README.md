# vectlite Kotlin

Experimental Kotlin/JVM bindings for vectlite generated with UniFFI.

The generated Kotlin source lives in `../uniffi/generated/kotlin` and is compiled into this Gradle package. The JVM binding uses JNA to load the native Rust library, so consumers must make `libvectlite_uniffi` available on the native library path or set:

```bash
-Duniffi.component.vectlite.libraryOverride=/absolute/path/to/libvectlite_uniffi.dylib
```

## Requirements

- **JDK 17+** -- the build uses `jvmToolchain(17)`, so Gradle will auto-provision a compatible JDK even if `JAVA_HOME` points to a newer version (e.g. JDK 25).
- **Rust toolchain** -- Cargo is invoked automatically by the `buildNative` Gradle task.

## Development

```bash
cd bindings/kotlin
gradle test
```

The `test` task depends on `buildNative`, which runs:

```bash
cargo build -p vectlite-uniffi
```

Use a release native library with:

```bash
gradle test -PnativeProfile=release
```

## Usage

```kotlin
import uniffi.vectlite.Database

val db = Database.openOrCreate("knowledge.vdb", 384u, "cosine")
db.upsert("doc1", embedding, """{"source":"docs"}""", null, null)

val results = db.search(
    query,
    5u,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
)

db.close()
```

### Tuning the HNSW index

```kotlin
import uniffi.vectlite.Database

val db = Database.openOrCreate("knowledge.vdb", 384u, "cosine")

// Higher recall, slightly slower build/search
db.bulkIngestTuned(
    recordsJson,
    batchSize = 5000u,
    m = 32u,
    efConstruction = 400u,
    efSearch = 200u,
    parallelInsertThreshold = null,
)

// Adjust tuning at any time without re-ingesting:
db.setIndexConfig(m = 32u, efConstruction = 400u, efSearch = null, parallelInsertThreshold = null)
db.setEfSearch(efSearch = 200u)            // query-time only, no rebuild
val cfg = db.indexConfig()                  // IndexConfigResult
println("m=${cfg.m} efC=${cfg.efConstruction} efS=${cfg.efSearch}")
```

Use higher `m` / `efConstruction` / `efSearch` to push Recall@10 toward `1.0`;
use lower values when latency or memory matter more than recall. Pass `0u` for
`efSearch` to revert to the auto-derived default.
