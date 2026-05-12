# vectlite Kotlin

Experimental Kotlin/JVM bindings for vectlite generated with UniFFI.

The generated Kotlin source lives in `../uniffi/generated/kotlin` and is compiled into this Gradle package. The JVM binding uses JNA to load the native Rust library, so consumers must make `libvectlite_uniffi` available on the native library path or set:

```bash
-Duniffi.component.vectlite.libraryOverride=/absolute/path/to/libvectlite_uniffi.dylib
```

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
