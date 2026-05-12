const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const vectlite = require('..')

function tempPath(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vectlite-otel-'))
  return path.join(dir, name)
}

// -- Fake tracer / span for testing (mimics @opentelemetry/api interface) --

function createFakeTracer() {
  const spans = []

  class FakeSpan {
    constructor(name, options) {
      this.name = name
      this.initialAttributes = { ...(options?.attributes ?? {}) }
      this.allAttributes = { ...(options?.attributes ?? {}) }
      this.ended = false
      this.exceptions = []
      this.status = null
    }

    setAttributes(attrs) {
      Object.assign(this.allAttributes, attrs)
    }

    recordException(err) {
      this.exceptions.push(err)
    }

    setStatus(status) {
      this.status = status
    }

    end() {
      this.ended = true
      spans.push(this)
    }
  }

  const tracer = {
    startActiveSpan(name, options, fn) {
      const span = new FakeSpan(name, options)
      return fn(span)
    },
  }

  return { tracer, spans }
}

test('configureOpenTelemetry with custom tracer', () => {
  const { tracer } = createFakeTracer()
  const result = vectlite.configureOpenTelemetry({ tracer })
  assert.strictEqual(result, tracer)

  // Disable
  const disabled = vectlite.configureOpenTelemetry(false)
  assert.strictEqual(disabled, null)
})

test('configureOpenTelemetry(false) disables tracing', () => {
  const result = vectlite.configureOpenTelemetry(false)
  assert.strictEqual(result, null)
})

test('configureOpenTelemetry({ enabled: false }) disables tracing', () => {
  const result = vectlite.configureOpenTelemetry({ enabled: false })
  assert.strictEqual(result, null)
})

test('configureOpenTelemetry() without @opentelemetry/api returns null', () => {
  // Since @opentelemetry/api is not installed in this test env, auto-detect returns null
  const result = vectlite.configureOpenTelemetry()
  assert.strictEqual(result, null)
})

test('search creates a span when tracer is configured', () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-search.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' })

  db.search([1, 0], { k: 5 })

  assert.equal(spans.length, 1)
  const span = spans[0]
  assert.equal(span.name, 'vectlite.search')
  assert.equal(span.ended, true)
  assert.equal(span.initialAttributes['db.system'], 'vectlite')
  assert.equal(span.initialAttributes['db.operation.name'], 'search')
  assert.equal(span.initialAttributes['vectlite.search.k'], 5)
  assert.equal(span.initialAttributes['vectlite.search.has_dense'], true)
  assert.equal(span.exceptions.length, 0)
  assert.strictEqual(span.status, null)

  db.close()
  vectlite.configureOpenTelemetry(false)
})

test('searchWithStats creates a span with stats attributes', () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-stats.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' })

  const outcome = db.searchWithStats([1, 0], { k: 5 })

  assert.equal(spans.length, 1)
  const span = spans[0]
  assert.equal(span.name, 'vectlite.search')
  // After completion, stats attributes should be set
  assert.equal(typeof span.allAttributes['vectlite.search.result_count'], 'number')
  assert.equal(typeof span.allAttributes['vectlite.search.total_us'], 'number')
  assert.equal(span.ended, true)

  db.close()
  vectlite.configureOpenTelemetry(false)
})

test('searchAsync creates a span for async search', async () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-async.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' })

  let nativeAvailable = true
  try {
    await db.searchAsync([1, 0], { k: 5 })
  } catch (err) {
    if (/not a function/.test(err.message)) {
      nativeAvailable = false
    } else {
      throw err
    }
  }

  if (nativeAvailable) {
    assert.equal(spans.length, 1)
    const span = spans[0]
    assert.equal(span.name, 'vectlite.search')
    assert.equal(span.ended, true)
  } else {
    // Native async not available in this build -- span still records the error
    assert.equal(spans.length, 1)
    assert.equal(spans[0].ended, true)
    assert.ok(spans[0].exceptions.length > 0)
  }

  db.close()
  vectlite.configureOpenTelemetry(false)
})

test('searchWithStatsAsync creates a span with stats', async () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-stats-async.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' })

  let nativeAvailable = true
  try {
    await db.searchWithStatsAsync([1, 0], { k: 5 })
  } catch (err) {
    if (/not a function/.test(err.message)) {
      nativeAvailable = false
    } else {
      throw err
    }
  }

  if (nativeAvailable) {
    assert.equal(spans.length, 1)
    const span = spans[0]
    assert.equal(span.name, 'vectlite.search')
    assert.equal(typeof span.allAttributes['vectlite.search.result_count'], 'number')
    assert.equal(span.ended, true)
  } else {
    assert.equal(spans.length, 1)
    assert.equal(spans[0].ended, true)
    assert.ok(spans[0].exceptions.length > 0)
  }

  db.close()
  vectlite.configureOpenTelemetry(false)
})

test('span records exception on search error', () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-error.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })

  // Search with wrong dimension vector should fail
  assert.throws(() => {
    db.search([1, 0, 3], { k: 5 })
  })

  assert.equal(spans.length, 1)
  const span = spans[0]
  assert.equal(span.name, 'vectlite.search')
  assert.equal(span.ended, true)
  assert.equal(span.exceptions.length, 1)
  assert.deepStrictEqual(span.status, { code: 2, message: span.exceptions[0]?.message ?? String(span.exceptions[0]) })

  db.close()
  vectlite.configureOpenTelemetry(false)
})

test('no span created when tracer is disabled', () => {
  vectlite.configureOpenTelemetry(false)

  const dbPath = tempPath('otel-none.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' })

  // This should work fine without any span overhead
  const results = db.search([1, 0], { k: 5 })
  assert.equal(results.length, 1)

  db.close()
})

test('span attributes include namespace and fusion', () => {
  const { tracer, spans } = createFakeTracer()
  vectlite.configureOpenTelemetry({ tracer })

  const dbPath = tempPath('otel-attrs.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })
  db.upsert('doc1', [1, 0], { source: 'docs' }, { namespace: 'ns1', sparse: { hello: 1 } })

  db.search([1, 0], {
    k: 3,
    namespace: 'ns1',
    sparse: { hello: 1 },
    fusion: 'rrf',
  })

  assert.equal(spans.length, 1)
  const attrs = spans[0].initialAttributes
  assert.equal(attrs['vectlite.search.k'], 3)
  assert.equal(attrs['vectlite.search.namespace'], 'ns1')
  assert.equal(attrs['vectlite.search.has_sparse'], true)
  assert.equal(attrs['vectlite.search.fusion'], 'rrf')

  db.close()
  vectlite.configureOpenTelemetry(false)
})
