const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const vectlite = require('..')

function tempPath(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vectlite-node-'))
  return path.join(dir, name)
}

test('database crud and search', () => {
  const dbPath = tempPath('knowledge.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })

  db.upsert('doc1', [1, 0], { source: 'docs', title: 'Auth' }, { sparse: { auth: 1 } })
  db.upsert('doc2', [0, 1], { source: 'notes', title: 'Billing' }, { sparse: { billing: 1 } })

  const record = db.get('doc1')
  assert.equal(record.id, 'doc1')
  assert.equal(record.metadata.source, 'docs')

  const results = db.search([1, 0], { k: 2, filter: { source: 'docs' } })
  assert.equal(results.length, 1)
  assert.equal(results[0].id, 'doc1')

  const objectResults = db.search({ query: [1, 0], k: 2, filter: { source: 'docs' } })
  assert.equal(objectResults.length, 1)
  assert.equal(objectResults[0].id, 'doc1')

  // search(query, k) shorthand
  const shorthandResults = db.search([1, 0], 2)
  assert.equal(shorthandResults.length, 2)
  assert.equal(shorthandResults[0].id, 'doc1')

  const shorthandStats = db.searchWithStats([1, 0], 2)
  assert.equal(shorthandStats.results.length, 2)
  assert.equal(typeof shorthandStats.stats.used_ann, 'boolean')

  const outcome = db.searchWithStats([1, 0], { k: 2, sparse: { auth: 1 }, fusion: 'rrf' })
  assert.equal(Array.isArray(outcome.results), true)
  assert.equal(typeof outcome.stats.used_ann, 'boolean')
})

test('quantization and multi-vector quantization are exposed on public wrapper', () => {
  const root = tempPath('quant-store')
  fs.mkdirSync(root, { recursive: true })
  const store = vectlite.openStore(root)
  const db = store.createCollection('c', 4)

  assert.equal(typeof db.enableQuantization, 'function')
  assert.equal(typeof db.disableQuantization, 'function')
  assert.equal(typeof db.enableMultiVectorQuantization, 'function')
  assert.equal(typeof db.disableMultiVectorQuantization, 'function')
  assert.equal(typeof db.isMultiVectorQuantized, 'function')

  for (let i = 0; i < 40; i += 1) {
    db.upsert(`doc${i}`, [i === 0 ? 1 : 0, i === 1 ? 1 : 0, 0, 0])
  }

  db.enableQuantization('scalar', { rescoreMultiplier: 1 })
  assert.equal(db.isQuantized, true)
  assert.equal(db.quantizationMethod, 'scalar')
  const outcome = db.searchWithStats([1, 0, 0, 0], { k: 5 })
  assert.equal(outcome.stats.ann_candidate_count, 5)

  db.disableQuantization()
  assert.equal(db.isQuantized, false)
  assert.equal(db.quantizationMethod, null)

  db.upsertMultiVectors(
    'mv1',
    [1, 0, 0, 0],
    { colbert: [[1, 0, 0, 0], [0.8, 0.2, 0, 0]] },
    { metadata: { source: 'mv' } },
  )
  const mvResults = db.searchMultiVector('colbert', [[1, 0, 0, 0]], { k: 1 })
  assert.equal(mvResults[0].id, 'mv1')

  db.enableMultiVectorQuantization('colbert', { rescoreMultiplier: 2 })
  assert.equal(db.isMultiVectorQuantized('colbert'), true)
  db.disableMultiVectorQuantization('colbert')
  assert.equal(db.isMultiVectorQuantized('colbert'), false)
})

test('product quantization default is dimension-compatible and invalid config is catchable', () => {
  const root = tempPath('pq-store')
  fs.mkdirSync(root, { recursive: true })
  const db = vectlite.openStore(root).createCollection('c', 146)

  const records = []
  for (let i = 0; i < 8; i += 1) {
    records.push({ id: `doc${i}`, vector: Array.from({ length: 146 }, (_, j) => (i + j) / 100) })
  }
  db.upsertMany(records)
  assert.deepEqual(db.validNumSubVectors(), [1, 2, 73, 146])

  assert.throws(
    () => db.enableQuantization('pq', { numSubVectors: 16, numCentroids: 4, trainingIterations: 1 }),
    /dimension \(146\) must be divisible by num_sub_vectors \(16\)/,
  )

  assert.doesNotThrow(() =>
    db.enableQuantization('PQ', { numCentroids: 4, trainingIterations: 1 }),
  )
  assert.equal(db.quantizationMethod, 'product')
})

test('store and text helpers', () => {
  const root = tempPath('store-root')
  fs.mkdirSync(root, { recursive: true })

  const store = vectlite.openStore(root)
  const db = store.createCollection('products', 2)

  vectlite.upsertText(db, 'p1', 'Auth guide', (text) => [text.length, 1], { source: 'docs' })
  const results = vectlite.searchText(db, 'auth', (text) => [text.length, 1], { k: 1 })

  assert.equal(results.length, 1)
  assert.equal(store.collections()[0], 'products')
})

test('count/list/deleteByFilter/close and lockTimeout are exposed in node wrapper', () => {
  const dbPath = tempPath('node-wrapper.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })

  db.upsert('user-1', [1, 0], { type: 'user', score: 10 }, { namespace: 'users' })
  db.upsert('user-2', [0.8, 0.2], { type: 'user', score: 3 }, { namespace: 'users' })
  db.upsert('feedback-1', [0, 1], { type: 'feedback', score: 2 }, { namespace: 'feedback' })

  assert.equal(db.count(), 3)
  assert.equal(db.count({ namespace: 'users' }), 2)
  assert.equal(db.count({ filter: { type: 'user' } }), 2)
  assert.equal(db.count({ namespace: 'users', filter: { score: { $gt: 5 } } }), 1)

  const listed = db.list({ namespace: 'users', filter: { type: 'user' }, limit: 1 })
  assert.equal(listed.length, 1)
  assert.equal(listed[0].namespace, 'users')

  assert.equal(db.deleteByFilter({ type: 'feedback' }, { namespace: 'feedback' }), 1)
  assert.equal(db.count(), 2)

  assert.throws(() => vectlite.open(dbPath, { lockTimeout: 0.05 }), /lock contention/)
  assert.throws(() => vectlite.open(dbPath, { lockTimeout: -1 }), /lock_timeout/)

  db.close()

  assert.throws(() => db.count(), /database is closed/)
  assert.throws(() => db.get('user-1'), /database is closed/)
  assert.throws(() => db.list(), /database is closed/)
})

test('text helpers support async embedders', async () => {
  const dbPath = tempPath('knowledge-async.vdb')
  const db = vectlite.open(dbPath, { dimension: 2 })

  await vectlite.upsertText(
    db,
    'doc1',
    'Auth guide',
    async (text) => [text.length, 1],
    { source: 'docs' },
  )

  const results = await vectlite.searchText(db, 'auth', async (text) => [text.length, 1], { k: 1 })
  const outcome = await vectlite.searchTextWithStats(
    db,
    'auth',
    async (text) => [text.length, 1],
    { k: 1 },
  )

  assert.equal(results.length, 1)
  assert.equal(results[0].id, 'doc1')
  assert.equal(Array.isArray(outcome.results), true)
})

// -------------------------------------------------------------------
// Bug #14: zero-norm query vector should be rejected for cosine
// -------------------------------------------------------------------

test('search with zero-norm query throws for cosine', () => {
  const dbPath = tempPath('zero-norm.vdb')
  const db = vectlite.open(dbPath, { dimension: 3 })
  db.upsert('a', [1, 0, 0])
  assert.throws(() => db.search([0, 0, 0], { k: 5 }), /zero norm/)
})

test('search with zero-norm query allowed for euclidean', () => {
  const dbPath = tempPath('zero-norm-euc.vdb')
  const db = vectlite.open(dbPath, { dimension: 3, metric: 'euclidean' })
  db.upsert('a', [1, 0, 0])
  const results = db.search([0, 0, 0], { k: 5 })
  assert.equal(results.length, 1)
})

// -------------------------------------------------------------------
// Bug #15: dimension mismatch in search query should be rejected
// -------------------------------------------------------------------

test('search with wrong dimension throws', () => {
  const dbPath = tempPath('dim-mismatch.vdb')
  const db = vectlite.open(dbPath, { dimension: 4 })
  db.upsert('a', [1, 0, 0, 0])

  // Undersized
  assert.throws(() => db.search([1, 0], { k: 5 }), /dimension mismatch/)
  // Oversized
  assert.throws(() => db.search([1, 0, 0, 0, 0, 0], { k: 5 }), /dimension mismatch/)
})

test('search undersized query with truncateDim succeeds', () => {
  const dbPath = tempPath('dim-trunc.vdb')
  const db = vectlite.open(dbPath, { dimension: 4 })
  db.upsert('a', [1, 0, 0, 0])
  const results = db.search([1, 0], { k: 5, truncateDim: 2 })
  assert.equal(results.length, 1)
})

// -------------------------------------------------------------------
// Bug #16: Store.close() should exist
// -------------------------------------------------------------------

test('store has close method', () => {
  const root = tempPath('store-close')
  fs.mkdirSync(root, { recursive: true })
  const store = vectlite.openStore(root)
  const db = store.createCollection('c', 3)
  db.upsert('a', [1, 0, 0])
  db.close()
  // Store.close() should not throw
  store.close()
})
