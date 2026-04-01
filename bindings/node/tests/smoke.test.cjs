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

  const outcome = db.searchWithStats([1, 0], { k: 2, sparse: { auth: 1 }, fusion: 'rrf' })
  assert.equal(Array.isArray(outcome.results), true)
  assert.equal(typeof outcome.stats.used_ann, 'boolean')
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
