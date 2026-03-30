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
