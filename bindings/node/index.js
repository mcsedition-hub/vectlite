'use strict'

const native = require('./vectlite.node')

const TOKEN_RE = /[a-z0-9]+/g

class VectLiteError extends Error {
  constructor(message, cause) {
    super(message)
    this.name = 'VectLiteError'
    if (cause !== undefined) {
      this.cause = cause
    }
  }
}

function wrapError(fn) {
  try {
    return fn()
  } catch (error) {
    if (error instanceof VectLiteError) {
      throw error
    }
    throw new VectLiteError(error?.message ?? String(error), error)
  }
}

function encode(value) {
  return value == null ? null : JSON.stringify(value)
}

function decode(value) {
  return value == null ? null : JSON.parse(value)
}

function asArray(values) {
  return Array.from(values)
}

function normalizeWriteOptions(options = {}) {
  return {
    namespace: options.namespace ?? null,
    sparse: options.sparse ?? null,
    vectors: options.vectors ?? null,
  }
}

function sparseTerms(text) {
  const counts = {}
  const tokens = String(text).toLowerCase().match(TOKEN_RE) ?? []
  if (tokens.length === 0) {
    return counts
  }

  const total = tokens.length
  for (const token of tokens) {
    counts[token] = (counts[token] ?? 0) + (1 / total)
  }
  return counts
}

class Transaction {
  constructor(nativeTx) {
    this._native = nativeTx
  }

  count() {
    return wrapError(() => this._native.count())
  }

  insert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.insert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors)),
    )
  }

  upsert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.upsert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors)),
    )
  }

  insertMany(records, options = {}) {
    return wrapError(() => this._native.insertMany(encode(records), options.namespace ?? null))
  }

  upsertMany(records, options = {}) {
    return wrapError(() => this._native.upsertMany(encode(records), options.namespace ?? null))
  }

  delete(id, options = {}) {
    return wrapError(() => this._native.delete(id, options.namespace ?? null))
  }

  deleteMany(ids, options = {}) {
    return wrapError(() => this._native.deleteMany(ids, options.namespace ?? null))
  }

  commit() {
    return wrapError(() => this._native.commit())
  }

  rollback() {
    return wrapError(() => this._native.rollback())
  }
}

class Database {
  constructor(nativeDb) {
    this._native = nativeDb
  }

  get path() {
    return wrapError(() => this._native.path)
  }

  get walPath() {
    return wrapError(() => this._native.walPath)
  }

  get dimension() {
    return wrapError(() => this._native.dimension)
  }

  get readOnly() {
    return wrapError(() => this._native.readOnly)
  }

  count() {
    return wrapError(() => this._native.count())
  }

  namespaces() {
    return wrapError(() => this._native.namespaces())
  }

  transaction() {
    return wrapError(() => new Transaction(this._native.transaction()))
  }

  insert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.insert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors)),
    )
  }

  upsert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.upsert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors)),
    )
  }

  insertMany(records, options = {}) {
    return wrapError(() => this._native.insertMany(encode(records), options.namespace ?? null))
  }

  upsertMany(records, options = {}) {
    return wrapError(() => this._native.upsertMany(encode(records), options.namespace ?? null))
  }

  bulkIngest(records, options = {}) {
    return wrapError(() =>
      this._native.bulkIngest(encode(records), options.namespace ?? null, options.batchSize ?? 10_000),
    )
  }

  get(id, options = {}) {
    return wrapError(() => decode(this._native.get(id, options.namespace ?? null)))
  }

  delete(id, options = {}) {
    return wrapError(() => this._native.delete(id, options.namespace ?? null))
  }

  deleteMany(ids, options = {}) {
    return wrapError(() => this._native.deleteMany(ids, options.namespace ?? null))
  }

  flush() {
    return wrapError(() => this._native.flush())
  }

  compact() {
    return wrapError(() => this._native.compact())
  }

  snapshot(dest) {
    return wrapError(() => this._native.snapshot(dest))
  }

  backup(dest) {
    return wrapError(() => this._native.backup(dest))
  }

  search(query = null, options = {}) {
    return wrapError(() => decode(this._native.search(query == null ? null : asArray(query), encode(options))))
  }

  searchWithStats(query = null, options = {}) {
    return wrapError(() =>
      decode(this._native.searchWithStats(query == null ? null : asArray(query), encode(options))),
    )
  }
}

class Store {
  constructor(nativeStore) {
    this._native = nativeStore
  }

  get root() {
    return wrapError(() => this._native.root)
  }

  createCollection(name, dimension) {
    return wrapError(() => new Database(this._native.createCollection(name, dimension)))
  }

  openCollection(name) {
    return wrapError(() => new Database(this._native.openCollection(name)))
  }

  openOrCreateCollection(name, dimension) {
    return wrapError(() => new Database(this._native.openOrCreateCollection(name, dimension)))
  }

  openCollectionReadOnly(name) {
    return wrapError(() => new Database(this._native.openCollectionReadOnly(name)))
  }

  dropCollection(name) {
    return wrapError(() => this._native.dropCollection(name))
  }

  collections() {
    return wrapError(() => this._native.collections())
  }
}

function open(path, options = {}) {
  return wrapError(() => new Database(native.open(path, options.dimension ?? null, options.readOnly ?? false)))
}

function openStore(root) {
  return wrapError(() => new Store(native.openStore(root)))
}

function restore(source, dest) {
  return wrapError(() => new Database(native.restore(source, dest)))
}

function upsertText(db, id, text, embed, metadata = null, options = {}) {
  const payload = { ...(metadata ?? {}) }
  if (payload.text === undefined) {
    payload.text = text
  }
  return db.upsert(id, asArray(embed(text)), payload, { ...options, sparse: sparseTerms(text) })
}

function searchText(db, text, embed, options = {}) {
  return db.search(asArray(embed(text)), { ...options, sparse: sparseTerms(text) })
}

function searchTextWithStats(db, text, embed, options = {}) {
  return db.searchWithStats(asArray(embed(text)), { ...options, sparse: sparseTerms(text) })
}

module.exports = {
  Database,
  Store,
  Transaction,
  VectLiteError,
  open,
  openStore,
  restore,
  sparseTerms,
  upsertText,
  searchText,
  searchTextWithStats,
}
