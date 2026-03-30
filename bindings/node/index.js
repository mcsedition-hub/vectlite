'use strict'

const fs = require('node:fs')
const path = require('node:path')

function linuxLibc() {
  if (process.platform !== 'linux') {
    return null
  }

  const report = process.report?.getReport?.()
  return report?.header?.glibcVersionRuntime ? 'gnu' : 'musl'
}

function runtimePrebuildTag() {
  switch (process.platform) {
    case 'darwin':
      if (process.arch === 'x64') return 'darwin-x64'
      if (process.arch === 'arm64') return 'darwin-arm64'
      return null
    case 'linux': {
      const libc = linuxLibc()
      if (process.arch === 'x64' && libc === 'gnu') return 'linux-x64-gnu'
      if (process.arch === 'arm64' && libc === 'gnu') return 'linux-arm64-gnu'
      return null
    }
    case 'win32':
      if (process.arch === 'x64') return 'win32-x64-msvc'
      if (process.arch === 'arm64') return 'win32-arm64-msvc'
      return null
    default:
      return null
  }
}

function loadNative() {
  const candidates = []
  const prebuildTag = runtimePrebuildTag()
  if (prebuildTag != null) {
    candidates.push(path.join(__dirname, 'prebuilds', prebuildTag, 'vectlite.node'))
  }
  candidates.push(path.join(__dirname, 'vectlite.node'))

  const errors = []
  for (const candidate of candidates) {
    if (!fs.existsSync(candidate)) {
      continue
    }
    try {
      return require(candidate)
    } catch (error) {
      errors.push(`${candidate}: ${error?.message ?? String(error)}`)
    }
  }

  const detail = errors.length === 0 ? 'No compatible prebuilt binary was found.' : errors.join('\n')
  throw new Error(
    `Unable to load the vectlite native addon.\n${detail}\n` +
      'If this platform is not covered by prebuilt binaries, install Rust/Cargo and reinstall the package.',
  )
}

const native = loadNative()

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

function wrapAsync(value) {
  return Promise.resolve(value).catch((error) => {
    if (error instanceof VectLiteError) {
      throw error
    }
    throw new VectLiteError(error?.message ?? String(error), error)
  })
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

function isPromiseLike(value) {
  return value != null && typeof value.then === 'function'
}

function embedText(text, embed) {
  const embedded = wrapError(() => embed(text))
  if (isPromiseLike(embedded)) {
    return wrapAsync(embedded).then((vector) => asArray(vector))
  }
  return asArray(embedded)
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
  const vector = embedText(text, embed)
  const writeOptions = { ...options, sparse: sparseTerms(text) }
  if (isPromiseLike(vector)) {
    return wrapAsync(vector.then((resolved) => db.upsert(id, resolved, payload, writeOptions)))
  }
  return db.upsert(id, vector, payload, writeOptions)
}

function searchText(db, text, embed, options = {}) {
  const vector = embedText(text, embed)
  const searchOptions = { ...options, sparse: sparseTerms(text) }
  if (isPromiseLike(vector)) {
    return wrapAsync(vector.then((resolved) => db.search(resolved, searchOptions)))
  }
  return db.search(vector, searchOptions)
}

function searchTextWithStats(db, text, embed, options = {}) {
  const vector = embedText(text, embed)
  const searchOptions = { ...options, sparse: sparseTerms(text) }
  if (isPromiseLike(vector)) {
    return wrapAsync(vector.then((resolved) => db.searchWithStats(resolved, searchOptions)))
  }
  return db.searchWithStats(vector, searchOptions)
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
