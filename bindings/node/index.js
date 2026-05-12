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
let otelTracer = null

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

function configureOpenTelemetry(options = {}) {
  if (options === false || options?.enabled === false) {
    otelTracer = null
    return null
  }
  if (options?.tracer != null) {
    otelTracer = options.tracer
    return otelTracer
  }
  try {
    const { trace } = require('@opentelemetry/api')
    otelTracer = trace.getTracer(options?.tracerName ?? 'vectlite')
    return otelTracer
  } catch {
    otelTracer = null
    return null
  }
}

function searchAttributes(query, options, stats = null) {
  const attrs = {
    'db.system': 'vectlite',
    'db.operation.name': 'search',
    'vectlite.search.k': options?.k ?? 10,
    'vectlite.search.namespace': options?.namespace ?? '',
    'vectlite.search.all_namespaces': Boolean(options?.allNamespaces),
    'vectlite.search.has_dense': query != null,
    'vectlite.search.has_sparse': options?.sparse != null,
    'vectlite.search.fusion': options?.fusion ?? 'linear',
  }
  if (options?.vectorName != null) attrs['vectlite.search.vector_name'] = options.vectorName
  if (options?.truncateDim != null) attrs['vectlite.search.truncate_dim'] = options.truncateDim
  if (stats != null) {
    attrs['vectlite.search.used_ann'] = Boolean(stats.used_ann)
    attrs['vectlite.search.exact_fallback'] = Boolean(stats.exact_fallback)
    attrs['vectlite.search.considered_count'] = stats.considered_count ?? 0
    attrs['vectlite.search.result_count'] = stats.result_count ?? 0
    attrs['vectlite.search.effective_dimension'] = stats.effective_dimension ?? 0
    attrs['vectlite.search.matryoshka_truncated'] = Boolean(stats.matryoshka_truncated)
    attrs['vectlite.search.total_us'] = stats.timings?.total_us ?? 0
  }
  return attrs
}

function withSearchSpan(query, options, fn) {
  if (otelTracer == null) {
    return fn()
  }
  return otelTracer.startActiveSpan('vectlite.search', { attributes: searchAttributes(query, options) }, (span) => {
    try {
      const value = fn()
      if (isPromiseLike(value)) {
        return value.then(
          (resolved) => {
            span.setAttributes(searchAttributes(query, options, resolved?.stats ?? null))
            span.end()
            return resolved
          },
          (error) => {
            span.recordException?.(error)
            span.setStatus?.({ code: 2, message: error?.message ?? String(error) })
            span.end()
            throw error
          },
        )
      }
      span.setAttributes(searchAttributes(query, options, value?.stats ?? null))
      span.end()
      return value
    } catch (error) {
      span.recordException?.(error)
      span.setStatus?.({ code: 2, message: error?.message ?? String(error) })
      span.end()
      throw error
    }
  })
}

function asArray(values) {
  if (
    values != null &&
    typeof values === 'object' &&
    !Array.isArray(values) &&
    !ArrayBuffer.isView(values) &&
    typeof values[Symbol.iterator] !== 'function' &&
    typeof values.length !== 'number'
  ) {
    throw new TypeError('vector must be an array-like or iterable of numbers')
  }
  return Array.from(values)
}

function encodeNativeOptions(value) {
  return typeof value === 'string' ? value : encode(value)
}

const SEARCH_OPTION_KEYS = new Set([
  'query',
  'k',
  'filter',
  'namespace',
  'allNamespaces',
  'sparse',
  'denseWeight',
  'sparseWeight',
  'fetchK',
  'mmrLambda',
  'vectorName',
  'fusion',
  'rrfK',
  'truncateDim',
  'explain',
  'queryVectors',
  'vectorWeights',
])

function isSearchRequestObject(value) {
  return (
    value != null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    !ArrayBuffer.isView(value) &&
    [...SEARCH_OPTION_KEYS].some((key) => Object.prototype.hasOwnProperty.call(value, key))
  )
}

function normalizeSearchArgs(query, options) {
  if (isSearchRequestObject(query) && (options == null || Object.keys(options).length === 0)) {
    const { query: normalizedQuery = null, ...normalizedOptions } = query
    return { query: normalizedQuery, options: normalizedOptions }
  }
  if (typeof options === 'number') {
    return { query, options: { k: options } }
  }
  return { query, options: options ?? {} }
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
    ttl: options.ttl ?? null,
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
    const { namespace, sparse, vectors, ttl } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.insert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors), ttl),
    )
  }

  upsert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors, ttl } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.upsert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors), ttl),
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

  get metric() {
    return wrapError(() => this._native.metric)
  }

  get readOnly() {
    return wrapError(() => this._native.readOnly)
  }

  count(options = {}) {
    return wrapError(() => this._native.count(options.namespace ?? null, encode(options.filter)))
  }

  namespaces() {
    return wrapError(() => this._native.namespaces())
  }

  close() {
    return wrapError(() => this._native.close())
  }

  list(options = {}) {
    return wrapError(() =>
      decode(
        this._native.list(
          options.namespace ?? null,
          encode(options.filter),
          options.limit ?? null,
          options.offset ?? null,
        ),
      ),
    )
  }

  listCursor(options = {}) {
    return wrapError(() => {
      const raw = decode(
        this._native.listCursor(
          options.namespace ?? null,
          encode(options.filter),
          options.limit ?? null,
          options.cursor ?? null,
        ),
      )
      return { records: raw.records, cursor: raw.cursor ?? null }
    })
  }

  transaction() {
    return wrapError(() => new Transaction(this._native.transaction()))
  }

  insert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors, ttl } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.insert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors), ttl),
    )
  }

  upsert(id, vector, metadata = null, options = {}) {
    const { namespace, sparse, vectors, ttl } = normalizeWriteOptions(options)
    return wrapError(() =>
      this._native.upsert(id, asArray(vector), encode(metadata), namespace, encode(sparse), encode(vectors), ttl),
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

  deleteByFilter(filter, options = {}) {
    return wrapError(() => this._native.deleteByFilter(encode(filter), options.namespace ?? null))
  }

  updateMetadata(id, metadata, options = {}) {
    return wrapError(() =>
      this._native.updateMetadata(id, encode(metadata), options.namespace ?? null),
    )
  }

  setTtl(id, ttl, options = {}) {
    return wrapError(() => this._native.setTtl(id, ttl, options.namespace ?? null))
  }

  clearTtl(id, options = {}) {
    return wrapError(() => this._native.clearTtl(id, options.namespace ?? null))
  }

  createIndex(field, indexType) {
    return wrapError(() => this._native.createIndex(field, indexType))
  }

  dropIndex(field) {
    return wrapError(() => this._native.dropIndex(field))
  }

  listIndexes() {
    return wrapError(() => decode(this._native.listIndexes()))
  }

  enableQuantization(method = 'scalar', options = {}) {
    return wrapError(() => this._native.enableQuantization(method, encodeNativeOptions(options)))
  }

  disableQuantization() {
    return wrapError(() => this._native.disableQuantization())
  }

  get isQuantized() {
    return wrapError(() => this._native.isQuantized)
  }

  get quantizationMethod() {
    return wrapError(() => this._native.quantizationMethod)
  }

  validNumSubVectors() {
    return wrapError(() => this._native.validNumSubVectors())
  }

  upsertMultiVectors(id, vector, multiVectors, options = {}) {
    return wrapError(() =>
      this._native.upsertMultiVectors(id, asArray(vector), encode(multiVectors), encode(options)),
    )
  }

  searchMultiVector(space, queryTokens, options = {}) {
    return wrapError(() =>
      decode(this._native.searchMultiVector(space, encode(queryTokens), encode(options))),
    )
  }

  enableMultiVectorQuantization(space, options = {}) {
    return wrapError(() =>
      this._native.enableMultiVectorQuantization(space, encodeNativeOptions(options)),
    )
  }

  disableMultiVectorQuantization(space) {
    return wrapError(() => this._native.disableMultiVectorQuantization(space))
  }

  isMultiVectorQuantized(space) {
    return wrapError(() => this._native.isMultiVectorQuantized(space))
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
    const normalized = normalizeSearchArgs(query, options)
    return withSearchSpan(normalized.query, normalized.options, () =>
      wrapError(() =>
        decode(
          this._native.search(
            normalized.query == null ? null : asArray(normalized.query),
            encode(normalized.options),
          ),
        ),
      ),
    )
  }

  searchWithStats(query = null, options = {}) {
    const normalized = normalizeSearchArgs(query, options)
    return withSearchSpan(normalized.query, normalized.options, () =>
      wrapError(() =>
        decode(
          this._native.searchWithStats(
            normalized.query == null ? null : asArray(normalized.query),
            encode(normalized.options),
          ),
        ),
      ),
    )
  }

  searchAsync(query = null, options = {}) {
    const normalized = normalizeSearchArgs(query, options)
    return withSearchSpan(normalized.query, normalized.options, () =>
      wrapAsync(
        this._native.searchAsync(
          normalized.query == null ? null : asArray(normalized.query),
          encode(normalized.options),
        ),
      ).then(decode),
    )
  }

  searchWithStatsAsync(query = null, options = {}) {
    const normalized = normalizeSearchArgs(query, options)
    return withSearchSpan(normalized.query, normalized.options, () =>
      wrapAsync(
        this._native.searchWithStatsAsync(
          normalized.query == null ? null : asArray(normalized.query),
          encode(normalized.options),
        ),
      ).then(decode),
    )
  }

  flushAsync() {
    return wrapAsync(this._native.flushAsync())
  }

  compactAsync() {
    return wrapAsync(this._native.compactAsync())
  }

  bulkIngestAsync(records, options = {}) {
    return wrapAsync(
      this._native.bulkIngestAsync(encode(records), options.namespace ?? null, options.batchSize ?? 10_000),
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

  close() {
    return wrapError(() => this._native.close())
  }
}

function open(path, options = {}) {
  return wrapError(() =>
    new Database(native.open(path, options.dimension ?? null, options.readOnly ?? false, options.lockTimeout ?? null, options.metric ?? null)),
  )
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
  configureOpenTelemetry,
  open,
  openStore,
  restore,
  sparseTerms,
  upsertText,
  searchText,
  searchTextWithStats,
}
