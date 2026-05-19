export type MetadataValue =
  | string
  | number
  | boolean
  | null
  | MetadataValue[]
  | { [key: string]: MetadataValue }

export type Metadata = { [key: string]: MetadataValue }
export type SparseVector = { [term: string]: number }
export type NamedVectors = { [name: string]: number[] }
export type MultiVectors = { [space: string]: number[][] }
export type Filter = { [key: string]: unknown }
export type TextEmbedding = ArrayLike<number>
export type TextEmbeddingResult = TextEmbedding | Promise<TextEmbedding>
export type TextEmbedder = (text: string) => TextEmbeddingResult

export interface Record {
  namespace: string
  id: string
  vector: number[]
  vectors: NamedVectors
  sparse: SparseVector
  metadata: Metadata
  expires_at: number | null
}

export interface SearchTimings {
  dense_us: number
  sparse_us: number
  fusion_us: number
  total_us: number
}

export interface SearchStats {
  used_ann: boolean
  ann_candidate_count: number
  exact_fallback: boolean
  considered_count: number
  fetch_k: number
  mmr_applied: boolean
  sparse_candidate_count: number
  ann_loaded_from_disk: boolean
  wal_entries_replayed: number
  fusion: string
  effective_dimension: number
  matryoshka_truncated: boolean
  rerank_applied: boolean
  rerank_count: number
  timings: SearchTimings
}

export interface ExplainDetails {
  fusion?: string
  dense_score?: number
  sparse_score?: number
  matched_terms?: string[]
  vector_name?: string | null
  dense_rank?: number | null
  sparse_rank?: number | null
  bm25_term_scores?: { [term: string]: number }
}

export interface SearchResult {
  namespace: string
  id: string
  score: number
  dense_score: number
  sparse_score: number
  vector_name: string | null
  matched_terms: string[]
  dense_rank: number | null
  sparse_rank: number | null
  metadata: Metadata
  explain?: ExplainDetails
}

export interface SearchResponse {
  results: SearchResult[]
  stats: SearchStats
}

export interface WriteOptions {
  namespace?: string | null
  sparse?: SparseVector | null
  vectors?: NamedVectors | null
  ttl?: number | null
}

export interface CountOptions {
  namespace?: string | null
  filter?: Filter | null
}

export interface ListOptions extends CountOptions {
  limit?: number | null
  offset?: number | null
}

export interface ListCursorOptions extends CountOptions {
  limit?: number | null
  cursor?: string | null
}

export interface ListCursorResult {
  records: Record[]
  cursor: string | null
}

export interface BulkIngestOptions {
  namespace?: string | null
  batchSize?: number
  /** Max bidirectional links per HNSW node (default 16). */
  m?: number | null
  /** Build-time search width (default 200). Higher = better recall, slower build. */
  efConstruction?: number | null
  /** Query-time search width. `null` = auto (derived from k). */
  efSearch?: number | null
  /** Minimum dataset size to engage Rayon-parallel HNSW insertion (default 256). */
  parallelInsertThreshold?: number | null
  /**
   * Percentage (0..=100) of dead nodes at which `compact()` triggers an
   * HNSW rebuild. Default 30. Set to 100 to disable automatic rebuild.
   */
  tombstoneRebuildPct?: number | null
  /**
   * Soft cap on the number of vectors per HNSW segment. Once the active
   * segment hits this size, a new segment is started for subsequent
   * inserts. Default 50_000.
   */
  segmentSizeThreshold?: number | null
}

export interface BulkIngestArrayOptions extends BulkIngestOptions {
  /** Per-record metadata (one entry per id, may contain nulls). */
  metadata?: Array<object | null>
}

export interface IndexConfig {
  m: number
  ef_construction: number
  ef_search: number | null
  parallel_insert_threshold: number
  tombstone_rebuild_pct: number
  segment_size_threshold: number
}

export interface SetIndexConfigOptions {
  m?: number | null
  efConstruction?: number | null
  efSearch?: number | null
  parallelInsertThreshold?: number | null
  tombstoneRebuildPct?: number | null
  segmentSizeThreshold?: number | null
}

/** WAL fsync mode. See `Database.setWalSyncMode`. */
export type WalSyncMode = 'per_op' | 'every_n' | 'on_flush'

export type WalSyncModeInfo =
  | { mode: 'per_op' }
  | { mode: 'every_n'; n: number }
  | { mode: 'on_flush' }

export interface TombstoneStats {
  /** Live (non-tombstoned) records across all HNSW graphs. */
  live: number
  /** Dead (tombstoned) records still in the graphs, awaiting compact(). */
  dead: number
}

export interface SearchOptions {
  k?: number
  filter?: Filter | null
  namespace?: string | null
  allNamespaces?: boolean
  sparse?: SparseVector | null
  denseWeight?: number
  sparseWeight?: number
  fetchK?: number
  mmrLambda?: number | null
  vectorName?: string | null
  fusion?: 'linear' | 'rrf'
  rrfK?: number
  truncateDim?: number | null
  explain?: boolean
  queryVectors?: { [name: string]: number[] } | null
  vectorWeights?: { [name: string]: number } | null
}

export interface SearchRequest extends SearchOptions {
  query?: number[] | null
}

export type QuantizationMethod = 'scalar' | 'int8' | 'binary' | 'product' | 'pq'
export interface QuantizationOptions {
  rescoreMultiplier?: number
  rescore_multiplier?: number
  numSubVectors?: number
  num_sub_vectors?: number
  numCentroids?: number
  num_centroids?: number
  trainingIterations?: number
  training_iterations?: number
}

export interface MultiVectorWriteOptions {
  namespace?: string | null
  metadata?: Metadata | null
}

export interface MultiVectorSearchOptions {
  k?: number
  filter?: Filter | null
  namespace?: string | null
}

export interface MultiVectorSearchResult {
  namespace: string
  id: string
  score: number
  metadata: Metadata
}

export interface MultiVectorQuantizationOptions {
  method?: 'two_bit'
  rescoreMultiplier?: number
  rescore_multiplier?: number
}

export type DistanceMetric = 'cosine' | 'euclidean' | 'dotproduct' | 'manhattan' | 'l2' | 'dot' | 'ip' | 'l1'

export interface OpenOptions {
  dimension?: number | null
  readOnly?: boolean
  lockTimeout?: number | null
  metric?: DistanceMetric | null
}

export class VectLiteError extends Error {}

export class Transaction {
  count(): number
  insert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  upsert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  insertMany(records: Record[], options?: { namespace?: string | null }): number
  upsertMany(records: Record[], options?: { namespace?: string | null }): number
  delete(id: string, options?: { namespace?: string | null }): boolean
  deleteMany(ids: string[], options?: { namespace?: string | null }): number
  commit(): void
  rollback(): void
}

export class Database {
  readonly path: string
  readonly walPath: string
  readonly dimension: number
  readonly metric: string
  readonly readOnly: boolean

  count(options?: CountOptions): number
  namespaces(): string[]
  close(): void
  list(options?: ListOptions): Record[]
  listCursor(options?: ListCursorOptions): ListCursorResult
  transaction(): Transaction
  insert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  upsert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  insertMany(records: Record[], options?: { namespace?: string | null }): number
  upsertMany(records: Record[], options?: { namespace?: string | null }): number
  bulkIngest(records: Record[], options?: BulkIngestOptions): number
  /**
   * Zero-copy bulk ingest from a Float32Array. 10–30× faster than
   * `bulkIngest(records)` because the vector data isn't JSON-serialised
   * between JS and Rust.
   *
   * @param ids          N record ids
   * @param vectorsFlat  Float32Array of length N * dim (row-major)
   * @param dim          must equal db.dimension
   */
  bulkIngestArray(
    ids: string[],
    vectorsFlat: Float32Array,
    dim: number,
    options?: BulkIngestArrayOptions,
  ): number
  /**
   * Number of HNSW segments in the active index for the given
   * (namespace, vectorName). Returns 0 when no graph exists. Useful for
   * tests and observability.
   */
  annSegmentCount(namespace?: string | null, vectorName?: string | null): number
  /** Get the current HNSW configuration. */
  indexConfig(): IndexConfig
  /** Adjust query-time `ef_search` only (no rebuild). `null` reverts to auto. */
  setEfSearch(efSearch: number | null): void
  /** Update HNSW parameters; rebuilds the ANN graph if `m`/`efConstruction` changed. */
  setIndexConfig(config: SetIndexConfigOptions): void
  /**
   * Configure WAL durability.
   *
   * - `"per_op"`   (default): fsync after every insert. Strongest durability.
   * - `"every_n"`  : fsync once every `n` inserts (pass `n` as 2nd arg).
   * - `"on_flush"` : only fsync at `flush()` / `compact()` / `close()`.
   *
   * On macOS APFS, `"on_flush"` typically multiplies ingestion throughput
   * by 5–10× at the cost of losing un-flushed writes on a crash.
   */
  setWalSyncMode(mode: WalSyncMode, n?: number | null): void
  /** Return the current WAL sync mode. */
  walSyncMode(): WalSyncModeInfo
  /** Total live and tombstoned record counts across every HNSW graph. */
  tombstoneStats(): TombstoneStats
  /**
   * Materialise the contiguous-vector arena up front for cache- and
   * SIMD-friendly scans. Normally built lazily on first use.
   */
  prepareForScan(): void
  /**
   * Number of vectors in the contiguous arena, or `null` if it hasn't
   * been materialised yet in this session.
   */
  vectorArenaLen(): number | null
  get(id: string, options?: { namespace?: string | null }): Record | null
  delete(id: string, options?: { namespace?: string | null }): boolean
  deleteMany(ids: string[], options?: { namespace?: string | null }): number
  deleteByFilter(filter: Filter, options?: { namespace?: string | null }): number
  updateMetadata(id: string, metadata: Metadata, options?: { namespace?: string | null }): boolean
  setTtl(id: string, ttl: number, options?: { namespace?: string | null }): boolean
  clearTtl(id: string, options?: { namespace?: string | null }): boolean
  createIndex(field: string, indexType: 'keyword' | 'numeric'): boolean
  dropIndex(field: string): boolean
  listIndexes(): Array<{ field: string; type: 'keyword' | 'numeric' }>
  readonly isQuantized: boolean
  readonly quantizationMethod: 'scalar' | 'binary' | 'product' | null
  enableQuantization(method?: QuantizationMethod, options?: QuantizationOptions | string): void
  disableQuantization(): void
  validNumSubVectors(): number[]
  upsertMultiVectors(id: string, vector: number[], multiVectors: MultiVectors, options?: MultiVectorWriteOptions): void
  searchMultiVector(space: string, queryTokens: number[][], options?: MultiVectorSearchOptions): MultiVectorSearchResult[]
  enableMultiVectorQuantization(space: string, options?: MultiVectorQuantizationOptions | string): void
  disableMultiVectorQuantization(space: string): void
  isMultiVectorQuantized(space: string): boolean
  flush(): void
  compact(): void
  snapshot(dest: string): void
  backup(dest: string): void
  search(request: SearchRequest): SearchResult[]
  search(query: number[], k: number): SearchResult[]
  search(query?: number[] | null, options?: SearchOptions): SearchResult[]
  searchWithStats(request: SearchRequest): SearchResponse
  searchWithStats(query: number[], k: number): SearchResponse
  searchWithStats(query?: number[] | null, options?: SearchOptions): SearchResponse
  searchAsync(request: SearchRequest): Promise<SearchResult[]>
  searchAsync(query: number[], k: number): Promise<SearchResult[]>
  searchAsync(query?: number[] | null, options?: SearchOptions): Promise<SearchResult[]>
  searchWithStatsAsync(request: SearchRequest): Promise<SearchResponse>
  searchWithStatsAsync(query: number[], k: number): Promise<SearchResponse>
  searchWithStatsAsync(query?: number[] | null, options?: SearchOptions): Promise<SearchResponse>
  flushAsync(): Promise<void>
  compactAsync(): Promise<void>
  bulkIngestAsync(records: Record[], options?: BulkIngestOptions): Promise<number>
}

export class Store {
  readonly root: string
  createCollection(name: string, dimension: number): Database
  openCollection(name: string): Database
  openOrCreateCollection(name: string, dimension: number): Database
  openCollectionReadOnly(name: string): Database
  dropCollection(name: string): boolean
  collections(): string[]
  close(): void
}

export function open(path: string, options?: OpenOptions): Database
export function openStore(root: string): Store
export function restore(source: string, dest: string): Database
export interface OpenTelemetryOptions {
  /** Pass `false` or `{ enabled: false }` to disable tracing. */
  enabled?: boolean
  /** Supply your own OTel `Tracer` instance. */
  tracer?: unknown
  /** Tracer name used when auto-resolving via `@opentelemetry/api`. Defaults to `'vectlite'`. */
  tracerName?: string
}

/**
 * Configure optional OpenTelemetry tracing for search operations.
 *
 * When a tracer is active, every `search`, `searchWithStats`, `searchAsync`,
 * and `searchWithStatsAsync` call is wrapped in a span with semantic
 * `db.system` / `db.operation.name` attributes and search-specific metrics.
 *
 * `@opentelemetry/api` is loaded lazily via `require()` -- it is **not** a
 * runtime dependency. If the package is not installed the function returns
 * `null` and search calls remain un-instrumented.
 *
 * @returns The resolved tracer, or `null` if tracing could not be configured.
 */
export function configureOpenTelemetry(options?: OpenTelemetryOptions | false): unknown | null

export function sparseTerms(text: string): SparseVector
export function upsertText(
  db: Database,
  id: string,
  text: string,
  embed: (text: string) => TextEmbedding,
  metadata?: Metadata | null,
  options?: WriteOptions,
): void
export function upsertText(
  db: Database,
  id: string,
  text: string,
  embed: (text: string) => Promise<TextEmbedding>,
  metadata?: Metadata | null,
  options?: WriteOptions,
): Promise<void>
export function searchText(
  db: Database,
  text: string,
  embed: (text: string) => TextEmbedding,
  options?: SearchOptions,
): SearchResult[]
export function searchText(
  db: Database,
  text: string,
  embed: (text: string) => Promise<TextEmbedding>,
  options?: SearchOptions,
): Promise<SearchResult[]>
export function searchTextWithStats(
  db: Database,
  text: string,
  embed: (text: string) => TextEmbedding,
  options?: SearchOptions,
): SearchResponse
export function searchTextWithStats(
  db: Database,
  text: string,
  embed: (text: string) => Promise<TextEmbedding>,
  options?: SearchOptions,
): Promise<SearchResponse>
