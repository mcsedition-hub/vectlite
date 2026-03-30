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
}

export interface BulkIngestOptions {
  namespace?: string | null
  batchSize?: number
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
  explain?: boolean
  queryVectors?: { [name: string]: number[] } | null
  vectorWeights?: { [name: string]: number } | null
}

export interface OpenOptions {
  dimension?: number | null
  readOnly?: boolean
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
  readonly readOnly: boolean

  count(): number
  namespaces(): string[]
  transaction(): Transaction
  insert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  upsert(id: string, vector: number[], metadata?: Metadata | null, options?: WriteOptions): void
  insertMany(records: Record[], options?: { namespace?: string | null }): number
  upsertMany(records: Record[], options?: { namespace?: string | null }): number
  bulkIngest(records: Record[], options?: BulkIngestOptions): number
  get(id: string, options?: { namespace?: string | null }): Record | null
  delete(id: string, options?: { namespace?: string | null }): boolean
  deleteMany(ids: string[], options?: { namespace?: string | null }): number
  flush(): void
  compact(): void
  snapshot(dest: string): void
  backup(dest: string): void
  search(query?: number[] | null, options?: SearchOptions): SearchResult[]
  searchWithStats(query?: number[] | null, options?: SearchOptions): SearchResponse
}

export class Store {
  readonly root: string
  createCollection(name: string, dimension: number): Database
  openCollection(name: string): Database
  openOrCreateCollection(name: string, dimension: number): Database
  openCollectionReadOnly(name: string): Database
  dropCollection(name: string): boolean
  collections(): string[]
}

export function open(path: string, options?: OpenOptions): Database
export function openStore(root: string): Store
export function restore(source: string, dest: string): Database
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
