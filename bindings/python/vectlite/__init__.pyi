from collections.abc import Callable, Sequence
from typing import Any, TypedDict

from . import analyzers, rerankers

__version__: str

MetadataValue = str | int | float | bool | None | list["MetadataValue"] | dict[str, "MetadataValue"]
Metadata = dict[str, MetadataValue]
Filter = dict[str, object]
RerankHook = Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]


class Record(TypedDict):
    namespace: str
    id: str
    vector: list[float]
    vectors: dict[str, list[float]]
    sparse: dict[str, float]
    metadata: Metadata


class ExplainDetails(TypedDict, total=False):
    fusion: str
    dense_score: float
    sparse_score: float
    matched_terms: list[str]
    vector_name: str | None
    dense_rank: int | None
    sparse_rank: int | None
    bm25_term_scores: dict[str, float]
    rerankers: list[dict[str, Any]]


class SearchResult(TypedDict, total=False):
    namespace: str
    id: str
    score: float
    dense_score: float
    sparse_score: float
    vector_name: str | None
    matched_terms: list[str]
    dense_rank: int | None
    sparse_rank: int | None
    bm25_term_scores: dict[str, float]
    rerank_score: float
    explain: ExplainDetails
    metadata: Metadata


class SearchTimings(TypedDict):
    dense_us: int
    sparse_us: int
    fusion_us: int
    total_us: int


class SearchStats(TypedDict):
    used_ann: bool
    ann_candidate_count: int
    exact_fallback: bool
    considered_count: int
    fetch_k: int
    mmr_applied: bool
    sparse_candidate_count: int
    ann_loaded_from_disk: bool
    wal_entries_replayed: int
    fusion: str
    rerank_applied: bool
    rerank_count: int
    timings: SearchTimings


class SearchResponse(TypedDict):
    results: list[SearchResult]
    stats: SearchStats


class Transaction:
    def __enter__(self) -> Transaction: ...
    def __exit__(self, exc_type: object | None, exc: object | None, tb: object | None) -> bool: ...
    def __len__(self) -> int: ...
    def insert(
        self,
        id: str,
        vector: list[float],
        metadata: Metadata | None = None,
        namespace: str | None = None,
        sparse: dict[str, float] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> None: ...
    def upsert(
        self,
        id: str,
        vector: list[float],
        metadata: Metadata | None = None,
        namespace: str | None = None,
        sparse: dict[str, float] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> None: ...
    def insert_many(self, records: list[Record], namespace: str | None = None) -> int: ...
    def upsert_many(self, records: list[Record], namespace: str | None = None) -> int: ...
    def delete(self, id: str, namespace: str | None = None) -> bool: ...
    def delete_many(self, ids: list[str], namespace: str | None = None) -> int: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class Database:
    @property
    def path(self) -> str: ...
    @property
    def wal_path(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    @property
    def read_only(self) -> bool: ...
    def __len__(self) -> int: ...
    def count(self) -> int: ...
    def namespaces(self) -> list[str]: ...
    def transaction(self) -> Transaction: ...
    def insert(
        self,
        id: str,
        vector: list[float],
        metadata: Metadata | None = None,
        namespace: str | None = None,
        sparse: dict[str, float] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> None: ...
    def upsert(
        self,
        id: str,
        vector: list[float],
        metadata: Metadata | None = None,
        namespace: str | None = None,
        sparse: dict[str, float] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> None: ...
    def insert_many(self, records: list[Record], namespace: str | None = None) -> int: ...
    def upsert_many(self, records: list[Record], namespace: str | None = None) -> int: ...
    def bulk_ingest(
        self,
        records: list[Record],
        namespace: str | None = None,
        batch_size: int = 10000,
    ) -> int: ...
    def get(self, id: str, namespace: str | None = None) -> Record | None: ...
    def delete(self, id: str, namespace: str | None = None) -> bool: ...
    def delete_many(self, ids: list[str], namespace: str | None = None) -> int: ...
    def flush(self) -> None: ...
    def compact(self) -> None: ...
    def snapshot(self, dest: str) -> None: ...
    def backup(self, dest: str) -> None: ...
    def search(
        self,
        query: list[float] | None = None,
        k: int = 10,
        filter: Filter | None = None,
        namespace: str | None = None,
        all_namespaces: bool = False,
        sparse: dict[str, float] | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fetch_k: int = 0,
        mmr_lambda: float | None = None,
        vector_name: str | None = None,
        fusion: str = "linear",
        rrf_k: int = 60,
        explain: bool = False,
        rerank: RerankHook | None = None,
        rerank_k: int = 0,
        query_vectors: dict[str, list[float]] | None = None,
        vector_weights: dict[str, float] | None = None,
    ) -> list[SearchResult]: ...
    def search_with_stats(
        self,
        query: list[float] | None = None,
        k: int = 10,
        filter: Filter | None = None,
        namespace: str | None = None,
        all_namespaces: bool = False,
        sparse: dict[str, float] | None = None,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fetch_k: int = 0,
        mmr_lambda: float | None = None,
        vector_name: str | None = None,
        fusion: str = "linear",
        rrf_k: int = 60,
        explain: bool = False,
        rerank: RerankHook | None = None,
        rerank_k: int = 0,
        query_vectors: dict[str, list[float]] | None = None,
        vector_weights: dict[str, float] | None = None,
    ) -> SearchResponse: ...


class Store:
    @property
    def root(self) -> str: ...
    def create_collection(self, name: str, dimension: int) -> Database: ...
    def open_collection(self, name: str) -> Database: ...
    def open_collection_read_only(self, name: str) -> Database: ...
    def open_or_create_collection(self, name: str, dimension: int) -> Database: ...
    def drop_collection(self, name: str) -> bool: ...
    def collections(self) -> list[str]: ...


class VectLiteError(Exception): ...


def open(path: str, dimension: int | None = None, read_only: bool = False) -> Database: ...
def open_store(root: str) -> Store: ...
def restore(source: str, dest: str) -> Database: ...
def upsert_text(
    db: Database,
    id: str,
    text: str,
    embed: Callable[[str], Sequence[float]],
    metadata: Metadata | None = None,
    namespace: str | None = None,
) -> None: ...
def search_text(
    db: Database,
    query: str,
    embed: Callable[[str], Sequence[float]],
    *,
    k: int = 10,
    filter: Filter | None = None,
    namespace: str | None = None,
    all_namespaces: bool = False,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    fetch_k: int = 0,
    mmr_lambda: float | None = None,
    vector_name: str | None = None,
    fusion: str = "linear",
    rrf_k: int = 60,
    explain: bool = False,
    rerank: RerankHook | None = None,
    rerank_k: int = 0,
) -> list[SearchResult]: ...
def search_text_with_stats(
    db: Database,
    query: str,
    embed: Callable[[str], Sequence[float]],
    *,
    k: int = 10,
    filter: Filter | None = None,
    namespace: str | None = None,
    all_namespaces: bool = False,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    fetch_k: int = 0,
    mmr_lambda: float | None = None,
    vector_name: str | None = None,
    fusion: str = "linear",
    rrf_k: int = 60,
    explain: bool = False,
    rerank: RerankHook | None = None,
    rerank_k: int = 0,
) -> SearchResponse: ...
def sparse_terms(text: str) -> dict[str, float]: ...
