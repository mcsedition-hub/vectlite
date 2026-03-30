"""Python bindings for the vectlite embedded vector store."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version as _pkg_version
from typing import Any

from . import analyzers, rerankers
from ._vectlite import Database, Store, Transaction, VectLiteError, open, open_store, restore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def sparse_terms(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return counts

    total = float(len(tokens))
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + (1.0 / total)
    return counts


def upsert_text(
    db: Database,
    id: str,
    text: str,
    embed: Callable[[str], Sequence[float]],
    metadata: Mapping[str, Any] | None = None,
    namespace: str | None = None,
) -> None:
    payload = dict(metadata or {})
    payload.setdefault("text", text)
    db.upsert(
        id,
        list(embed(text)),
        payload,
        namespace=namespace,
        sparse=sparse_terms(text),
    )


def _wrap_rerank_with_text(
    query_text: str,
    rerank: Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]] | None,
) -> Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]] | None:
    if rerank is None:
        return None

    def wrapped(
        query_payload: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = dict(query_payload)
        payload["text"] = query_text
        return rerank(payload, results)

    return wrapped


def search_text(
    db: Database,
    query: str,
    embed: Callable[[str], Sequence[float]],
    *,
    k: int = 10,
    filter: dict[str, object] | None = None,
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
    rerank: Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    rerank_k: int = 0,
) -> list[dict[str, Any]]:
    return db.search(
        list(embed(query)),
        k=k,
        filter=filter,
        namespace=namespace,
        all_namespaces=all_namespaces,
        sparse=sparse_terms(query),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
        vector_name=vector_name,
        fusion=fusion,
        rrf_k=rrf_k,
        explain=explain,
        rerank=_wrap_rerank_with_text(query, rerank),
        rerank_k=rerank_k,
    )


def search_text_with_stats(
    db: Database,
    query: str,
    embed: Callable[[str], Sequence[float]],
    *,
    k: int = 10,
    filter: dict[str, object] | None = None,
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
    rerank: Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
    rerank_k: int = 0,
) -> dict[str, Any]:
    return db.search_with_stats(
        list(embed(query)),
        k=k,
        filter=filter,
        namespace=namespace,
        all_namespaces=all_namespaces,
        sparse=sparse_terms(query),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        fetch_k=fetch_k,
        mmr_lambda=mmr_lambda,
        vector_name=vector_name,
        fusion=fusion,
        rrf_k=rrf_k,
        explain=explain,
        rerank=_wrap_rerank_with_text(query, rerank),
        rerank_k=rerank_k,
    )

try:
    __version__ = _pkg_version("vectlite")
except Exception:  # package not installed (editable / dev)
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "analyzers",
    "Database",
    "Store",
    "Transaction",
    "VectLiteError",
    "open",
    "open_store",
    "rerankers",
    "restore",
    "search_text",
    "search_text_with_stats",
    "sparse_terms",
    "upsert_text",
]
