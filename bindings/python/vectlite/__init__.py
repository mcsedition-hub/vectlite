"""Python bindings for the vectlite embedded vector store."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version as _pkg_version
from typing import Any

from . import analyzers, embedders, rerankers, schema
from ._vectlite import (
    Database,
    Store,
    Transaction,
    VectLiteError,
    VectLiteLockError,
    open,
    open_store,
    restore,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# ---------------------------------------------------------------------------
# Optional OpenTelemetry tracing
# ---------------------------------------------------------------------------

_otel_tracer: Any = None


def configure_opentelemetry(
    options: dict[str, Any] | bool | None = None,
) -> Any:
    """Configure optional OpenTelemetry tracing for search operations.

    When a tracer is active, ``search``, ``search_with_stats``,
    ``search_text``, and ``search_text_with_stats`` calls are wrapped in a
    span carrying semantic ``db.system`` / ``db.operation.name`` attributes
    and search-specific metrics.

    ``opentelemetry-api`` is imported lazily -- it is **not** a runtime
    dependency.  If the package is not installed the function returns
    ``None`` and search calls remain un-instrumented.

    Args:
        options: ``False`` to disable, a dict with optional keys
                 ``tracer`` (a pre-built ``Tracer``), ``tracer_name``
                 (defaults to ``"vectlite"``), or ``True``/``None``/``{}``
                 to auto-resolve from ``opentelemetry.trace``.

    Returns:
        The resolved tracer, or ``None`` if tracing could not be configured.
    """
    global _otel_tracer

    if options is False or (isinstance(options, dict) and options.get("enabled") is False):
        _otel_tracer = None
        return None

    if isinstance(options, dict) and options.get("tracer") is not None:
        _otel_tracer = options["tracer"]
        return _otel_tracer

    try:
        from opentelemetry import trace  # type: ignore[import-untyped]

        tracer_name = "vectlite"
        if isinstance(options, dict) and options.get("tracer_name"):
            tracer_name = options["tracer_name"]
        _otel_tracer = trace.get_tracer(tracer_name)
        return _otel_tracer
    except Exception:
        _otel_tracer = None
        return None


def _search_attributes(
    query: Any,
    kwargs: dict[str, Any],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "db.system": "vectlite",
        "db.operation.name": "search",
        "vectlite.search.k": kwargs.get("k", 10),
        "vectlite.search.namespace": kwargs.get("namespace") or "",
        "vectlite.search.all_namespaces": bool(kwargs.get("all_namespaces")),
        "vectlite.search.has_dense": query is not None,
        "vectlite.search.has_sparse": kwargs.get("sparse") is not None,
        "vectlite.search.fusion": kwargs.get("fusion", "linear"),
    }
    if kwargs.get("vector_name") is not None:
        attrs["vectlite.search.vector_name"] = kwargs["vector_name"]
    if kwargs.get("truncate_dim") is not None:
        attrs["vectlite.search.truncate_dim"] = kwargs["truncate_dim"]
    if stats is not None:
        attrs["vectlite.search.used_ann"] = bool(stats.get("used_ann"))
        attrs["vectlite.search.exact_fallback"] = bool(stats.get("exact_fallback"))
        attrs["vectlite.search.considered_count"] = stats.get("considered_count", 0)
        attrs["vectlite.search.result_count"] = stats.get("result_count", 0)
        attrs["vectlite.search.effective_dimension"] = stats.get("effective_dimension", 0)
        attrs["vectlite.search.matryoshka_truncated"] = bool(stats.get("matryoshka_truncated"))
        timings = stats.get("timings") or {}
        attrs["vectlite.search.total_us"] = timings.get("total_us", 0)
    return attrs


def _with_search_span(query: Any, kwargs: dict[str, Any], fn: Callable[..., Any]) -> Any:
    if _otel_tracer is None:
        return fn()
    span = _otel_tracer.start_span("vectlite.search", attributes=_search_attributes(query, kwargs))
    try:
        result = fn()
        # If result is a SearchResponse dict with stats, enrich the span
        if isinstance(result, dict) and "stats" in result:
            span.set_attributes(_search_attributes(query, kwargs, result["stats"]))
        span.end()
        return result
    except Exception as exc:
        span.record_exception(exc)
        try:
            from opentelemetry.trace import StatusCode  # type: ignore[import-untyped]

            span.set_status(StatusCode.ERROR, str(exc))
        except Exception:
            # Fallback when opentelemetry isn't available (custom tracer)
            try:
                span.set_status(2, str(exc))  # 2 == ERROR
            except Exception:
                pass
        span.end()
        raise


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
    search_kwargs: dict[str, Any] = {
        "k": k, "namespace": namespace, "all_namespaces": all_namespaces,
        "fusion": fusion, "vector_name": vector_name,
    }
    return _with_search_span(
        query,
        search_kwargs,
        lambda: db.search(
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
        ),
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
    search_kwargs: dict[str, Any] = {
        "k": k, "namespace": namespace, "all_namespaces": all_namespaces,
        "fusion": fusion, "vector_name": vector_name,
    }
    return _with_search_span(
        query,
        search_kwargs,
        lambda: db.search_with_stats(
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
        ),
    )

try:
    __version__ = _pkg_version("vectlite")
except Exception:  # package not installed (editable / dev)
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "analyzers",
    "configure_opentelemetry",
    "embedders",
    "Database",
    "Store",
    "Transaction",
    "VectLiteError",
    "VectLiteLockError",
    "open",
    "open_store",
    "rerankers",
    "restore",
    "schema",
    "search_text",
    "search_text_with_stats",
    "sparse_terms",
    "upsert_text",
]
