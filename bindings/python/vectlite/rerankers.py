"""Built-in rerankers for vectlite search results."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

RerankHook = Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _identity(result: Mapping[str, Any]) -> tuple[str, str]:
    namespace = result.get("namespace")
    if not isinstance(namespace, str):
        namespace = ""
    identifier = result.get("id")
    if not isinstance(identifier, str):
        raise ValueError("rerank results must include a string 'id'")
    return namespace, identifier


def _copy_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(result) for result in results]


def _query_terms(query: Mapping[str, Any]) -> list[str]:
    text = query.get("text")
    if isinstance(text, str):
        terms = _tokenize(text)
        if terms:
            return list(dict.fromkeys(terms))

    sparse = query.get("sparse")
    if isinstance(sparse, Mapping):
        terms = [str(term).lower() for term in sparse if str(term)]
        return list(dict.fromkeys(terms))

    return []


def _append_explain(result: dict[str, Any], payload: dict[str, Any]) -> None:
    explain = result.get("explain")
    if isinstance(explain, dict):
        rerankers = explain.setdefault("rerankers", [])
        if isinstance(rerankers, list):
            rerankers.append(payload)


def text_match(
    *,
    text_key: str = "text",
    title_key: str | None = "title",
    text_weight: float = 1.0,
    title_weight: float = 1.5,
    matched_term_weight: float = 0.25,
    phrase_boost: float = 1.0,
) -> RerankHook:
    """Boost candidates whose metadata text overlaps the sparse/text query."""

    def rerank(query: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_terms = _query_terms(query)
        if not query_terms:
            return _copy_results(results)

        query_term_set = set(query_terms)
        query_text = query.get("text")
        query_phrase = query_text.lower() if isinstance(query_text, str) else ""
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for index, raw_result in enumerate(results):
            result = dict(raw_result)
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}

            title_text = ""
            if title_key is not None:
                candidate = metadata.get(title_key)
                if isinstance(candidate, str):
                    title_text = candidate

            body_text = ""
            candidate = metadata.get(text_key)
            if isinstance(candidate, str):
                body_text = candidate

            title_overlap = 0.0
            if title_text:
                title_terms = set(_tokenize(title_text))
                title_overlap = len(query_term_set & title_terms) / len(query_term_set)

            body_overlap = 0.0
            if body_text:
                body_terms = set(_tokenize(body_text))
                body_overlap = len(query_term_set & body_terms) / len(query_term_set)

            matched_terms = result.get("matched_terms")
            matched_terms = matched_terms if isinstance(matched_terms, list) else []
            matched_ratio = len(query_term_set & {str(term).lower() for term in matched_terms}) / len(
                query_term_set
            )

            phrase_score = 0.0
            if query_phrase:
                haystacks = [title_text.lower(), body_text.lower()]
                if any(query_phrase in haystack for haystack in haystacks if haystack):
                    phrase_score = phrase_boost

            base_score = float(result.get("score", 0.0))
            rerank_score = (
                base_score
                + (text_weight * body_overlap)
                + (title_weight * title_overlap)
                + (matched_term_weight * matched_ratio)
                + phrase_score
            )
            result["rerank_score"] = rerank_score
            _append_explain(
                result,
                {
                    "name": "text_match",
                    "base_score": base_score,
                    "body_overlap": body_overlap,
                    "title_overlap": title_overlap,
                    "matched_term_ratio": matched_ratio,
                    "phrase_score": phrase_score,
                    "score": rerank_score,
                },
            )
            scored.append((rerank_score, index, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in scored]

    rerank.__name__ = "text_match"
    return rerank


def metadata_boost(
    field: str,
    boosts: Mapping[Any, float],
    *,
    default: float = 0.0,
) -> RerankHook:
    """Boost candidates according to a metadata field value."""

    def rerank(query: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del query
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for index, raw_result in enumerate(results):
            result = dict(raw_result)
            metadata = result.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            boost = float(boosts.get(metadata.get(field), default))
            base_score = float(result.get("rerank_score", result.get("score", 0.0)))
            rerank_score = base_score + boost
            result["rerank_score"] = rerank_score
            _append_explain(
                result,
                {
                    "name": "metadata_boost",
                    "field": field,
                    "value": metadata.get(field),
                    "boost": boost,
                    "base_score": base_score,
                    "score": rerank_score,
                },
            )
            scored.append((rerank_score, index, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in scored]

    rerank.__name__ = "metadata_boost"
    return rerank


def compose(
    *rerankers: RerankHook,
    strategy: str = "sequential",
    rank_constant: int = 60,
) -> RerankHook:
    """Compose several rerankers either sequentially or with reciprocal rank fusion."""

    if strategy not in {"sequential", "rrf"}:
        raise ValueError("strategy must be 'sequential' or 'rrf'")

    def rerank(query: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rerankers:
            return _copy_results(results)

        if strategy == "sequential":
            current = _copy_results(results)
            for step in rerankers:
                current = step(query, current)
            return current

        base_results = _copy_results(results)
        original_order = {_identity(result): index for index, result in enumerate(base_results)}
        fused_scores = {key: 0.0 for key in original_order}
        component_details: dict[tuple[str, str], list[dict[str, Any]]] = {
            key: [] for key in original_order
        }

        for step in rerankers:
            name = getattr(step, "__name__", step.__class__.__name__)
            ranked = step(query, _copy_results(results))
            for rank, result in enumerate(ranked, start=1):
                key = _identity(result)
                if key not in fused_scores:
                    continue
                contribution = 1.0 / (max(rank_constant, 1) + rank)
                fused_scores[key] += contribution
                component_details[key].append(
                    {
                        "reranker": name,
                        "rank": rank,
                        "contribution": contribution,
                    }
                )

        base_results.sort(
            key=lambda result: (-fused_scores[_identity(result)], original_order[_identity(result)])
        )

        for result in base_results:
            key = _identity(result)
            result["rerank_score"] = fused_scores[key]
            _append_explain(
                result,
                {
                    "name": "compose",
                    "strategy": "rrf",
                    "rank_constant": max(rank_constant, 1),
                    "components": component_details[key],
                    "score": fused_scores[key],
                },
            )

        return base_results

    rerank.__name__ = f"compose_{strategy}"
    return rerank


def cross_encoder(
    model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    *,
    text_key: str = "text",
    batch_size: int = 32,
    device: str | None = None,
) -> RerankHook:
    """Create a reranker that uses a cross-encoder model to re-score results.

    Requires ``sentence-transformers``.  Install with::

        pip install sentence-transformers

    Args:
        model_name_or_path: HuggingFace model name or local path.
        text_key: Metadata field containing the document text.
        batch_size: Batch size for cross-encoder inference.
        device: Device to run the model on (e.g. "cpu", "cuda").
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for cross_encoder reranker. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = CrossEncoder(model_name_or_path, device=device)

    def rerank(query: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_text = query.get("text", "")
        if not isinstance(query_text, str) or not query_text:
            return _copy_results(results)

        pairs: list[tuple[str, str]] = []
        for result in results:
            metadata = result.get("metadata")
            doc_text = ""
            if isinstance(metadata, Mapping):
                candidate = metadata.get(text_key)
                if isinstance(candidate, str):
                    doc_text = candidate
            pairs.append((query_text, doc_text))

        scores = model.predict(pairs, batch_size=batch_size)

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, raw_result in enumerate(results):
            result = dict(raw_result)
            ce_score = float(scores[index])
            base_score = float(result.get("rerank_score", result.get("score", 0.0)))
            rerank_score = base_score + ce_score
            result["rerank_score"] = rerank_score
            _append_explain(
                result,
                {
                    "name": "cross_encoder",
                    "model": model_name_or_path,
                    "ce_score": ce_score,
                    "base_score": base_score,
                    "score": rerank_score,
                },
            )
            scored.append((rerank_score, index, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in scored]

    rerank.__name__ = "cross_encoder"
    return rerank


def bi_encoder(
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    text_key: str = "text",
    batch_size: int = 64,
    device: str | None = None,
) -> RerankHook:
    """Create a reranker that uses a bi-encoder to re-score results by embedding
    similarity between the query and document text.

    Requires ``sentence-transformers``.  Install with::

        pip install sentence-transformers

    Args:
        model_name_or_path: HuggingFace model name or local path.
        text_key: Metadata field containing the document text.
        batch_size: Batch size for encoding.
        device: Device to run the model on (e.g. "cpu", "cuda").
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for bi_encoder reranker. "
            "Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name_or_path, device=device)

    def rerank(query: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_text = query.get("text", "")
        if not isinstance(query_text, str) or not query_text:
            return _copy_results(results)

        doc_texts: list[str] = []
        for result in results:
            metadata = result.get("metadata")
            doc_text = ""
            if isinstance(metadata, Mapping):
                candidate = metadata.get(text_key)
                if isinstance(candidate, str):
                    doc_text = candidate
            doc_texts.append(doc_text)

        query_embedding = model.encode(query_text, convert_to_numpy=True)
        doc_embeddings = model.encode(doc_texts, batch_size=batch_size, convert_to_numpy=True)

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index, raw_result in enumerate(results):
            result = dict(raw_result)
            import numpy as np

            q_norm = np.linalg.norm(query_embedding)
            d_norm = np.linalg.norm(doc_embeddings[index])
            if q_norm > 0 and d_norm > 0:
                sim = float(np.dot(query_embedding, doc_embeddings[index]) / (q_norm * d_norm))
            else:
                sim = 0.0
            base_score = float(result.get("rerank_score", result.get("score", 0.0)))
            rerank_score = base_score + sim
            result["rerank_score"] = rerank_score
            _append_explain(
                result,
                {
                    "name": "bi_encoder",
                    "model": model_name_or_path,
                    "similarity": sim,
                    "base_score": base_score,
                    "score": rerank_score,
                },
            )
            scored.append((rerank_score, index, result))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [result for _, _, result in scored]

    rerank.__name__ = "bi_encoder"
    return rerank


__all__ = ["bi_encoder", "compose", "cross_encoder", "metadata_boost", "text_match"]
