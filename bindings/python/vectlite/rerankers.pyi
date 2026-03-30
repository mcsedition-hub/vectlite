from collections.abc import Callable, Mapping
from typing import Any

RerankHook = Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]


def text_match(
    *,
    text_key: str = "text",
    title_key: str | None = "title",
    text_weight: float = 1.0,
    title_weight: float = 1.5,
    matched_term_weight: float = 0.25,
    phrase_boost: float = 1.0,
) -> RerankHook: ...
def metadata_boost(
    field: str,
    boosts: Mapping[Any, float],
    *,
    default: float = 0.0,
) -> RerankHook: ...
def compose(
    *rerankers: RerankHook,
    strategy: str = "sequential",
    rank_constant: int = 60,
) -> RerankHook: ...
