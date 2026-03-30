"""Text analyzers for configurable sparse vector generation."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English stopwords (top 25). Users can supply their own list.
ENGLISH_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "with",
    }
)

FRENCH_STOPWORDS = frozenset(
    {
        "le", "la", "les", "de", "des", "du", "un", "une", "et", "en",
        "est", "que", "qui", "dans", "pour", "par", "sur", "au", "aux",
        "ce", "se", "ne", "pas", "son", "sa", "ses",
    }
)

_STOPWORD_SETS: dict[str, frozenset[str]] = {
    "en": ENGLISH_STOPWORDS,
    "english": ENGLISH_STOPWORDS,
    "fr": FRENCH_STOPWORDS,
    "french": FRENCH_STOPWORDS,
}

TokenFilter = Callable[[list[str]], list[str]]


def _try_import_stemmer() -> Any | None:
    """Lazily import PyStemmer / Snowball stemmer."""
    try:
        import Stemmer  # type: ignore[import-untyped]

        return Stemmer
    except ImportError:
        return None


class Analyzer:
    """Configurable text analyzer that produces sparse term vectors.

    Example usage::

        analyzer = (
            Analyzer()
            .lowercase()
            .stopwords("en")
            .stemmer("en")
        )
        terms = analyzer.sparse_terms("How to authenticate users")
    """

    def __init__(self) -> None:
        self._filters: list[TokenFilter] = []
        self._tokenizer: Callable[[str], list[str]] = _default_tokenize

    def tokenizer(self, fn: Callable[[str], list[str]]) -> Analyzer:
        """Replace the default tokenizer with a custom one."""
        self._tokenizer = fn
        return self

    def lowercase(self) -> Analyzer:
        """Add a lowercase token filter."""

        def _lower(tokens: list[str]) -> list[str]:
            return [t.lower() for t in tokens]

        self._filters.append(_lower)
        return self

    def stopwords(self, lang_or_set: str | frozenset[str] | set[str]) -> Analyzer:
        """Remove stopwords. Pass a language code ('en', 'fr') or a custom set."""
        if isinstance(lang_or_set, str):
            stop_set = _STOPWORD_SETS.get(lang_or_set.lower(), frozenset())
        else:
            stop_set = frozenset(lang_or_set)

        def _remove_stopwords(tokens: list[str]) -> list[str]:
            return [t for t in tokens if t not in stop_set]

        self._filters.append(_remove_stopwords)
        return self

    def stemmer(self, lang: str = "english") -> Analyzer:
        """Add a Snowball stemmer. Requires the ``PyStemmer`` package."""
        stemmer_mod = _try_import_stemmer()
        if stemmer_mod is None:
            raise ImportError(
                "PyStemmer is required for stemming. Install it with: pip install PyStemmer"
            )
        stem = stemmer_mod.Stemmer(lang)

        def _stem(tokens: list[str]) -> list[str]:
            return stem.stemWords(tokens)

        self._filters.append(_stem)
        return self

    def ngrams(self, n: int = 2) -> Analyzer:
        """Add character n-gram generation."""
        if n < 1:
            raise ValueError("ngram size must be >= 1")

        def _ngrams(tokens: list[str]) -> list[str]:
            result: list[str] = []
            for token in tokens:
                if len(token) <= n:
                    result.append(token)
                else:
                    for i in range(len(token) - n + 1):
                        result.append(token[i : i + n])
            return result

        self._filters.append(_ngrams)
        return self

    def filter(self, fn: TokenFilter) -> Analyzer:
        """Add a custom token filter function."""
        self._filters.append(fn)
        return self

    def tokenize(self, text: str) -> list[str]:
        """Run text through the full pipeline (tokenizer + filters)."""
        tokens = self._tokenizer(text)
        for f in self._filters:
            tokens = f(tokens)
        return tokens

    def sparse_terms(self, text: str) -> dict[str, float]:
        """Produce a sparse term-frequency vector from text."""
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        total = float(len(tokens))
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + (1.0 / total)
        return counts

    def sparse_terms_weighted(
        self,
        fields: dict[str, str],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Produce a sparse vector from multiple text fields with per-field weights.

        Args:
            fields: Mapping of field_name -> text content.
            weights: Mapping of field_name -> weight multiplier (default 1.0).
        """
        weights = weights or {}
        combined: dict[str, float] = {}
        for field_name, text in fields.items():
            w = weights.get(field_name, 1.0)
            terms = self.sparse_terms(text)
            for term, score in terms.items():
                combined[term] = combined.get(term, 0.0) + w * score
        return combined


def _default_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


__all__ = [
    "ENGLISH_STOPWORDS",
    "FRENCH_STOPWORDS",
    "Analyzer",
]
