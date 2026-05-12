"""Tests for optional OpenTelemetry integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import vectlite


# ---------------------------------------------------------------------------
# Fake tracer / span (mimics opentelemetry-api interface)
# ---------------------------------------------------------------------------


class FakeSpan:
    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.initial_attributes: dict[str, Any] = dict(attributes or {})
        self.all_attributes: dict[str, Any] = dict(attributes or {})
        self.ended = False
        self.exceptions: list[BaseException] = []
        self.status: Any = None

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self.all_attributes.update(attrs)

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        self.status = (args, kwargs)

    def end(self) -> None:
        self.ended = True


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(
        self, name: str, attributes: dict[str, Any] | None = None, **kwargs: Any
    ) -> FakeSpan:
        span = FakeSpan(name, attributes)
        self.spans.append(span)
        return span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed(_text: str) -> list[float]:
    return [1.0, 0.0]


def _setup_db(tmp_path: Path, name: str = "otel.vdb") -> vectlite.Database:
    db = vectlite.open(str(tmp_path / name), dimension=2)
    db.upsert("doc1", [1, 0], {"source": "docs"}, sparse={"hello": 1.0})
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_configure_with_custom_tracer() -> None:
    tracer = FakeTracer()
    result = vectlite.configure_opentelemetry({"tracer": tracer})
    assert result is tracer
    vectlite.configure_opentelemetry(False)


def test_configure_false_disables() -> None:
    result = vectlite.configure_opentelemetry(False)
    assert result is None


def test_configure_enabled_false_disables() -> None:
    result = vectlite.configure_opentelemetry({"enabled": False})
    assert result is None


def test_configure_auto_without_otel_returns_none() -> None:
    """When opentelemetry-api is not installed, auto-detect returns None."""
    # We monkey-patch the import to simulate missing package
    import sys

    saved = sys.modules.get("opentelemetry")
    sys.modules["opentelemetry"] = None  # type: ignore[assignment]
    try:
        result = vectlite.configure_opentelemetry()
        assert result is None
    finally:
        if saved is not None:
            sys.modules["opentelemetry"] = saved
        else:
            sys.modules.pop("opentelemetry", None)
        vectlite.configure_opentelemetry(False)


def test_search_text_creates_span(tmp_path: Path) -> None:
    tracer = FakeTracer()
    vectlite.configure_opentelemetry({"tracer": tracer})
    try:
        db = _setup_db(tmp_path)
        vectlite.upsert_text(db, "doc2", "hello world", _embed)
        vectlite.search_text(db, "hello", _embed, k=5)

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.name == "vectlite.search"
        assert span.ended is True
        assert span.initial_attributes["db.system"] == "vectlite"
        assert span.initial_attributes["db.operation.name"] == "search"
        assert span.initial_attributes["vectlite.search.k"] == 5
        assert len(span.exceptions) == 0
        db.close()
    finally:
        vectlite.configure_opentelemetry(False)


def test_search_text_with_stats_enriches_span(tmp_path: Path) -> None:
    tracer = FakeTracer()
    vectlite.configure_opentelemetry({"tracer": tracer})
    try:
        db = _setup_db(tmp_path)
        vectlite.upsert_text(db, "doc2", "hello world", _embed)
        result = vectlite.search_text_with_stats(db, "hello", _embed, k=5)

        assert "stats" in result
        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.name == "vectlite.search"
        assert span.ended is True
        # Stats attributes should have been set after completion
        assert "vectlite.search.result_count" in span.all_attributes
        assert "vectlite.search.total_us" in span.all_attributes
        assert isinstance(span.all_attributes["vectlite.search.result_count"], int)
        db.close()
    finally:
        vectlite.configure_opentelemetry(False)


def test_no_span_when_disabled(tmp_path: Path) -> None:
    vectlite.configure_opentelemetry(False)
    db = _setup_db(tmp_path)
    vectlite.upsert_text(db, "doc2", "hello world", _embed)

    # Should work fine without any tracing
    results = vectlite.search_text(db, "hello", _embed, k=5)
    assert isinstance(results, list)
    db.close()


def test_span_attributes_include_namespace_and_fusion(tmp_path: Path) -> None:
    tracer = FakeTracer()
    vectlite.configure_opentelemetry({"tracer": tracer})
    try:
        db = _setup_db(tmp_path)
        vectlite.upsert_text(
            db, "doc2", "hello world", _embed, namespace="ns1"
        )
        vectlite.search_text(
            db, "hello", _embed, k=3, namespace="ns1", fusion="rrf"
        )

        assert len(tracer.spans) == 1
        attrs = tracer.spans[0].initial_attributes
        assert attrs["vectlite.search.k"] == 3
        assert attrs["vectlite.search.namespace"] == "ns1"
        assert attrs["vectlite.search.fusion"] == "rrf"
        db.close()
    finally:
        vectlite.configure_opentelemetry(False)


def test_span_records_exception_on_error(tmp_path: Path) -> None:
    tracer = FakeTracer()
    vectlite.configure_opentelemetry({"tracer": tracer})
    try:
        db = _setup_db(tmp_path)

        def bad_embed(_text: str) -> list[float]:
            raise ValueError("embedding failed")

        try:
            vectlite.search_text(db, "hello", bad_embed, k=5)
        except ValueError:
            pass

        assert len(tracer.spans) == 1
        span = tracer.spans[0]
        assert span.ended is True
        assert len(span.exceptions) == 1
        assert isinstance(span.exceptions[0], ValueError)
        assert span.status is not None
        db.close()
    finally:
        vectlite.configure_opentelemetry(False)
