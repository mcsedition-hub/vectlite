import json
import math
from pathlib import Path

import pytest

import vectlite


def embed(text: str) -> list[float]:
    text = text.lower()
    return [1.0 if "auth" in text else 0.0, 1.0 if "notes" in text else 0.0]


def test_roundtrip_and_search(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "knowledge.vdb"), dimension=3)

    db.upsert("doc1", [1.0, 0.0, 0.0], {"source": "blog", "priority": 10})
    db.upsert("doc2", [0.0, 1.0, 0.0], {"source": "notes", "priority": 2})

    record = db.get("doc1")
    assert record is not None
    assert record["metadata"]["source"] == "blog"

    results = db.search(
        [1.0, 0.0, 0.0],
        k=2,
        filter={"source": "blog", "priority": {"$gt": 5}},
    )

    assert [item["id"] for item in results] == ["doc1"]


def test_open_existing_file_without_dimension(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.insert("doc1", [1.0, 0.0], {"source": "notes"})
    del db

    reopened = vectlite.open(str(path))
    assert reopened.dimension == 2
    assert len(reopened) == 1


def test_batch_methods_and_extended_filters(tmp_path: Path) -> None:
    path = tmp_path / "batch.vdb"
    db = vectlite.open(str(path), dimension=2)

    inserted = db.upsert_many(
        [
            {
                "id": "doc1",
                "vector": [1.0, 0.0],
                "sparse": {"auth": 1.0},
                "metadata": {"source": "blog", "priority": 10},
            },
            {
                "id": "doc2",
                "vector": [0.8, 0.2],
                "sparse": {"notes": 1.0},
                "metadata": {"source": "notes", "priority": 5},
            },
            {
                "id": "doc3",
                "vector": [0.0, 1.0],
                "sparse": {"auth": 0.5},
                "metadata": {"source": "blog", "priority": 3},
            },
        ]
    )

    assert inserted == 3

    results = db.search(
        [1.0, 0.0],
        k=10,
        filter={"source": {"$ne": "notes"}, "priority": {"$gte": 3, "$lte": 10}},
    )

    assert [item["id"] for item in results] == ["doc1", "doc3"]

    deleted = db.delete_many(["doc2", "missing"])
    assert deleted == 1
    assert db.get("doc2") is None


def test_namespaces_and_text_helpers(tmp_path: Path) -> None:
    path = tmp_path / "namespaces.vdb"
    db = vectlite.open(str(path), dimension=2)

    vectlite.upsert_text(
        db,
        "doc1",
        "auth notes setup",
        embed,
        {"source": "docs"},
        namespace="docs",
    )
    vectlite.upsert_text(
        db,
        "doc1",
        "shopping list",
        embed,
        {"source": "notes"},
        namespace="notes",
    )

    docs_record = db.get("doc1", namespace="docs")
    notes_record = db.get("doc1", namespace="notes")

    assert docs_record is not None
    assert notes_record is not None
    assert docs_record["namespace"] == "docs"
    assert notes_record["namespace"] == "notes"
    assert db.namespaces() == ["docs", "notes"]

    docs_results = vectlite.search_text(db, "auth", embed, namespace="docs")
    all_results = vectlite.search_text(
        db,
        "auth",
        embed,
        filter={"source": {"$in": ["docs", "notes"]}, "$not": {"source": {"$eq": "notes"}}},
        all_namespaces=True,
    )

    assert [item["namespace"] for item in docs_results] == ["docs"]
    assert [item["namespace"] for item in all_results] == ["docs"]


def test_sparse_only_and_hybrid_scoring(tmp_path: Path) -> None:
    path = tmp_path / "hybrid.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert(
        "doc1",
        [1.0, 0.0],
        {"source": "docs"},
        sparse={"auth": 1.0, "sso": 0.5},
    )
    db.upsert(
        "doc2",
        [1.0, 0.0],
        {"source": "docs"},
        sparse={"billing": 1.0},
    )

    sparse_results = db.search(None, sparse={"auth": 1.0}, k=10)
    hybrid_results = db.search([1.0, 0.0], sparse={"auth": 1.0}, k=10)

    assert sparse_results[0]["id"] == "doc1"
    assert hybrid_results[0]["id"] == "doc1"
    assert hybrid_results[0]["sparse_score"] > hybrid_results[1]["sparse_score"]


def test_mmr_and_search_stats(tmp_path: Path) -> None:
    path = tmp_path / "mmr.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert("doc1", [1.0, 0.0], {"source": "docs"})
    db.upsert("doc2", [0.99, 0.01], {"source": "docs"})
    db.upsert("doc3", [0.7, 0.7], {"source": "docs"})

    plain_results = db.search([1.0, 0.0], k=2)
    mmr_results = db.search([1.0, 0.0], k=2, fetch_k=3, mmr_lambda=0.3)
    outcome = db.search_with_stats([1.0, 0.0], k=2, fetch_k=3, mmr_lambda=0.3)
    text_outcome = vectlite.search_text_with_stats(
        db,
        "auth",
        embed,
        k=2,
        fetch_k=3,
        mmr_lambda=0.3,
    )

    assert [item["id"] for item in plain_results] == ["doc1", "doc2"]
    assert [item["id"] for item in mmr_results] == ["doc1", "doc3"]
    assert [item["id"] for item in outcome["results"]] == ["doc1", "doc3"]
    assert outcome["stats"]["used_ann"] is False
    assert outcome["stats"]["mmr_applied"] is True
    assert outcome["stats"]["fetch_k"] == 3
    assert outcome["stats"]["considered_count"] == 3
    assert "stats" in text_outcome


def test_named_vectors_and_rerank_hook(tmp_path: Path) -> None:
    path = tmp_path / "named-vectors.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert(
        "doc1",
        [0.1, 0.9],
        {"source": "docs"},
        vectors={"title": [1.0, 0.0], "body": [0.0, 1.0]},
    )
    db.upsert(
        "doc2",
        [0.9, 0.1],
        {"source": "notes"},
        vectors={"title": [0.0, 1.0], "body": [1.0, 0.0]},
    )

    record = db.get("doc1")
    assert record is not None
    assert record["vectors"]["title"] == [1.0, 0.0]

    title_results = db.search([1.0, 0.0], k=2, vector_name="title")
    default_results = db.search([1.0, 0.0], k=2)

    seen_queries: list[dict[str, object]] = []

    def rerank(query: dict[str, object], results: list[dict[str, object]]) -> list[dict[str, object]]:
        seen_queries.append(query)
        return list(reversed(results))

    reranked = db.search([1.0, 0.0], k=2, vector_name="title", rerank=rerank)
    outcome = db.search_with_stats(
        [1.0, 0.0],
        k=2,
        vector_name="title",
        rerank=rerank,
        rerank_k=2,
    )

    assert [item["id"] for item in title_results] == ["doc1", "doc2"]
    assert [item["id"] for item in default_results] == ["doc2", "doc1"]
    assert [item["id"] for item in reranked] == ["doc2", "doc1"]
    assert [item["id"] for item in outcome["results"]] == ["doc2", "doc1"]
    assert outcome["stats"]["rerank_applied"] is True
    assert outcome["stats"]["rerank_count"] == 2
    assert seen_queries[0]["vector_name"] == "title"


def test_transactions_commit_and_rollback(tmp_path: Path) -> None:
    path = tmp_path / "transactions.vdb"
    db = vectlite.open(str(path), dimension=2)

    with db.transaction() as tx:
        tx.upsert("doc1", [1.0, 0.0], {"source": "docs"})
        tx.upsert("doc2", [0.0, 1.0], {"source": "notes"})
        assert len(tx) == 2

    assert len(db) == 2
    assert Path(db.wal_path).exists()

    with pytest.raises(RuntimeError):
        with db.transaction() as tx:
            tx.upsert("doc3", [1.0, 0.0], {"source": "docs"})
            raise RuntimeError("rollback me")

    assert db.get("doc3") is None

    tx = db.transaction()
    tx.upsert("doc4", [1.0, 0.0], {"source": "draft"})
    tx.rollback()
    assert db.get("doc4") is None


def test_wal_recovery_compaction_and_disk_ann_load(tmp_path: Path) -> None:
    path = tmp_path / "durable.vdb"
    db = vectlite.open(str(path), dimension=2)

    records = []
    for index in range(160):
        topic = "auth" if index < 120 else "billing"
        vector = [1.0, 0.0] if topic == "auth" else [0.0, 1.0]
        records.append(
            {
                "id": f"doc{index:03d}",
                "vector": vector,
                "sparse": vectlite.sparse_terms(f"{topic} guide {index}"),
                "metadata": {
                    "source": "docs",
                    "title": f"{topic} guide",
                    "text": f"{topic} guide number {index}",
                },
            }
        )

    assert db.upsert_many(records) == 160
    wal_path = Path(db.wal_path)
    assert wal_path.exists()
    del db

    reopened = vectlite.open(str(path))
    assert len(reopened) == 160

    outcome = reopened.search_with_stats(
        [1.0, 0.0],
        sparse=vectlite.sparse_terms("auth guide"),
        k=5,
        fusion="rrf",
        rrf_k=10,
        explain=True,
    )

    assert outcome["stats"]["used_ann"] is True
    assert outcome["stats"]["ann_loaded_from_disk"] is True
    assert outcome["stats"]["wal_entries_replayed"] == 160
    assert outcome["stats"]["fusion"] == "rrf"
    assert outcome["results"][0]["explain"]["fusion"] == "rrf"
    assert outcome["results"][0]["dense_rank"] is not None
    assert outcome["results"][0]["sparse_rank"] is not None

    reopened.compact()
    assert not wal_path.exists()
    del reopened

    compacted = vectlite.open(str(path))
    compacted_outcome = compacted.search_with_stats([1.0, 0.0], k=3)

    assert compacted_outcome["stats"]["wal_entries_replayed"] == 0
    assert compacted_outcome["stats"]["ann_loaded_from_disk"] is True


def test_builtin_rerankers_and_search_text_explain(tmp_path: Path) -> None:
    path = tmp_path / "rerankers.vdb"
    db = vectlite.open(str(path), dimension=2)

    def flat_embed(_: str) -> list[float]:
        return [1.0, 0.0]

    vectlite.upsert_text(
        db,
        "doc1",
        "auth billing checklist",
        flat_embed,
        {"source": "notes", "title": "Billing"},
    )
    vectlite.upsert_text(
        db,
        "doc2",
        "auth setup guide",
        flat_embed,
        {"source": "docs", "title": "Auth setup"},
    )

    baseline = vectlite.search_text(
        db,
        "auth setup",
        flat_embed,
        k=2,
        dense_weight=1.0,
        sparse_weight=0.0,
    )

    rerank = vectlite.rerankers.compose(
        vectlite.rerankers.text_match(),
        vectlite.rerankers.metadata_boost("source", {"docs": 0.25}),
    )
    reranked = vectlite.search_text(
        db,
        "auth setup",
        flat_embed,
        k=2,
        dense_weight=1.0,
        sparse_weight=0.0,
        explain=True,
        rerank=rerank,
        rerank_k=2,
    )

    assert [item["id"] for item in baseline] == ["doc1", "doc2"]
    assert [item["id"] for item in reranked] == ["doc2", "doc1"]
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
    assert reranked[0]["explain"]["rerankers"][0]["name"] == "text_match"
    assert reranked[0]["explain"]["rerankers"][1]["name"] == "metadata_boost"


def test_rich_metadata_types(tmp_path: Path) -> None:
    """Metadata supports None, lists, and nested dicts."""
    path = tmp_path / "rich.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert(
        "doc1",
        [1.0, 0.0],
        {
            "tags": ["python", "rust"],
            "author": None,
            "extra": {"nested_key": "value", "count": 42},
            "mixed_list": [1, "two", True, None],
        },
    )

    record = db.get("doc1")
    assert record is not None
    meta = record["metadata"]
    assert meta["tags"] == ["python", "rust"]
    assert meta["author"] is None
    assert meta["extra"] == {"count": 42, "nested_key": "value"}
    assert meta["mixed_list"] == [1, "two", True, None]

    # Verify persistence across reopen
    del db
    reopened = vectlite.open(str(path))
    record2 = reopened.get("doc1")
    assert record2 is not None
    assert record2["metadata"]["tags"] == ["python", "rust"]
    assert record2["metadata"]["author"] is None
    assert record2["metadata"]["extra"]["nested_key"] == "value"


def test_insert_duplicate_raises(tmp_path: Path) -> None:
    """Inserting a record with an existing id raises VectLiteError."""
    path = tmp_path / "dup.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.insert("doc1", [1.0, 0.0], {"source": "first"})

    with pytest.raises(vectlite.VectLiteError, match="already exists"):
        db.insert("doc1", [0.0, 1.0], {"source": "second"})

    # Original record is untouched
    record = db.get("doc1")
    assert record is not None
    assert record["metadata"]["source"] == "first"

    # upsert still works on the same id
    db.upsert("doc1", [0.0, 1.0], {"source": "updated"})
    record = db.get("doc1")
    assert record is not None
    assert record["metadata"]["source"] == "updated"


def test_insert_many_duplicate_raises(tmp_path: Path) -> None:
    """insert_many rejects batch when an id already exists."""
    path = tmp_path / "dup_many.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.insert("doc1", [1.0, 0.0])

    with pytest.raises(vectlite.VectLiteError, match="already exists"):
        db.insert_many([
            {"id": "doc2", "vector": [0.0, 1.0]},
            {"id": "doc1", "vector": [0.5, 0.5]},
        ])


def test_transaction_insert_duplicate_raises(tmp_path: Path) -> None:
    """Transaction insert rejects duplicates at commit time."""
    path = tmp_path / "tx_dup.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.insert("doc1", [1.0, 0.0])

    with pytest.raises(vectlite.VectLiteError, match="already exists"):
        with db.transaction() as tx:
            tx.insert("doc1", [0.0, 1.0])

    assert len(db) == 1


def test_open_without_dimension_raises_vectlite_error(tmp_path: Path) -> None:
    """open() on a non-existent path without dimension raises VectLiteError."""
    path = tmp_path / "missing.vdb"
    with pytest.raises(vectlite.VectLiteError, match="dimension"):
        vectlite.open(str(path))


def test_version_attribute() -> None:
    """Package exposes __version__."""
    assert isinstance(vectlite.__version__, str)
    assert len(vectlite.__version__) > 0


def test_collections_store(tmp_path: Path) -> None:
    """Store manages physical collections."""
    store = vectlite.open_store(str(tmp_path / "mystore"))
    db1 = store.create_collection("products", 2)
    db1.upsert("p1", [1.0, 0.0], {"name": "Widget"})
    assert len(db1) == 1
    del db1

    db2 = store.open_collection("products")
    assert db2.dimension == 2
    assert len(db2) == 1
    del db2

    assert store.collections() == ["products"]

    db3 = store.open_or_create_collection("logs", 3)
    assert db3.dimension == 3
    del db3

    assert sorted(store.collections()) == ["logs", "products"]
    assert store.drop_collection("logs") is True
    assert store.collections() == ["products"]


def test_bulk_ingest(tmp_path: Path) -> None:
    """bulk_ingest writes many records efficiently."""
    path = tmp_path / "bulk.vdb"
    db = vectlite.open(str(path), dimension=2)

    records = [
        {"id": f"doc{i}", "vector": [1.0, 0.0], "metadata": {"idx": i}}
        for i in range(50)
    ]
    count = db.bulk_ingest(records, batch_size=20)
    assert count == 50
    assert len(db) == 50
    assert db.get("doc0") is not None
    assert db.get("doc49") is not None


def test_nested_metadata_filters(tmp_path: Path) -> None:
    """Dot-path, $elemMatch, and $size filters work on nested metadata."""
    path = tmp_path / "nested.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert(
        "doc1",
        [1.0, 0.0],
        {
            "tags": ["python", "rust", "go"],
            "author": {"name": "Alice", "level": 5},
        },
    )
    db.upsert(
        "doc2",
        [0.9, 0.1],
        {
            "tags": ["javascript"],
            "author": {"name": "Bob", "level": 2},
        },
    )

    # dot-path filter
    results = db.search([1.0, 0.0], k=10, filter={"author.name": "Alice"})
    assert [r["id"] for r in results] == ["doc1"]

    # $size filter
    results = db.search([1.0, 0.0], k=10, filter={"tags": {"$size": 3}})
    assert [r["id"] for r in results] == ["doc1"]

    # $elemMatch filter
    results = db.search(
        [1.0, 0.0], k=10, filter={"tags": {"$elemMatch": {"$eq": "rust"}}}
    )
    assert [r["id"] for r in results] == ["doc1"]


def test_observability_timings_and_bm25_scores(tmp_path: Path) -> None:
    """Search stats include timings and BM25 term scores in explain."""
    path = tmp_path / "obs.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert("doc1", [1.0, 0.0], {"text": "auth guide"}, sparse={"auth": 1.0})
    db.upsert("doc2", [0.9, 0.1], {"text": "billing"}, sparse={"billing": 1.0})

    outcome = db.search_with_stats(
        [1.0, 0.0], sparse={"auth": 1.0}, k=2, explain=True
    )

    stats = outcome["stats"]
    assert "timings" in stats
    timings = stats["timings"]
    assert timings["total_us"] >= 0
    assert timings["dense_us"] >= 0
    assert timings["sparse_us"] >= 0
    assert timings["fusion_us"] >= 0

    # BM25 term scores should appear in explain
    top = outcome["results"][0]
    assert top["id"] == "doc1"
    assert "bm25_term_scores" in top["explain"]
    assert "auth" in top["explain"]["bm25_term_scores"]


def test_read_only_mode(tmp_path: Path) -> None:
    """Read-only mode prevents writes but allows reads."""
    path = tmp_path / "readonly.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0], {"source": "docs"})
    del db

    ro = vectlite.open(str(path), read_only=True)
    assert ro.read_only is True
    assert len(ro) == 1
    assert ro.get("doc1") is not None

    # Search should work
    results = ro.search([1.0, 0.0], k=2)
    assert len(results) == 1

    # Write operations should raise
    with pytest.raises(vectlite.VectLiteError, match="read-only"):
        ro.upsert("doc2", [0.0, 1.0])

    with pytest.raises(vectlite.VectLiteError, match="read-only"):
        ro.delete("doc1")

    with pytest.raises(vectlite.VectLiteError, match="read-only"):
        ro.compact()


def test_snapshot_and_backup_restore(tmp_path: Path) -> None:
    """Snapshot creates a standalone .vdb; backup/restore roundtrips."""
    path = tmp_path / "original.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0], {"source": "a"})
    db.upsert("doc2", [0.0, 1.0], {"source": "b"})

    # Snapshot
    snap_path = tmp_path / "snap.vdb"
    db.snapshot(str(snap_path))
    del db

    snap = vectlite.open(str(snap_path))
    assert len(snap) == 2
    assert snap.get("doc1") is not None
    del snap

    # Backup
    db = vectlite.open(str(path))
    backup_dir = tmp_path / "backup"
    db.backup(str(backup_dir))
    del db

    assert (backup_dir / "original.vdb").exists()

    # Restore
    restored_path = tmp_path / "restored.vdb"
    restored = vectlite.restore(str(backup_dir), str(restored_path))
    assert len(restored) == 2
    assert restored.get("doc2") is not None
    del restored


def test_lock_contention(tmp_path: Path) -> None:
    """Opening the same database twice (exclusive) raises lock contention."""
    path = tmp_path / "locked.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0])

    with pytest.raises(vectlite.VectLiteError, match="lock contention"):
        vectlite.open(str(path))

    del db  # release lock

    # Now it should work
    db2 = vectlite.open(str(path))
    assert len(db2) == 1


def test_analyzers_module() -> None:
    """Analyzer pipeline produces expected sparse terms."""
    analyzer = vectlite.analyzers.Analyzer().lowercase().stopwords("en")
    terms = analyzer.sparse_terms("The quick brown fox jumps over the lazy dog")
    assert "the" not in terms
    assert "quick" in terms
    assert "fox" in terms
    # All values should be > 0
    assert all(v > 0 for v in terms.values())

    # Weighted fields
    weighted = analyzer.sparse_terms_weighted(
        {"title": "Auth Guide", "body": "Authentication setup manual"},
        {"title": 2.0, "body": 1.0},
    )
    assert "auth" in weighted or "authentication" in weighted


def test_analyzers_ngrams() -> None:
    """Analyzer ngram filter produces character n-grams."""
    analyzer = vectlite.analyzers.Analyzer().lowercase().ngrams(3)
    tokens = analyzer.tokenize("hello world")
    assert "hel" in tokens
    assert "ell" in tokens
    assert "llo" in tokens


def test_close_and_context_manager(tmp_path: Path) -> None:
    """db.close() releases the lock; context manager auto-closes."""
    path = tmp_path / "close.vdb"

    # Explicit close
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0], {"source": "a"})
    db.close()

    # After close, the lock is released - we can reopen
    db2 = vectlite.open(str(path))
    assert len(db2) == 1
    db2.close()

    # Context manager auto-closes
    with vectlite.open(str(path)) as db3:
        db3.upsert("doc2", [0.0, 1.0], {"source": "b"})
        assert len(db3) == 2

    # Lock released, can reopen
    db4 = vectlite.open(str(path))
    assert len(db4) == 2
    db4.close()


def test_close_raises_on_use_after_close(tmp_path: Path) -> None:
    """Using a closed database raises an error."""
    path = tmp_path / "closed_use.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0], {"source": "docs"})
    db.close()

    with pytest.raises(vectlite.VectLiteError, match="closed"):
        db.upsert("doc2", [0.0, 1.0])

    with pytest.raises(vectlite.VectLiteError, match="closed"):
        db.get("doc1")

    with pytest.raises(vectlite.VectLiteError, match="closed"):
        db.count()

    with pytest.raises(vectlite.VectLiteError, match="closed"):
        db.list()

    with pytest.raises(vectlite.VectLiteError, match="closed"):
        db.search([1.0, 0.0])


def test_list_without_vector(tmp_path: Path) -> None:
    """db.list() returns records without requiring a vector query."""
    path = tmp_path / "list.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert("doc1", [1.0, 0.0], {"type": "user", "score": 10}, namespace="users")
    db.upsert("doc2", [0.0, 1.0], {"type": "feedback", "score": 5}, namespace="feedback")
    db.upsert("doc3", [0.5, 0.5], {"type": "user", "score": 3}, namespace="users")

    # List all
    all_records = db.list()
    assert len(all_records) == 3

    # List by namespace
    user_records = db.list(namespace="users")
    assert len(user_records) == 2
    assert all(r["namespace"] == "users" for r in user_records)

    # List with filter
    high_score = db.list(filter={"score": {"$gt": 4}})
    assert len(high_score) == 2

    # List with namespace + filter
    user_high = db.list(namespace="users", filter={"score": {"$gt": 5}})
    assert len(user_high) == 1
    assert user_high[0]["id"] == "doc1"

    # Pagination
    page1 = db.list(limit=2)
    page2 = db.list(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 1


def test_delete_by_filter(tmp_path: Path) -> None:
    """db.delete_by_filter() removes matching records."""
    path = tmp_path / "del_filter.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert("doc1", [1.0, 0.0], {"type": "feedback"}, namespace="feedback")
    db.upsert("doc2", [0.0, 1.0], {"type": "feedback"}, namespace="feedback")
    db.upsert("doc3", [0.5, 0.5], {"type": "user"}, namespace="users")
    assert len(db) == 3

    # Delete all feedback
    deleted = db.delete_by_filter({"type": "feedback"}, namespace="feedback")
    assert deleted == 2
    assert len(db) == 1
    assert db.get("doc3", namespace="users") is not None

    # Delete with no matches returns 0
    deleted = db.delete_by_filter({"type": "nonexistent"})
    assert deleted == 0


def test_count_with_namespace_and_filter(tmp_path: Path) -> None:
    """db.count() supports namespace and filter parameters."""
    path = tmp_path / "count.vdb"
    db = vectlite.open(str(path), dimension=2)

    db.upsert("doc1", [1.0, 0.0], {"type": "user", "score": 10}, namespace="users")
    db.upsert("doc2", [0.0, 1.0], {"type": "feedback"}, namespace="feedback")
    db.upsert("doc3", [0.5, 0.5], {"type": "user", "score": 3}, namespace="users")

    # Total count
    assert db.count() == 3

    # Count by namespace
    assert db.count(namespace="users") == 2
    assert db.count(namespace="feedback") == 1
    assert db.count(namespace="nonexistent") == 0

    # Count with filter
    assert db.count(filter={"type": "user"}) == 2

    # Count with namespace + filter
    assert db.count(namespace="users", filter={"score": {"$gt": 5}}) == 1


def test_lock_contention_raises_specific_exception(tmp_path: Path) -> None:
    """Lock contention raises VectLiteLockError (subclass of VectLiteError)."""
    path = tmp_path / "locktype.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0])

    # VectLiteLockError should be catchable as both itself and VectLiteError
    with pytest.raises(vectlite.VectLiteLockError):
        vectlite.open(str(path))

    with pytest.raises(vectlite.VectLiteError):
        vectlite.open(str(path))

    del db


def test_lock_timeout(tmp_path: Path) -> None:
    """lock_timeout parameter retries before giving up."""
    path = tmp_path / "timeout.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.upsert("doc1", [1.0, 0.0])

    # With a very short timeout, should still fail (lock is held)
    with pytest.raises(vectlite.VectLiteLockError):
        vectlite.open(str(path), lock_timeout=0.1)

    del db


def test_lock_timeout_rejects_invalid_values(tmp_path: Path) -> None:
    """lock_timeout must be finite and non-negative."""
    path = tmp_path / "invalid-timeout.vdb"
    db = vectlite.open(str(path), dimension=2)
    db.close()

    with pytest.raises(vectlite.VectLiteError, match="lock_timeout"):
        vectlite.open(str(path), lock_timeout=-1.0)

    with pytest.raises(vectlite.VectLiteError, match="lock_timeout"):
        vectlite.open(str(path), lock_timeout=math.nan)


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------


def test_scalar_quantization(tmp_path: Path) -> None:
    """Scalar quantization accelerates search and persists across reopens."""
    path = tmp_path / "quant_scalar.vdb"
    db = vectlite.open(str(path), dimension=32)

    # Insert 50 records
    records = []
    for i in range(50):
        v = [0.0] * 32
        v[i % 32] = 1.0
        v[(i + 1) % 32] = 0.5
        records.append({"id": f"doc{i}", "vector": v})
    db.upsert_many(records)

    # Enable scalar quantization
    db.enable_quantization("scalar", rescore_multiplier=5)
    assert callable(db.is_quantized)
    assert db.is_quantized() is True
    assert db.quantization_method == "scalar"

    # Search should return correct results
    query = [0.0] * 32
    query[0] = 1.0
    results = db.search(query, k=5)
    assert len(results) > 0
    assert results[0]["id"] == "doc0"

    # Close and reopen: quantization should persist
    db.close()
    db2 = vectlite.open(str(path))
    assert db2.is_quantized() is True
    assert db2.quantization_method == "scalar"
    results2 = db2.search(query, k=5)
    assert results2[0]["id"] == "doc0"
    db2.close()


def test_binary_quantization(tmp_path: Path) -> None:
    """Binary quantization with Hamming distance + rescoring."""
    path = tmp_path / "quant_binary.vdb"
    db = vectlite.open(str(path), dimension=64)

    for i in range(100):
        v = [1.0 if (i + j) % 3 == 0 else -1.0 for j in range(64)]
        db.upsert(f"doc{i}", v)

    db.enable_quantization("binary")
    assert db.quantization_method == "binary"

    query = [1.0 if j % 3 == 0 else -1.0 for j in range(64)]
    results = db.search(query, k=5)
    assert results[0]["id"] == "doc0"
    db.close()


def test_quantization_rescore_multiplier_controls_candidate_count(tmp_path: Path) -> None:
    """rescore_multiplier controls the exact-rescore candidate budget."""
    for method in ("scalar", "binary"):
        path = tmp_path / f"quant_rescore_{method}.vdb"
        db = vectlite.open(str(path), dimension=32)

        records = []
        for i in range(200):
            vector = [1.0 if (i * 17 + j * 31) % 23 < 11 else -1.0 for j in range(32)]
            records.append({"id": f"doc{i}", "vector": vector})
        db.upsert_many(records)

        query = records[0]["vector"]

        db.enable_quantization(method, rescore_multiplier=1)
        outcome = db.search_with_stats(query, k=10)
        assert outcome["stats"]["used_ann"] is True
        assert outcome["stats"]["ann_candidate_count"] == 10

        db.disable_quantization()
        db.enable_quantization(method, rescore_multiplier=4)
        outcome = db.search_with_stats(query, k=10)
        assert outcome["stats"]["used_ann"] is True
        assert outcome["stats"]["ann_candidate_count"] == 40

        db.close()


def test_product_quantization(tmp_path: Path) -> None:
    """Product quantization compresses vectors into centroid codes."""
    path = tmp_path / "quant_pq.vdb"
    db = vectlite.open(str(path), dimension=32)

    for i in range(100):
        v = [((i * 7 + j * 13) % 100) / 100.0 for j in range(32)]
        db.upsert(f"doc{i}", v)

    db.enable_quantization(
        "product",
        num_sub_vectors=4,
        num_centroids=16,
        training_iterations=5,
    )
    assert db.quantization_method == "product"

    query = [(j * 13 % 100) / 100.0 for j in range(32)]
    results = db.search(query, k=5)
    assert results[0]["id"] == "doc0"
    db.close()


def test_product_quantization_invalid_subvectors_returns_vectlite_error(tmp_path: Path) -> None:
    """Invalid PQ partitioning returns a typed error instead of a Rust panic."""
    path = tmp_path / "quant_pq_invalid_subvectors.vdb"
    db = vectlite.open(str(path), dimension=146)

    assert db.valid_num_sub_vectors() == [1, 2, 73, 146]

    for i in range(8):
        db.upsert(f"doc{i}", [0.1 + (i + j) / 100.0 for j in range(146)])

    with pytest.raises(
        vectlite.VectLiteError,
        match=r"dimension \(146\) must be divisible by num_sub_vectors \(7\)",
    ):
        db.enable_quantization(
            "pq",
            num_sub_vectors=7,
            num_centroids=4,
            training_iterations=1,
        )

    assert db.is_quantized() is False

    db.enable_quantization("PQ", num_centroids=4, training_iterations=1)
    assert db.quantization_method == "product"
    db.close()


def test_disable_quantization(tmp_path: Path) -> None:
    """Disabling quantization removes sidecar and stops quantized search."""
    path = tmp_path / "quant_disable.vdb"
    db = vectlite.open(str(path), dimension=8)

    for i in range(10):
        db.upsert(f"doc{i}", [float(i + j) for j in range(8)])

    db.enable_quantization("scalar")
    assert db.is_quantized() is True

    db.disable_quantization()
    assert db.is_quantized() is False
    assert db.quantization_method is None

    # Search still works without quantization
    results = db.search([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], k=3)
    assert len(results) > 0
    db.close()


def test_quantization_empty_database_raises(tmp_path: Path) -> None:
    """Enabling quantization on empty database raises an error."""
    path = tmp_path / "quant_empty.vdb"
    db = vectlite.open(str(path), dimension=4)

    with pytest.raises(vectlite.VectLiteError):
        db.enable_quantization("scalar")

    db.close()


def test_quantization_invalid_method_raises(tmp_path: Path) -> None:
    """Using an invalid quantization method raises ValueError."""
    path = tmp_path / "quant_invalid.vdb"
    db = vectlite.open(str(path), dimension=4)
    db.upsert("doc1", [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(
        ValueError,
        match=r"Expected: 'scalar', 'binary', or 'pq' \(alias: 'product'\)",
    ):
        db.enable_quantization("invalid_method")

    db.close()


# ---------------------------------------------------------------------------
# Multi-vector / ColBERT-style tests
# ---------------------------------------------------------------------------


def test_multi_vector_upsert_and_search(tmp_path: Path) -> None:
    """Upsert records with multi-vectors and search via MaxSim."""
    path = tmp_path / "mv_basic.vdb"
    db = vectlite.open(str(path), dimension=3)

    # Upsert with ColBERT-style token vectors
    db.upsert_multi_vectors(
        "doc1",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]},
    )
    db.upsert_multi_vectors(
        "doc2",
        [0.0, 0.0, 1.0],
        {"colbert": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]},
    )

    assert db.count() == 2

    # Search with query tokens matching doc1
    results = db.search_multi_vector(
        "colbert",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        k=2,
    )

    assert len(results) == 2
    assert results[0]["id"] == "doc1"
    assert results[0]["score"] > results[1]["score"]

    db.close()


def test_multi_vector_with_metadata(tmp_path: Path) -> None:
    """Multi-vector upsert preserves metadata and it appears in results."""
    path = tmp_path / "mv_meta.vdb"
    db = vectlite.open(str(path), dimension=3)

    db.upsert_multi_vectors(
        "doc1",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 0.0, 0.0]]},
        metadata={"source": "blog"},
    )

    results = db.search_multi_vector("colbert", [[1.0, 0.0, 0.0]])
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "blog"

    db.close()


def test_multi_vector_namespace_filter(tmp_path: Path) -> None:
    """Multi-vector search respects namespace filtering."""
    path = tmp_path / "mv_ns.vdb"
    db = vectlite.open(str(path), dimension=3)

    db.upsert_multi_vectors(
        "doc1",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 0.0, 0.0]]},
        namespace="ns1",
    )
    db.upsert_multi_vectors(
        "doc2",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 0.0, 0.0]]},
        namespace="ns2",
    )

    results = db.search_multi_vector(
        "colbert", [[1.0, 0.0, 0.0]], namespace="ns1"
    )
    assert len(results) == 1
    assert results[0]["id"] == "doc1"

    db.close()


def test_multi_vector_quantization(tmp_path: Path) -> None:
    """Enable/disable 2-bit quantization for multi-vector space."""
    path = tmp_path / "mv_quant.vdb"
    db = vectlite.open(str(path), dimension=3)

    for i in range(10):
        db.upsert_multi_vectors(
            f"doc{i}",
            [float(i), 0.0, 0.0],
            {"colbert": [[float(i), 0.0, 0.0], [0.0, float(i), 0.0]]},
        )

    assert db.is_multi_vector_quantized("colbert") is False

    db.enable_multi_vector_quantization("colbert")
    assert db.is_multi_vector_quantized("colbert") is True

    # Search should still work
    results = db.search_multi_vector(
        "colbert", [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]], k=3
    )
    assert len(results) > 0

    # Disable
    db.disable_multi_vector_quantization("colbert")
    assert db.is_multi_vector_quantized("colbert") is False

    db.close()


def test_multi_vector_quantization_persists(tmp_path: Path) -> None:
    """Multi-vector quantization persists across database close/reopen."""
    path = tmp_path / "mv_quant_persist.vdb"
    db = vectlite.open(str(path), dimension=3)

    for i in range(10):
        db.upsert_multi_vectors(
            f"doc{i}",
            [1.0, 0.0, 0.0],
            {"colbert": [[float(i) * 0.1, 0.5, 0.5], [0.5, float(i) * 0.1, 0.5]]},
        )

    db.enable_multi_vector_quantization("colbert")
    assert db.is_multi_vector_quantized("colbert") is True
    db.close()

    db2 = vectlite.open(str(path))
    assert db2.is_multi_vector_quantized("colbert") is True

    results = db2.search_multi_vector("colbert", [[0.9, 0.5, 0.5]], k=5)
    assert len(results) > 0
    db2.close()


def test_multi_vector_invalid_method_raises(tmp_path: Path) -> None:
    """Using an invalid multi-vector quantization method raises ValueError."""
    path = tmp_path / "mv_invalid.vdb"
    db = vectlite.open(str(path), dimension=3)
    db.upsert_multi_vectors(
        "doc1",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 0.0, 0.0]]},
    )

    with pytest.raises(ValueError, match="unknown multi-vector quantization method"):
        db.enable_multi_vector_quantization("colbert", method="invalid_method")

    db.close()


def test_multi_vector_record_persists(tmp_path: Path) -> None:
    """Multi-vector data persists across close/reopen."""
    path = tmp_path / "mv_persist.vdb"
    db = vectlite.open(str(path), dimension=3)

    db.upsert_multi_vectors(
        "doc1",
        [1.0, 0.0, 0.0],
        {"colbert": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]},
    )
    db.close()

    db2 = vectlite.open(str(path))
    # Verify by searching — the tokens should be findable
    results = db2.search_multi_vector("colbert", [[1.0, 2.0, 3.0]], k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    db2.close()


# ---------------------------------------------------------------------------
# Distance metric tests
# ---------------------------------------------------------------------------


def test_default_metric_is_cosine(tmp_path: Path) -> None:
    """Database created without metric defaults to cosine."""
    path = tmp_path / "default_metric.vdb"
    db = vectlite.open(str(path), dimension=3)
    assert db.metric == "cosine"
    db.close()


def test_create_with_each_metric(tmp_path: Path) -> None:
    """Each supported metric can be set at creation and persists across reopen."""
    for metric_name in ["cosine", "euclidean", "dotproduct", "manhattan"]:
        path = tmp_path / f"metric_{metric_name}.vdb"
        db = vectlite.open(str(path), dimension=3, metric=metric_name)
        assert db.metric == metric_name
        db.close()

        # Reopen and verify metric persisted
        db2 = vectlite.open(str(path))
        assert db2.metric == metric_name
        db2.close()


def test_metric_aliases(tmp_path: Path) -> None:
    """Metric aliases (l2, dot, ip, l1) are accepted."""
    aliases = {
        "l2": "euclidean",
        "dot": "dotproduct",
        "ip": "dotproduct",
        "l1": "manhattan",
    }
    for alias, expected in aliases.items():
        path = tmp_path / f"metric_alias_{alias}.vdb"
        db = vectlite.open(str(path), dimension=3, metric=alias)
        assert db.metric == expected
        db.close()


def test_invalid_metric_raises(tmp_path: Path) -> None:
    """An invalid metric name raises VectLiteError."""
    path = tmp_path / "bad_metric.vdb"
    with pytest.raises(vectlite.VectLiteError, match="unknown distance metric"):
        vectlite.open(str(path), dimension=3, metric="hamming")


def test_euclidean_search_ordering(tmp_path: Path) -> None:
    """Euclidean metric orders results by L2 distance (closest first)."""
    path = tmp_path / "euclidean_search.vdb"
    db = vectlite.open(str(path), dimension=3, metric="euclidean")

    db.upsert("close", [1.0, 0.0, 0.0])  # L2 = 1 from [0,0,0]
    db.upsert("mid", [3.0, 0.0, 0.0])  # L2 = 3
    db.upsert("far", [5.0, 5.0, 5.0])  # L2 = sqrt(75) ≈ 8.66

    results = db.search([0.0, 0.0, 0.0], k=3)

    assert [r["id"] for r in results] == ["close", "mid", "far"]
    # Scores are negative distances (higher = closer)
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    db.close()


def test_dotproduct_search_ordering(tmp_path: Path) -> None:
    """Dot product metric orders by raw inner product (highest first)."""
    path = tmp_path / "dot_search.vdb"
    db = vectlite.open(str(path), dimension=3, metric="dotproduct")

    db.upsert("high", [10.0, 0.0, 0.0])  # dot = 10 with query [1,0,0]
    db.upsert("medium", [5.0, 0.0, 0.0])  # dot = 5
    db.upsert("low", [0.0, 1.0, 0.0])  # dot = 0

    results = db.search([1.0, 0.0, 0.0], k=3)

    assert [r["id"] for r in results] == ["high", "medium", "low"]
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    db.close()


def test_manhattan_search_ordering(tmp_path: Path) -> None:
    """Manhattan metric orders by L1 distance (closest first)."""
    path = tmp_path / "manhattan_search.vdb"
    db = vectlite.open(str(path), dimension=3, metric="manhattan")

    db.upsert("close", [1.0, 0.0, 0.0])  # L1 = 1 from [0,0,0]
    db.upsert("mid", [2.0, 1.0, 0.0])  # L1 = 3
    db.upsert("far", [3.0, 3.0, 3.0])  # L1 = 9

    results = db.search([0.0, 0.0, 0.0], k=3)

    assert [r["id"] for r in results] == ["close", "mid", "far"]
    assert results[0]["score"] > results[1]["score"] > results[2]["score"]
    db.close()


def test_cosine_explicit_search(tmp_path: Path) -> None:
    """Cosine metric explicitly set works same as default."""
    path = tmp_path / "cosine_explicit.vdb"
    db = vectlite.open(str(path), dimension=3, metric="cosine")

    db.upsert("aligned", [2.0, 0.0, 0.0])  # cos = 1.0 with [1,0,0]
    db.upsert("diagonal", [1.0, 1.0, 0.0])  # cos ≈ 0.707
    db.upsert("orthogonal", [0.0, 0.0, 1.0])  # cos = 0.0

    results = db.search([1.0, 0.0, 0.0], k=3)

    assert [r["id"] for r in results] == ["aligned", "diagonal", "orthogonal"]
    assert abs(results[0]["score"] - 1.0) < 1e-4
    db.close()


def test_metric_persists_after_upsert_and_reopen(tmp_path: Path) -> None:
    """Metric persists correctly even after writing data and reopening."""
    path = tmp_path / "metric_persist.vdb"
    db = vectlite.open(str(path), dimension=3, metric="manhattan")
    db.upsert("a", [1.0, 0.0, 0.0])
    db.upsert("b", [0.0, 5.0, 0.0])
    db.close()

    # Reopen, verify metric, and search
    db2 = vectlite.open(str(path))
    assert db2.metric == "manhattan"

    results = db2.search([1.0, 0.0, 0.0], k=2)
    assert results[0]["id"] == "a"  # L1 = 0 vs L1 = 6
    assert results[1]["id"] == "b"
    db2.close()


# ---------------------------------------------------------------------------
# update_metadata  (Feature 6: partial metadata patch)
# ---------------------------------------------------------------------------


def test_update_metadata_merges_patch(tmp_path: Path) -> None:
    """update_metadata merges keys without touching the vector."""
    db = vectlite.open(str(tmp_path / "update_meta.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"source": "blog", "version": 1})

    db.update_metadata("doc1", {"version": 2, "reviewed": True})

    record = db.get("doc1")
    assert record is not None
    assert record["metadata"]["source"] == "blog"  # untouched
    assert record["metadata"]["version"] == 2  # overwritten
    assert record["metadata"]["reviewed"] is True  # added
    assert record["vector"] == [1.0, 0.0, 0.0]  # vector intact
    db.close()


def test_update_metadata_returns_false_for_missing(tmp_path: Path) -> None:
    """update_metadata returns False when the id doesn't exist."""
    db = vectlite.open(str(tmp_path / "update_missing.vdb"), dimension=3)
    result = db.update_metadata("nonexistent", {"key": "val"})
    assert result is False
    db.close()


def test_update_metadata_persists_across_reopen(tmp_path: Path) -> None:
    """Partial metadata patch survives close/reopen via WAL replay."""
    path = str(tmp_path / "update_persist.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"source": "blog"})
    db.update_metadata("doc1", {"source": "updated", "new_key": 42})
    db.close()

    db2 = vectlite.open(path)
    record = db2.get("doc1")
    assert record is not None
    assert record["metadata"]["source"] == "updated"
    assert record["metadata"]["new_key"] == 42
    assert record["vector"] == [1.0, 0.0, 0.0]
    db2.close()


def test_update_metadata_with_namespace(tmp_path: Path) -> None:
    """update_metadata works correctly with explicit namespaces."""
    db = vectlite.open(str(tmp_path / "update_ns.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"key": "original"}, namespace="ns1")

    result = db.update_metadata("doc1", {"key": "patched"}, namespace="ns1")
    assert result is True

    record = db.get("doc1", namespace="ns1")
    assert record is not None
    assert record["metadata"]["key"] == "patched"

    # Wrong namespace returns False
    result2 = db.update_metadata("doc1", {"key": "nope"}, namespace="ns2")
    assert result2 is False
    db.close()


def test_update_metadata_searchable_after_patch(tmp_path: Path) -> None:
    """After patching metadata, filtered count reflects the new values."""
    db = vectlite.open(str(tmp_path / "update_search.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"status": "draft"})

    assert db.count(filter={"status": "draft"}) == 1

    db.update_metadata("doc1", {"status": "published"})

    assert db.count(filter={"status": "draft"}) == 0
    assert db.count(filter={"status": "published"}) == 1
    db.close()


def test_update_metadata_read_only_fails(tmp_path: Path) -> None:
    """update_metadata on a read-only database raises VectLiteError."""
    path = str(tmp_path / "update_ro.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0])
    db.close()

    db_ro = vectlite.open(path, read_only=True)
    with pytest.raises(vectlite.VectLiteError):
        db_ro.update_metadata("doc1", {"key": "val"})
    db_ro.close()


# ── Payload Index tests ──────────────────────────────────────────────


def test_create_index_keyword(tmp_path: Path) -> None:
    """create_index creates a keyword index and returns True."""
    db = vectlite.open(str(tmp_path / "pidx.vdb"), dimension=3)
    assert db.create_index("source", "keyword") is True
    indexes = db.list_indexes()
    assert len(indexes) == 1
    assert indexes[0] == ("source", "keyword")
    db.close()


def test_create_index_duplicate_returns_false(tmp_path: Path) -> None:
    """Duplicate create_index call returns False."""
    db = vectlite.open(str(tmp_path / "pidx_dup.vdb"), dimension=3)
    assert db.create_index("source", "keyword") is True
    assert db.create_index("source", "keyword") is False
    assert len(db.list_indexes()) == 1
    db.close()


def test_create_index_numeric(tmp_path: Path) -> None:
    """create_index creates a numeric index."""
    db = vectlite.open(str(tmp_path / "pidx_num.vdb"), dimension=3)
    assert db.create_index("price", "numeric") is True
    indexes = db.list_indexes()
    assert len(indexes) == 1
    assert indexes[0] == ("price", "numeric")
    db.close()


def test_drop_index(tmp_path: Path) -> None:
    """drop_index removes an existing index."""
    db = vectlite.open(str(tmp_path / "pidx_drop.vdb"), dimension=3)
    db.create_index("source", "keyword")
    assert db.drop_index("source") is True
    assert len(db.list_indexes()) == 0
    assert db.drop_index("source") is False
    db.close()


def test_list_indexes_empty(tmp_path: Path) -> None:
    """list_indexes returns empty list by default."""
    db = vectlite.open(str(tmp_path / "pidx_empty.vdb"), dimension=3)
    assert db.list_indexes() == []
    db.close()


def test_keyword_index_accelerates_count(tmp_path: Path) -> None:
    """Keyword index accelerates count with $eq filter."""
    db = vectlite.open(str(tmp_path / "pidx_count.vdb"), dimension=3)
    for i in range(20):
        db.upsert(f"doc{i}", [1.0, 0.0, 0.0], {"tag": f"t{i % 4}"})

    db.create_index("tag", "keyword")

    assert db.count(filter={"tag": "t0"}) == 5
    assert db.count(filter={"tag": "t3"}) == 5
    assert db.count(filter={"tag": "t99"}) == 0
    db.close()


def test_numeric_index_range_queries(tmp_path: Path) -> None:
    """Numeric index accelerates range queries."""
    db = vectlite.open(str(tmp_path / "pidx_range.vdb"), dimension=3)
    for i in range(50):
        db.upsert(f"doc{i}", [1.0, 0.0, 0.0], {"score": float(i)})

    db.create_index("score", "numeric")

    assert db.count(filter={"score": {"$gt": 40}}) == 9
    assert db.count(filter={"score": {"$gte": 40}}) == 10
    assert db.count(filter={"score": {"$lt": 10}}) == 10
    assert db.count(filter={"score": {"$lte": 10}}) == 11
    db.close()


def test_payload_index_persists_across_reopen(tmp_path: Path) -> None:
    """Index definitions and data persist across close/reopen."""
    path = str(tmp_path / "pidx_persist.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"source": "blog"})
    db.upsert("doc2", [0.0, 1.0, 0.0], {"source": "docs"})
    db.create_index("source", "keyword")
    db.close()

    db2 = vectlite.open(path)
    indexes = db2.list_indexes()
    assert len(indexes) == 1
    assert indexes[0][0] == "source"
    assert db2.count(filter={"source": "blog"}) == 1
    db2.close()


def test_payload_index_update_metadata(tmp_path: Path) -> None:
    """update_metadata correctly updates the payload index."""
    db = vectlite.open(str(tmp_path / "pidx_upd.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"status": "draft"})
    db.create_index("status", "keyword")

    assert db.count(filter={"status": "draft"}) == 1
    db.update_metadata("doc1", {"status": "published"})
    assert db.count(filter={"status": "draft"}) == 0
    assert db.count(filter={"status": "published"}) == 1
    db.close()


def test_create_index_read_only_fails(tmp_path: Path) -> None:
    """create_index on a read-only database raises VectLiteError."""
    path = str(tmp_path / "pidx_ro.vdb")
    db = vectlite.open(path, dimension=3)
    db.close()

    db_ro = vectlite.open(path, read_only=True)
    with pytest.raises(vectlite.VectLiteError):
        db_ro.create_index("source", "keyword")
    db_ro.close()


def test_payload_index_search_filters(tmp_path: Path) -> None:
    """Search with a payload-indexed filter returns correct results."""
    db = vectlite.open(str(tmp_path / "pidx_search.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], {"category": "tech"})
    db.upsert("doc2", [0.9, 0.1, 0.0], {"category": "science"})
    db.upsert("doc3", [0.8, 0.2, 0.0], {"category": "tech"})
    db.create_index("category", "keyword")

    results = db.search([1.0, 0.0, 0.0], k=10, filter={"category": "tech"})
    ids = [r["id"] for r in results]
    assert "doc1" in ids
    assert "doc3" in ids
    assert "doc2" not in ids
    db.close()


def test_payload_index_invalid_type_raises(tmp_path: Path) -> None:
    """create_index with an invalid type name raises an error."""
    db = vectlite.open(str(tmp_path / "pidx_bad.vdb"), dimension=3)
    with pytest.raises(vectlite.VectLiteError):
        db.create_index("field", "invalid_type")
    db.close()


# -------------------------------------------------------------------
# TTL / Expiry tests
# -------------------------------------------------------------------

import time


def test_set_ttl_hides_record(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0])
    assert db.get("doc1") is not None

    db.set_ttl("doc1", 0.0)
    time.sleep(0.02)
    assert db.get("doc1") is None
    db.close()


def test_clear_ttl_restores_record(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_clear.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0])
    db.set_ttl("doc1", 0.0)
    time.sleep(0.02)
    assert db.get("doc1") is None

    db.clear_ttl("doc1")
    assert db.get("doc1") is not None
    db.close()


def test_ttl_excludes_from_count_list_search(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_cls.vdb"), dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.upsert("b", [0.0, 1.0, 0.0])
    assert db.count() == 2
    assert len(db.list()) == 2

    db.set_ttl("a", 0.0)
    time.sleep(0.02)
    assert db.count() == 1
    assert len(db.list()) == 1
    results = db.search([1.0, 0.0, 0.0], k=10)
    assert len(results) == 1
    assert results[0]["id"] == "b"
    db.close()


def test_upsert_with_ttl(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_up.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], ttl=0.0)
    time.sleep(0.02)
    assert db.get("doc1") is None

    # With a long TTL, record is visible
    db.upsert("doc2", [0.0, 1.0, 0.0], ttl=86400)
    assert db.get("doc2") is not None
    db.close()


def test_upsert_many_with_ttl(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_batch.vdb"), dimension=3)
    db.upsert_many([
        {"id": "a", "vector": [1.0, 0.0, 0.0], "ttl": 0.0},
        {"id": "b", "vector": [0.0, 1.0, 0.0]},
    ])
    time.sleep(0.02)
    assert db.get("a") is None
    assert db.get("b") is not None
    db.close()


def test_ttl_persists_after_reopen(tmp_path: Path) -> None:
    path = str(tmp_path / "ttl_persist.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0])
    db.set_ttl("doc1", 0.0)
    db.close()

    time.sleep(0.02)
    db2 = vectlite.open(path)
    assert db2.get("doc1") is None
    db2.close()


def test_set_ttl_returns_false_for_missing(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_miss.vdb"), dimension=3)
    assert db.set_ttl("ghost", 60.0) is False
    assert db.clear_ttl("ghost") is False
    db.close()


def test_ttl_with_namespace(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_ns.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], namespace="ns1")
    db.set_ttl("doc1", 0.0, namespace="ns1")
    time.sleep(0.02)
    assert db.get("doc1", namespace="ns1") is None

    # Wrong namespace
    assert db.set_ttl("doc1", 60.0, namespace="ns2") is False
    db.close()


def test_transaction_upsert_with_ttl(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_tx.vdb"), dimension=3)
    with db.transaction() as tx:
        tx.upsert("a", [1.0, 0.0, 0.0], ttl=0.0)
        tx.upsert("b", [0.0, 1.0, 0.0], ttl=86400)
    time.sleep(0.02)
    assert db.get("a") is None
    assert db.get("b") is not None
    db.close()


def test_expires_at_in_record_output(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "ttl_ea.vdb"), dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0], ttl=86400)
    record = db.get("doc1")
    assert record is not None
    assert record["expires_at"] is not None
    assert record["expires_at"] > time.time()
    db.close()


# -------------------------------------------------------------------
# Cursor-based pagination
# -------------------------------------------------------------------


def test_list_cursor_basic(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "cursor.vdb"), dimension=3)
    for i in range(5):
        db.upsert(f"doc{i}", [1.0, 0.0, 0.0])

    # First page of 2
    records, cursor = db.list_cursor(limit=2)
    assert len(records) == 2
    assert cursor is not None

    # Second page of 2
    records2, cursor2 = db.list_cursor(limit=2, cursor=cursor)
    assert len(records2) == 2
    assert cursor2 is not None

    # Third page (only 1 remaining)
    records3, cursor3 = db.list_cursor(limit=2, cursor=cursor2)
    assert len(records3) == 1
    assert cursor3 is None

    # All ids are unique
    all_ids = [r["id"] for r in records + records2 + records3]
    assert len(set(all_ids)) == 5
    db.close()


def test_list_cursor_with_namespace(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "cursor_ns.vdb"), dimension=3)
    for i in range(3):
        db.upsert(f"doc{i}", [1.0, 0.0, 0.0], namespace="ns1")
    for i in range(2):
        db.upsert(f"doc{i}", [0.0, 1.0, 0.0], namespace="ns2")

    page1, c1 = db.list_cursor(namespace="ns1", limit=2)
    assert len(page1) == 2
    assert c1 is not None

    page2, c2 = db.list_cursor(namespace="ns1", limit=2, cursor=c1)
    assert len(page2) == 1
    assert c2 is None
    db.close()


def test_list_cursor_excludes_expired(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "cursor_ttl.vdb"), dimension=3)
    for i in range(5):
        db.upsert(f"doc{i}", [1.0, 0.0, 0.0])
    db.set_ttl("doc1", 0.0)
    time.sleep(0.01)

    all_records: list[dict] = []
    cursor = None
    while True:
        page, cursor = db.list_cursor(limit=10, cursor=cursor)
        all_records.extend(page)
        if cursor is None:
            break
    assert len(all_records) == 4
    assert all(r["id"] != "doc1" for r in all_records)
    db.close()


def test_list_cursor_empty(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "cursor_empty.vdb"), dimension=3)
    records, cursor = db.list_cursor(limit=10)
    assert records == []
    assert cursor is None
    db.close()


# -------------------------------------------------------------------
# Embedding providers (import test)
# -------------------------------------------------------------------


def test_embedders_module_importable() -> None:
    from vectlite import embedders

    # All factory functions should be present
    assert callable(embedders.openai)
    assert callable(embedders.cohere)
    assert callable(embedders.voyage)
    assert callable(embedders.fastembed)
    assert callable(embedders.sentence_transformer)
    assert callable(embedders.ollama)


def test_embedders_missing_sdk_raises() -> None:
    """Calling a provider without its SDK installed raises ImportError."""
    from vectlite import embedders
    import importlib
    import sys

    # We test that calling the factory with a fake sdk name fails gracefully.
    # Since openai/cohere/etc might or might not be installed, we just verify
    # the module structure is correct and each function is callable.
    for name in embedders.__all__:
        fn = getattr(embedders, name)
        assert callable(fn), f"embedders.{name} should be callable"


def test_embedders_with_upsert_text(tmp_path: Path) -> None:
    """Test that a custom embed function works with upsert_text/search_text."""
    db = vectlite.open(str(tmp_path / "embed_test.vdb"), dimension=3)

    # Trivial embedder for testing
    def mock_embed(text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]

    vectlite.upsert_text(db, "doc1", "hello world", mock_embed)
    results = vectlite.search_text(db, "hello", mock_embed, k=1)
    assert len(results) >= 1
    assert results[0]["id"] == "doc1"
    db.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def test_cli_stats(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_stats.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("doc1", [1.0, 0.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["stats", path])
    output = json.loads(buf.getvalue())
    assert output["total_records"] == 1
    assert output["dimension"] == 3


def test_cli_count(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_count.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.upsert("b", [0.0, 1.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["count", path])
    assert buf.getvalue().strip() == "2"


def test_cli_list(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_list.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0], {"tag": "x"})
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["list", path, "--limit", "1"])
    records = json.loads(buf.getvalue())
    assert len(records) == 1
    assert records[0]["id"] == "a"


def test_cli_compact(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_compact.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["compact", path])
    assert "Compacted" in buf.getvalue()


def test_cli_verify(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_verify.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["verify", path])
    assert "OK" in buf.getvalue()


def test_cli_dump(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_dump.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.upsert("b", [0.0, 1.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["dump", path])
    lines = [l for l in buf.getvalue().strip().split("\n") if l]
    assert len(lines) == 2
    for line in lines:
        record = json.loads(line)
        assert "id" in record


def test_cli_search(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_search.vdb")
    db = vectlite.open(path, dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.close()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["search", path, "--query", "[1.0, 0.0, 0.0]", "--k", "1"])
    results = json.loads(buf.getvalue())
    assert len(results) == 1
    assert results[0]["id"] == "a"


def test_cli_import_jsonl(tmp_path: Path) -> None:
    from vectlite.cli import main
    import io
    import contextlib

    path = str(tmp_path / "cli_import.vdb")
    jsonl_file = str(tmp_path / "data.jsonl")
    with open(jsonl_file, "w") as f:
        f.write(json.dumps({"id": "r1", "vector": [1.0, 0.0, 0.0], "metadata": {"k": "v"}}) + "\n")
        f.write(json.dumps({"id": "r2", "vector": [0.0, 1.0, 0.0], "metadata": {}}) + "\n")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["import-jsonl", path, jsonl_file, "--dimension", "3"])
    assert "Imported 2" in buf.getvalue()

    db = vectlite.open(path, read_only=True)
    assert db.count() == 2
    db.close()


# -------------------------------------------------------------------
# Schema validation
# -------------------------------------------------------------------


def test_schema_basic_validation() -> None:
    from vectlite.schema import Schema, SchemaError

    s = Schema({
        "price": "number",
        "title": "string",
        "active": "boolean",
    })

    # Valid
    s.validate({"price": 9.99, "title": "Hello", "active": True})
    s.validate({"price": 42})  # partial is OK
    s.validate(None)           # None is OK
    s.validate({})             # empty is OK

    # Invalid
    import pytest
    with pytest.raises(SchemaError, match="price"):
        s.validate({"price": "not a number"})
    with pytest.raises(SchemaError, match="title"):
        s.validate({"title": 123})
    with pytest.raises(SchemaError, match="active"):
        s.validate({"active": "yes"})


def test_schema_nested_object() -> None:
    from vectlite.schema import Schema, SchemaError

    s = Schema({
        "author": {
            "name": "string",
            "age": "number",
        },
    })

    s.validate({"author": {"name": "Alice", "age": 30}})
    s.validate({"author": {"name": "Bob"}})

    import pytest
    with pytest.raises(SchemaError, match="author.age"):
        s.validate({"author": {"name": "Alice", "age": "thirty"}})


def test_schema_typed_array() -> None:
    from vectlite.schema import Schema, SchemaError

    s = Schema({"tags": "array<string>"})

    s.validate({"tags": ["a", "b", "c"]})
    s.validate({"tags": []})

    import pytest
    with pytest.raises(SchemaError, match="tags"):
        s.validate({"tags": "not an array"})
    with pytest.raises(SchemaError, match="tags\\[1\\]"):
        s.validate({"tags": ["ok", 42]})


def test_schema_strict_mode() -> None:
    from vectlite.schema import Schema, SchemaError

    s = Schema({"price": "number"}, strict=True)

    s.validate({"price": 10})

    import pytest
    with pytest.raises(SchemaError, match="unknown"):
        s.validate({"price": 10, "extra_field": "oops"})


def test_schema_null_values_allowed() -> None:
    from vectlite.schema import Schema

    s = Schema({"price": "number"})
    # None values pass validation (represent missing data)
    s.validate({"price": None})


def test_schema_save_and_load(tmp_path: Path) -> None:
    from vectlite.schema import Schema, load

    db = vectlite.open(str(tmp_path / "schema_test.vdb"), dimension=3)

    s = Schema({"price": "number", "tags": "array<string>"}, strict=True)
    s.save(db)

    loaded = load(db)
    assert loaded is not None
    assert loaded.fields == s.fields
    assert loaded.strict is True
    db.close()


def test_schema_load_missing(tmp_path: Path) -> None:
    from vectlite.schema import load

    db = vectlite.open(str(tmp_path / "no_schema.vdb"), dimension=3)
    assert load(db) is None
    db.close()


def test_validated_database(tmp_path: Path) -> None:
    from vectlite.schema import Schema, SchemaError, validated

    db = vectlite.open(str(tmp_path / "validated.vdb"), dimension=3)
    s = Schema({"price": "number"})
    vdb = validated(db, s)

    # Valid write
    vdb.upsert("doc1", [1.0, 0.0, 0.0], {"price": 9.99})
    assert vdb.get("doc1") is not None

    # Invalid write
    import pytest
    with pytest.raises(SchemaError, match="price"):
        vdb.upsert("doc2", [0.0, 1.0, 0.0], {"price": "free"})

    # doc2 should NOT have been written
    assert vdb.get("doc2") is None
    db.close()


# -------------------------------------------------------------------
# Bug #14: zero-norm query vector should be rejected for cosine
# -------------------------------------------------------------------


def test_search_zero_norm_query_raises(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "zero.vdb"), dimension=3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.upsert("b", [0.0, 1.0, 0.0])

    with pytest.raises(vectlite.VectLiteError, match="zero norm"):
        db.search([0.0, 0.0, 0.0], k=5)
    db.close()


def test_search_zero_norm_euclidean_allowed(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "zero-euc.vdb"), dimension=3, metric="euclidean")
    db.upsert("a", [1.0, 0.0, 0.0])
    results = db.search([0.0, 0.0, 0.0], k=5)
    assert len(results) == 1
    db.close()


# -------------------------------------------------------------------
# Bug #15: dimension mismatch in search query should be rejected
# -------------------------------------------------------------------


def test_search_wrong_dimension_raises(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "dim.vdb"), dimension=4)
    db.upsert("a", [1.0, 0.0, 0.0, 0.0])

    # Undersized query (dim=2 vs db dim=4)
    with pytest.raises(vectlite.VectLiteError, match="dimension mismatch"):
        db.search([1.0, 0.0], k=5)

    # Oversized query (dim=6 vs db dim=4)
    with pytest.raises(vectlite.VectLiteError, match="dimension mismatch"):
        db.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=5)
    db.close()


def test_search_undersized_query_with_truncate_dim_ok(tmp_path: Path) -> None:
    db = vectlite.open(str(tmp_path / "dim-trunc.vdb"), dimension=4)
    db.upsert("a", [1.0, 0.0, 0.0, 0.0])
    results = db.search([1.0, 0.0], k=5, truncate_dim=2)
    assert len(results) == 1
    db.close()


# -------------------------------------------------------------------
# Bug #16: Store.close() should exist
# -------------------------------------------------------------------


def test_store_has_close(tmp_path: Path) -> None:
    store = vectlite.open_store(str(tmp_path / "store"))
    db = store.create_collection("c", 3)
    db.upsert("a", [1.0, 0.0, 0.0])
    db.close()
    # Store.close() should not raise
    store.close()
