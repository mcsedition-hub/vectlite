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
