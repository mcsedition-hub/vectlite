#!/usr/bin/env python3
"""
Bench single-record `insert` throughput before/after the ingestion fix.

Usage:
    # one-shot run:
    python3 scripts/bench_ingest.py

    # custom run:
    python3 scripts/bench_ingest.py --records 20000 --dim 384

Requires the local Python binding to be built and importable, e.g.:
    cd bindings/python && maturin develop --release
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import tempfile
import time
from pathlib import Path


def bench_single_inserts(n_records: int, dim: int) -> dict:
    import vectlite  # provided by `bindings/python` via maturin

    tmpdir = Path(tempfile.mkdtemp(prefix="vectlite-bench-"))
    db_path = tmpdir / "bench.vdb"
    try:
        db = vectlite.open(str(db_path), dimension=dim)

        rng = random.Random(42)
        vectors = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(n_records)]

        # Warm-up: a few inserts so the WAL file exists and the cached writer
        # is primed.
        for i in range(min(8, n_records)):
            db.insert(f"warmup-{i}", vectors[i], {})

        t0 = time.perf_counter()
        for i in range(n_records):
            db.insert(f"doc-{i}", vectors[i], {"src": "bench"})
        t_inserts = time.perf_counter() - t0

        # Force a flush to measure the deferred cost too.
        t1 = time.perf_counter()
        db.flush()
        t_flush = time.perf_counter() - t1

        return {
            "n_records": n_records,
            "dim": dim,
            "insert_seconds": t_inserts,
            "flush_seconds": t_flush,
            "inserts_per_sec": n_records / t_inserts if t_inserts > 0 else float("inf"),
            "total_seconds": t_inserts + t_flush,
            "throughput_with_flush": n_records / (t_inserts + t_flush),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_bulk_ingest(n_records: int, dim: int, batch_size: int = 1024) -> dict:
    import vectlite

    tmpdir = Path(tempfile.mkdtemp(prefix="vectlite-bench-bulk-"))
    db_path = tmpdir / "bench.vdb"
    try:
        db = vectlite.open(str(db_path), dimension=dim)
        rng = random.Random(42)

        records = [
            {
                "id": f"doc-{i}",
                "vector": [rng.uniform(-1.0, 1.0) for _ in range(dim)],
                "metadata": {"src": "bench"},
            }
            for i in range(n_records)
        ]

        t0 = time.perf_counter()
        db.bulk_ingest(records, batch_size=batch_size)
        elapsed = time.perf_counter() - t0
        return {
            "n_records": n_records,
            "dim": dim,
            "batch_size": batch_size,
            "bulk_seconds": elapsed,
            "bulk_per_sec": n_records / elapsed,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_delete_throughput(n_records: int, n_deletes: int, dim: int) -> dict:
    """Measure delete throughput: insert N, then delete the first M."""
    import vectlite

    tmpdir = Path(tempfile.mkdtemp(prefix="vectlite-bench-delete-"))
    db_path = tmpdir / "bench.vdb"
    try:
        db = vectlite.open(str(db_path), dimension=dim)
        rng = random.Random(42)
        records = [
            {
                "id": f"doc-{i}",
                "vector": [rng.uniform(-1.0, 1.0) for _ in range(dim)],
                "metadata": {},
            }
            for i in range(n_records)
        ]
        db.bulk_ingest(records, batch_size=1024)

        t0 = time.perf_counter()
        for i in range(n_deletes):
            db.delete(f"doc-{i}")
        t_del = time.perf_counter() - t0
        return {
            "n_deletes": n_deletes,
            "delete_seconds": t_del,
            "deletes_per_sec": n_deletes / t_del if t_del > 0 else float("inf"),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--skip-bulk", action="store_true")
    parser.add_argument("--skip-delete", action="store_true")
    args = parser.parse_args()

    print(f"== single insert() with WalSyncMode=PerOp (default): "
          f"{args.records} records of dim {args.dim} ==")
    r = bench_single_inserts(args.records, args.dim)
    print(f"  insert loop : {r['insert_seconds']:.3f}s "
          f"= {r['inserts_per_sec']:>10.1f} vec/s")
    print(f"  flush()      : {r['flush_seconds']:.3f}s")
    print(f"  including flush: {r['throughput_with_flush']:.1f} vec/s")

    if not args.skip_bulk:
        print()
        print(f"== bulk_ingest(): {args.records} records of dim {args.dim} ==")
        r = bench_bulk_ingest(args.records, args.dim)
        print(f"  bulk_ingest : {r['bulk_seconds']:.3f}s "
              f"= {r['bulk_per_sec']:>10.1f} vec/s")

    if not args.skip_delete:
        print()
        n_del = min(1000, args.records // 4)
        print(f"== delete() throughput (tombstoning): {n_del} deletes after "
              f"{args.records} inserts ==")
        r = bench_delete_throughput(args.records, n_del, args.dim)
        print(f"  delete loop : {r['delete_seconds']:.3f}s "
              f"= {r['deletes_per_sec']:>10.1f} del/s")


if __name__ == "__main__":
    main()
