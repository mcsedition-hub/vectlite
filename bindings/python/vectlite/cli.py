"""Command-line interface for vectlite.

Usage::

    vectlite stats my.vdb
    vectlite count my.vdb --namespace blog
    vectlite list my.vdb --limit 5
    vectlite dump my.vdb
    vectlite import-jsonl my.vdb data.jsonl --dimension 384
    vectlite import-csv my.vdb data.csv --dimension 384 --vector-col embedding
    vectlite compact my.vdb
    vectlite verify my.vdb
    vectlite bench my.vdb --queries 1000
    vectlite search my.vdb --query '[1.0, 0.0, 0.5]' --k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any

import vectlite


def _open_db(path: str, read_only: bool = False, dimension: int | None = None) -> vectlite.Database:
    return vectlite.open(path, dimension=dimension, read_only=read_only)


def _json_out(data: Any) -> None:
    json.dump(data, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


# -------------------------------------------------------------------
# stats
# -------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    """Print database statistics."""
    db = _open_db(args.path, read_only=True)
    path = db.path
    wal_path = db.wal_path
    namespaces = db.namespaces()
    counts = {}
    total = 0
    for ns in namespaces:
        c = db.count(namespace=ns)
        counts[ns] = c
        total += c

    file_size = os.path.getsize(path) if os.path.isfile(path) else 0
    wal_size = os.path.getsize(wal_path) if os.path.isfile(wal_path) else 0

    info = {
        "path": path,
        "wal_path": wal_path,
        "dimension": db.dimension,
        "metric": db.metric,
        "read_only": db.read_only,
        "total_records": total,
        "namespaces": counts,
        "file_size_bytes": file_size,
        "wal_size_bytes": wal_size,
        "indexes": db.list_indexes(),
    }
    _json_out(info)
    db.close()


# -------------------------------------------------------------------
# count
# -------------------------------------------------------------------

def cmd_count(args: argparse.Namespace) -> None:
    """Print record count."""
    db = _open_db(args.path, read_only=True)
    filt = json.loads(args.filter) if args.filter else None
    c = db.count(namespace=args.namespace, filter=filt)
    print(c)
    db.close()


# -------------------------------------------------------------------
# list
# -------------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> None:
    """List records."""
    db = _open_db(args.path, read_only=True)
    filt = json.loads(args.filter) if args.filter else None
    records = db.list(
        namespace=args.namespace,
        filter=filt,
        limit=args.limit,
        offset=args.offset,
    )
    _json_out(records)
    db.close()


# -------------------------------------------------------------------
# dump
# -------------------------------------------------------------------

def cmd_dump(args: argparse.Namespace) -> None:
    """Dump all records as JSONL to stdout."""
    db = _open_db(args.path, read_only=True)
    cursor = None
    while True:
        page, cursor = db.list_cursor(
            namespace=args.namespace,
            limit=500,
            cursor=cursor,
        )
        for record in page:
            json.dump(record, sys.stdout, default=str)
            sys.stdout.write("\n")
        if cursor is None:
            break
    db.close()


# -------------------------------------------------------------------
# import-jsonl
# -------------------------------------------------------------------

def cmd_import_jsonl(args: argparse.Namespace) -> None:
    """Import records from a JSONL file."""
    db = _open_db(args.path, dimension=args.dimension)
    count = 0
    batch: list[dict[str, Any]] = []

    with open(args.file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            batch.append(record)
            if len(batch) >= args.batch_size:
                db.upsert_many(batch, namespace=args.namespace)
                count += len(batch)
                batch.clear()

    if batch:
        db.upsert_many(batch, namespace=args.namespace)
        count += len(batch)

    print(f"Imported {count} records")
    db.close()


# -------------------------------------------------------------------
# import-csv
# -------------------------------------------------------------------

def cmd_import_csv(args: argparse.Namespace) -> None:
    """Import records from a CSV file."""
    db = _open_db(args.path, dimension=args.dimension)
    count = 0
    id_col = args.id_col
    vector_col = args.vector_col

    with open(args.file, "r", newline="") as f:
        reader = csv.DictReader(f)
        batch: list[dict[str, Any]] = []
        for row in reader:
            record_id = row.get(id_col, f"row_{count}")
            vector_str = row.get(vector_col, "")
            try:
                vector = json.loads(vector_str)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: skipping row {count}, cannot parse vector column '{vector_col}'", file=sys.stderr)
                count += 1
                continue

            metadata = {k: v for k, v in row.items() if k not in (id_col, vector_col)}
            # Try to parse numeric values
            for k, v in metadata.items():
                try:
                    metadata[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        metadata[k] = float(v)
                    except (ValueError, TypeError):
                        pass

            batch.append({
                "id": str(record_id),
                "vector": vector,
                "metadata": metadata,
            })
            if len(batch) >= args.batch_size:
                db.upsert_many(batch, namespace=args.namespace)
                batch.clear()
            count += 1

    if batch:
        db.upsert_many(batch, namespace=args.namespace)

    print(f"Imported {count} records")
    db.close()


# -------------------------------------------------------------------
# compact
# -------------------------------------------------------------------

def cmd_compact(args: argparse.Namespace) -> None:
    """Compact the database (merge WAL into main file)."""
    db = _open_db(args.path)
    t0 = time.monotonic()
    db.compact()
    elapsed = time.monotonic() - t0
    print(f"Compacted in {elapsed:.3f}s")
    db.close()


# -------------------------------------------------------------------
# verify
# -------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    """Verify database integrity by opening and reading all records."""
    t0 = time.monotonic()
    try:
        db = _open_db(args.path, read_only=True)
    except Exception as e:
        print(f"FAIL: cannot open database: {e}", file=sys.stderr)
        sys.exit(1)

    total = 0
    errors = 0
    cursor = None
    while True:
        try:
            page, cursor = db.list_cursor(limit=1000, cursor=cursor)
        except Exception as e:
            print(f"FAIL: error during iteration: {e}", file=sys.stderr)
            errors += 1
            break
        total += len(page)
        if cursor is None:
            break

    elapsed = time.monotonic() - t0
    if errors == 0:
        print(f"OK: {total} records verified in {elapsed:.3f}s")
    else:
        print(f"FAIL: {errors} error(s), {total} records read in {elapsed:.3f}s", file=sys.stderr)
        sys.exit(1)
    db.close()


# -------------------------------------------------------------------
# bench
# -------------------------------------------------------------------

def cmd_bench(args: argparse.Namespace) -> None:
    """Run a simple search benchmark."""
    import random

    db = _open_db(args.path, read_only=True)
    dim = db.dimension
    n_queries = args.queries
    k = args.k

    print(f"Benchmarking: {n_queries} queries, k={k}, dimension={dim}")

    # Generate random query vectors
    random.seed(42)
    queries = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(n_queries)]

    t0 = time.monotonic()
    for query in queries:
        db.search(query, k=k)
    elapsed = time.monotonic() - t0

    qps = n_queries / elapsed if elapsed > 0 else float("inf")
    avg_ms = (elapsed / n_queries) * 1000 if n_queries > 0 else 0

    result = {
        "queries": n_queries,
        "k": k,
        "total_seconds": round(elapsed, 3),
        "queries_per_second": round(qps, 1),
        "avg_latency_ms": round(avg_ms, 2),
    }
    _json_out(result)
    db.close()


# -------------------------------------------------------------------
# search
# -------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    """Run a search query."""
    db = _open_db(args.path, read_only=True)
    query = json.loads(args.query)
    filt = json.loads(args.filter) if args.filter else None

    if args.stats:
        response = db.search_with_stats(
            query,
            k=args.k,
            filter=filt,
            namespace=args.namespace,
        )
        _json_out(response)
    else:
        results = db.search(
            query,
            k=args.k,
            filter=filt,
            namespace=args.namespace,
        )
        _json_out(results)
    db.close()


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vectlite",
        description="vectlite - Embedded vector store CLI",
    )
    parser.add_argument("--version", action="version", version=f"vectlite {vectlite.__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # stats
    p = subparsers.add_parser("stats", help="Print database statistics")
    p.add_argument("path", help="Path to .vdb file")
    p.set_defaults(func=cmd_stats)

    # count
    p = subparsers.add_parser("count", help="Count records")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("--namespace", "-n", default=None, help="Namespace to count")
    p.add_argument("--filter", "-f", default=None, help="JSON filter expression")
    p.set_defaults(func=cmd_count)

    # list
    p = subparsers.add_parser("list", help="List records")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("--namespace", "-n", default=None, help="Namespace")
    p.add_argument("--filter", "-f", default=None, help="JSON filter expression")
    p.add_argument("--limit", "-l", type=int, default=10, help="Max records to return")
    p.add_argument("--offset", type=int, default=0, help="Offset")
    p.set_defaults(func=cmd_list)

    # dump
    p = subparsers.add_parser("dump", help="Dump all records as JSONL")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("--namespace", "-n", default=None, help="Namespace")
    p.set_defaults(func=cmd_dump)

    # import-jsonl
    p = subparsers.add_parser("import-jsonl", help="Import from JSONL file")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("file", help="JSONL file to import")
    p.add_argument("--dimension", "-d", type=int, default=None, help="Vector dimension")
    p.add_argument("--namespace", "-n", default=None, help="Target namespace")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    p.set_defaults(func=cmd_import_jsonl)

    # import-csv
    p = subparsers.add_parser("import-csv", help="Import from CSV file")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("file", help="CSV file to import")
    p.add_argument("--dimension", "-d", type=int, default=None, help="Vector dimension")
    p.add_argument("--namespace", "-n", default=None, help="Target namespace")
    p.add_argument("--id-col", default="id", help="Column name for record ID")
    p.add_argument("--vector-col", default="vector", help="Column name for vector (JSON array)")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    p.set_defaults(func=cmd_import_csv)

    # compact
    p = subparsers.add_parser("compact", help="Compact database (merge WAL)")
    p.add_argument("path", help="Path to .vdb file")
    p.set_defaults(func=cmd_compact)

    # verify
    p = subparsers.add_parser("verify", help="Verify database integrity")
    p.add_argument("path", help="Path to .vdb file")
    p.set_defaults(func=cmd_verify)

    # bench
    p = subparsers.add_parser("bench", help="Run search benchmark")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("--queries", "-q", type=int, default=100, help="Number of queries")
    p.add_argument("--k", type=int, default=10, help="Top-k results per query")
    p.set_defaults(func=cmd_bench)

    # search
    p = subparsers.add_parser("search", help="Run a search query")
    p.add_argument("path", help="Path to .vdb file")
    p.add_argument("--query", "-q", required=True, help="Query vector as JSON array")
    p.add_argument("--k", type=int, default=10, help="Top-k results")
    p.add_argument("--namespace", "-n", default=None, help="Namespace")
    p.add_argument("--filter", "-f", default=None, help="JSON filter expression")
    p.add_argument("--stats", action="store_true", help="Include search stats")
    p.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except vectlite.VectLiteError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
