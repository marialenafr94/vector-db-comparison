#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import h5py
import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "sift1m": (ROOT / "datasets/real/sift1m/sift-128-euclidean.hdf5", 128),
    "vibe_yahoo": (ROOT / "datasets/real/vibe/yahoo-minilm-384-normalized.hdf5", 384),
}
DBS = {"pgvector": (5433, "benchmark-pgvector"), "vectorchord": (5434, "benchmark-vectorchord")}
TABLE = "bench_loading_vec4"
COLUMNS = ["dataset", "size_label", "target_rows", "db", "repeat", "elapsed_s", "table_mib", "rows", "peak_container_mib", "memory_samples", "sample_interval_s", "status", "error"]


def mem_bytes(container):
    """Docker stats memory usage, as reported by Docker, in bytes."""
    p = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container], capture_output=True, text=True, timeout=15)
    if p.returncode != 0:
        return None
    value = p.stdout.strip().split("/")[0].strip().split()
    if not value:
        return None
    raw = value[0]
    units = {"B": 1, "kB": 1000, "MB": 1000**2, "GB": 1000**3, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    for unit in ("GiB", "MiB", "KiB", "GB", "MB", "kB", "B"):
        if raw.endswith(unit):
            return float(raw[:-len(unit)]) * units[unit]
    return None


def monitor(container, stop, readings, interval):
    while not stop.is_set():
        try:
            v = mem_bytes(container)
            if v is not None:
                readings.append(v)
        except Exception:
            pass
        stop.wait(interval)


def conn_for(db):
    port, _ = DBS[db]
    return psycopg.connect(host="127.0.0.1", port=port, dbname="vectors", user="benchmark", password=os.environ.get("BENCH_DB_PASSWORD", "benchmark_local"), autocommit=True, connect_timeout=10)


def vec_literal(v):
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def assert_table_absent(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{TABLE}",))
        if cur.fetchone()[0] is not None:
            raise RuntimeError(
                f"Safety stop: public.{TABLE} already exists. "
                "Inspect it manually; this benchmark will not overwrite it."
            )


def safe_drop(conn):
    """Only call after this process successfully created the benchmark table."""
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE public.{TABLE}")


def run_case(conn, db, path, dim, count, batch, interval):
    _, container = DBS[db]
    assert_table_absent(conn)
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {TABLE} (id integer PRIMARY KEY, embedding vector({dim}))")
    try:
        stop = threading.Event()
        readings = []
        worker = threading.Thread(target=monitor, args=(container, stop, readings, interval), daemon=True)
        worker.start()
        start = time.perf_counter()
        try:
            with h5py.File(path, "r") as f:
                train = f["train"]
                with conn.cursor() as cur:
                    for offset in range(0, count, batch):
                        data = np.asarray(train[offset:min(offset + batch, count)], dtype=np.float32)
                        # PostgreSQL COPY text mode; same insertion method in both systems.
                        with cur.copy(f"COPY {TABLE} (id, embedding) FROM STDIN") as copy:
                            for j, v in enumerate(data):
                                copy.write_row((offset + j, vec_literal(v)))
            elapsed = time.perf_counter() - start
        finally:
            stop.set()
            worker.join(timeout=20)
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*), pg_total_relation_size('{TABLE}') FROM {TABLE}")
            rows, bytes_used = cur.fetchone()
        return elapsed, rows, bytes_used / (1024 * 1024), (max(readings) / (1024 * 1024) if readings else None), len(readings)

    finally:
        safe_drop(conn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="sift1m")
    parser.add_argument("--db", choices=[*DBS, "all"], default="all")
    parser.add_argument("--size", choices=["small", "medium", "full", "all"], default="small")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output", default=str(ROOT / "results/real/loading/loading_8runs.csv"))
    args = parser.parse_args()
    if args.repeats < 1 or args.batch < 1 or args.sample_interval <= 0:
        parser.error("repeats, batch and sample interval must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "ok":
                    completed.add((row["dataset"], row["size_label"], row["db"], int(row["repeat"])))
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    dbs = list(DBS) if args.db == "all" else [args.db]
    sizes = ["small", "medium", "full"] if args.size == "all" else [args.size]
    with output.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fh.tell() == 0:
            writer.writeheader()
        for dataset in datasets:
            path, dim = DATASETS[dataset]
            if not path.exists():
                raise FileNotFoundError(path)
            with h5py.File(path, "r") as f:
                total = len(f["train"])
                assert f["train"].shape[1] == dim
            for size_label in sizes:
                count = min({"small": 10000, "medium": 100000, "full": total}[size_label], total)
                for db in dbs:
                    with conn_for(db) as conn:
                        for rep in range(1, args.repeats + 1):
                            key = (dataset, size_label, db, rep)
                            if key in completed:
                                print(f"SKIP completed {key}", flush=True)
                                continue
                            print(f"START dataset={dataset} size={size_label} rows={count} db={db} repeat={rep}/{args.repeats}", flush=True)
                            row = dict(dataset=dataset, size_label=size_label, target_rows=count, db=db, repeat=rep, elapsed_s="", table_mib="", rows="", peak_container_mib="", memory_samples=0, sample_interval_s=args.sample_interval, status="error", error="")
                            try:
                                assert_table_absent(conn)
                                elapsed, actual, mib, peak, samples = run_case(conn, db, path, dim, count, args.batch, args.sample_interval)
                                if actual != count:
                                    raise RuntimeError(f"Expected {count} rows, got {actual}")
                                row.update(elapsed_s=round(elapsed, 4), table_mib=round(mib, 3), rows=actual, peak_container_mib=round(peak, 3) if peak is not None else "", memory_samples=samples, status="ok")
                            except Exception as exc:
                                row["error"] = repr(exc)
                            writer.writerow(row)
                            fh.flush()
                            os.fsync(fh.fileno())
                            print("RESULT", json.dumps(row), flush=True)
                            if row["status"] != "ok":
                                raise RuntimeError(f"Benchmark failed; see {output}: {row['error']}")
    print(f"Finished. CSV: {output}", flush=True)


if __name__ == "__main__":
    main()
