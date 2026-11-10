#!/usr/bin/env python3

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import psycopg

ROOT = Path(__file__).resolve().parents[1]
DBS = {"pgvector": (5433, "benchmark-pgvector"), "vectorchord": (5434, "benchmark-vectorchord")}
DATASETS = {"sift1m": ("sift_1m", 1_000_000), "vibe_yahoo": ("vibe_yahoo_384", 677_305)}
TABLE = "bench_index_vec4"
INDEX = "bench_index_vec4_ann"
FIELDS = ["dataset", "target_rows", "db", "repeat", "copy_s", "build_s", "index_mib", "table_mib", "rows", "peak_container_mib", "memory_samples", "sample_interval_s", "status", "error"]


def connect(db):
    return psycopg.connect(host="127.0.0.1", port=DBS[db][0], dbname="vectors", user="benchmark", password=os.environ.get("BENCH_DB_PASSWORD", "benchmark_local"), autocommit=True, connect_timeout=15)


def memory_mib(container):
    p = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container], capture_output=True, text=True, timeout=15)
    if p.returncode != 0:
        return None
    raw = p.stdout.strip().split("/")[0].strip().split()[0]
    units = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "GB": 1000**3 / 1024**2, "MB": 1000**2 / 1024**2, "kB": 1000 / 1024**2, "B": 1 / 1024**2}
    for suffix in sorted(units, key=len, reverse=True):
        if raw.endswith(suffix):
            return float(raw[:-len(suffix)]) * units[suffix]
    return None


def monitor(container, stop, samples, interval):
    while not stop.is_set():
        try:
            value = memory_mib(container)
            if value is not None:
                samples.append(value)
        except Exception:
            pass
        stop.wait(interval)


def drop_temp(conn):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS public.{TABLE}")


def one_run(conn, db, source, count, interval, smoke):
    drop_temp(conn)
    # Server-side copy, intentionally untimed relative to index build.
    # ORDER BY id gives a reproducible subset; full copies include every source row.
    start = time.perf_counter()
    with conn.cursor() as cur:
        if smoke:
            cur.execute(f"CREATE TABLE public.{TABLE} AS SELECT id, embedding FROM public.{source} ORDER BY id LIMIT %s", (count,))
        else:
            cur.execute(f"CREATE TABLE public.{TABLE} AS SELECT id, embedding FROM public.{source}")
        cur.execute(f"SELECT count(*) FROM public.{TABLE}")
        actual = cur.fetchone()[0]
    copy_s = time.perf_counter() - start
    if actual != count:
        raise RuntimeError(f"Copy row count mismatch: expected {count}, got {actual}")

    container = DBS[db][1]
    samples = []
    stop = threading.Event()
    thread = threading.Thread(target=monitor, args=(container, stop, samples, interval), daemon=True)
    thread.start()
    try:
        with conn.cursor() as cur:
            if db == "pgvector":
                sql = f"CREATE INDEX {INDEX} ON public.{TABLE} USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)"
            else:
                sql = f"CREATE INDEX {INDEX} ON public.{TABLE} USING vchordrq (embedding vector_l2_ops)"
            started = time.perf_counter()
            cur.execute(sql)
            build_s = time.perf_counter() - started
    finally:
        stop.set()
        thread.join(timeout=20)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_relation_size(%s::regclass), pg_total_relation_size(%s::regclass)", (f"public.{INDEX}", f"public.{TABLE}"))
        index_bytes, table_bytes = cur.fetchone()
    return dict(copy_s=round(copy_s, 4), build_s=round(build_s, 4), index_mib=round(index_bytes / 1048576, 3), table_mib=round(table_bytes / 1048576, 3), rows=actual, peak_container_mib=round(max(samples), 3) if samples else "", memory_samples=len(samples))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--db", choices=[*DBS, "all"], default="all")
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--max-new-runs", type=int, default=None, help="Stop after this many new measurements")
    p.add_argument("--smoke", action="store_true", help="Build on 10,000 rows instead of full dataset")
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--output", default=str(ROOT / "results/real/index_build/index_build_8runs.csv"))
    a = p.parse_args()
    if a.repeats < 1 or a.sample_interval <= 0:
        p.error("repeats and sample interval must be positive")
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    completed.add((row["dataset"], row["db"], int(row["repeat"])))
    datasets = DATASETS if a.dataset == "all" else {a.dataset: DATASETS[a.dataset]}
    dbs = DBS if a.db == "all" else {a.db: DBS[a.db]}
    new_runs = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            w.writeheader()
        for dataset, (source, total) in datasets.items():
            target = 10000 if a.smoke else total
            for db in dbs:
                with connect(db) as conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT to_regclass(%s)", (f"public.{source}",))
                            if cur.fetchone()[0] is None:
                                raise RuntimeError(f"Missing source table public.{source} in {db}")
                            cur.execute("SELECT to_regclass(%s)", (f"public.{TABLE}",))
                            if cur.fetchone()[0] is not None:
                                raise RuntimeError(f"Safety stop: public.{TABLE} already exists in {db}. Inspect it manually before running.")
                        for rep in range(1, a.repeats + 1):
                            if (dataset, db, rep) in completed:
                                print(f"SKIP {dataset} {db} repeat={rep}", flush=True)
                                continue
                            print(f"START dataset={dataset} db={db} rows={target} repeat={rep}/{a.repeats}", flush=True)
                            row = dict(dataset=dataset, target_rows=target, db=db, repeat=rep, copy_s="", build_s="", index_mib="", table_mib="", rows="", peak_container_mib="", memory_samples=0, sample_interval_s=a.sample_interval, status="error", error="")
                            try:
                                row.update(one_run(conn, db, source, target, a.sample_interval, a.smoke))
                                row["status"] = "ok"
                            except Exception as exc:
                                row["error"] = repr(exc)
                            finally:
                                try:
                                    drop_temp(conn)
                                except Exception as exc:
                                    print(f"WARNING: temporary table cleanup failed: {exc}", flush=True)
                            w.writerow(row)
                            f.flush()
                            os.fsync(f.fileno())
                            print("RESULT", json.dumps(row), flush=True)
                            if row["status"] != "ok":
                                raise RuntimeError(f"Stopped after failed run; inspect {path}")
                            new_runs += 1
                            if a.max_new_runs is not None and new_runs >= a.max_new_runs:
                                print("STOP: requested number of new runs completed", flush=True)
                                return
                    finally:
                        # No broad cleanup: never touch original tables or indexes.
                        pass
    print(f"COMPLETE {path}", flush=True)


if __name__ == "__main__":
    main()
