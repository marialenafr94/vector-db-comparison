
#!/usr/bin/env python3
"""
Filtered QPS + concurrent latency benchmark.

Datasets:
    sift1m
    vibe
    
Outputs:
    results/real/sift1m_filtered_qps_8runs.csv
    results/real/vibe_yahoo_filtered_qps_8runs.csv
"""

import argparse
import csv
import random
import statistics
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import psycopg2


REPEATS = 8
WARMUP_SECONDS = 10
MEASURE_SECONDS = 30
CLIENT_COUNTS = [1, 8]

CONFIGS = [
    ("pgvector", "strict_order", 60),
    ("pgvector", "relaxed_order", 60),
    ("vectorchord", "default", 0.5),
    ("vectorchord", "default", 0.7),
]

DATASETS = {
    "sift1m": {
        "hdf5": (
            "datasets/real/sift1m/"
            "sift-128-euclidean.hdf5"
        ),
        "table": "sift_1m",
        "output": (
            "results/real/"
            "sift1m_filtered_qps_8runs.csv"
        ),
    },
    "vibe": {
        "hdf5": (
            "datasets/real/vibe/"
            "yahoo-minilm-384-normalized.hdf5"
        ),
        "table": "vibe_yahoo_384",
        "output": (
            "results/real/"
            "vibe_yahoo_filtered_qps_8runs.csv"
        ),
    },
}

FIELDS = [
    "dataset",
    "repeat",
    "seed",
    "database",
    "filter",
    "selectivity_percent",
    "scan_mode",
    "search_value",
    "clients",
    "warmup_seconds",
    "measure_seconds",
    "completed_queries",
    "errors",
    "elapsed_seconds",
    "qps",
    "mean_returned",
    "latency_samples",
    "mean_latency_ms",
    "sd_latency_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "min_latency_ms",
    "max_latency_ms",
]


def connect(database):
    port = 5433 if database == "pgvector" else 5434

    conn = psycopg2.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        connect_timeout=15,
    )
    conn.autocommit = True
    return conn


def configure(conn, database, scan_mode, search_value):
    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")

        if database == "pgvector":
            cur.execute(
                "SELECT set_config("
                "'hnsw.ef_search', %s, false)",
                (str(int(search_value)),),
            )

            cur.execute(
                "SELECT set_config("
                "'hnsw.iterative_scan', %s, false)",
                (scan_mode,),
            )

            cur.execute(
                "SET hnsw.max_scan_tuples = 20000"
            )

            cur.execute(
                "SET hnsw.scan_mem_multiplier = 1"
            )
        else:
            cur.execute(
                "SELECT set_config("
                "'vchordrq.epsilon', %s, false)",
                (str(float(search_value)),),
            )


def load_queries(path):
    with h5py.File(path, "r") as f:
        data = np.asarray(
            f["test"][:100],
            dtype=np.float32,
        )

    return [
        "[" + ",".join(
            format(float(value), ".9g")
            for value in vector
        ) + "]"
        for vector in data
    ]


def run_phase(
    database,
    table,
    scan_mode,
    search_value,
    query_vectors,
    clients,
    duration_seconds,
    seed,
):
    """
    Each client sends the next query immediately
    after receiving the previous query's result.

    All clients start together.

    Requests started before the deadline are allowed
    to finish. Actual elapsed time includes this
    final in-flight completion period.
    """

    sql = (
        f"SELECT id FROM {table} "
        "WHERE category_1 = 0 "
        "ORDER BY embedding <-> %s::vector "
        "LIMIT 10"
    )

    start_event = threading.Event()
    ready = threading.Barrier(clients + 1)

    timing = {
        "deadline": None,
    }

    def worker(worker_id):
        rng = random.Random(
            seed + worker_id * 100003
        )

        conn = None
        latencies_ms = []
        completed = 0
        errors = 0
        returned = 0

        try:
            conn = connect(database)

            configure(
                conn,
                database,
                scan_mode,
                search_value,
            )

            with conn.cursor() as cur:
                ready.wait(timeout=60)
                start_event.wait()

                while (
                    time.monotonic()
                    < timing["deadline"]
                ):
                    query = query_vectors[
                        rng.randrange(
                            len(query_vectors)
                        )
                    ]

                    query_start = (
                        time.perf_counter_ns()
                    )

                    try:
                        cur.execute(sql, (query,))
                        rows = cur.fetchall()

                    except Exception:
                        errors += 1
                        raise

                    query_end = (
                        time.perf_counter_ns()
                    )

                    latency_ms = (
                        query_end - query_start
                    ) / 1_000_000.0

                    latencies_ms.append(
                        latency_ms
                    )

                    completed += 1
                    returned += len(rows)

        finally:
            if conn is not None:
                conn.close()

        return {
            "completed": completed,
            "errors": errors,
            "returned": returned,
            "latencies_ms": latencies_ms,
        }

    with ThreadPoolExecutor(
        max_workers=clients
    ) as pool:

        futures = [
            pool.submit(worker, worker_id)
            for worker_id in range(clients)
        ]

        try:
            ready.wait(timeout=60)

        except threading.BrokenBarrierError:
            for future in futures:
                if future.done():
                    future.result()

            raise RuntimeError(
                "A worker failed during setup. "
                "Check the database connection "
                "and search settings."
            )

        start_time = time.monotonic()

        timing["deadline"] = (
            start_time + duration_seconds
        )

        start_event.set()

        results = [
            future.result()
            for future in futures
        ]

        elapsed_seconds = (
            time.monotonic() - start_time
        )

    completed = sum(
        r["completed"] for r in results
    )

    errors = sum(
        r["errors"] for r in results
    )

    returned = sum(
        r["returned"] for r in results
    )

    all_latencies = [
        latency
        for result in results
        for latency in result["latencies_ms"]
    ]

    if completed == 0:
        raise RuntimeError(
            "No queries completed during "
            "the measurement phase."
        )

    if len(all_latencies) != completed:
        raise RuntimeError(
            "Latency sample count mismatch."
        )

    latency_array = np.asarray(
        all_latencies,
        dtype=np.float64,
    )

    return {
        "completed_queries": completed,
        "errors": errors,
        "elapsed_seconds": elapsed_seconds,
        "qps": (
            completed / elapsed_seconds
        ),
        "mean_returned": (
            returned / completed
        ),
        "latency_samples": len(
            all_latencies
        ),
        "mean_latency_ms": float(
            np.mean(latency_array)
        ),
        "sd_latency_ms": (
            statistics.stdev(all_latencies)
            if len(all_latencies) > 1
            else 0.0
        ),
        "p50_ms": float(
            np.percentile(latency_array, 50)
        ),
        "p95_ms": float(
            np.percentile(latency_array, 95)
        ),
        "p99_ms": float(
            np.percentile(latency_array, 99)
        ),
        "min_latency_ms": float(
            np.min(latency_array)
        ),
        "max_latency_ms": float(
            np.max(latency_array)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
    )

    args = parser.parse_args()

    dataset = args.dataset
    info = DATASETS[dataset]

    queries = load_queries(
        info["hdf5"]
    )

    output = Path(info["output"])
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = set()

    if output.exists() and output.stat().st_size:
        with output.open(newline="") as f:
            reader = csv.DictReader(f)

            # Prevent mixing the old QPS-only CSV
            # with the updated latency-enabled CSV.
            if reader.fieldnames != FIELDS:
                raise RuntimeError(
                    f"Existing CSV has a different "
                    f"schema: {output}\n"
                    "Rename the old file before "
                    "running this updated script."
                )

            for row in reader:
                key = (
                    int(row["repeat"]),
                    row["database"],
                    row["scan_mode"],
                    row["search_value"],
                    int(row["clients"]),
                )

                if key in existing:
                    raise RuntimeError(
                        f"Duplicate CSV key: {key}"
                    )

                existing.add(key)

    jobs = []

    for repeat in range(
        1, REPEATS + 1
    ):
        for (
            database,
            scan_mode,
            search_value,
        ) in CONFIGS:

            for clients in CLIENT_COUNTS:
                key = (
                    repeat,
                    database,
                    scan_mode,
                    str(search_value),
                    clients,
                )

                if key not in existing:
                    jobs.append(
                        (
                            repeat,
                            database,
                            scan_mode,
                            search_value,
                            clients,
                        )
                    )

    random.Random(
        20261104
    ).shuffle(jobs)

    print(
        f"\nDataset: {dataset}",
        flush=True,
    )

    print(
        f"Queries loaded: {len(queries)}",
        flush=True,
    )

    print(
        f"Completed rows: {len(existing)}",
        flush=True,
    )

    print(
        f"Remaining rows: {len(jobs)}",
        flush=True,
    )

    with output.open(
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        if output.stat().st_size == 0:
            writer.writeheader()
            f.flush()

        for number, job in enumerate(
            jobs, 1
        ):
            (
                repeat,
                database,
                scan_mode,
                search_value,
                clients,
            ) = job

            seed = 42 + repeat

            print(
                f"\n[{number}/{len(jobs)}] "
                f"repeat={repeat} "
                f"database={database} "
                f"mode={scan_mode} "
                f"value={search_value} "
                f"clients={clients}",
                flush=True,
            )

            # Warmup: discarded entirely.
            run_phase(
                database=database,
                table=info["table"],
                scan_mode=scan_mode,
                search_value=search_value,
                query_vectors=queries,
                clients=clients,
                duration_seconds=WARMUP_SECONDS,
                seed=seed + 1000000,
            )

            # Measurement: all request latencies
            # are retained for percentile calculation.
            result = run_phase(
                database=database,
                table=info["table"],
                scan_mode=scan_mode,
                search_value=search_value,
                query_vectors=queries,
                clients=clients,
                duration_seconds=MEASURE_SECONDS,
                seed=seed,
            )

            row = {
                "dataset": dataset,
                "repeat": repeat,
                "seed": seed,
                "database": database,
                "filter": "category_1",
                "selectivity_percent": 1,
                "scan_mode": scan_mode,
                "search_value": search_value,
                "clients": clients,
                "warmup_seconds": (
                    WARMUP_SECONDS
                ),
                "measure_seconds": (
                    MEASURE_SECONDS
                ),
                "completed_queries": (
                    result["completed_queries"]
                ),
                "errors": result["errors"],
                "elapsed_seconds": round(
                    result["elapsed_seconds"], 6
                ),
                "qps": round(
                    result["qps"], 4
                ),
                "mean_returned": round(
                    result["mean_returned"], 4
                ),
                "latency_samples": (
                    result["latency_samples"]
                ),
                "mean_latency_ms": round(
                    result["mean_latency_ms"], 4
                ),
                "sd_latency_ms": round(
                    result["sd_latency_ms"], 4
                ),
                "p50_ms": round(
                    result["p50_ms"], 4
                ),
                "p95_ms": round(
                    result["p95_ms"], 4
                ),
                "p99_ms": round(
                    result["p99_ms"], 4
                ),
                "min_latency_ms": round(
                    result["min_latency_ms"], 4
                ),
                "max_latency_ms": round(
                    result["max_latency_ms"], 4
                ),
            }

            writer.writerow(row)
            f.flush()

            print(
                f"QPS={row['qps']:.2f} | "
                f"p50={row['p50_ms']:.2f}ms | "
                f"p95={row['p95_ms']:.2f}ms | "
                f"p99={row['p99_ms']:.2f}ms | "
                f"completed={row['completed_queries']} | "
                f"returned={row['mean_returned']:.2f} | "
                f"errors={row['errors']}",
                flush=True,
            )

    print(
        f"\nFINISHED: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
