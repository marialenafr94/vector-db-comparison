
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import psycopg

# Benchmark settings
DIM = 384
K = 10
QUERY_POOL = 200
WARMUP_PER_CLIENT = 10
MEASURE_SECONDS = 15
CLIENT_COUNTS = [1, 8]
REPEATS = 3

CONFIGURATIONS = [
    ("pgvector", 5433, 300),
    ("pgvector", 5433, 600),
    ("vectorchord", 5434, None),
]

SQL = """
    SELECT id
    FROM items
    ORDER BY embedding <-> %s::vector
    LIMIT 10
"""

OUTPUT = Path("results/qps_results.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
queries = rng.normal(size=(QUERY_POOL, DIM)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

query_literals = [
    "[" + ",".join(map(str, q)) + "]"
    for q in queries
]


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    )


def worker(port, ef_search, client_id, barrier, duration):
    latencies = []
    completed = 0

    with connect(port) as conn:
        with conn.cursor() as cur:
            if ef_search is not None:
                cur.execute(
                    "SELECT set_config(%s, %s, false)",
                    ("hnsw.ef_search", str(ef_search)),
                )

            # Warm up each connection before measuring.
            for i in range(WARMUP_PER_CLIENT):
                q = query_literals[
                    (client_id * 17 + i) % QUERY_POOL
                ]
                cur.execute(SQL, (q,))
                cur.fetchall()

            # Synchronize clients after warm-up.
            barrier.wait()

            start = time.perf_counter()
            deadline = start + duration

            i = 0
            while time.perf_counter() < deadline:
                q = query_literals[
                    (client_id * 17 + i) % QUERY_POOL
                ]

                query_start = time.perf_counter()
                cur.execute(SQL, (q,))
                cur.fetchall()

                latencies.append(
                    (time.perf_counter() - query_start) * 1000
                )
                completed += 1
                i += 1

    return completed, latencies


def run_trial(name, port, ef_search, clients, repeat):
    barrier = Barrier(clients + 1)

    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [
            pool.submit(
                worker,
                port,
                ef_search,
                client_id,
                barrier,
                MEASURE_SECONDS,
            )
            for client_id in range(clients)
        ]

        # Wait until all clients finish connecting and warming up.
        barrier.wait()
        wall_start = time.perf_counter()

        outputs = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - wall_start

    completed = sum(count for count, _ in outputs)
    latencies = [
        latency
        for _, client_latencies in outputs
        for latency in client_latencies
    ]

    result = {
        "database": name,
        "ef_search": ef_search if ef_search is not None else "default",
        "clients": clients,
        "repeat": repeat,
        "completed_queries": completed,
        "wall_seconds": round(wall_seconds, 3),
        "qps": round(completed / wall_seconds, 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
    }

    print(result, flush=True)
    return result


def main():
    results = []

    for clients in CLIENT_COUNTS:
        for repeat in range(1, REPEATS + 1):
            # Alternate order between repetitions.
            configurations = (
                CONFIGURATIONS
                if repeat % 2
                else list(reversed(CONFIGURATIONS))
            )

            for name, port, ef_search in configurations:
                print(
                    f"\nRunning {name}, "
                    f"ef={ef_search}, "
                    f"clients={clients}, "
                    f"repeat={repeat}",
                    flush=True,
                )

                result = run_trial(
                    name, port, ef_search, clients, repeat
                )
                results.append(result)

                # Save incrementally, so completed trials survive
                # if the experiment is interrupted.
                with OUTPUT.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=results[0].keys()
                    )
                    writer.writeheader()
                    writer.writerows(results)

    print(f"\nFinished. Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
