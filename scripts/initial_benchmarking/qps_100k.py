
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import psycopg

# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

DIM = 384
K = 10
QUERY_POOL = 200
WARMUP_PER_CLIENT = 10
MEASURE_SECONDS = 15
CLIENT_COUNTS = [1, 8]
REPEATS = 3

CONFIGURATIONS = [
    ("pgvector", 5433, 300),
    ("pgvector", 5433, 1000),
    ("vectorchord", 5434, None),
]

SQL = """
    SELECT id
    FROM items_100k
    ORDER BY embedding <-> %s::vector
    LIMIT 10
"""

OUTPUT = Path("results/qps_100k.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)

queries = rng.normal(
    size=(QUERY_POOL, DIM)
).astype(np.float32)

queries /= np.linalg.norm(
    queries,
    axis=1,
    keepdims=True,
)

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


def worker(
    port,
    ef_search,
    client_id,
    ready_barrier,
    finish_time,
):
    latencies = []
    completed = 0

    with connect(port) as conn:
        with conn.cursor() as cur:

            # Keep index-focused measurements consistent.
            cur.execute("SET enable_seqscan = off")

            if ef_search is not None:
                cur.execute(
                    "SELECT set_config(%s, %s, false)",
                    ("hnsw.ef_search", str(ef_search)),
                )

                cur.execute(
                    "SELECT current_setting(%s)",
                    ("hnsw.ef_search",),
                )

                actual = int(cur.fetchone()[0])

                if actual != ef_search:
                    raise RuntimeError(
                        f"Expected ef_search={ef_search}, "
                        f"got {actual}"
                    )

            # Warm up each connection.
            for i in range(WARMUP_PER_CLIENT):
                q = query_literals[
                    (client_id * 17 + i) % QUERY_POOL
                ]

                cur.execute(SQL, (q,))
                cur.fetchall()

            # All workers start together.
            ready_barrier.wait()

            i = 0

            while time.perf_counter() < finish_time:
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

        # Shared deadline is set after all warmups.
        # Use a shared mutable object for the deadline.
        timing = {"deadline": None}

        def timed_worker(client_id):
            with connect(port) as conn:
                with conn.cursor() as cur:

                    cur.execute("SET enable_seqscan = off")

                    if ef_search is not None:
                        cur.execute(
                            "SELECT set_config(%s, %s, false)",
                            ("hnsw.ef_search", str(ef_search)),
                        )

                        cur.execute(
                            "SELECT current_setting(%s)",
                            ("hnsw.ef_search",),
                        )

                        actual = int(cur.fetchone()[0])

                        if actual != ef_search:
                            raise RuntimeError(
                                f"Expected ef_search={ef_search}, "
                                f"got {actual}"
                            )

                    for i in range(WARMUP_PER_CLIENT):
                        q = query_literals[
                            (client_id * 17 + i) % QUERY_POOL
                        ]

                        cur.execute(SQL, (q,))
                        cur.fetchall()

                    barrier.wait()

                    # Main thread publishes deadline after warmup.
                    barrier.wait()

                    latencies = []
                    completed = 0
                    i = 0

                    while time.perf_counter() < timing["deadline"]:
                        q = query_literals[
                            (client_id * 17 + i) % QUERY_POOL
                        ]

                        start = time.perf_counter()

                        cur.execute(SQL, (q,))
                        cur.fetchall()

                        latencies.append(
                            (time.perf_counter() - start) * 1000
                        )

                        completed += 1
                        i += 1

                    return completed, latencies

        futures = [
            pool.submit(timed_worker, client_id)
            for client_id in range(clients)
        ]

        # Wait until every client has warmed up.
        barrier.wait()

        start_time = time.perf_counter()
        timing["deadline"] = start_time + MEASURE_SECONDS

        # Release all workers.
        barrier.wait()

        outputs = [future.result() for future in futures]

    completed = sum(count for count, _ in outputs)

    latencies = [
        latency
        for _, client_latencies in outputs
        for latency in client_latencies
    ]

    # Fixed measurement window. A query already in flight
    # at the deadline is allowed to finish.
    wall_seconds = time.perf_counter() - start_time

    result = {
        "database": name,
        "ef_search": (
            ef_search if ef_search is not None else "default"
        ),
        "clients": clients,
        "repeat": repeat,
        "completed_queries": completed,
        "wall_seconds": round(wall_seconds, 3),
        "qps": round(completed / wall_seconds, 2),
        "p50_ms": round(
            float(np.percentile(latencies, 50)), 3
        ),
        "p95_ms": round(
            float(np.percentile(latencies, 95)), 3
        ),
        "p99_ms": round(
            float(np.percentile(latencies, 99)), 3
        ),
    }

    print(result, flush=True)
    return result


def main():
    results = []

    for clients in CLIENT_COUNTS:
        for repeat in range(1, REPEATS + 1):

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
                    name,
                    port,
                    ef_search,
                    clients,
                    repeat,
                )

                results.append(result)

                with OUTPUT.open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=results[0].keys(),
                    )
                    writer.writeheader()
                    writer.writerows(results)

    print(
        f"\nFinished. Results saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
