
import time
import statistics
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/sift1m/sift-128-euclidean.hdf5"
)

NUM_QUERIES = 100


def vector_literal(vector):
    return "[" + ",".join(
        str(float(x)) for x in vector
    ) + "]"


def run_pass(cur, queries, label):
    times = []

    for query in queries:
        start = time.perf_counter()

        cur.execute(
            """
            SELECT id
            FROM sift_1m
            ORDER BY embedding <-> %s::vector
            LIMIT 10
            """,
            (vector_literal(query),),
        )
        cur.fetchall()

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed)

    print(f"\n{label}")
    print(
        f"p50: {np.percentile(times, 50):.2f} ms"
    )
    print(
        f"p95: {np.percentile(times, 95):.2f} ms"
    )
    print(
        f"p99: {np.percentile(times, 99):.2f} ms"
    )
    print(
        f"mean: {statistics.mean(times):.2f} ms"
    )


def main():
    with h5py.File(DATASET, "r") as file:
        queries = file["test"][:NUM_QUERIES]

    with psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")

            run_pass(
                cur,
                queries,
                "Pass 1 (initial)",
            )

            run_pass(
                cur,
                queries,
                "Pass 2 (warmed)",
            )


if __name__ == "__main__":
    main()
