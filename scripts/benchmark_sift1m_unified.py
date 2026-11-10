
import csv
import random
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/sift1m/sift-128-euclidean.hdf5"
)
OUTPUT = Path(
    "results/real/sift1m_unified_recall_latency.csv"
)

K = 10
NUM_QUERIES = 1000
WARMUP_QUERIES = 50

# Use a fixed seed for reproducible ordering.
SEED = 42

CONFIGURATIONS = [
    ("pgvector", 5433, "hnsw.ef_search", 20),
    ("pgvector", 5433, "hnsw.ef_search", 40),
    ("pgvector", 5433, "hnsw.ef_search", 60),
    ("pgvector", 5433, "hnsw.ef_search", 100),
    ("pgvector", 5433, "hnsw.ef_search", 200),

    ("vectorchord", 5434, "vchordrq.epsilon", 0.3),
    ("vectorchord", 5434, "vchordrq.epsilon", 0.5),
    ("vectorchord", 5434, "vchordrq.epsilon", 0.7),
    ("vectorchord", 5434, "vchordrq.epsilon", 1.0),
    ("vectorchord", 5434, "vchordrq.epsilon", 1.9),
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


def vector_literal(vector):
    return "[" + ",".join(
        str(float(value)) for value in vector
    ) + "]"


def run_query(cur, vector):
    cur.execute(
        """
        SELECT id
        FROM sift_1m
        ORDER BY embedding <-> %s::vector
        LIMIT 10
        """,
        (vector,),
    )
    return [row[0] for row in cur.fetchall()]


def benchmark(
    database,
    port,
    parameter,
    value,
    vectors,
    expected,
    query_order,
    warmup_vectors,
):
    print(
        f"\nTesting {database}: {parameter}={value}",
        flush=True,
    )

    latencies = []
    recalls = []
    incomplete = 0

    with connect(port) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")

            # set_config works with both integer
            # and floating-point setting values.
            cur.execute(
                "SELECT set_config(%s, %s, false)",
                (parameter, str(value)),
            )

            # Warm-up queries are not timed.
            print("  Warming up...", flush=True)
            for vector in warmup_vectors:
                run_query(cur, vector)

            print("  Measuring...", flush=True)

            for position, index in enumerate(query_order):
                start = time.perf_counter()

                found = run_query(
                    cur,
                    vectors[index],
                )

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                correct = len(
                    set(found).intersection(
                        expected[index]
                    )
                )

                recalls.append(correct / K)
                latencies.append(elapsed_ms)

                if len(found) < K:
                    incomplete += 1

                if (position + 1) % 200 == 0:
                    print(
                        f"  {position + 1}/"
                        f"{len(query_order)} queries",
                        flush=True,
                    )

    result = {
        "database": database,
        "parameter": parameter,
        "value": value,
        "queries": len(query_order),
        "recall_at_10": round(
            statistics.mean(recalls), 4
        ),
        "p50_ms": round(
            float(np.percentile(latencies, 50)), 3
        ),
        "p95_ms": round(
            float(np.percentile(latencies, 95)), 3
        ),
        "p99_ms": round(
            float(np.percentile(latencies, 99)), 3
        ),
        "mean_ms": round(
            statistics.mean(latencies), 3
        ),
        "incomplete_queries": incomplete,
    }

    print("  Result:", result, flush=True)
    return result


def save_results(results):
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save after each completed configuration,
    # so partial progress is preserved.
    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    with h5py.File(DATASET, "r") as file:
        # Separate warm-up and measured queries.
        # SIFT1M has 10,000 test queries.
        raw_vectors = file["test"][:(
            NUM_QUERIES + WARMUP_QUERIES
        )]

        raw_neighbors = file["neighbors"][
            :NUM_QUERIES
        ]

    vectors = [
        vector_literal(vector)
        for vector in raw_vectors[:NUM_QUERIES]
    ]

    warmup_vectors = [
        vector_literal(vector)
        for vector in raw_vectors[NUM_QUERIES:]
    ]

    expected = [
        set(int(x) for x in row[:K])
        for row in raw_neighbors
    ]

    query_order = list(range(NUM_QUERIES))
    random.Random(SEED).shuffle(query_order)

    # Shuffle configuration order to reduce
    # systematic first-run bias.
    configurations = CONFIGURATIONS.copy()
    random.Random(SEED).shuffle(configurations)

    print(
        f"Measured queries: {NUM_QUERIES}",
        flush=True,
    )
    print(
        f"Warm-up queries/configuration: "
        f"{WARMUP_QUERIES}",
        flush=True,
    )

    results = []

    for configuration in configurations:
        result = benchmark(
            *configuration,
            vectors,
            expected,
            query_order,
            warmup_vectors,
        )

        results.append(result)
        save_results(results)

    print(
        f"\nFinished. Results saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
