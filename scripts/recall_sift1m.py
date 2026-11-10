
import csv
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/sift1m/sift-128-euclidean.hdf5"
)
OUTPUT = Path("results/real/sift1m_recall.csv")

# Start with 100 queries as a pilot.
# Later, increase this to 1,000 or 10,000.
NUM_QUERIES = 100
K = 10

PGVECTOR_EF_VALUES = [40, 100, 200, 400, 700, 1000]


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


def percentile(values, p):
    return float(np.percentile(values, p))


def run_configuration(
    name,
    port,
    queries,
    ground_truth,
    ef_search=None,
):
    print(
        f"\nTesting {name}, ef_search={ef_search}",
        flush=True,
    )

    recalls = []
    latencies_ms = []

    with connect(port) as conn:
        with conn.cursor() as cur:
            # Force the approximate index to be used
            # during this index-focused experiment.
            cur.execute("SET enable_seqscan = off")

            if ef_search is not None:
                cur.execute(
                    f"SET hnsw.ef_search = {ef_search}"
                )

            for i, query in enumerate(queries):
                vector = vector_literal(query)

                start = time.perf_counter()

                cur.execute(
                    """
                    SELECT id
                    FROM sift_1m
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (vector, K),
                )

                result_ids = [
                    row[0] for row in cur.fetchall()
                ]

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                expected_ids = set(
                    int(x)
                    for x in ground_truth[i][:K]
                )

                correct = len(
                    expected_ids.intersection(result_ids)
                )

                recalls.append(correct / K)
                latencies_ms.append(elapsed_ms)

                if (i + 1) % 25 == 0:
                    print(
                        f"  Completed {i + 1}/"
                        f"{len(queries)} queries",
                        flush=True,
                    )

    return {
        "database": name,
        "ef_search": (
            ef_search if ef_search is not None
            else "default"
        ),
        "queries": len(queries),
        "recall_at_10": round(
            statistics.mean(recalls), 4
        ),
        "p50_ms": round(
            percentile(latencies_ms, 50), 3
        ),
        "p95_ms": round(
            percentile(latencies_ms, 95), 3
        ),
        "p99_ms": round(
            percentile(latencies_ms, 99), 3
        ),
    }


def main():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(DATASET, "r") as file:
        queries = file["test"][:NUM_QUERIES]
        ground_truth = file["neighbors"][
            :NUM_QUERIES
        ]

    print(
        f"Loaded {len(queries)} SIFT1M queries",
        flush=True,
    )

    results = []

    for ef in PGVECTOR_EF_VALUES:
        result = run_configuration(
            "pgvector",
            5433,
            queries,
            ground_truth,
            ef_search=ef,
        )

        results.append(result)
        print(result, flush=True)

    result = run_configuration(
        "vectorchord",
        5434,
        queries,
        ground_truth,
    )

    results.append(result)
    print(result, flush=True)

    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
