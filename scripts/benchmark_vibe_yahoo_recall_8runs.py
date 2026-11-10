
import csv
import random
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/vibe/yahoo-minilm-384-normalized.hdf5"
)
OUTPUT = Path(
    "results/real/vibe_yahoo_recall_latency.csv"
)

TABLE = "vibe_yahoo_384"
WARMUP_QUERIES = 50
RANDOM_SEED = 42

DATABASES = {
    "pgvector": 5433,
    "vectorchord": 5434,
}

CONFIGURATIONS = [
    ("pgvector", "hnsw.ef_search", 20),
    ("pgvector", "hnsw.ef_search", 40),
    ("pgvector", "hnsw.ef_search", 60),
    ("pgvector", "hnsw.ef_search", 100),
    ("pgvector", "hnsw.ef_search", 200),

    ("vectorchord", "vchordrq.epsilon", 0.3),
    ("vectorchord", "vchordrq.epsilon", 0.5),
    ("vectorchord", "vchordrq.epsilon", 0.7),
    ("vectorchord", "vchordrq.epsilon", 1.0),
    ("vectorchord", "vchordrq.epsilon", 1.9),
]


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
    )


def vector_literal(vector):
    return "[" + ",".join(
        str(float(x)) for x in vector
    ) + "]"


def percentile(values, percent):
    return float(np.percentile(values, percent))


def verify_index_plan(cur, vector, database):
    cur.execute(
        f"""
        EXPLAIN
        SELECT id
        FROM {TABLE}
        ORDER BY embedding <-> %s::vector
        LIMIT 10
        """,
        (vector,),
    )

    plan = "\n".join(
        row[0] for row in cur.fetchall()
    )

    expected_index = (
        "vibe_yahoo_384_embedding_hnsw"
        if database == "pgvector"
        else "vibe_yahoo_384_embedding_vchord"
    )

    if expected_index not in plan:
        raise RuntimeError(
            f"Expected index {expected_index} "
            f"was not found in query plan:\n{plan}"
        )

    print(
        f"Verified index: {expected_index}",
        flush=True,
    )


def benchmark_configuration(
    database,
    parameter,
    value,
    queries,
    ground_truth,
):
    port = DATABASES[database]
    total = len(queries)

    print(
        f"\n=== {database}: {parameter}={value} ===",
        flush=True,
    )

    rng = random.Random(RANDOM_SEED)
    order = list(range(total))
    rng.shuffle(order)

    warmup_order = [
        order[i % total]
        for i in range(WARMUP_QUERIES)
    ]

    latencies_ms = []
    recalls = []
    incomplete = 0

    with connect(port) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")

            # Parameter names come from our fixed configuration
            # list, not from external user input.
            cur.execute(
                f"SET {parameter} = {value}"
            )

            if database == "pgvector":
                cur.execute(
                    "SET hnsw.iterative_scan = off"
                )
            else:
                cur.execute(
                    "SET vchordrq.prefilter = off"
                )

            verify_index_plan(
                cur,
                queries[0],
                database,
            )

            sql = f"""
                SELECT id
                FROM {TABLE}
                ORDER BY embedding <-> %s::vector
                LIMIT 10
            """

            print(
                f"Warming up with {WARMUP_QUERIES} queries...",
                flush=True,
            )

            for query_id in warmup_order:
                cur.execute(
                    sql,
                    (queries[query_id],),
                )
                cur.fetchall()

            print(
                f"Measuring {total} queries...",
                flush=True,
            )

            for position, query_id in enumerate(
                order, start=1
            ):
                start = time.perf_counter()

                cur.execute(
                    sql,
                    (queries[query_id],),
                )
                returned_ids = [
                    row[0]
                    for row in cur.fetchall()
                ]

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                exact_top10 = set(
                    ground_truth[query_id, :10]
                )

                found = len(
                    exact_top10.intersection(
                        returned_ids
                    )
                )

                recalls.append(found / 10.0)
                latencies_ms.append(elapsed_ms)

                if len(returned_ids) < 10:
                    incomplete += 1

                if position % 200 == 0:
                    print(
                        f"  {position}/{total} queries",
                        flush=True,
                    )

    result = {
        "database": database,
        "parameter": parameter,
        "value": value,
        "queries": total,
        "recall_at_10": round(
            statistics.mean(recalls), 5
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
        "mean_ms": round(
            statistics.mean(latencies_ms), 3
        ),
        "incomplete_queries": incomplete,
    }

    print("RESULT:", result, flush=True)

    return result


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    with h5py.File(DATASET, "r") as f:
        raw_queries = f["test"][:]
        ground_truth = f["neighbors"][:]

    if raw_queries.shape != (1000, 384):
        raise RuntimeError(
            f"Unexpected query shape: "
            f"{raw_queries.shape}"
        )

    if ground_truth.shape[0] != 1000:
        raise RuntimeError(
            f"Unexpected ground-truth shape: "
            f"{ground_truth.shape}"
        )

    # Convert once, outside the timed query loop.
    queries = [
        vector_literal(row)
        for row in raw_queries
    ]

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    fieldnames = [
        "database",
        "parameter",
        "value",
        "queries",
        "recall_at_10",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "incomplete_queries",
    ]

    completed = set()

    if OUTPUT.exists():
        with OUTPUT.open(newline="") as file:
            for row in csv.DictReader(file):
                completed.add(
                    (
                        row["database"],
                        row["parameter"],
                        str(row["value"]),
                    )
                )

    configurations = CONFIGURATIONS.copy()

    # Randomize configuration order to reduce systematic
    # bias from always testing one engine first.
    random.Random(RANDOM_SEED).shuffle(
        configurations
    )

    write_header = not OUTPUT.exists()

    with OUTPUT.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if write_header:
            writer.writeheader()
            file.flush()

        for database, parameter, value in configurations:
            key = (
                database,
                parameter,
                str(value),
            )

            if key in completed:
                print(
                    f"Skipping completed: {key}",
                    flush=True,
                )
                continue

            result = benchmark_configuration(
                database,
                parameter,
                value,
                queries,
                ground_truth,
            )

            writer.writerow(result)
            file.flush()

    print(
        f"\nFinished. Results saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
