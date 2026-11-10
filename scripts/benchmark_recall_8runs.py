
import argparse
import csv
import random
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg


REPEATS = 8
NUM_QUERIES = 1000
WARMUP_QUERIES = 50
K = 10
SEED = 42

DATABASES = {
    "pgvector": 5433,
    "vectorchord": 5434,
}

DATASETS = {
    "sift1m": {
        "file": "datasets/real/sift1m/sift-128-euclidean.hdf5",
        "table": "sift_1m",
        "output": "results/real/sift1m_recall_8runs.csv",
        "indexes": {
            "pgvector": "sift_1m_embedding_hnsw",
            "vectorchord": "sift_1m_embedding_vchord",
        },
    },
    "vibe": {
        "file": "datasets/real/vibe/yahoo-minilm-384-normalized.hdf5",
        "table": "vibe_yahoo_384",
        "output": "results/real/vibe_yahoo_recall_8runs.csv",
        "indexes": {
            "pgvector": "vibe_yahoo_384_embedding_hnsw",
            "vectorchord": "vibe_yahoo_384_embedding_vchord",
        },
    },
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

FIELDS = [
    "dataset",
    "database",
    "parameter",
    "value",
    "repeat",
    "seed",
    "queries",
    "warmup_queries",
    "recall_at_10",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
    "incomplete_queries",
]


def vector_literal(vector):
    return "[" + ",".join(
        str(float(value)) for value in vector
    ) + "]"


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
    )


def load_dataset(config):
    with h5py.File(config["file"], "r") as file:
        raw_queries = file["test"][:NUM_QUERIES]
        neighbors = file["neighbors"][:NUM_QUERIES, :K]

    if len(raw_queries) != NUM_QUERIES:
        raise RuntimeError("Dataset has too few test queries.")

    if neighbors.shape != (NUM_QUERIES, K):
        raise RuntimeError(
            f"Unexpected ground-truth shape: {neighbors.shape}"
        )

    queries = [
        vector_literal(vector)
        for vector in raw_queries
    ]

    ground_truth = [
        set(int(value) for value in row)
        for row in neighbors
    ]

    return queries, ground_truth


def configure_connection(cur, database, parameter, value):
    cur.execute("SET enable_seqscan = off")

    # These names and values are defined in CONFIGURATIONS.
    cur.execute(f"SET {parameter} = {value}")

    if database == "pgvector":
        cur.execute("SET hnsw.iterative_scan = off")
    else:
        cur.execute("SET vchordrq.prefilter = off")


def verify_index(cur, sql, query, expected_index):
    cur.execute("EXPLAIN " + sql, (query,))
    plan = "\n".join(row[0] for row in cur.fetchall())

    if expected_index not in plan:
        raise RuntimeError(
            f"Expected index {expected_index} not found:\n{plan}"
        )


def run_one(
    dataset_name,
    dataset,
    database,
    parameter,
    value,
    repeat,
    queries,
    ground_truth,
):
    run_seed = SEED + repeat

    rng = random.Random(run_seed)
    order = list(range(NUM_QUERIES))
    rng.shuffle(order)

    warmup_order = [
        order[i % NUM_QUERIES]
        for i in range(WARMUP_QUERIES)
    ]

    sql = f"""
        SELECT id
        FROM {dataset['table']}
        ORDER BY embedding <-> %s::vector
        LIMIT {K}
    """

    latencies = []
    recalls = []
    incomplete = 0

    print(
        f"\n{dataset_name} | {database} | "
        f"{parameter}={value} | repeat {repeat}/{REPEATS}",
        flush=True,
    )

    with connect(DATABASES[database]) as conn:
        with conn.cursor() as cur:
            configure_connection(
                cur, database, parameter, value
            )

            verify_index(
                cur,
                sql,
                queries[0],
                dataset["indexes"][database],
            )

            # Warm-up is excluded from all reported timings.
            for query_id in warmup_order:
                cur.execute(sql, (queries[query_id],))
                cur.fetchall()

            for position, query_id in enumerate(
                order, start=1
            ):
                start = time.perf_counter()

                cur.execute(sql, (queries[query_id],))
                returned_ids = [
                    row[0] for row in cur.fetchall()
                ]

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                found = len(
                    ground_truth[query_id].intersection(
                        returned_ids
                    )
                )

                recalls.append(found / K)
                latencies.append(elapsed_ms)

                if len(returned_ids) < K:
                    incomplete += 1

                if position % 250 == 0:
                    print(
                        f"  {position}/{NUM_QUERIES}",
                        flush=True,
                    )

    result = {
        "dataset": dataset_name,
        "database": database,
        "parameter": parameter,
        "value": value,
        "repeat": repeat,
        "seed": run_seed,
        "queries": NUM_QUERIES,
        "warmup_queries": WARMUP_QUERIES,
        "recall_at_10": round(
            statistics.mean(recalls), 5
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

    print("RESULT:", result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        required=True,
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset = DATASETS[dataset_name]
    output = Path(dataset["output"])

    queries, ground_truth = load_dataset(dataset)

    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    completed = set()

    if output.exists():
        with output.open(newline="") as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != FIELDS:
                raise RuntimeError(
                    "Existing CSV has an incompatible header. "
                    "Refusing to append."
                )

            for row in reader:
                completed.add(
                    (
                        row["database"],
                        row["parameter"],
                        row["value"],
                        int(row["repeat"]),
                    )
                )

    # Each repetition gets a different deterministic
    # configuration order.
    jobs = []

    for repeat in range(1, REPEATS + 1):
        configs = CONFIGURATIONS.copy()

        random.Random(
            SEED + 1000 + repeat
        ).shuffle(configs)

        for database, parameter, value in configs:
            jobs.append(
                (
                    database,
                    parameter,
                    value,
                    repeat,
                )
            )

    write_header = not output.exists()

    with output.open("a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )

        if write_header:
            writer.writeheader()
            file.flush()

        for database, parameter, value, repeat in jobs:
            key = (
                database,
                parameter,
                str(value),
                repeat,
            )

            if key in completed:
                print(
                    f"Skipping completed: {key}",
                    flush=True,
                )
                continue

            result = run_one(
                dataset_name,
                dataset,
                database,
                parameter,
                value,
                repeat,
                queries,
                ground_truth,
            )

            writer.writerow(result)
            file.flush()

    print(
        f"\nFinished: {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
