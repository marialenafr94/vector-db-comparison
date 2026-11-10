import csv
import random
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg


DATASET = Path("datasets/real/sift1m/sift-128-euclidean.hdf5")
GROUND_TRUTH = Path("results/real/sift1m_filtered_groundtruth.npz")
OUTPUT = Path("results/real/sift1m_filtered_8runs.csv")

K = 10
NUM_QUERIES = 100
WARMUP_QUERIES = 10
REPEATS = 8
SEED = 42

FILTERS = [
    ("category_50", 50),
    ("category_10", 10),
    ("category_1", 1),
]

CONFIGS = [
    ("pgvector", 5433, "off", 60),
    ("pgvector", 5433, "strict_order", 60),
    ("pgvector", 5433, "relaxed_order", 60),
    ("vectorchord", 5434, "default", 0.5),
    ("vectorchord", 5434, "default", 0.7),
]

FIELDNAMES = [
    "repeat",
    "seed",
    "warmup_queries",
    "database",
    "filter",
    "selectivity_percent",
    "scan_mode",
    "search_value",
    "queries",
    "recall_at_10",
    "mean_returned",
    "incomplete_queries",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "mean_ms",
]


def vector_literal(vector):
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    )


def make_sql(column):
    # Column names come only from the fixed FILTERS list.
    return f"""
        SELECT id
        FROM sift_1m
        WHERE {column} = 0
        ORDER BY embedding <-> %s::vector
        LIMIT {K}
    """


def configure(cur, database, scan_mode, search_value):
    cur.execute("SET enable_seqscan = off")

    if database == "pgvector":
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(search_value),),
        )
        cur.execute(
            "SELECT set_config('hnsw.iterative_scan', %s, false)",
            (scan_mode,),
        )
        cur.execute("SET hnsw.max_scan_tuples = 20000")
        cur.execute("SET hnsw.scan_mem_multiplier = 1")

    else:
        cur.execute(
            "SELECT set_config('vchordrq.epsilon', %s, false)",
            (str(search_value),),
        )


def verify_plan(cur, sql, vector, database):
    cur.execute("EXPLAIN " + sql, (vector,))
    plan = "\n".join(row[0] for row in cur.fetchall())

    expected_index = (
        "sift_1m_embedding_hnsw"
        if database == "pgvector"
        else "sift_1m_embedding_vchord"
    )

    if expected_index not in plan:
        raise RuntimeError(
            f"Expected ANN index {expected_index} not found.\n"
            f"Query plan:\n{plan}"
        )


def run_configuration(
    database,
    port,
    scan_mode,
    search_value,
    column,
    selectivity,
    queries,
    ground_truth,
    query_order,
    repeat,
):
    print(
        f"\nRepeat {repeat}/{REPEATS} | "
        f"{database} | {column}=0 | "
        f"scan={scan_mode} | search={search_value}",
        flush=True,
    )

    sql = make_sql(column)

    latencies = []
    recalls = []
    result_counts = []

    with connect(port) as conn:
        with conn.cursor() as cur:
            configure(cur, database, scan_mode, search_value)

            verify_plan(
                cur,
                sql,
                queries[0],
                database,
            )

            # Warm-up queries are not timed.
            for query_id in query_order[:WARMUP_QUERIES]:
                cur.execute(sql, (queries[query_id],))
                cur.fetchall()

            # Measure 100 queries.
            for position, query_id in enumerate(
                query_order, start=1
            ):
                start = time.perf_counter()

                cur.execute(sql, (queries[query_id],))
                returned_ids = [
                    row[0] for row in cur.fetchall()
                ]

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                correct = len(
                    set(returned_ids).intersection(
                        ground_truth[query_id]
                    )
                )

                recalls.append(correct / K)
                result_counts.append(len(returned_ids))
                latencies.append(elapsed_ms)

                if position % 25 == 0:
                    print(
                        f"  {position}/{NUM_QUERIES} queries",
                        flush=True,
                    )

    result = {
        "repeat": repeat,
        "seed": SEED + repeat,
        "warmup_queries": WARMUP_QUERIES,
        "database": database,
        "filter": column,
        "selectivity_percent": selectivity,
        "scan_mode": scan_mode,
        "search_value": search_value,
        "queries": len(query_order),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mean_returned": round(float(np.mean(result_counts)), 3),
        "incomplete_queries": sum(
            count < K for count in result_counts
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
            float(np.mean(latencies)), 3
        ),
    }

    print("  Result:", result, flush=True)
    return result


def load_data():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(GROUND_TRUTH)

    with h5py.File(DATASET, "r") as file:
        queries = [
            vector_literal(vector)
            for vector in file["test"][:NUM_QUERIES]
        ]

    if len(queries) != NUM_QUERIES:
        raise RuntimeError("Dataset has too few queries.")

    with np.load(GROUND_TRUTH) as data:
        if len(data["query_indices"]) < NUM_QUERIES:
            raise RuntimeError(
                "Ground truth has too few queries."
            )

        if not np.array_equal(
            data["query_indices"][:NUM_QUERIES],
            np.arange(NUM_QUERIES),
        ):
            raise RuntimeError(
                "Ground truth query order mismatch."
            )

        exact = {}

        for column, _ in FILTERS:
            if column not in data:
                raise RuntimeError(
                    f"Missing ground truth for {column}"
                )

            neighbors = data[column][:NUM_QUERIES]

            if neighbors.shape != (NUM_QUERIES, K):
                raise RuntimeError(
                    f"Unexpected ground-truth shape "
                    f"for {column}: {neighbors.shape}"
                )

            exact[column] = [
                set(int(value) for value in row)
                for row in neighbors
            ]

    return queries, exact


def experiment_key(
    database,
    column,
    scan_mode,
    search_value,
    repeat,
):
    return (
        database,
        column,
        scan_mode,
        str(search_value),
        int(repeat),
    )


def load_completed():
    completed = set()

    if not OUTPUT.exists():
        return completed

    with OUTPUT.open(newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != FIELDNAMES:
            raise RuntimeError(
                "Existing output CSV has an incompatible header. "
                "Refusing to append."
            )

        for row in reader:
            key = experiment_key(
                row["database"],
                row["filter"],
                row["scan_mode"],
                row["search_value"],
                row["repeat"],
            )

            if key in completed:
                raise RuntimeError(
                    f"Duplicate experiment in CSV: {key}"
                )

            completed.add(key)

    return completed


def main():
    queries, exact = load_data()

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed = load_completed()

    # Build 15 configurations for each of 8 repetitions.
    jobs = []

    for repeat in range(1, REPEATS + 1):
        configurations = [
            (
                column,
                selectivity,
                database,
                port,
                scan_mode,
                search_value,
            )
            for column, selectivity in FILTERS
            for database, port, scan_mode, search_value in CONFIGS
        ]

        # Randomize configuration order for each repetition.
        random.Random(
            SEED + 1000 + repeat
        ).shuffle(configurations)

        for configuration in configurations:
            jobs.append((repeat, *configuration))

    write_header = not OUTPUT.exists()

    with OUTPUT.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if write_header:
            writer.writeheader()
            file.flush()

        for (
            repeat,
            column,
            selectivity,
            database,
            port,
            scan_mode,
            search_value,
        ) in jobs:

            key = experiment_key(
                database,
                column,
                scan_mode,
                search_value,
                repeat,
            )

            if key in completed:
                print(
                    f"Skipping completed: {key}",
                    flush=True,
                )
                continue

            query_order = list(range(NUM_QUERIES))

            # Reproducible query order per repetition.
            random.Random(
                SEED + repeat
            ).shuffle(query_order)

            result = run_configuration(
                database=database,
                port=port,
                scan_mode=scan_mode,
                search_value=search_value,
                column=column,
                selectivity=selectivity,
                queries=queries,
                ground_truth=exact[column],
                query_order=query_order,
                repeat=repeat,
            )

            writer.writerow(result)
            file.flush()

            completed.add(key)

    print(
        f"\nFinished. Results saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()