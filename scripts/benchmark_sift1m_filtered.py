
import csv
import random
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path("datasets/real/sift1m/sift-128-euclidean.hdf5")
GROUND_TRUTH = Path("results/real/sift1m_filtered_groundtruth.npz")
OUTPUT = Path("results/real/sift1m_filtered_results.csv")

K = 10
NUM_QUERIES = 100
WARMUP_QUERIES = 10

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
    # Column names are taken only from the fixed FILTERS list.
    return f"""
        SELECT id
        FROM sift_1m
        WHERE {column} = 0
        ORDER BY embedding <-> %s::vector
        LIMIT 10
    """


def configure(cur, database, scan_mode, search_value):
    cur.execute("SET enable_seqscan = off")

    # Give pgvector iterative scans an explicit search budget.
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

    print("  Query plan:")
    print("  " + plan.replace("\n", "\n  "), flush=True)

    if expected_index not in plan:
        raise RuntimeError(
            f"Expected ANN index {expected_index} not found in plan"
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
):
    print(
        f"\n{database} | {column}={0} | "
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
            verify_plan(cur, sql, queries[0], database)

            # Warm up without recording latency or recall.
            for i in query_order[:WARMUP_QUERIES]:
                cur.execute(sql, (queries[i],))
                cur.fetchall()

            for position, i in enumerate(query_order, 1):
                start = time.perf_counter()

                cur.execute(sql, (queries[i],))
                found = [row[0] for row in cur.fetchall()]

                elapsed_ms = (time.perf_counter() - start) * 1000

                correct = len(
                    set(found).intersection(ground_truth[i])
                )

                recalls.append(correct / K)
                result_counts.append(len(found))
                latencies.append(elapsed_ms)

                if position % 25 == 0:
                    print(
                        f"  {position}/{len(query_order)} queries",
                        flush=True,
                    )

    result = {
        "database": database,
        "filter": column,
        "selectivity_percent": selectivity,
        "scan_mode": scan_mode,
        "search_value": search_value,
        "queries": len(query_order),
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "mean_returned": round(float(np.mean(result_counts)), 3),
        "incomplete_queries": sum(n < K for n in result_counts),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "mean_ms": round(float(np.mean(latencies)), 3),
    }

    print("  Result:", result, flush=True)
    return result


def main():
    with h5py.File(DATASET, "r") as file:
        queries = [
            vector_literal(v)
            for v in file["test"][:NUM_QUERIES]
        ]

    with np.load(GROUND_TRUTH) as data:
        exact = {
            column: [
                set(int(x) for x in row)
                for row in data[column][:NUM_QUERIES]
            ]
            for column, _ in FILTERS
        }

        if len(data["query_indices"]) < NUM_QUERIES:
            raise RuntimeError("Ground truth has too few queries")

        if not np.array_equal(
            data["query_indices"][:NUM_QUERIES],
            np.arange(NUM_QUERIES),
        ):
            raise RuntimeError("Ground truth query order mismatch")

    order = list(range(NUM_QUERIES))
    random.Random(42).shuffle(order)

    # Save each completed combination immediately.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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

    with OUTPUT.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        file.flush()

        for column, selectivity in FILTERS:
            for database, port, scan_mode, search_value in CONFIGS:
                result = run_configuration(
                    database,
                    port,
                    scan_mode,
                    search_value,
                    column,
                    selectivity,
                    queries,
                    exact[column],
                    order,
                )

                writer.writerow(result)
                file.flush()

    print(f"\nFinished. Results saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
