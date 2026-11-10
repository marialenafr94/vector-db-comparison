
import csv
import time
from pathlib import Path

import numpy as np
import psycopg

# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

DIM = 384
K = 10
N_QUERIES = 100
WARMUP = 10

PGVECTOR_PORT = 5433
VECTORCHORD_PORT = 5434

PGVECTOR_EF_SEARCH = 300
PGVECTOR_MAX_SCAN_TUPLES = 20000

FILTERS = [
    ("50_percent", "category_50", 0),
    ("10_percent", "category_10", 0),
    ("1_percent", "category_1", 0),
]

CONFIGURATIONS = [
    ("pgvector", PGVECTOR_PORT, "off"),
    ("pgvector", PGVECTOR_PORT, "strict_order"),
    ("pgvector", PGVECTOR_PORT, "relaxed_order"),
    ("vectorchord", VECTORCHORD_PORT, "default"),
]

OUTPUT = Path("results/filtered_100k_index_focused.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


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
    return "[" + ",".join(map(str, vector)) + "]"


# --------------------------------------------------
# Load the exact dataset
# --------------------------------------------------

print("Loading vectors for ground truth...", flush=True)

with connect(PGVECTOR_PORT) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                category_50,
                category_10,
                category_1,
                embedding::text
            FROM items_100k
            ORDER BY id
        """)
        rows = cur.fetchall()

ids = np.array([r[0] for r in rows], dtype=np.int32)

categories = {
    "category_50": np.array([r[1] for r in rows]),
    "category_10": np.array([r[2] for r in rows]),
    "category_1": np.array([r[3] for r in rows]),
}

vectors = np.array(
    [
        np.fromstring(
            r[4].strip("[]"),
            sep=",",
            dtype=np.float32,
        )
        for r in rows
    ],
    dtype=np.float32,
)

assert vectors.shape == (100000, DIM)

# Same query generation as the previous experiments.
rng = np.random.default_rng(2026)
queries = rng.normal(
    size=(N_QUERIES, DIM)
).astype(np.float32)

queries /= np.linalg.norm(queries, axis=1, keepdims=True)

query_literals = [vector_literal(q) for q in queries]


# --------------------------------------------------
# Exact filtered ground truth
# --------------------------------------------------

# This computes the same nearest-neighbor ranking using squared L2 distances, without creating a large temporary matrix for every query.
def calculate_ground_truth(column, value):
    mask = categories[column] == value

    filtered_ids = ids[mask]
    filtered_vectors = vectors[mask].astype(np.float64)

    squared_norms = np.einsum(
        "ij,ij->i",
        filtered_vectors,
        filtered_vectors,
    )

    truth = []

    for query in queries:
        q = query.astype(np.float64)

        distances = (
            squared_norms - 2.0 * (filtered_vectors @ q)
        )

        nearest = np.argsort(
            distances,
            kind="stable",
        )[:K]

        truth.append(set(filtered_ids[nearest]))

    return truth, len(filtered_ids)

# --------------------------------------------------
# Run a configuration
# --------------------------------------------------

def run_configuration(
    database,
    port,
    scan_mode,
    filter_name,
    column,
    value,
    ground_truth,
    candidate_count,
):
    # Column names are selected only from the fixed FILTERS
    # configuration above, never from user input.
    sql = f"""
        SELECT id
        FROM items_100k
        WHERE {column} = %s
        ORDER BY embedding <-> %s::vector
        LIMIT {K}
    """

    latencies = []
    recalls = []
    result_counts = []

    with connect(port) as conn:
        with conn.cursor() as cur:

            # Experimental setting: encourage vector index scans.
            cur.execute("SET enable_seqscan = off")

            if database == "pgvector":
                settings = {
                    "hnsw.ef_search": str(PGVECTOR_EF_SEARCH),
                    "hnsw.iterative_scan": scan_mode,
                    "hnsw.max_scan_tuples": str(
                        PGVECTOR_MAX_SCAN_TUPLES
                    ),
                }

                for setting, setting_value in settings.items():
                    cur.execute(
                        "SELECT set_config(%s, %s, false)",
                        (setting, setting_value),
                    )

            # Warm up the current configuration.
            for q in query_literals[:WARMUP]:
                cur.execute(sql, (value, q))
                cur.fetchall()

            for i, q in enumerate(query_literals):
                start = time.perf_counter()

                cur.execute(sql, (value, q))
                found = [row[0] for row in cur.fetchall()]

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(elapsed_ms)

                # Missing results count as missed neighbors.
                recalls.append(
                    len(set(found) & ground_truth[i]) / K
                )

                result_counts.append(len(found))

    result = {
        "database": database,
        "scan_mode": scan_mode,
        "filter": filter_name,
        "candidate_count": candidate_count,
        "ef_search": (
            PGVECTOR_EF_SEARCH
            if database == "pgvector"
            else "default"
        ),
        "recall_at_10": round(
            float(np.mean(recalls)), 4
        ),
        "mean_results_returned": round(
            float(np.mean(result_counts)), 2
        ),
        "queries_with_fewer_than_10": sum(
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
        "queries": N_QUERIES,
    }

    print(result, flush=True)
    return result


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    results = []

    for filter_name, column, value in FILTERS:
        print(
            f"\n===== FILTER: {filter_name} =====",
            flush=True,
        )

        ground_truth, candidate_count = (
            calculate_ground_truth(column, value)
        )

        print(
            f"Exact ground truth computed from "
            f"{candidate_count} candidates.",
            flush=True,
        )

        for database, port, scan_mode in CONFIGURATIONS:
            print(
                f"\nRunning {database}, mode={scan_mode}",
                flush=True,
            )

            result = run_configuration(
                database,
                port,
                scan_mode,
                filter_name,
                column,
                value,
                ground_truth,
                candidate_count,
            )

            results.append(result)

            # Save after each completed configuration.
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
