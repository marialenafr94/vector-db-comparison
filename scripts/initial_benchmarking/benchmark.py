
import csv
import time
from pathlib import Path

import numpy as np
import psycopg

K = 10
N_QUERIES = 100
DIM = 384

DATABASES = {
    "pgvector": 5433,
    "vectorchord": 5434,
}

# Independent queries, not copied from the database.
rng = np.random.default_rng(2026)
queries = rng.normal(size=(N_QUERIES, DIM)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

def vector_literal(v):
    return "[" + ",".join(map(str, v)) + "]"

def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
    )

# Load the stored vectors from PostgreSQL.
with connect(5433) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT id, embedding::text FROM items ORDER BY id")
        rows = cur.fetchall()

ids = np.array([row[0] for row in rows])
vectors = np.array([
    np.fromstring(row[1].strip("[]"), sep=",", dtype=np.float32)
    for row in rows
])

assert vectors.shape[1] == DIM

# Exact L2 ground truth, calculated in chunks.
# Squared L2 gives the same ranking as L2.
ground_truth = []
for q in queries:
    distances = np.sum((vectors - q) ** 2, axis=1)
    nearest = np.argsort(distances, kind="stable")[:K]
    ground_truth.append(set(ids[nearest]))

Path("results").mkdir(exist_ok=True)

summary = []

for db_name, port in DATABASES.items():
    print(f"\nBenchmarking {db_name}...")

    with connect(port) as conn:
        with conn.cursor() as cur:
            # Set a consistent initial search configuration.
            if db_name == "pgvector":
                cur.execute("SET hnsw.ef_search = 400")

            sql = """
                SELECT id
                FROM items
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """

            # Warm-up runs are excluded from measurement.
            for q in queries[:20]:
                cur.execute(sql, (vector_literal(q), K))
                cur.fetchall()

            latencies = []
            recalls = []

            for i, q in enumerate(queries):
                start = time.perf_counter()

                cur.execute(sql, (vector_literal(q), K))
                found = [row[0] for row in cur.fetchall()]

                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

                recall = len(set(found) & ground_truth[i]) / K
                recalls.append(recall)

    result = {
        "database": db_name,
        "vectors": len(vectors),
        "dimensions": DIM,
        "queries": N_QUERIES,
        "k": K,
        "recall_at_10": float(np.mean(recalls)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    summary.append(result)
    print(result)

with open("results/benchmark_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("\nSaved results/benchmark_summary.csv")
