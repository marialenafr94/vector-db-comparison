
import csv
import hashlib
import time
from pathlib import Path

import numpy as np
import psycopg

# Experiment configuration
K = 10
N_QUERIES = 100
DIM = 384
WARMUP = 20

PGVECTOR_EF_VALUES = [850, 900, 950, 1000]

DATABASES = {
    "pgvector": 5433,
    "vectorchord": 5434,
}

OUTPUT = Path("results/tuning_100k_high_recall.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
    )


def vector_literal(vector):
    return "[" + ",".join(map(str, vector)) + "]"


def load_dataset(port):
    with connect(port) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding::text FROM items_100k ORDER BY id"
            )
            rows = cur.fetchall()

    ids = np.array([row[0] for row in rows], dtype=np.int32)
    vectors = np.array(
        [
            np.fromstring(
                row[1].strip("[]"),
                sep=",",
                dtype=np.float32,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    return ids, vectors


def dataset_digest(ids, vectors):
    digest = hashlib.sha256()
    digest.update(ids.tobytes())
    digest.update(vectors.tobytes())
    return digest.hexdigest()


print("Loading and verifying datasets...", flush=True)

ids, vectors = load_dataset(DATABASES["pgvector"])
other_ids, other_vectors = load_dataset(DATABASES["vectorchord"])

assert len(ids) == 100_000, "Unexpected dataset size"
assert vectors.shape == (len(ids), DIM)
assert np.array_equal(ids, other_ids), "IDs differ"
assert np.array_equal(vectors, other_vectors), "Vectors differ"

print("Both databases contain identical vectors.")
print("Dataset SHA256:", dataset_digest(ids, vectors), flush=True)

# Generate reproducible, independent query vectors.
rng = np.random.default_rng(2026)
queries = rng.normal(
    size=(N_QUERIES, DIM)
).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

query_literals = [vector_literal(q) for q in queries]

# Exact L2 top-10 ground truth.
print("Computing exact ground truth...", flush=True)

vector_norms = np.sum(vectors.astype(np.float64) ** 2, axis=1)
vectors64 = vectors.astype(np.float64)

ground_truth = []
for q in queries:
    q64 = q.astype(np.float64)
    distances = vector_norms - 2.0 * (vectors64 @ q64)
    nearest = np.argsort(distances, kind="stable")[:K]
    ground_truth.append(set(ids[nearest]))

SQL = """
    SELECT id
    FROM items_100k
    ORDER BY embedding <-> %s::vector
    LIMIT %s
"""


def benchmark_configuration(name, port, setting, value):
    latencies = []
    recalls = []

    with connect(port) as conn:
        with conn.cursor() as cur:
            # Keep all runs in read-only transaction mode.
            if setting is not None:
                cur.execute(
                    "SELECT set_config(%s, %s, false)",
                    (setting, str(value)),
                )

            # Warm up this configuration.
            for q in query_literals[:WARMUP]:
                cur.execute(SQL, (q, K))
                cur.fetchall()

            for i, q in enumerate(query_literals):
                start = time.perf_counter()
                cur.execute(SQL, (q, K))
                found = [row[0] for row in cur.fetchall()]
                elapsed_ms = (time.perf_counter() - start) * 1000

                latencies.append(elapsed_ms)
                recalls.append(
                    len(set(found) & ground_truth[i]) / K
                )

    result = {
        "database": name,
        "setting": setting or "default",
        "value": str(value) if value is not None else "default",
        "recall_at_10": round(float(np.mean(recalls)), 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "queries": N_QUERIES,
        "dimensions": DIM,
        "vectors": len(ids),
    }

    print(result, flush=True)
    return result


results = []

# pgvector HNSW sweep
for ef in PGVECTOR_EF_VALUES:
    print(f"\npgvector ef_search={ef}", flush=True)
    results.append(
        benchmark_configuration(
            "pgvector",
            DATABASES["pgvector"],
            "hnsw.ef_search",
            ef,
        )
    )

# VectorChord default configuration.
# The current 10k index may be listless, meaning probes
# should remain at the default (empty) setting.
print("\nVectorChord default configuration", flush=True)
results.append(
    benchmark_configuration(
        "vectorchord",
        DATABASES["vectorchord"],
        None,
        None,
    )
)

with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT}", flush=True)
