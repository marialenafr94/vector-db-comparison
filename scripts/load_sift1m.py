
import csv
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/sift1m/sift-128-euclidean.hdf5"
)

OUTPUT = Path("results/real/sift1m_ingestion.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5000
DIM = 128

DATABASES = [
    ("pgvector", 5433),
    ("vectorchord", 5434),
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
    return "[" + ",".join(map(str, vector)) + "]"


def create_table(conn):
    with conn.cursor() as cur:
        # Refuse to overwrite an existing table.
        cur.execute(
            "SELECT to_regclass(%s)",
            ("public.sift_1m",),
        )

        if cur.fetchone()[0] is not None:
            raise RuntimeError(
                "Table sift_1m already exists. "
                "Stopping to protect existing data and indexes."
            )

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            CREATE TABLE sift_1m (
                id INTEGER PRIMARY KEY,
                category_50 INTEGER NOT NULL,
                category_10 INTEGER NOT NULL,
                category_1 INTEGER NOT NULL,
                embedding vector(128) NOT NULL
            )
        """)


def load_database(name, port, dataset):
    print(f"\nLoading SIFT1M into {name}...", flush=True)

    with connect(port) as conn:
        create_table(conn)

        start = time.perf_counter()

        with conn.cursor() as cur:
            for offset in range(0, len(dataset), BATCH_SIZE):
                batch = dataset[
                    offset : offset + BATCH_SIZE
                ]

                rows = [
                    (
                        offset + i,
                        (offset + i) % 2,
                        (offset + i) % 10,
                        (offset + i) % 100,
                        vector_literal(vector),
                    )
                    for i, vector in enumerate(batch)
                ]

                with cur.copy("""
                    COPY sift_1m (
                        id,
                        category_50,
                        category_10,
                        category_1,
                        embedding
                    )
                    FROM STDIN
                """) as copy:
                    for row in rows:
                        copy.write_row(row)

                if (offset + len(batch)) % 100000 == 0:
                    print(
                        f"{name}: "
                        f"{offset + len(batch):,} vectors loaded",
                        flush=True,
                    )

        elapsed = time.perf_counter() - start

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sift_1m")
            count = cur.fetchone()[0]

            cur.execute("""
                SELECT pg_total_relation_size('sift_1m')
            """)
            table_bytes = cur.fetchone()[0]

    result = {
        "database": name,
        "vectors": count,
        "dimensions": DIM,
        "ingestion_seconds": round(elapsed, 3),
        "table_mb": round(
            table_bytes / 1024 / 1024, 2
        ),
    }

    print(result, flush=True)
    return result


def main():
    with h5py.File(DATASET, "r") as f:
        dataset = f["train"]

        results = []

        for name, port in DATABASES:
            result = load_database(
                name,
                port,
                dataset,
            )
            results.append(result)

            with OUTPUT.open("x", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=results[0].keys(),
                )
                writer.writeheader()
                writer.writerows(results)

    print(f"\nResults saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
