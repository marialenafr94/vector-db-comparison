
import csv
import time
from pathlib import Path

import numpy as np
import psycopg

N = 100_000
DIM = 384
BATCH_SIZE = 1_000
SEED = 42

TABLE = "items_100k"
DATASET = Path("datasets/synthetic_100000_384d.npy")
RESULTS = Path("results/ingestion_100k.csv")

DATABASES = {
    "pgvector": {
        "port": 5433,
        "index": "items_100k_embedding_hnsw",
        "sql": """
            CREATE INDEX items_100k_embedding_hnsw
            ON items_100k
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64)
        """,
    },
    "vectorchord": {
        "port": 5434,
        "index": "items_100k_embedding_vchord",
        "sql": """
            CREATE INDEX items_100k_embedding_vchord
            ON items_100k
            USING vchordrq (embedding vector_l2_ops)
        """,
    },
}


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


def create_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)

    if DATASET.exists():
        data = np.load(DATASET, mmap_mode="r")
        if data.shape != (N, DIM) or data.dtype != np.float32:
            raise ValueError("Existing dataset has incorrect shape or dtype")
        print(f"Reusing existing dataset: {DATASET}", flush=True)
        return 0.0

    print("Generating 100,000 vectors...", flush=True)
    start = time.perf_counter()

    rng = np.random.default_rng(SEED)

    data = np.lib.format.open_memmap(
        DATASET,
        mode="w+",
        dtype=np.float32,
        shape=(N, DIM),
    )

    for offset in range(0, N, BATCH_SIZE):
        end = min(offset + BATCH_SIZE, N)

        batch = rng.normal(
            size=(end - offset, DIM)
        ).astype(np.float32)

        batch /= np.linalg.norm(
            batch, axis=1, keepdims=True
        )

        data[offset:end] = batch

        if end % 10_000 == 0:
            print(f"Generated {end:,}/{N:,}", flush=True)

    data.flush()
    del data

    elapsed = time.perf_counter() - start
    print(f"Generation completed in {elapsed:.2f} s", flush=True)
    return elapsed


def prepare_database(name, config, generation_seconds):
    print(f"\n===== {name.upper()} =====", flush=True)

    data = np.load(DATASET, mmap_mode="r")

    with connect(config["port"]) as conn:
        with conn.cursor() as cur:
            # Do not overwrite previous work if this script
            # is accidentally run a second time.
            cur.execute(
                "SELECT to_regclass(%s)",
                (f"public.{TABLE}",),
            )

            if cur.fetchone()[0] is not None:
                raise RuntimeError(
                    f"{TABLE} already exists in {name}. "
                    "Stopping to avoid overwriting data."
                )

            cur.execute(f"""
                CREATE TABLE {TABLE} (
                    id INTEGER PRIMARY KEY,
                    category_50 INTEGER NOT NULL,
                    category_10 INTEGER NOT NULL,
                    category_1 INTEGER NOT NULL,
                    embedding vector({DIM}) NOT NULL
                )
            """)

            print("Loading vectors...", flush=True)
            start = time.perf_counter()

            for offset in range(0, N, BATCH_SIZE):
                end = min(offset + BATCH_SIZE, N)

                with cur.copy(f"""
                    COPY {TABLE} (
                        id,
                        category_50,
                        category_10,
                        category_1,
                        embedding
                    ) FROM STDIN
                """) as copy:
                    for i in range(offset, end):
                        copy.write_row((
                            i,
                            i % 2,
                            i % 10,
                            i % 100,
                            vector_literal(data[i]),
                        ))

                if end % 10_000 == 0:
                    print(f"Loaded {end:,}/{N:,}", flush=True)

        # Commit table creation and all copied rows.
        conn.commit()

        ingest_seconds = time.perf_counter() - start
        print(
            f"Ingestion completed in {ingest_seconds:.2f} s",
            flush=True,
        )

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
            row_count = cur.fetchone()[0]

            if row_count != N:
                raise RuntimeError(
                    f"Expected {N} rows, found {row_count}"
                )

            print("Building vector index...", flush=True)
            start = time.perf_counter()

            cur.execute(config["sql"])
            conn.commit()

            index_seconds = time.perf_counter() - start

            print(
                f"Index built in {index_seconds:.2f} s",
                flush=True,
            )

            # Refresh planner statistics.
            cur.execute(f"ANALYZE {TABLE}")
            conn.commit()

            cur.execute("""
                SELECT pg_relation_size(%s::regclass)
            """, (f"public.{TABLE}",))
            table_bytes = cur.fetchone()[0]

            cur.execute("""
                SELECT pg_relation_size(%s::regclass)
            """, (f"public.{config['index']}",))
            index_bytes = cur.fetchone()[0]

    return {
        "database": name,
        "vectors": N,
        "dimensions": DIM,
        "generation_seconds": round(generation_seconds, 3),
        "ingestion_seconds": round(ingest_seconds, 3),
        "index_seconds": round(index_seconds, 3),
        "table_mb": round(table_bytes / 1024**2, 2),
        "vector_index_mb": round(index_bytes / 1024**2, 2),
    }


def main():
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    generation_seconds = create_dataset()

    results = []

    for name, config in DATABASES.items():
        result = prepare_database(
            name, config, generation_seconds
        )

        results.append(result)
        print(result, flush=True)

        # Preserve measurements even if the next database
        # fails or runs out of space.
        with RESULTS.open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=results[0].keys()
            )
            writer.writeheader()
            writer.writerows(results)

    print("\nFinished!")
    print(f"Results saved to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
