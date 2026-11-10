
import csv
import time
from pathlib import Path

import h5py
import psycopg

DATASET = Path(
    "datasets/real/vibe/yahoo-minilm-384-normalized.hdf5"
)
OUTPUT = Path("results/real/vibe_yahoo_ingestion.csv")

TABLE = "vibe_yahoo_384"
BATCH_SIZE = 5000

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
    )


def vector_literal(vector):
    return "[" + ",".join(
        str(float(value)) for value in vector
    ) + "]"


def create_table(conn):
    with conn.cursor() as cur:
        # Protect existing benchmark results.
        cur.execute(
            """
            SELECT to_regclass(%s)
            """,
            (f"public.{TABLE}",),
        )
        if cur.fetchone()[0] is not None:
            raise RuntimeError(
                f"Table {TABLE} already exists. "
                "Stopping to avoid overwriting data."
            )

        cur.execute(
            f"""
            CREATE TABLE {TABLE} (
                id INTEGER PRIMARY KEY,
                category_50 INTEGER NOT NULL,
                category_10 INTEGER NOT NULL,
                category_1 INTEGER NOT NULL,
                embedding vector(384) NOT NULL
            )
            """
        )

    conn.commit()


def load_database(name, port, train):
    total = len(train)
    print(f"\nLoading {name}: {total:,} vectors", flush=True)

    with connect(port) as conn:
        create_table(conn)

        start_time = time.perf_counter()

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            with conn.cursor() as cur:
                with cur.copy(
                    f"""
                    COPY {TABLE} (
                        id,
                        category_50,
                        category_10,
                        category_1,
                        embedding
                    ) FROM STDIN
                    """
                ) as copy:
                    for row_id in range(start, end):
                        copy.write_row(
                            (
                                row_id,
                                row_id % 2,
                                row_id % 10,
                                row_id % 100,
                                vector_literal(train[row_id]),
                            )
                        )

            conn.commit()

            if end % 50000 < BATCH_SIZE or end == total:
                print(
                    f"  Loaded {end:,}/{total:,}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start_time

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
            count = cur.fetchone()[0]

            cur.execute(
                "SELECT pg_total_relation_size(%s::regclass)",
                (TABLE,),
            )
            table_bytes = cur.fetchone()[0]

        if count != total:
            raise RuntimeError(
                f"Row count mismatch: {count} != {total}"
            )

        table_mb = table_bytes / (1024 * 1024)

        print(
            f"{name}: {elapsed:.3f} seconds, "
            f"{table_mb:.2f} MiB",
            flush=True,
        )

        return {
            "database": name,
            "vectors": count,
            "dimensions": 384,
            "ingestion_seconds": round(elapsed, 3),
            "table_mb": round(table_mb, 2),
        }


def main():
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    with h5py.File(DATASET, "r") as file:
        train = file["train"][:]

    assert train.shape == (677305, 384), train.shape

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "database",
        "vectors",
        "dimensions",
        "ingestion_seconds",
        "table_mb",
    ]

    # Append completed measurements so a failure in the
    # second database does not lose the first result.
    write_header = not OUTPUT.exists()

    with OUTPUT.open("a", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fieldnames
        )

        if write_header:
            writer.writeheader()
            output_file.flush()

        for name, port in DATABASES:
            result = load_database(name, port, train)
            writer.writerow(result)
            output_file.flush()

    print(f"\nResults saved to {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
