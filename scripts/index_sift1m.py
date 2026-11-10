
import csv
import time
from pathlib import Path

import psycopg

OUTPUT = Path("results/real/sift1m_index_build.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

CONFIGURATIONS = [
    (
        "pgvector",
        5433,
        "sift_1m_embedding_hnsw",
        """
        CREATE INDEX sift_1m_embedding_hnsw
        ON sift_1m
        USING hnsw (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64)
        """,
    ),
    (
        "vectorchord",
        5434,
        "sift_1m_embedding_vchord",
        """
        CREATE INDEX sift_1m_embedding_vchord
        ON sift_1m
        USING vchordrq (embedding vector_l2_ops)
        """,
    ),
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


def build_index(name, port, index_name, sql):
    print(f"\nBuilding {name} index...", flush=True)

    with connect(port) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM sift_1m"
            )
            count = cur.fetchone()[0]

            if count != 1_000_000:
                raise RuntimeError(
                    f"{name}: expected 1,000,000 "
                    f"vectors, found {count}"
                )

            # Prevent accidentally treating an existing
            # index as a new index build.
            cur.execute(
                """
                SELECT to_regclass(%s)
                """,
                (index_name,),
            )

            if cur.fetchone()[0] is not None:
                raise RuntimeError(
                    f"{name}: index {index_name} "
                    "already exists. Skipping to avoid "
                    "overwriting or misleading timings."
                )

            start = time.perf_counter()
            cur.execute(sql)
            build_seconds = time.perf_counter() - start

            print(
                f"{name}: index built in "
                f"{build_seconds:.2f} seconds",
                flush=True,
            )

            cur.execute("ANALYZE sift_1m")

            cur.execute(
                """
                SELECT pg_relation_size(%s::regclass)
                """,
                (index_name,),
            )
            index_bytes = cur.fetchone()[0]

            cur.execute(
                """
                SELECT pg_total_relation_size('sift_1m')
                """
            )
            total_bytes = cur.fetchone()[0]

    result = {
        "database": name,
        "vectors": count,
        "dimensions": 128,
        "index_name": index_name,
        "index_build_seconds": round(
            build_seconds, 3
        ),
        "index_mb": round(
            index_bytes / 1024 / 1024, 2
        ),
        "total_table_and_indexes_mb": round(
            total_bytes / 1024 / 1024, 2
        ),
    }

    print(result, flush=True)
    return result


def main():
    results = []

    for name, port, index_name, sql in CONFIGURATIONS:
        result = build_index(
            name,
            port,
            index_name,
            sql,
        )

        results.append(result)

        with OUTPUT.open("x", newline="") as file:
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
