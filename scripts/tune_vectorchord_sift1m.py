
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path(
    "datasets/real/sift1m/sift-128-euclidean.hdf5"
)

K = 10
NUM_QUERIES = 100
EPSILON_VALUES = [0.5, 1.0, 1.9]


def vector_literal(vector):
    return "[" + ",".join(
        str(float(x)) for x in vector
    ) + "]"


def main():
    with h5py.File(DATASET, "r") as file:
        queries = file["test"][:NUM_QUERIES]
        neighbors = file["neighbors"][:NUM_QUERIES]

    with psycopg.connect(
        host="127.0.0.1",
        port=5434,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")

            for epsilon in EPSILON_VALUES:
                cur.execute(
                    "SELECT set_config("
                    "'vchordrq.epsilon', %s, false)",
                    (str(epsilon),),
                )

                recalls = []
                latencies = []

                print(
                    f"\nTesting epsilon={epsilon}",
                    flush=True,
                )

                for i, query in enumerate(queries):
                    start = time.perf_counter()

                    cur.execute(
                        """
                        SELECT id
                        FROM sift_1m
                        ORDER BY embedding <-> %s::vector
                        LIMIT 10
                        """,
                        (vector_literal(query),),
                    )

                    result_ids = {
                        row[0] for row in cur.fetchall()
                    }

                    latencies.append(
                        (time.perf_counter() - start)
                        * 1000
                    )

                    expected_ids = set(
                        int(x) for x in neighbors[i][:K]
                    )

                    recalls.append(
                        len(
                            result_ids.intersection(
                                expected_ids
                            )
                        ) / K
                    )

                print(
                    f"Recall@10: "
                    f"{np.mean(recalls):.4f}"
                )
                print(
                    f"p50: "
                    f"{np.percentile(latencies, 50):.2f} ms"
                )
                print(
                    f"p95: "
                    f"{np.percentile(latencies, 95):.2f} ms"
                )
                print(
                    f"p99: "
                    f"{np.percentile(latencies, 99):.2f} ms"
                )


if __name__ == "__main__":
    main()
