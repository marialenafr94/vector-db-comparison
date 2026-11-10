
import time
from pathlib import Path

import h5py
import numpy as np


DATASET = Path(
    "datasets/real/vibe/yahoo-minilm-384-normalized.hdf5"
)
OUTPUT = Path(
    "results/real/vibe_yahoo_filtered_groundtruth.npz"
)

NUM_QUERIES = 100
K = 10
BATCH_SIZE = 8192

FILTERS = [
    ("category_50", 2),
    ("category_10", 10),
    ("category_1", 100),
]


def exact_top_k(train, query, candidate_ids):
    """Exact L2 top-K over the filtered candidate set."""
    best_ids = np.empty(0, dtype=np.int64)
    best_distances = np.empty(0, dtype=np.float64)

    query64 = np.asarray(query, dtype=np.float64)

    for start in range(0, len(candidate_ids), BATCH_SIZE):
        ids = candidate_ids[start:start + BATCH_SIZE]

        vectors = np.asarray(train[ids], dtype=np.float64)
        differences = vectors - query64
        distances = np.einsum(
            "ij,ij->i", differences, differences
        )

        all_ids = np.concatenate((best_ids, ids))
        all_distances = np.concatenate(
            (best_distances, distances)
        )

        # Stable deterministic tie-breaking by ID.
        order = np.lexsort((all_ids, all_distances))
        selected = order[:K]

        best_ids = all_ids[selected]
        best_distances = all_distances[selected]

    if len(best_ids) != K:
        raise RuntimeError(
            f"Expected {K} exact neighbors, got {len(best_ids)}"
        )

    return best_ids


def main():
    start_time = time.perf_counter()

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    with h5py.File(DATASET, "r") as file:
        train = file["train"]
        test = file["test"]

        if train.shape[1] != 384:
            raise RuntimeError(
                f"Unexpected train dimensions: {train.shape}"
            )

        if len(test) < NUM_QUERIES:
            raise RuntimeError(
                "Dataset contains too few test queries"
            )

        queries = np.asarray(
            test[:NUM_QUERIES], dtype=np.float32
        )

        n_vectors = len(train)
        all_ids = np.arange(n_vectors, dtype=np.int64)

        results = {
            "query_indices": np.arange(
                NUM_QUERIES, dtype=np.int64
            )
        }

        for column, modulus in FILTERS:
            candidate_ids = all_ids[
                all_ids % modulus == 0
            ]

            print(
                f"\n{column}: {len(candidate_ids):,} "
                f"eligible vectors",
                flush=True,
            )

            if len(candidate_ids) < K:
                raise RuntimeError(
                    f"Not enough candidates for {column}"
                )

            neighbors = np.empty(
                (NUM_QUERIES, K),
                dtype=np.int64,
            )

            for i, query in enumerate(queries):
                neighbors[i] = exact_top_k(
                    train, query, candidate_ids
                )

                if (i + 1) % 10 == 0:
                    print(
                        f"  {i + 1}/{NUM_QUERIES} queries",
                        flush=True,
                    )

            results[column] = neighbors

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **results)

    elapsed = time.perf_counter() - start_time

    print(f"\nSaved: {OUTPUT}", flush=True)
    print(f"Elapsed: {elapsed:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
