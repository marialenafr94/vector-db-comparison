
import time
from pathlib import Path

import h5py
import numpy as np

DATASET = Path("datasets/real/sift1m/sift-128-euclidean.hdf5")
OUTPUT = Path("results/real/sift1m_filtered_groundtruth.npz")

NUM_QUERIES = 100
K = 10
CHUNK_SIZE = 20000

FILTERS = [
    ("category_50", 2),
    ("category_10", 10),
    ("category_1", 100),
]


def exact_topk(train, queries, modulus):
    """
    Find exact L2 nearest neighbors among train IDs
    satisfying id % modulus == 0.

    Uses chunked matrix multiplication to limit memory.
    """
    eligible_ids = np.arange(
        0, len(train), modulus, dtype=np.int32
    )

    best_distances = np.full(
        (len(queries), K), np.inf, dtype=np.float64
    )
    best_ids = np.full(
        (len(queries), K), -1, dtype=np.int32
    )

    q = np.asarray(queries, dtype=np.float64)
    q_norms = np.sum(q * q, axis=1, keepdims=True)

    total_chunks = (
        len(eligible_ids) + CHUNK_SIZE - 1
    ) // CHUNK_SIZE

    for chunk_number, start in enumerate(
        range(0, len(eligible_ids), CHUNK_SIZE), 1
    ):
        ids = eligible_ids[start:start + CHUNK_SIZE]
        vectors = np.asarray(train[ids], dtype=np.float64)

        # Squared Euclidean distance:
        # ||q - x||^2 = ||q||^2 + ||x||^2 - 2(q dot x)
        x_norms = np.sum(vectors * vectors, axis=1)
        distances = (
            q_norms
            + x_norms[np.newaxis, :]
            - 2.0 * (q @ vectors.T)
        )
        np.maximum(distances, 0, out=distances)

        # Keep K nearest candidates from this chunk.
        local_positions = np.argpartition(
            distances, K - 1, axis=1
        )[:, :K]

        local_distances = np.take_along_axis(
            distances, local_positions, axis=1
        )
        local_ids = ids[local_positions]

        combined_distances = np.concatenate(
            [best_distances, local_distances], axis=1
        )
        combined_ids = np.concatenate(
            [best_ids, local_ids], axis=1
        )

        positions = np.argpartition(
            combined_distances, K - 1, axis=1
        )[:, :K]

        best_distances = np.take_along_axis(
            combined_distances, positions, axis=1
        )
        best_ids = np.take_along_axis(
            combined_ids, positions, axis=1
        )

        if chunk_number % 5 == 0 or chunk_number == total_chunks:
            print(
                f"  Chunk {chunk_number}/{total_chunks}",
                flush=True,
            )

    # Sort the final K candidates by exact distance.
    order = np.argsort(
        best_distances, axis=1, kind="stable"
    )

    return np.take_along_axis(best_ids, order, axis=1)


def main():
    start_time = time.perf_counter()

    print("Reading SIFT1M...", flush=True)

    with h5py.File(DATASET, "r") as file:
        train = file["train"][:]
        queries = file["test"][:NUM_QUERIES]

    print(
        f"Train: {train.shape}; Queries: {queries.shape}",
        flush=True,
    )

    results = {}

    for name, modulus in FILTERS:
        print(
            f"\nGenerating {name}=0 "
            f"(approximately {100/modulus:g}% eligible)",
            flush=True,
        )

        start = time.perf_counter()
        neighbors = exact_topk(train, queries, modulus)

        # Every returned ID must satisfy the filter.
        assert np.all(neighbors >= 0)
        assert np.all(neighbors % modulus == 0)
        assert neighbors.shape == (NUM_QUERIES, K)

        results[name] = neighbors

        print(
            f"  Completed in "
            f"{time.perf_counter()-start:.1f} seconds",
            flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT,
        **results,
        query_indices=np.arange(NUM_QUERIES),
    )

    print(
        f"\nSaved: {OUTPUT}\n"
        f"Total time: {time.perf_counter()-start_time:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
