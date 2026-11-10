
import time
import numpy as np
import psycopg

N = 10_000
DIM = 384
BATCH_SIZE = 500

rng = np.random.default_rng(42)

# Generate float32 vectors and normalize them.
vectors = rng.normal(size=(N, DIM)).astype(np.float32)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# Categories allow future filtered queries.
# 50%, 10%, and 1% selectivity.
records = []
for i, vec in enumerate(vectors):
    records.append((
        i + 1,
        i % 2,
        i % 10,
        i % 100,
        "[" + ",".join(map(str, vec)) + "]",
    ))

databases = {
    "pgvector": 5433,
    "vectorchord": 5434,
}

for name, port in databases.items():
    print(f"\nLoading {name}...")

    start = time.perf_counter()

    with psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
    ) as conn:

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY,
                    category_50 INTEGER NOT NULL,
                    category_10 INTEGER NOT NULL,
                    category_1 INTEGER NOT NULL,
                    embedding vector(384) NOT NULL
                )
            """)

            # Recreate the small test dataset.
            cur.execute("TRUNCATE TABLE items")

            for start_idx in range(0, N, BATCH_SIZE):
                batch = records[start_idx:start_idx + BATCH_SIZE]

                with cur.copy("""
                    COPY items (
                        id, category_50, category_10,
                        category_1, embedding
                    ) FROM STDIN
                """) as copy:
                    for record in batch:
                        copy.write_row(record)

            cur.execute("ANALYZE items")
            cur.execute("SELECT COUNT(*) FROM items")
            count = cur.fetchone()[0]

    elapsed = time.perf_counter() - start
    print(f"{name}: {count:,} vectors loaded in {elapsed:.2f}s")

print("\nDataset preparation complete.")
