
import re
import csv
import json
import math
import random
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import psycopg

DATASET = Path("datasets/real/vibe/yahoo-minilm-384-normalized.hdf5")
OUTPUT = Path("results/real/vibe_yahoo_qps_final.csv")

WARMUP_SECONDS = 10
MEASURE_SECONDS = 30
CLIENT_COUNTS = [1, 8]
REPEATS = 8
QUERY_COUNT = 1000

CONFIGS = [
    ("pgvector", 5433, "hnsw.ef_search", 20),
    ("pgvector", 5433, "hnsw.ef_search", 40),
    ("pgvector", 5433, "hnsw.ef_search", 60),
    ("pgvector", 5433, "hnsw.ef_search", 100),
    ("pgvector", 5433, "hnsw.ef_search", 200),
    ("vectorchord", 5434, "vchordrq.epsilon", 0.3),
    ("vectorchord", 5434, "vchordrq.epsilon", 0.5),
    ("vectorchord", 5434, "vchordrq.epsilon", 0.7),
    ("vectorchord", 5434, "vchordrq.epsilon", 1.0),
    ("vectorchord", 5434, "vchordrq.epsilon", 1.9),
]

CONTAINERS = {
    "pgvector": "benchmark-pgvector",
    "vectorchord": "benchmark-vectorchord",
}

SQL = """
SELECT id
FROM vibe_yahoo_384
ORDER BY embedding <-> %s::vector
LIMIT 10
"""


def vector_literal(v):
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def connect(port):
    return psycopg.connect(
        host="127.0.0.1",
        port=port,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    )


def read_memory_bytes(value):
    value = value.strip()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)",
        value,
    )

    if match is None:
        raise ValueError(
            f"Unexpected Docker memory format: {value!r}"
        )

    number = float(match.group(1))
    unit = match.group(2)

    factors = {
        "B": 1,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "TiB": 1024**4,
    }

    return number * factors[unit]


def monitor_container(container, stop, samples):
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "docker", "stats", "--no-stream",
                    "--format", "{{json .}}", container
                ],
                capture_output=True,
                text=True,
                timeout=8,
                check=True,
            )
            data = json.loads(result.stdout.strip())
            cpu = float(data["CPUPerc"].replace("%", ""))
            memory_text = data["MemUsage"].split("/")[0]
            memory = read_memory_bytes(memory_text)

            samples.append((time.perf_counter(), cpu, memory))
        except Exception as exc:
            print(f"  Resource sample failed: {exc}", flush=True)

        stop.wait(1)


def worker(
    port,
    parameter,
    value,
    vectors,
    client_id,
    warmup_end,
    measure_end,
    barrier,
):
    rng = random.Random(1000 + client_id)
    order = list(range(len(vectors)))
    rng.shuffle(order)
    position = 0

    measured_latencies = []
    completed = 0
    errors = 0

    with connect(port) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            cur.execute(
                "SELECT set_config(%s, %s, false)",
                (parameter, str(value)),
            )

            barrier.wait()

            while time.perf_counter() < measure_end:
                now = time.perf_counter()

                if position >= len(order):
                    rng.shuffle(order)
                    position = 0

                vector = vectors[order[position]]
                position += 1

                start = time.perf_counter()

                try:
                    cur.execute(SQL, (vector,))
                    cur.fetchall()
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"  Query error: {exc}", flush=True)
                    continue

                end = time.perf_counter()

                # Only count queries that started and
                # completed inside the measurement window.
                if start >= warmup_end and end <= measure_end:
                    completed += 1
                    measured_latencies.append((end - start) * 1000)

    return completed, errors, measured_latencies


def run_one(config, clients, repeat, vectors):
    database, port, parameter, value = config
    container = CONTAINERS[database]

    print(
        f"\n{database} | {parameter}={value} | "
        f"clients={clients} | repeat={repeat}",
        flush=True,
    )

    samples = []
    stop_monitor = threading.Event()

    barrier = threading.Barrier(clients + 1)
    with ThreadPoolExecutor(max_workers=clients) as pool:
        # Start all workers and wait until each has connected
        # and configured its PostgreSQL session.
        futures = [
            pool.submit(
                worker,
                port,
                parameter,
                value,
                vectors,
                client_id,
                # Times are filled below using a shared clock.
                # Workers wait at the barrier first.
                0,
                float("inf"),
                barrier,
            )
            for client_id in range(clients)
        ]

        # We need shared deadlines that are known to workers.
        # Set them through the following clock holder instead
        # of the placeholder values above.
        # This is handled by the worker wrapper in main().
        raise RuntimeError("Replace run_one with run_trial below")


def run_trial(config, clients, repeat, vectors):
    database, port, parameter, value = config
    container = CONTAINERS[database]

    print(
        f"\n{database} | {parameter}={value} | "
        f"clients={clients} | repeat={repeat}",
        flush=True,
    )

    barrier = threading.Barrier(clients + 1)
    clock = {}
    samples = []
    stop_monitor = threading.Event()

    def client_task(client_id):
        rng = random.Random(1000 + client_id)
        order = list(range(len(vectors)))
        rng.shuffle(order)
        position = 0
        completed = 0
        errors = 0
        latencies = []

        with connect(port) as conn:
            with conn.cursor() as cur:
                cur.execute("SET enable_seqscan = off")
                cur.execute(
                    "SELECT set_config(%s, %s, false)",
                    (parameter, str(value)),
                )

                barrier.wait()

                warmup_end = clock["warmup_end"]
                measure_end = clock["measure_end"]

                while time.perf_counter() < measure_end:
                    if position >= len(order):
                        rng.shuffle(order)
                        position = 0

                    vector = vectors[order[position]]
                    position += 1
                    start = time.perf_counter()

                    try:
                        cur.execute(SQL, (vector,))
                        cur.fetchall()
                    except Exception as exc:
                        errors += 1
                        if errors <= 3:
                            print(f"  Query error: {exc}", flush=True)
                        continue

                    end = time.perf_counter()

                    if start >= warmup_end and end <= measure_end:
                        completed += 1
                        latencies.append((end - start) * 1000)

        return completed, errors, latencies

    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [
            pool.submit(client_task, i)
            for i in range(clients)
        ]

        # All clients must be ready before the clock starts.
        # The barrier action initializes deadlines.
        # Use a separate start event to release them safely.
        # Since the workers wait at the barrier, set the
        # deadlines immediately before releasing it.
        start = time.perf_counter() + 0.1
        clock["warmup_end"] = start + WARMUP_SECONDS
        clock["measure_end"] = (
            clock["warmup_end"] + MEASURE_SECONDS
        )

        monitor = threading.Thread(
            target=monitor_container,
            args=(container, stop_monitor, samples),
            daemon=True,
        )
        monitor.start()

        barrier.wait()

        results = [future.result() for future in futures]

        stop_monitor.set()
        monitor.join(timeout=10)

    total_queries = sum(r[0] for r in results)
    total_errors = sum(r[1] for r in results)
    latencies = [x for r in results for x in r[2]]

    # QPS uses the full fixed measurement duration.
    qps = total_queries / MEASURE_SECONDS

    # Keep only samples taken during measurement.
    measurement_samples = [
        s for s in samples
        if clock["warmup_end"] <= s[0] <= clock["measure_end"]
    ]

    cpu_values = [s[1] for s in measurement_samples]
    memory_values = [s[2] for s in measurement_samples]

    row = {
        "database": database,
        "parameter": parameter,
        "value": value,
        "clients": clients,
        "repeat": repeat,
        "warmup_seconds": WARMUP_SECONDS,
        "measure_seconds": MEASURE_SECONDS,
        "completed_queries": total_queries,
        "errors": total_errors,
        "qps": round(qps, 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3)
        if latencies else "",
        "p95_ms": round(float(np.percentile(latencies, 95)), 3)
        if latencies else "",
        "p99_ms": round(float(np.percentile(latencies, 99)), 3)
        if latencies else "",
        "mean_cpu_percent": round(statistics.mean(cpu_values), 2)
        if cpu_values else "",
        "peak_memory_mib": round(max(memory_values) / 1024**2, 2)
        if memory_values else "",
        "resource_samples": len(measurement_samples),
    }

    print(
        f"  QPS={row['qps']} | p50={row['p50_ms']} ms | "
        f"p95={row['p95_ms']} ms | "
        f"CPU={row['mean_cpu_percent']}% | "
        f"peak memory={row['peak_memory_mib']} MiB",
        flush=True,
    )

    return row


def main():
    with h5py.File(DATASET, "r") as file:
        vectors = [
            vector_literal(v)
            for v in file["test"][:QUERY_COUNT]
        ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    trials = [
        (config, clients, repeat)
        for config in CONFIGS
        for clients in CLIENT_COUNTS
        for repeat in range(1, REPEATS + 1)
    ]

    random.Random(42).shuffle(trials)

    with OUTPUT.open("x", newline="") as file:
        writer = None

        for config, clients, repeat in trials:
            row = run_trial(config, clients, repeat, vectors)

            if writer is None:
                writer = csv.DictWriter(
                    file,
                    fieldnames=list(row.keys()),
                )
                writer.writeheader()

            writer.writerow(row)
            file.flush()

    print(f"\nFinished: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
