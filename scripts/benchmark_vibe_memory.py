import argparse
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
import psycopg2

DATASET = Path("datasets/real/vibe/yahoo-minilm-384-normalized.hdf5")
OUTPUT_DIR = Path("results/real/memory_constraint")

REPEATS = 8
WARMUP_SECONDS = 10
MEASURE_SECONDS = 30
CLIENT_COUNTS = [1, 8]

DATABASES = {
    "pgvector": {
        "port": 5433,
        "container": "benchmark-pgvector",
        "setting": "ef_search=60",
    },
    "vectorchord": {
        "port": 5434,
        "container": "benchmark-vectorchord",
        "setting": "epsilon=1.0",
    },
}

FIELDS = [
    "regime", "database", "setting", "repeat", "seed",
    "clients", "warmup_seconds", "measure_seconds",
    "completed_queries", "errors", "elapsed_seconds", "qps",
    "mean_returned", "mean_latency_ms",
    "p50_ms", "p95_ms", "p99_ms",
    "cpu_mean_percent", "cpu_peak_percent",
    "memory_mean_mib", "memory_peak_mib",
    "monitor_samples", "container_memory_limit_bytes",
    "container_memory_swap_limit_bytes",
]


def docker_inspect(container):
    result = subprocess.run(
        [
            "docker", "inspect", container,
            "--format",
            "{{json .HostConfig.Memory}} "
            "{{json .HostConfig.MemorySwap}} "
            "{{json .State.Running}} "
            "{{json .State.OOMKilled}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = result.stdout.strip().split()
    memory, swap = int(values[0]), int(values[1])
    running = values[2].lower() == "true"
    oom = values[3].lower() == "true"
    if not running or oom:
        raise RuntimeError(
            f"{container}: running={running}, oom_killed={oom}"
        )
    return memory, swap


def parse_size_mib(value):
    number = ""
    unit = ""
    for char in value.strip():
        if char.isdigit() or char == ".":
            number += char
        else:
            unit += char
    factors = {
        "B": 1 / 1048576,
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "TiB": 1024 * 1024,
        "kB": 1000 / 1048576,
        "MB": 1000000 / 1048576,
        "GB": 1000000000 / 1048576,
    }
    return float(number) * factors[unit.strip()]


def sample_stats(container):
    result = subprocess.run(
        [
            "docker", "stats", "--no-stream",
            "--format", "{{.CPUPerc}}|{{.MemUsage}}",
            container,
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    line = result.stdout.strip().splitlines()[-1]
    cpu_text, memory_text = line.split("|", 1)
    cpu = float(cpu_text.strip().rstrip("%"))
    memory = parse_size_mib(memory_text.split("/")[0])
    return cpu, memory


def monitor(container, stop, measurements):
    while not stop.is_set():
        try:
            measurements.append(sample_stats(container))
        except Exception as exc:
            print(f"Monitoring warning: {exc}", flush=True)
        stop.wait(1.0)


def connect(database):
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=DATABASES[database]["port"],
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        connect_timeout=15,
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        if database == "pgvector":
            cur.execute("SET hnsw.ef_search = 60")
            cur.execute("SET hnsw.iterative_scan = off")
        else:
            cur.execute("SET vchordrq.epsilon = 1.0")

    return conn


def worker(database, vectors, seed, start_barrier, stop_event,
           warmup_end, deadline):
    rng = random.Random(seed)
    conn = connect(database)
    timings = []
    returned = []
    errors = 0

    sql = (
        "SELECT id FROM vibe_yahoo_384 "
        "ORDER BY embedding <-> %s::vector LIMIT 10"
    )

    try:
        with conn.cursor() as cur:
            start_barrier.wait(timeout=60)

            while not stop_event.is_set():
                now = time.perf_counter()
                if now >= deadline:
                    break

                vector = vectors[rng.randrange(len(vectors))]
                vector_text = "[" + ",".join(
                    str(float(x)) for x in vector
                ) + "]"

                started = time.perf_counter_ns()
                try:
                    cur.execute(sql, (vector_text,))
                    rows = cur.fetchall()
                    finished = time.perf_counter_ns()

                    # Discard all warmup queries.
                    # Requests starting during measurement are included.
                    if started / 1e9 >= warmup_end:
                        timings.append(
                            (finished - started) / 1_000_000
                        )
                        returned.append(len(rows))
                except Exception:
                    errors += 1
                    raise
    finally:
        conn.close()

    return timings, returned, errors


def percentile(values, q):
    return float(np.percentile(values, q)) if values else math.nan


def run_one(regime, database, clients, repeat, vectors):
    container = DATABASES[database]["container"]
    memory_limit, swap_limit = docker_inspect(container)

    # Connect and initialize workers before starting the timed window.
    barrier = threading.Barrier(clients + 1)
    stop = threading.Event()
    measurements = []
    monitor_stop = threading.Event()

    seed = 1000 + repeat
    with ThreadPoolExecutor(max_workers=clients) as pool:
        # Workers wait at barrier after configuring their connections.
        # The deadline is assigned conservatively before barrier release.
        # Actual warmup starts immediately after barrier release.
        start = time.perf_counter()
        warmup_end = start + WARMUP_SECONDS
        deadline = warmup_end + MEASURE_SECONDS

        futures = [
            pool.submit(
                worker, database, vectors, seed * 100 + i,
                barrier, stop, warmup_end, deadline
            )
            for i in range(clients)
        ]

        barrier.wait(timeout=60)

        # Rebase the measurement clock after workers are ready.
        # The first few milliseconds of scheduling delay are negligible
        # compared with the 10-second discarded warmup.
        monitor_thread = threading.Thread(
            target=monitor,
            args=(container, monitor_stop, measurements),
            daemon=True,
        )

        while time.perf_counter() < warmup_end:
            time.sleep(0.05)

        measure_start = time.perf_counter()
        monitor_thread.start()

        while time.perf_counter() < deadline:
            time.sleep(0.05)

        stop.set()
        results = [future.result() for future in futures]
        measure_end = time.perf_counter()

        monitor_stop.set()
        monitor_thread.join(timeout=12)

    timings = [x for item in results for x in item[0]]
    returned = [x for item in results for x in item[1]]
    errors = sum(item[2] for item in results)

    if not timings:
        raise RuntimeError("No measured queries completed")

    if not measurements:
        raise RuntimeError("No Docker monitoring samples collected")

    cpu = [x[0] for x in measurements]
    memory = [x[1] for x in measurements]
    elapsed = measure_end - measure_start

    row = {
        "regime": regime,
        "database": database,
        "setting": DATABASES[database]["setting"],
        "repeat": repeat,
        "seed": seed,
        "clients": clients,
        "warmup_seconds": WARMUP_SECONDS,
        "measure_seconds": MEASURE_SECONDS,
        "completed_queries": len(timings),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 6),
        "qps": round(len(timings) / elapsed, 4),
        "mean_returned": round(statistics.mean(returned), 4),
        "mean_latency_ms": round(statistics.mean(timings), 4),
        "p50_ms": round(percentile(timings, 50), 4),
        "p95_ms": round(percentile(timings, 95), 4),
        "p99_ms": round(percentile(timings, 99), 4),
        "cpu_mean_percent": round(statistics.mean(cpu), 3),
        "cpu_peak_percent": round(max(cpu), 3),
        "memory_mean_mib": round(statistics.mean(memory), 3),
        "memory_peak_mib": round(max(memory), 3),
        "monitor_samples": len(measurements),
        "container_memory_limit_bytes": memory_limit,
        "container_memory_swap_limit_bytes": swap_limit,
    }

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=["unlimited", "limited_1536m"],
        required=True,
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"vibe_memory_{args.regime}.csv"

    with h5py.File(DATASET, "r") as h5:
        vectors = np.asarray(h5["test"][:100], dtype=np.float32)

    expected_limit = (
        0 if args.regime == "unlimited" else 1536 * 1024 * 1024
    )

    for database, config in DATABASES.items():
        memory, swap = docker_inspect(config["container"])
        if memory != expected_limit:
            raise RuntimeError(
                f"{database}: expected memory limit {expected_limit}, "
                f"found {memory}. Refusing to run wrong regime."
            )
        if args.regime == "limited_1536m" and swap != expected_limit:
            raise RuntimeError(
                f"{database}: expected memory+swap {expected_limit}, "
                f"found {swap}."
            )

    existing = set()
    if output.exists():
        with output.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise RuntimeError("Existing CSV schema mismatch")
            for row in reader:
                existing.add((
                    row["database"],
                    int(row["clients"]),
                    int(row["repeat"]),
                ))

    with output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if output.stat().st_size == 0:
            writer.writeheader()
            f.flush()

        for repeat in range(1, REPEATS + 1):
            for database in DATABASES:
                for clients in CLIENT_COUNTS:
                    key = (database, clients, repeat)
                    if key in existing:
                        print(f"SKIP {key}", flush=True)
                        continue

                    print(
                        f"RUN regime={args.regime} "
                        f"repeat={repeat}/{REPEATS} "
                        f"database={database} clients={clients}",
                        flush=True,
                    )

                    row = run_one(
                        args.regime, database, clients,
                        repeat, vectors
                    )

                    writer.writerow(row)
                    f.flush()

                    print(
                        f"  QPS={row['qps']:.2f} "
                        f"p50={row['p50_ms']:.2f}ms "
                        f"p95={row['p95_ms']:.2f}ms "
                        f"p99={row['p99_ms']:.2f}ms "
                        f"CPU={row['cpu_mean_percent']:.1f}% "
                        f"MEM={row['memory_peak_mib']:.1f}MiB",
                        flush=True,
                    )

                    time.sleep(3)

    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()