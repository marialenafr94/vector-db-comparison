
import csv
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import psycopg

TABLE = "vibe_yahoo_384"
OUTPUT = Path("results/real/vibe_yahoo_index_build.csv")

MAINTENANCE_WORK_MEM = "512MB"

DATABASES = [
    {
        "name": "pgvector",
        "port": 5433,
        "container": "benchmark-pgvector",
        "index": "vibe_yahoo_384_embedding_hnsw",
        "sql": """
            CREATE INDEX vibe_yahoo_384_embedding_hnsw
            ON vibe_yahoo_384
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 16, ef_construction = 64)
        """,
    },
    {
        "name": "vectorchord",
        "port": 5434,
        "container": "benchmark-vectorchord",
        "index": "vibe_yahoo_384_embedding_vchord",
        "sql": """
            CREATE INDEX vibe_yahoo_384_embedding_vchord
            ON vibe_yahoo_384
            USING vchordrq (embedding vector_l2_ops)
        """,
    },
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


def parse_memory_mib(value):
    # Example: "1024MiB", "1.25GiB", "832.1MiB"
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)\s*",
        value,
    )

    if not match:
        raise ValueError(f"Unknown memory format: {value!r}")

    number = float(match.group(1))
    unit = match.group(2)

    factors = {
        "B": 1 / (1024 * 1024),
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "TiB": 1024 * 1024,
        "kB": 1000 / (1024 * 1024),
        "MB": 1000 * 1000 / (1024 * 1024),
        "GB": 1000 * 1000 * 1000 / (1024 * 1024),
    }

    if unit not in factors:
        raise ValueError(f"Unknown memory unit: {unit}")

    return number * factors[unit]


def sample_container(container):
    result = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            container,
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )

    line = result.stdout.strip().splitlines()[0]
    stats = json.loads(line)

    # Docker reports usage / limit, for example:
    # "1.2GiB / 9.7GiB"
    used = stats["MemUsage"].split("/")[0].strip()

    cpu_text = stats["CPUPerc"].strip().rstrip("%")

    return parse_memory_mib(used), float(cpu_text)


def monitor_container(container, stop_event, samples, errors):
    while not stop_event.is_set():
        try:
            memory_mib, cpu_percent = sample_container(container)
            samples.append((memory_mib, cpu_percent))
        except Exception as exc:
            errors.append(str(exc))

        stop_event.wait(2)


def index_exists(cur, index_name):
    cur.execute(
        "SELECT to_regclass(%s)",
        (f"public.{index_name}",),
    )
    return cur.fetchone()[0] is not None


def relation_size_mib(cur, name):
    cur.execute(
        "SELECT pg_relation_size(%s::regclass)",
        (name,),
    )
    return cur.fetchone()[0] / (1024 * 1024)


def total_size_mib(cur, name):
    cur.execute(
        "SELECT pg_total_relation_size(%s::regclass)",
        (name,),
    )
    return cur.fetchone()[0] / (1024 * 1024)


def build_index(config):
    name = config["name"]
    print(f"\n=== {name}: VIBE index build ===", flush=True)

    with connect(config["port"]) as conn:
        with conn.cursor() as cur:
            if index_exists(cur, config["index"]):
                raise RuntimeError(
                    f"Index {config['index']} already exists. "
                    "Stopping rather than overwriting it."
                )

            cur.execute(
                "SELECT COUNT(*) FROM vibe_yahoo_384"
            )
            count = cur.fetchone()[0]

            if count != 677305:
                raise RuntimeError(
                    f"Unexpected row count: {count}"
                )

            cur.execute(
                "SELECT set_config("
                "'maintenance_work_mem', %s, false)",
                (MAINTENANCE_WORK_MEM,),
            )

            cur.execute("SHOW maintenance_work_mem")
            print(
                "maintenance_work_mem:",
                cur.fetchone()[0],
                flush=True,
            )

            stop_event = threading.Event()
            samples = []
            errors = []

            monitor = threading.Thread(
                target=monitor_container,
                args=(
                    config["container"],
                    stop_event,
                    samples,
                    errors,
                ),
                daemon=True,
            )

            # Capture one initial sample, if possible.
            try:
                memory, cpu = sample_container(
                    config["container"]
                )
                samples.append((memory, cpu))
            except Exception as exc:
                errors.append(str(exc))

            monitor.start()

            start = time.perf_counter()

            try:
                print(
                    "Creating index... this may take a while.",
                    flush=True,
                )
                cur.execute(config["sql"])
                elapsed = time.perf_counter() - start
            finally:
                stop_event.set()
                monitor.join(timeout=20)

            index_mib = relation_size_mib(
                cur, config["index"]
            )
            total_mib = total_size_mib(cur, TABLE)

            peak_memory = (
                max(row[0] for row in samples)
                if samples
                else None
            )
            mean_cpu = (
                sum(row[1] for row in samples) / len(samples)
                if samples
                else None
            )

            result = {
                "database": name,
                "vectors": count,
                "dimensions": 384,
                "index_name": config["index"],
                "maintenance_work_mem": MAINTENANCE_WORK_MEM,
                "index_build_seconds": round(elapsed, 3),
                "index_mib": round(index_mib, 2),
                "total_table_and_indexes_mib": round(
                    total_mib, 2
                ),
                "sampled_peak_container_memory_mib": (
                    round(peak_memory, 2)
                    if peak_memory is not None
                    else ""
                ),
                "mean_sampled_cpu_percent": (
                    round(mean_cpu, 2)
                    if mean_cpu is not None
                    else ""
                ),
                "resource_samples": len(samples),
                "monitor_errors": len(errors),
            }

            print("Result:", result, flush=True)

            if errors:
                print(
                    "Monitoring warnings:",
                    errors[:3],
                    flush=True,
                )

            return result


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "database",
        "vectors",
        "dimensions",
        "index_name",
        "maintenance_work_mem",
        "index_build_seconds",
        "index_mib",
        "total_table_and_indexes_mib",
        "sampled_peak_container_memory_mib",
        "mean_sampled_cpu_percent",
        "resource_samples",
        "monitor_errors",
    ]

    # Keep successful rows if the next build fails.
    existing = set()

    if OUTPUT.exists():
        with OUTPUT.open(newline="") as file:
            existing = {
                row["database"]
                for row in csv.DictReader(file)
            }

    write_header = not OUTPUT.exists()

    with OUTPUT.open("a", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        if write_header:
            writer.writeheader()
            file.flush()

        for config in DATABASES:
            if config["name"] in existing:
                print(
                    f"Skipping recorded build: "
                    f"{config['name']}",
                    flush=True,
                )
                continue

            result = build_index(config)
            writer.writerow(result)
            file.flush()

    print(
        f"\nResults saved to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
