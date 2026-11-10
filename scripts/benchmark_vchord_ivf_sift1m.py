"""
VectorChord VChordRQ *with IVF partitioning* on SIFT1M.

Supplementary experiment: the main benchmark used the vchord 0.5.0 default
build options (lists = [], i.e. no partitioning). This script rebuilds the
VectorChord index with `lists = [N]` and sweeps `vchordrq.probes`, using the
same methodology as the main scripts:

  recall : 1000 queries, 50 warm-up queries (discarded), 8 repetitions,
           query order seed = 42 + repeat, configuration order seed = 1042 + repeat
  qps    : closed loop, 1 and 8 clients, 10 s warm-up + 30 s measurement,
           8 repetitions, client i uses seed 1000 + i, trial order seed 42,
           CPU / memory sampled with `docker stats`

Only the VectorChord container (port 5434) is touched.

Steps (run from the repository root, containers up, venv active):

  python scripts/benchmark_vchord_ivf_sift1m.py build      # drops default index, builds IVF index
  python scripts/benchmark_vchord_ivf_sift1m.py recall
  python scripts/benchmark_vchord_ivf_sift1m.py qps
  python scripts/benchmark_vchord_ivf_sift1m.py restore    # back to the default index

`build` DROPS the default index `sift_1m_embedding_vchord`; `restore` drops the
IVF index and recreates the default one exactly as scripts/index_sift1m.py does.
"""

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import psycopg

PORT = 5434
CONTAINER = "benchmark-vectorchord"
TABLE = "sift_1m"
EXPECTED_ROWS = 1_000_000
DEFAULT_INDEX = "sift_1m_embedding_vchord"
IVF_INDEX = "sift_1m_embedding_vchord_ivf"
DATASET = Path("datasets/real/sift1m/sift-128-euclidean.hdf5")
OUT_DIR = Path("results/real/vchord_ivf")

LISTS = 1000                       # ~ sqrt(N) for N = 1M
PROBES = [5, 10, 20, 40, 80]       # 0.5% .. 8% of the lists
EPSILON = 1.9                      # vchord default; override with --epsilon

NUM_QUERIES = 1000
WARMUP_QUERIES = 50
K = 10
SEED = 42
WARMUP_SECONDS = 10
MEASURE_SECONDS = 30
CLIENT_COUNTS = [1, 8]

SQL = f"""
SELECT id
FROM {TABLE}
ORDER BY embedding <-> %s::vector
LIMIT {K}
"""

DEFAULT_INDEX_SQL = f"""
CREATE INDEX {DEFAULT_INDEX}
ON {TABLE}
USING vchordrq (embedding vector_l2_ops)
"""


def ivf_index_sql(lists):
    return f"""
CREATE INDEX {IVF_INDEX}
ON {TABLE}
USING vchordrq (embedding vector_l2_ops)
WITH (options = $$
[build.internal]
lists = [{int(lists)}]
$$)
"""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def connect():
    return psycopg.connect(
        host="127.0.0.1",
        port=PORT,
        dbname="vectors",
        user="benchmark",
        password="benchmark_local",
        autocommit=True,
    )


def vector_literal(v):
    return "[" + ",".join(str(float(x)) for x in v) + "]"


def load_queries():
    with h5py.File(DATASET, "r") as f:
        raw = f["test"][:NUM_QUERIES]
        neighbors = f["neighbors"][:NUM_QUERIES, :K]
    if len(raw) != NUM_QUERIES or neighbors.shape != (NUM_QUERIES, K):
        raise RuntimeError("Unexpected dataset shape.")
    queries = [vector_literal(v) for v in raw]
    truth = [set(int(x) for x in row) for row in neighbors]
    return queries, truth


def index_exists(cur, name):
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{name}",))
    return cur.fetchone()[0]


def read_memory_bytes(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", value.strip())
    if match is None:
        raise ValueError(f"Unexpected Docker memory format: {value!r}")
    factors = {
        "B": 1, "kB": 1000, "MB": 1000**2, "GB": 1000**3,
        "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3,
    }
    return float(match.group(1)) * factors[match.group(2)]


def monitor_container(stop, samples):
    while not stop.is_set():
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", CONTAINER],
                capture_output=True, text=True, timeout=8, check=True,
            )
            data = json.loads(result.stdout.strip())
            cpu = float(data["CPUPerc"].replace("%", ""))
            mem = read_memory_bytes(data["MemUsage"].split("/")[0])
            samples.append((time.perf_counter(), cpu, mem))
        except Exception as exc:  # keep measuring even if one sample fails
            print(f"  Resource sample failed: {exc}", flush=True)
        stop.wait(0.5)


def configure_session(cur, probes):
    cur.execute("SET enable_seqscan = off")
    cur.execute("SELECT set_config('vchordrq.probes', %s, false)", (str(probes),))
    cur.execute("SELECT set_config('vchordrq.epsilon', %s, false)", (str(EPSILON),))
    cur.execute("SET vchordrq.prefilter = off")


def read_index_lists(cur):
    cur.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (IVF_INDEX,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"{IVF_INDEX} does not exist. Run `build` first.")
    match = re.search(r"lists\s*=\s*\[\s*(\d+)\s*\]", row[0])
    if match is None:
        raise SystemExit(f"Cannot read lists from index definition:\n{row[0]}")
    return int(match.group(1))


def verify_ivf_index(cur, query):
    cur.execute("EXPLAIN " + SQL, (query,))
    plan = "\n".join(r[0] for r in cur.fetchall())
    if IVF_INDEX not in plan:
        raise RuntimeError(f"Planner does not use {IVF_INDEX}:\n{plan}")


def suffixed(name):
    """Add an epsilon tag to output names when epsilon differs from 1.9."""
    if EPSILON == 1.9:
        return OUT_DIR / f"{name}.csv"
    return OUT_DIR / f"{name}_eps{EPSILON:g}.csv"


def open_new_csv(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("x", newline="")
    except FileExistsError:
        raise SystemExit(
            f"{path} already exists. Refusing to overwrite; "
            "move it away or delete it first."
        )
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


# --------------------------------------------------------------------------
# build / restore
# --------------------------------------------------------------------------

def cmd_build(args):
    out = OUT_DIR / "sift1m_ivf_index_build.csv"
    run = 2
    while out.exists():
        out = OUT_DIR / f"sift1m_ivf_index_build_run{run}.csv"
        run += 1
    fields = ["lists", "build_s", "index_mib", "peak_container_mib",
              "memory_samples", "maintenance_work_mem"]
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        rows = cur.fetchone()[0]
        if rows != EXPECTED_ROWS:
            raise SystemExit(f"Expected {EXPECTED_ROWS} rows in {TABLE}, found {rows}.")
        if index_exists(cur, IVF_INDEX):
            raise SystemExit(f"{IVF_INDEX} already exists. Run `restore` first.")

        f, writer = open_new_csv(out, fields)

        if index_exists(cur, DEFAULT_INDEX):
            print(f"Dropping default index {DEFAULT_INDEX} ...", flush=True)
            cur.execute(f"DROP INDEX {DEFAULT_INDEX}")

        cur.execute("SHOW maintenance_work_mem")
        mwm = cur.fetchone()[0]

        samples, stop = [], threading.Event()
        mon = threading.Thread(target=monitor_container, args=(stop, samples), daemon=True)
        mon.start()

        print(f"Building {IVF_INDEX} with lists=[{args.lists}] ...", flush=True)
        t0 = time.perf_counter()
        cur.execute(ivf_index_sql(args.lists))
        build_s = time.perf_counter() - t0

        stop.set()
        mon.join(timeout=10)

        cur.execute("SELECT pg_relation_size(%s)", (IVF_INDEX,))
        index_mib = cur.fetchone()[0] / 1024**2

    row = {
        "lists": args.lists,
        "build_s": round(build_s, 3),
        "index_mib": round(index_mib, 3),
        "peak_container_mib": round(max(s[2] for s in samples) / 1024**2, 1) if samples else "",
        "memory_samples": len(samples),
        "maintenance_work_mem": mwm,
    }
    writer.writerow(row)
    f.close()
    print("RESULT", row, flush=True)


def cmd_restore(args):
    with connect() as conn, conn.cursor() as cur:
        if index_exists(cur, IVF_INDEX):
            print(f"Dropping {IVF_INDEX} ...", flush=True)
            cur.execute(f"DROP INDEX {IVF_INDEX}")
        if not index_exists(cur, DEFAULT_INDEX):
            print(f"Recreating default index {DEFAULT_INDEX} ...", flush=True)
            cur.execute(DEFAULT_INDEX_SQL)
    print("Default VectorChord index restored.", flush=True)


# --------------------------------------------------------------------------
# recall / latency (1 client, sequential)
# --------------------------------------------------------------------------

def cmd_recall(args):
    global LISTS_USED
    queries, truth = load_queries()
    with connect() as conn, conn.cursor() as cur:
        LISTS_USED = read_index_lists(cur)
    print(f"{IVF_INDEX}: lists = [{LISTS_USED}]", flush=True)
    out = suffixed("sift1m_ivf_recall_8runs")
    fields = ["dataset", "database", "lists", "probes", "epsilon", "repeat", "seed",
              "queries", "warmup_queries", "recall_at_10", "p50_ms", "p95_ms",
              "p99_ms", "mean_ms", "incomplete_queries"]
    f, writer = open_new_csv(out, fields)

    jobs = []
    for repeat in range(1, args.repeats + 1):
        probes = PROBES.copy()
        random.Random(SEED + 1000 + repeat).shuffle(probes)
        jobs += [(p, repeat) for p in probes]

    for probes, repeat in jobs:
        seed = SEED + repeat
        order = list(range(NUM_QUERIES))
        random.Random(seed).shuffle(order)
        latencies, recalls, incomplete = [], [], 0

        with connect() as conn, conn.cursor() as cur:
            configure_session(cur, probes)
            verify_ivf_index(cur, queries[0])
            for qid in order[:WARMUP_QUERIES]:          # discarded
                cur.execute(SQL, (queries[qid],))
                cur.fetchall()
            for qid in order:
                t0 = time.perf_counter()
                cur.execute(SQL, (queries[qid],))
                ids = [r[0] for r in cur.fetchall()]
                latencies.append((time.perf_counter() - t0) * 1000)
                recalls.append(len(truth[qid].intersection(ids)) / K)
                incomplete += len(ids) < K

        row = {
            "dataset": "sift1m", "database": "vectorchord_ivf", "lists": LISTS_USED,
            "probes": probes, "epsilon": EPSILON, "repeat": repeat, "seed": seed,
            "queries": NUM_QUERIES, "warmup_queries": WARMUP_QUERIES,
            "recall_at_10": round(statistics.mean(recalls), 5),
            "p50_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies, 99)), 3),
            "mean_ms": round(statistics.mean(latencies), 3),
            "incomplete_queries": incomplete,
        }
        writer.writerow(row)
        f.flush()
        print(f"probes={probes:>3} repeat={repeat} recall={row['recall_at_10']:.4f} "
              f"p50={row['p50_ms']} ms p95={row['p95_ms']} ms", flush=True)
    f.close()


# --------------------------------------------------------------------------
# throughput (closed loop, 1 and 8 clients)
# --------------------------------------------------------------------------

def run_trial(probes, clients, repeat, queries):
    barrier = threading.Barrier(clients + 1)
    clock = {}

    def client_task(client_id):
        order = list(range(len(queries)))
        random.Random(1000 + client_id).shuffle(order)
        pos, completed, errors, lats = 0, 0, 0, []
        with connect() as conn, conn.cursor() as cur:
            configure_session(cur, probes)
            barrier.wait()
            warmup_end, measure_end = clock["warmup_end"], clock["measure_end"]
            while time.perf_counter() < measure_end:
                if pos >= len(order):
                    random.Random(1000 + client_id).shuffle(order)
                    pos = 0
                q = queries[order[pos]]
                pos += 1
                t0 = time.perf_counter()
                try:
                    cur.execute(SQL, (q,))
                    cur.fetchall()
                except Exception as exc:
                    errors += 1
                    if errors <= 3:
                        print(f"  Query error: {exc}", flush=True)
                    continue
                t1 = time.perf_counter()
                if t0 >= warmup_end and t1 <= measure_end:
                    completed += 1
                    lats.append((t1 - t0) * 1000)
        return completed, errors, lats

    samples, stop = [], threading.Event()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [pool.submit(client_task, i) for i in range(clients)]
        start = time.perf_counter() + 0.1
        clock["warmup_end"] = start + WARMUP_SECONDS
        clock["measure_end"] = clock["warmup_end"] + MEASURE_SECONDS
        mon = threading.Thread(target=monitor_container, args=(stop, samples), daemon=True)
        mon.start()
        barrier.wait()
        results = [fu.result() for fu in futures]
        stop.set()
        mon.join(timeout=10)

    lats = [x for r in results for x in r[2]]
    meas = [s for s in samples if clock["warmup_end"] <= s[0] <= clock["measure_end"]]
    return {
        "database": "vectorchord_ivf", "lists": LISTS_USED, "probes": probes,
        "epsilon": EPSILON, "clients": clients, "repeat": repeat,
        "warmup_seconds": WARMUP_SECONDS, "measure_seconds": MEASURE_SECONDS,
        "completed_queries": sum(r[0] for r in results),
        "errors": sum(r[1] for r in results),
        "qps": round(sum(r[0] for r in results) / MEASURE_SECONDS, 2),
        "p50_ms": round(float(np.percentile(lats, 50)), 3) if lats else "",
        "p95_ms": round(float(np.percentile(lats, 95)), 3) if lats else "",
        "p99_ms": round(float(np.percentile(lats, 99)), 3) if lats else "",
        "mean_cpu_percent": round(statistics.mean(s[1] for s in meas), 2) if meas else "",
        "peak_memory_mib": round(max(s[2] for s in meas) / 1024**2, 2) if meas else "",
        "resource_samples": len(meas),
    }


def cmd_qps(args):
    global LISTS_USED
    queries, _ = load_queries()
    with connect() as conn, conn.cursor() as cur:
        LISTS_USED = read_index_lists(cur)
        configure_session(cur, PROBES[0])
        verify_ivf_index(cur, queries[0])

    out = suffixed("sift1m_ivf_qps_8runs")
    fields = ["database", "lists", "probes", "epsilon", "clients", "repeat",
              "warmup_seconds", "measure_seconds", "completed_queries", "errors",
              "qps", "p50_ms", "p95_ms", "p99_ms", "mean_cpu_percent",
              "peak_memory_mib", "resource_samples"]
    f, writer = open_new_csv(out, fields)

    trials = [(p, c, r) for p in PROBES for c in CLIENT_COUNTS
              for r in range(1, args.repeats + 1)]
    random.Random(SEED).shuffle(trials)
    total = len(trials)
    for n, (probes, clients, repeat) in enumerate(trials, start=1):
        print(f"[{n}/{total}] probes={probes} clients={clients} repeat={repeat}", flush=True)
        row = run_trial(probes, clients, repeat, queries)
        writer.writerow(row)
        f.flush()
        print(f"  QPS={row['qps']} p50={row['p50_ms']} ms p95={row['p95_ms']} ms "
              f"CPU={row['mean_cpu_percent']}%", flush=True)
    f.close()


# --------------------------------------------------------------------------

LISTS_USED = LISTS


def main():
    global EPSILON
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "recall", "qps", "restore"])
    parser.add_argument("--lists", type=int, default=LISTS,
                        help=f"number of IVF lists for `build` (default {LISTS})")
    parser.add_argument("--repeats", type=int, default=8,
                        help="repetitions for recall/qps (default 8)")
    parser.add_argument("--epsilon", type=float, default=EPSILON,
                        help="vchordrq.epsilon for recall/qps (default 1.9); "
                             "non-default values are written to *_eps<value>.csv")
    args = parser.parse_args()
    EPSILON = args.epsilon

    {"build": cmd_build, "recall": cmd_recall,
     "qps": cmd_qps, "restore": cmd_restore}[args.command](args)


if __name__ == "__main__":
    main()
