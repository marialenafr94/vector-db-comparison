# Vec4 — pgvector HNSW vs VectorChord VChordRQ

A reproducible experimental comparison of PostgreSQL pgvector HNSW
with iterative scans and VectorChord VChordRQ for approximate nearest
neighbor (ANN) vector search.

This repository contains benchmark scripts, experimental results,
and visualizations for the Comparison Between Vector Databases-Vec4 assignment.
Assignments link here: 
[Συστήματα Διαχείρισης Πληροφοριών - ClassProjects](https://helios.ntua.gr/pluginfile.php/171907/mod_resource/content/7/Class_Projects_2026-27.pdf)


## 1. Systems compared

| System | PostgreSQL | Extensions |
|---|---|---|
| pgvector HNSW | 17.11 | vector 0.8.7 |
| VectorChord VChordRQ | 17.5 | vchord 0.5.0, vector 0.8.0 |

Docker images are specified in `docker/docker-compose.yml`.

The PostgreSQL minor versions and vector extension versions differ
between the two environments since they were built from separate docker images.

## 2. Hardware and software

- CPU: AMD Ryzen 7 7730U with Radeon Graphics
- WSL2: 4 cores / 8 logical processors
- SIMD: AVX2, AVX, FMA, SSE4.2
- Host RAM: 16 GB
- WSL2 memory allocation: 10 GB
- Storage: 128 GB external USB SSD
- OS: Ubuntu 26.04.1 LTS under WSL2
- Python: 3.14.4
- Docker: 29.9.2
- Docker Compose: v5.6.1

Exact Python dependency versions are recorded in `requirements.txt`.

## 3. Datasets

| Dataset | Vectors | Dimensions |
|---|---:|---:|
| SIFT1M | 1,000,000 | 128 |
| VIBE Yahoo MiniLM | 677,305 | 384 |

The dataset files must be placed at:

- `datasets/real/sift1m/sift-128-euclidean.hdf5`
- `datasets/real/vibe/yahoo-minilm-384-normalized.hdf5`

The datasets are excluded from Git because of their size.

Sources:

- SIFT1M: ANN-Benchmarks, `https://github.com/erikbern/ann-benchmarks`
- VIBE Yahoo MiniLM: VIBE benchmark, `https://github.com/vector-index-bench/vibe`

Both databases store embeddings using PostgreSQL's `vector` type.

## 4. Installation

Requirements:

- Linux or WSL2
- Docker and Docker Compose
- Python 3.14
- Sufficient disk space for datasets and indexes

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
docker compose -f docker/docker-compose.yml up -d
```

Database connections:

| System | Host | Port | Database |
|---|---|---:|---|
| pgvector | 127.0.0.1 | 5433 | vectors |
| VectorChord | 127.0.0.1 | 5434 | vectors |

The Docker configuration contains local demonstration credentials.
These must not be used for publicly accessible databases.

## 5. Database preparation

Dataset loaders:

- `scripts/load_sift1m.py`
- `scripts/load_vibe_yahoo.py`

Index creation scripts:

- `scripts/index_sift1m.py`
- `scripts/index_vibe_yahoo.py`

## 6. Experimental methodology

The experiments measure:

- Data loading time at small, medium, and full dataset sizes
- Index construction time and index storage
- Peak container memory during index construction
- Recall@10 against ground truth
- Query latency: p50, p95, p99
- Throughput in queries per second (QPS)
- Concurrent clients: 1 and 8
- Filter selectivity: 50%, 10%, and 1%
- pgvector HNSW iterative scans
- Recall–throughput tradeoffs across search parameters
- Performance under Docker memory limits: unlimited, 1536, 1024 and 768 MiB
  (768 MiB is below both the VIBE table size and either index size)

Final benchmark configurations were measured eight times.

### Unfiltered QPS methodology

- Workload: closed-loop
- Warmup: 10 seconds
- Measurement interval: 30 seconds
- Clients: 1 and 8
- Repetitions: 8
- Query order: deterministic random seeds
- CPU and memory monitored during measurement

Open-loop throughput was not evaluated.

### Index configuration

pgvector:

- Index type: HNSW
- Distance: L2
- `m = 16`
- `ef_construction = 64`

VectorChord:

- Index type: VChordRQ
- Distance: L2
- Default index-build parameters in the benchmark script

## 7. Benchmark scripts

| Experiment | Script |
|---|---|
| Data loading | `scripts/benchmark_loading_8runs.py` |
| Index construction | `scripts/benchmark_index_build_8runs.py` |
| SIFT1M recall | `scripts/benchmark_sift1m_recall_8runs.py` |
| VIBE Yahoo recall | `scripts/benchmark_vibe_yahoo_recall_8runs.py` |
| SIFT1M QPS | `scripts/benchmark_sift1m_qps_8runs.py` |
| VIBE Yahoo QPS | `scripts/benchmark_vibe_yahoo_qps.py` |
| SIFT1M filtered recall | `scripts/benchmark_sift1m_filtered_8runs.py` |
| VIBE Yahoo filtered recall | `scripts/benchmark_vibe_yahoo_filtered.py` |
| Filtered QPS | `scripts/benchmark_filtered_qps_8runs.py` |
| Memory-limited QPS (unlimited, 1536 MiB) | `scripts/benchmark_vibe_memory.py --regime {unlimited,limited_1536m}` |
| Memory-limited QPS (1024, 768 MiB) | `scripts/benchmark_vibe_memory_stress.py --regime {limited_1024m,limited_768m}` |
| Filtered ground truth | `scripts/generate_{sift1m,vibe_yahoo}_filtered_groundtruth.py` |
| All plots | `scripts/generate_all_plots.py`, `scripts/plot_vibe_memory_stress.py` |

Some scripts use command-line arguments; others use constants
defined in the source code.

Inspect prerequisites and output paths before execution.

## 8. Final results

### Loading and index construction

- `results/real/loading/loading_8runs.csv`
- `results/real/index_build/index_build_8runs.csv`

### Unfiltered recall

- `results/real/sift1m_recall_8runs.csv`
- `results/real/vibe_yahoo_recall_8runs.csv`

### Unfiltered throughput

- `results/real/sift1m_qps_8runs.csv`
- `results/real/vibe_yahoo_qps_final.csv`

### Filtered recall

- `results/real/sift1m_filtered_8runs.csv`
- `results/real/vibe_yahoo_filtered_8runs.csv`

### Filtered throughput

- `results/real/sift1m_filtered_qps_8runs.csv`
- `results/real/vibe_yahoo_filtered_qps_8runs.csv`

### Memory-constrained experiments

- `results/real/memory_constraint/vibe_memory_unlimited.csv`
- `results/real/memory_constraint/vibe_memory_limited_1536m.csv`
- `results/real/memory_constraint/vibe_memory_limited_1024m.csv`
- `results/real/memory_constraint/vibe_memory_limited_768m.csv`
- `results/real/memory_constraint/diagnostics/` (cgroup v2 counters at 768 MiB)
- `results/real/memory_constraint/plots/`

The memory limit is applied with
`docker update --memory <X> --memory-swap <X> <container>`, so no extra swap
is available. The scripts refuse to run if the container limit does not match
the requested regime.

### Plots

Figures are available in `results/real/plots/` and, with annotations, in
`results/real/plots_annotated/`.

## 9. Reproducibility and limitations

- Eight repetitions were used for final configurations.
- Fixed random seeds were used for query ordering and workloads.
- Benchmark timing depends on caching, CPU scheduling, and system load.
- PostgreSQL and vector extension versions differ between systems.
- The memory limits do not by themselves prove paging; at 768 MiB the
  cgroup `memory.events:max` counter shows heavy reclaim for pgvector,
  but block I/O and major page faults were not measured.
- VectorChord was built with default options (`lists = []`, no IVF
  partitioning), so every query scans all quantized codes. Results do not
  represent VChordRQ with tuned `lists`/`probes`.
- Filtered ANN queries can return incomplete results when
  iterative scanning is disabled.
- Search benchmarks use closed-loop clients.
- Earlier exploratory experiments are separate from final results.

## 10. Repository structure

```text
docker/                    Docker Compose configuration
scripts/                   Benchmark and preparation scripts
scripts/initial_benchmarking/
                           Preliminary experiments
results/real/              Final benchmark CSV files
results/real/plots/        Final visualizations
results/real/plots_annotated/
                           Annotated visualizations
results/real/memory_constraint/
                           Memory-limit results, diagnostics, plots
report/                    Final report (PDF, LaTeX source, figures)
requirements.txt           Pinned Python dependencies
datasets/                  Local datasets (not committed)
logs/                      Execution logs (not committed; logs cited by
                           the report are kept under results/real/)
```

## 11. Safe reproduction

Before running benchmarks:

1. Obtain the datasets and place them in the expected directories.
2. Start fresh Docker database containers.
3. Initialize the dataset tables and indexes.
4. Run the relevant benchmark scripts.
5. Use new output CSV paths to preserve submitted results.
6. Aggregate repeated measurements and compare against the included CSVs.

Some scripts create or drop temporary benchmark tables.
Review their database operations before running them.

End-to-end order (SIFT1M shown; use the `vibe_yahoo` scripts for VIBE):

```bash
docker compose -f docker/docker-compose.yml up -d
python scripts/load_sift1m.py
python scripts/index_sift1m.py
python scripts/benchmark_loading_8runs.py --output results/reproduction/loading.csv
python scripts/benchmark_index_build_8runs.py --output results/reproduction/index_build.csv
python scripts/generate_sift1m_filtered_groundtruth.py
python scripts/benchmark_sift1m_recall_8runs.py
python scripts/benchmark_sift1m_qps_8runs.py
python scripts/benchmark_sift1m_filtered_8runs.py
python scripts/benchmark_filtered_qps_8runs.py
# memory experiments (VIBE): set the limit first, e.g.
docker update --memory 768m --memory-swap 768m benchmark-pgvector benchmark-vectorchord
python scripts/benchmark_vibe_memory_stress.py --regime limited_768m
python scripts/generate_all_plots.py
```

Run `python <script> --help` first where available; some scripts use
constants in the source instead of arguments.


## Maintainer safety review (post-experiment)

The source review made defensive changes **after** the submitted measurements
were collected. These changes do not alter ANN search settings or timing
parameters. They have been syntax-checked, but have **not** been validated
by rerunning database benchmarks. See `REPRODUCIBILITY.md` for exact scope.

Use a fresh clone/database and preserve the included final CSVs.
The SHA-256 checksums in section 12 identify the exact HDF5 files used for
the submitted results. The upstream sources do not publish checksums, so a
freshly downloaded file should be compared against these values.

## 12. Submission

- Repository: https://github.com/marialenafr94/vector-db-comparison
- Tag: `v1.0-submission` (commit: `git rev-parse v1.0-submission`)
- Random seeds (all fixed, so every run is reproducible):
  - Recall (unfiltered and filtered) and filtered QPS: seed = 42 + repeat
    (43-50), so each repetition uses a different query order.
  - Unfiltered QPS: client *i* uses seed 1000 + i in every repetition;
    the order of trials is shuffled with seed 42.
  - Memory-limited QPS: seed = 1000 + repeat (1001-1008).
  - Loading and index build involve no randomness.
- Dataset checksums:

  `sift-128-euclidean.hdf5` (525,128,288 bytes)

  - SHA-256: `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`
  - SHA-512: `e25e3cfa1e0863b0104e20fb67f4a98abcb0875ef324b1ded253a3534e9e09f150d5d5a9d190f0a24c926e6b6f957c3ca4a729238579b27aed558397f07bc8bd`

  `yahoo-minilm-384-normalized.hdf5` (1,043,492,672 bytes)

  - SHA-256: `8b0519850249d7ea4aea258eb43ae837dca1bb2f7a85ee0a5145afdb699c4136`
  - SHA-512: `4ea19f8b0c8eafefd5765aa340a971dfb85806647de467f60b6b7f34f62f3311455d5738ed9cdf82708509f1965a8d5bd189c7c5fc034728b98764526c320449`

  Verify on Linux/WSL with `sha256sum` or `sha512sum`, or on Windows with
  `(Get-FileHash <file> -Algorithm SHA512).Hash` (PowerShell prints
  uppercase hex; the comparison is case-insensitive).
