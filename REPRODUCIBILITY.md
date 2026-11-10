# Reproduction and safety notes

## Important: existing data

Run database initialization and benchmarks only in a disposable PostgreSQL
instance. The scripts are not designed for production databases.

- `load_sift1m.py` and `load_vibe_yahoo.py` refuse to replace existing source tables.
- `benchmark_loading_8runs.py` refuses to start if its dedicated
  `public.bench_loading_vec4` table already exists, creates that table for
  each measurement, and drops it in the same function after that run.
- `benchmark_index_build_8runs.py` uses `public.bench_index_vec4` and
  `public.bench_index_vec4_ann`. It refuses to start when the temporary
  table already exists, but it does drop its own table during a run.
  Do not execute concurrent benchmark processes against the same database.
- Scripts opening final CSV outputs in exclusive-create mode now refuse to
  overwrite an existing file. Select a fresh output location or archive
  existing results before a new run.
- Loading/index-build benchmark scripts append to their output CSV and skip
  completed configurations. Use `--output` with a new path for an independent
  experiment. Existing CSVs are not overwritten, but may be appended to.

## Minimal environment setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
docker compose -f docker/docker-compose.yml up -d
```

Place HDF5 files at the paths in `README.md`. Dataset sources are listed in
`README.md` (section 3) and the SHA-256 / SHA-512 checksums of the exact files
used for the submitted results in section 12. Verify a downloaded file with
`sha256sum` or `sha512sum` before running anything. Dataset preparation
scripts may create tables; review them before execution.

## Validation without running a benchmark

```bash
python -m compileall -q scripts
python scripts/benchmark_index_build_8runs.py --help
python scripts/benchmark_loading_8runs.py --help
```

These checks do not connect to the databases or execute benchmarks.

## Running new experiments

Do not use the included `results/real/` CSV filenames as destinations for
new measurements. Keep new outputs separate, for example under
`results/reproduction/`. A full index-build run is expensive; only run it
when time, storage, and permission to modify disposable benchmark tables
are available. The provided `--smoke` option for the index-build benchmark
uses 10,000 rows but still creates and drops a temporary benchmark table.

The supplementary IVF script `benchmark_vchord_ivf_sift1m.py build` drops the
default VectorChord index on `sift_1m` and builds an IVF index instead. Run
`benchmark_vchord_ivf_sift1m.py restore` afterwards to return the database to
the state used by the main experiments.

## Methodology and limitations

The QPS scripts use closed-loop workers, 10-second warmups, 30-second
measurement windows, 1/8 clients and eight repetitions. Query and trial
ordering use fixed seeds. Closed-loop performance is not open-loop throughput.
PostgreSQL minor and vector extension versions differ across the two images.
Memory-limit runs use container limits of 1536, 1024 and 768 MiB (plus an
unlimited baseline) on VIBE (table 1074 MiB, indexes 1179-1322 MiB). The
1024 and 768 MiB limits are below both the table size and either index size;
pgvector throughput collapses only at 768 MiB, where the cgroup
`memory.events:max` counter shows heavy reclaim for pgvector and none for
VectorChord. Block I/O and major page
faults were not measured, so paging is inferred, not directly observed.

SIFT1M is the integer-valued dataset (every value is a whole number, 0-212)
and VIBE the floating-point one. Neither pgvector nor VectorChord has an
integer vector type, so both are stored as float32 `vector`, which represents
these integers exactly. Smaller storage types such as `halfvec` (float16)
were not evaluated.

## Provenance

The safety edits were made after original measurements and have been checked
by Python compilation, not by rerunning benchmarks. Reproduced numbers may
differ due to caching, OS, container state, and hardware. Compare aggregate
statistics and recall, not exact floating-point equality of timing samples.
