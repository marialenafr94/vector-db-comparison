
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$HOME/vector-db-comparison"
source .venv/bin/activate

mkdir -p logs

echo "Starting SIFT1M filtered QPS + latency"

python3 -u scripts/benchmark_filtered_qps_8runs.py \
    --dataset sift1m \
    2>&1 | tee logs/sift1m_filtered_qps.log

echo "SIFT1M complete. Starting VIBE Yahoo."

python3 -u scripts/benchmark_filtered_qps_8runs.py \
    --dataset vibe \
    2>&1 | tee logs/vibe_filtered_qps.log

echo "Both filtered QPS benchmarks completed!"
