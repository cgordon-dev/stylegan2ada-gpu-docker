#!/usr/bin/env bash
# metrics/monitor.sh
set -euo pipefail

# Paths inside the metrics-watcher container
DATASET=/workspace/datasets/cifar10.zip
RESULTS_ROOT=/workspace/results
METRICS=${METRICS:-fid50k_full,kid50k_full,pr50k3,is50k,ppl2_wu}
SLEEP_INTERVAL=${SLEEP_INTERVAL:-300}

declare -A LAST_CHECKPOINT

# Process one run directory
process_run() {
  local run="$1"
  local run_dir="${RESULTS_ROOT}/${run}"
  # Find newest network snapshot
  local latest
  latest=$(ls "${run_dir}"/network-snapshot-*.pkl 2>/dev/null | sort | tail -n1 || true)
  if [[ -n "$latest" && "${LAST_CHECKPOINT[$run]}" != "$latest" ]]; then
    echo "[$(date)] Detected new snapshot for '$run': $latest"
    python /metrics/calc_metrics.py \
      --network="$latest" \
      --metrics="$METRICS" \
      --data="$DATASET" \
      --result-dir="${run_dir}/metrics"
    LAST_CHECKPOINT[$run]="$latest"
  fi
}

# Main loop: scan all subdirs under RESULTS_ROOT
while true; do
  for dir in "${RESULTS_ROOT}"/*; do
    if [[ -d "$dir" ]]; then
      run_name=$(basename "$dir")
      process_run "$run_name"
    fi
  done
  sleep "$SLEEP_INTERVAL"
done