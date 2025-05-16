#!/usr/bin/env bash
set -e

# Top‑level result directories to watch
RESULTS_DIRS=(
  /workspace/results/train-auto
  /workspace/results/train-cifar
  /workspace/results/train-mixed
)

OUTDIR=/workspace/results/infer-standard
PORT=8001

echo "[infer-standard] Watching for snapshots in: ${RESULTS_DIRS[*]}"
while true; do
  # find newest snapshot across all RESULTS_DIRS
  latest=$(find "${RESULTS_DIRS[@]}" -type f -name 'network-snapshot-*.pkl' \
           -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -n "$latest" ]]; then
    echo "[infer-standard] Found snapshot: $latest"
    mkdir -p "$OUTDIR"
    python generate.py \
      --outdir="$OUTDIR" \
      --network="$latest" \
      --trunc=1 \
      --seeds=0-35         \
      # add any other generate.py flags you need here
    exit 0
  else
    echo "[infer-standard] No snapshot yet. Sleeping 10s…"
    sleep 10
  fi
done