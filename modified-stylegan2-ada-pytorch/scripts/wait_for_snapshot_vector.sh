#!/usr/bin/env bash
set -e

# Top‑level result directories to watch
RESULTS_DIRS=(
  /workspace/results/train-auto
  /workspace/results/train-cifar
  /workspace/results/train-mixed
)

OUTDIR=/workspace/results/infer-vector
TARGET_IMAGE=/workspace/targets/mytarget.png   # mount your target(s) here

echo "[infer-vector] Watching for snapshots in: ${RESULTS_DIRS[*]}"
while true; do
  latest=$(find "${RESULTS_DIRS[@]}" -type f -name 'network-snapshot-*.pkl' \
           -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
  if [[ -n "$latest" ]]; then
    echo "[infer-vector] Found snapshot: $latest"
    mkdir -p "$OUTDIR"
    python projector.py \
      --outdir="$OUTDIR" \
      --target="$TARGET_IMAGE" \
      --network="$latest"
    exit 0
  else
    echo "[infer-vector] No snapshot yet. Sleeping 10s…"
    sleep 10
  fi
done