#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PYTHON=/home/ubuntu/miniconda3/envs/stylegan2-ada/bin/python
CONDA_LIB="$($PYTHON -c 'import sys; print(sys.prefix)')/lib"

# —— Pre‑flight checks —— #
# Ensure the Conda env’s lib dir is on the loader path
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "▶︎ Pre‑flight checks…"

# 1) Get torch.cuda version
TORCH_CUDA_FULL=$("$PYTHON" - <<EOF
import torch
print(torch.version.cuda or "")
EOF
)
TORCH_CUDA_MAJOR=${TORCH_CUDA_FULL%%.*}
echo "  • PyTorch CUDA version: $TORCH_CUDA_FULL (major=$TORCH_CUDA_MAJOR)"

# 2) Check Python can find libcudart.so.<major>
CUDA_LIB_FOUND=$("$PYTHON" - <<EOF
import ctypes.util
print(ctypes.util.find_library("cudart") or "")
EOF
)
if [[ "$CUDA_LIB_FOUND" == *".so.${TORCH_CUDA_MAJOR}"* ]]; then
  echo "  ✓ Python sees libcudart.so.${TORCH_CUDA_MAJOR} (${CUDA_LIB_FOUND##*/})"
else
  echo "  ✗ ERROR: Python did NOT find libcudart.so.${TORCH_CUDA_MAJOR} (found: $CUDA_LIB_FOUND)"
  echo "    Confirm that '$CONDA_LIB/libcudart.so.${TORCH_CUDA_MAJOR}.*' exists"
  exit 1
fi

# 3) Verify torch.version.cuda matches
if "$PYTHON" - <<EOF
import torch, sys
if not torch.version.cuda or not torch.version.cuda.startswith("$TORCH_CUDA_MAJOR"):
    sys.exit(1)
EOF
then
  echo "  ✓ PyTorch built for CUDA $TORCH_CUDA_FULL"
else
  echo "  ✗ ERROR: PyTorch CUDA mismatch: $TORCH_CUDA_FULL"
  exit 1
fi

# 4) Check metrics.calc_metrics imports
if "$PYTHON" - <<EOF
import metrics.calc_metrics
EOF
then
  echo "  ✓ metrics.calc_metrics module importable"
else
  echo "  ✗ ERROR: Unable to import metrics.calc_metrics"
  exit 1
fi

# 5) Check for snapshots
SNAPS=(outputs/train-mixed/*/network-snapshot-*.pkl)
if (( ${#SNAPS[@]} > 0 )); then
  echo "  ✓ Found ${#SNAPS[@]} snapshot(s)"
else
  echo "  ✗ ERROR: No network-snapshot-*.pkl files under outputs/train-mixed/"
  exit 1
fi

echo "▶︎ All pre‑flight checks passed."
echo

# —— Metrics evaluation loop —— #

# Ensure output dir exists
mkdir -p results/metrics

# The metrics you want
METRICS="fid50k_full,kid50k_full,pr50k3_full"

for SNAP in outputs/train-mixed/*/network-snapshot-*.pkl; do
  [[ -f "$SNAP" ]] || continue

  echo "=== Evaluating snapshot: $SNAP ==="
  RUN_NAME=$(basename "$(dirname "$SNAP")")
  OUT_FILE=results/metrics/"$RUN_NAME".jsonl

  echo "  → writing JSONL to $OUT_FILE"
  "$PYTHON" -m metrics.calc_metrics \
    --metrics="$METRICS" \
    --network="$SNAP" \
    --data=datasets/cifar10.zip \
    --mirror=1 \
  >> "$OUT_FILE"
  echo
done