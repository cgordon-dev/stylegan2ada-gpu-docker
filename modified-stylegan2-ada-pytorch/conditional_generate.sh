#!/bin/bash

# Set common variables
NETWORK="outputs/network-snapshot-000100.pkl"
BASE_OUTDIR="results_class_conditional"
SEEDS="0-35"  # or any range you want
TRUNC=1

# Loop over class labels 0 to 9
for CLASS_IDX in {0..9}
do
    echo "============================================="
    echo "Generating images for class ${CLASS_IDX}..."
    echo "============================================="

    # Create output directory per class
    OUTDIR="${BASE_OUTDIR}/class_${CLASS_IDX}"
    mkdir -p "$OUTDIR"

    # Run the Python script
    python generate.py \
      --network="$NETWORK" \
      --outdir="$OUTDIR" \
      --seeds="$SEEDS" \
      --class="$CLASS_IDX" \
      --trunc="$TRUNC"

    echo "Done generating for class ${CLASS_IDX}."
    echo ""
done

echo "============================================="
echo "All classes generated successfully!"
echo "============================================="