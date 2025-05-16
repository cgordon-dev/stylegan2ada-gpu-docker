#!/bin/bash

# Set common variables
NETWORK="outputs/network-snapshot-000100.pkl"
BASE_OUTDIR="results_style_mixing"
TRUNC=1
STYLES="0-6"  # Layers to style mix
NOISE_MODE="const"
MASTER_LOG="${BASE_OUTDIR}/master_timing_log.csv"

# Clear previous master log if it exists
if [ -f "$MASTER_LOG" ]; then
    rm "$MASTER_LOG"
fi

# Flag to track first timing log
FIRST_LOG=true

# Define seed combinations you want to test
declare -a ROW_SEEDS_LIST=(
    "1-5"
    "6-10"
)

declare -a COL_SEEDS_LIST=(
    "11-15"
    "16-20"
)

# Loop over seed combinations
for idx in "${!ROW_SEEDS_LIST[@]}"
do
    ROW_SEEDS="${ROW_SEEDS_LIST[$idx]}"
    COL_SEEDS="${COL_SEEDS_LIST[$idx]}"

    echo "=================================================="
    echo "Generating style mixing grid for rows [$ROW_SEEDS] and cols [$COL_SEEDS]..."
    echo "=================================================="

    # Create unique output folder for this run
    OUTDIR="${BASE_OUTDIR}/mix_${idx}"
    mkdir -p "$OUTDIR"

    # Run the style mixing script
    python style_mixing.py \
      --network="$NETWORK" \
      --outdir="$OUTDIR" \
      --rows="$ROW_SEEDS" \
      --cols="$COL_SEEDS" \
      --styles="$STYLES" \
      --trunc="$TRUNC" \
      --noise-mode="$NOISE_MODE"

    echo "Done style mixing for combination $idx."
    echo ""

    # Append timing log to master log
    TIMING_LOG="${OUTDIR}/timing_log.csv"

    if [ -f "$TIMING_LOG" ]; then
        if $FIRST_LOG; then
            # Copy header and first row
            cat "$TIMING_LOG" >> "$MASTER_LOG"
            FIRST_LOG=false
        else
            # Skip header line for subsequent files
            tail -n +2 "$TIMING_LOG" >> "$MASTER_LOG"
        fi
    else
        echo "⚠️ Warning: No timing_log.csv found for style mixing combo $idx"
    fi
done

echo "=================================================="
echo "All style mixing runs completed successfully!"
echo "Master timing log saved to: $MASTER_LOG"
echo "=================================================="