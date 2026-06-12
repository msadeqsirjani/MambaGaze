#!/usr/bin/env bash
# Shared launcher sourced by every scripts/benchmark/benchmark_<model>.sh wrapper.
# The wrapper sets MODEL, then this script handles arguments and benchmarking.
# Run ON the target device.
#
# Usage: benchmark_<model>.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]
#   CKPT_DIR  directory with the model's .pt checkpoints
#             (default: every outputs/<model>_*_<split> directory)
#   OUT_DIR   output directory (default: outputs/benchmark/<device>)

set -euo pipefail

USAGE="Usage: $(basename "$0") {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]"
DEVICE=${1:?$USAGE}
SPLIT=${2:?$USAGE}

case "$DEVICE" in
    jetson-orin-nano|jetson-orin-nx|jetson-agx-orin) ;;
    *) echo "Unknown device: '$DEVICE'"; echo "$USAGE"; exit 1 ;;
esac

case "$SPLIT" in
    loso|kfold) ;;
    *) echo "Unknown split: '$SPLIT' (expected loso or kfold)"; echo "$USAGE"; exit 1 ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

OUT_DIR=${4:-outputs/benchmark/$DEVICE}

if [[ -n "${3:-}" ]]; then
    CKPT_DIRS=("$3")
else
    mapfile -t CKPT_DIRS < <(find outputs -maxdepth 1 -type d \
        \( -name "${MODEL}_${SPLIT}" -o -name "${MODEL}_*_${SPLIT}" \) 2>/dev/null | sort)
    if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
        echo "No '$SPLIT' checkpoint directories found under outputs/ for model '$MODEL'."
        echo "Train first (scripts/train/train_${MODEL}.sh <dataset> $SPLIT) or pass CKPT_DIR explicitly."
        echo "$USAGE"
        exit 1
    fi
fi

for CKPT_DIR in "${CKPT_DIRS[@]}"; do
    echo "=== $MODEL | device=$DEVICE | split=$SPLIT | checkpoints=$CKPT_DIR | out=$OUT_DIR ==="
    PYTHONPATH=src python -m benchmark \
        --data   "$CKPT_DIR" \
        --arch   "$MODEL"    \
        --device "$DEVICE"   \
        --out    "$OUT_DIR/$(basename "$CKPT_DIR")" \
        --seq_len 500 --in_dim 30 --batch 1 \
        --warmup  20  --iters 100
done
