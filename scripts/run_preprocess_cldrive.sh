#!/usr/bin/env bash
# Preprocess the CL-Drive dataset into NPZ windows.

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT=${1:-datasets/cldrive}
OUT_DIR=${2:-datasets/cldrive/processed}

PYTHONPATH=src python -m preprocess.cldrive \
    --root    "$DATA_ROOT" \
    --out     "$OUT_DIR"   \
    --hz      50           \
    --win     10           \
    --workers 8

echo "Preprocessing done. LOSO splits: $OUT_DIR/loso_splits.json"
