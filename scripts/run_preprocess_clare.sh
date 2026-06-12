#!/usr/bin/env bash
# Preprocess the CLARE dataset into NPZ windows.
# Edit DATA_ROOT and OUT_DIR to match your paths.

set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT=${1:-datasets/clare}
OUT_DIR=${2:-datasets/clare/processed}

PYTHONPATH=src python -m preprocess.clare \
    --root    "$DATA_ROOT" \
    --out     "$OUT_DIR"   \
    --hz      50           \
    --win     10           \
    --workers 8

echo "Preprocessing done. LOSO splits: $OUT_DIR/loso_splits.json"
