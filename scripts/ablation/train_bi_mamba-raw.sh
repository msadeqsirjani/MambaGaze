#!/usr/bin/env bash
# Usage: train_bi_mamba-raw.sh [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
# Ablation baseline: bi_mamba with NO post-processing or loss weighting
# (no threshold calibration, no post-hoc auto-flip, no positive class
# weight). Hyperparameters follow the shared dataset config in _common.sh.

MODEL=bi_mamba
MODEL_ARGS="--no-optimize_threshold --no-auto_flip --no_pos_weight"
OUT_SUFFIX="-raw"
source "$(dirname "${BASH_SOURCE[0]}")/../train/_common.sh"
