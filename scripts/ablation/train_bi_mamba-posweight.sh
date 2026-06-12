#!/usr/bin/env bash
# Usage: train_bi_mamba-posweight.sh [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
# Ablation: bi_mamba with ONLY the positive class weight in the BCE loss
# (no threshold calibration, no post-hoc auto-flip). --no-no_pos_weight is
# the BooleanOptionalAction negation of --no_pos_weight, i.e. pos_weight ON.
# Hyperparameters follow the shared dataset config in _common.sh.

MODEL=bi_mamba
MODEL_ARGS="--no-no_pos_weight --no-optimize_threshold --no-auto_flip"
OUT_SUFFIX="-posweight"
source "$(dirname "${BASH_SOURCE[0]}")/../train/_common.sh"
