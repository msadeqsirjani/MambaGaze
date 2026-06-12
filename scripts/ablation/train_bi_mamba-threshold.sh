#!/usr/bin/env bash
# Usage: train_bi_mamba-threshold.sh [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
# Ablation: bi_mamba with ONLY per-fold decision-threshold calibration
# (no post-hoc auto-flip, no positive class weight). Hyperparameters follow
# the shared dataset config in _common.sh; flags here override its recipe.

MODEL=bi_mamba
MODEL_ARGS="--optimize_threshold --no-auto_flip --no_pos_weight"
OUT_SUFFIX="-threshold"
source "$(dirname "${BASH_SOURCE[0]}")/../train/_common.sh"
