#!/usr/bin/env bash
# Usage: train_bi_mamba-posthoc.sh [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
# Ablation: bi_mamba with ONLY the post-hoc auto-flip (prediction flip when
# test AUC < 0.5; no threshold calibration, no positive class weight).
# Hyperparameters follow the shared dataset config in _common.sh; flags here
# override its recipe (including CL-Drive's --no-auto_flip).

MODEL=bi_mamba
MODEL_ARGS="--auto_flip --no-optimize_threshold --no_pos_weight"
OUT_SUFFIX="-posthoc"
source "$(dirname "${BASH_SOURCE[0]}")/../train/_common.sh"
