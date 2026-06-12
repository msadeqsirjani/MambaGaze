#!/usr/bin/env bash
# Usage: train_uni_s4d.sh [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
# Fair comparison: all models use the shared dataset config in _common.sh
# (same epochs/batch/lr/wd/dropout and d_model/n_layers). No per-model args.

MODEL=uni_s4d
MODEL_ARGS=""
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
