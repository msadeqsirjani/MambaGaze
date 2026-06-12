#!/usr/bin/env bash
# Benchmark cnn. Usage: benchmark_cnn.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=cnn
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
