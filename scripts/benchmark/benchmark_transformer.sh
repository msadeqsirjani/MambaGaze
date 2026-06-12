#!/usr/bin/env bash
# Benchmark transformer. Usage: benchmark_transformer.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=transformer
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
