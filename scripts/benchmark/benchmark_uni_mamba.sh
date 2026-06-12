#!/usr/bin/env bash
# Benchmark uni_mamba. Usage: benchmark_uni_mamba.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=uni_mamba
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
