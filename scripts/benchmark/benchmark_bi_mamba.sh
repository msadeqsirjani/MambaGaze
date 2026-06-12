#!/usr/bin/env bash
# Benchmark bi_mamba. Usage: benchmark_bi_mamba.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=bi_mamba
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
