#!/usr/bin/env bash
# Benchmark bi_s4d. Usage: benchmark_bi_s4d.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=bi_s4d
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
