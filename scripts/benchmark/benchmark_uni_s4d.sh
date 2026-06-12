#!/usr/bin/env bash
# Benchmark uni_s4d. Usage: benchmark_uni_s4d.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=uni_s4d
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
