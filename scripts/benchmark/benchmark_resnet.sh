#!/usr/bin/env bash
# Benchmark resnet. Usage: benchmark_resnet.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=resnet
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
