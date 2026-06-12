#!/usr/bin/env bash
# Benchmark vgg. Usage: benchmark_vgg.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} {loso|kfold} [CKPT_DIR] [OUT_DIR]

MODEL=vgg
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
