#!/usr/bin/env bash
# Run benchmarks for every model × every split, one at a time.
# Usage: benchmark_all.sh {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} [OUT_DIR]

set -euo pipefail

DEVICE=${1:?"Usage: $(basename "$0") {jetson-orin-nano|jetson-orin-nx|jetson-agx-orin} [OUT_DIR]"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
    bi_mamba 
    uni_mamba
    bi_s4d 
    uni_s4d
    bi_s4  
    uni_s4
    transformer
    cnn 
    resnet 
    vgg
)

for SPLIT in loso kfold; do
    for MODEL in "${MODELS[@]}"; do
        bash "$SCRIPT_DIR/benchmark_${MODEL}.sh" "$DEVICE" "$SPLIT" ${2:+"$2"}
    done
done
