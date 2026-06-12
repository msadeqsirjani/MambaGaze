#!/usr/bin/env bash
# Shared launcher sourced by scripts/train/train_<model>.sh wrappers.
# Wrapper sets MODEL and MODEL_ARGS (or MODEL_ARGS_CLARE / MODEL_ARGS_CLDRIVE),
# and optionally OUT_SUFFIX to keep ablation outputs apart from the main runs.

set -euo pipefail

USAGE="Usage: $(basename "$0") [clare|cldrive] [loso|kfold] [DATA_DIR] [OUT_DIR]
  no args:           all four dataset/split combinations
  one arg:           e.g. 'loso' = both datasets LOSO; 'clare' = clare on both splits
  dataset + split:   single combination"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

run_one() {
    local DATASET=$1 SPLIT=$2 DATA_DIR=$3 OUT_DIR=$4 SKIP_MISSING=$5

    case "$DATASET" in
        clare|cldrive) ;;
        *) echo "Unknown dataset: '$DATASET' (expected clare or cldrive)"; echo "$USAGE"; exit 1 ;;
    esac
    case "$SPLIT" in
        loso|kfold) ;;
        *) echo "Unknown split: '$SPLIT' (expected loso or kfold)"; echo "$USAGE"; exit 1 ;;
    esac

    [[ -n "$DATA_DIR" ]] || DATA_DIR=datasets/$DATASET/processed
    [[ -n "$OUT_DIR"  ]] || OUT_DIR=outputs/${MODEL}${OUT_SUFFIX:-}_${DATASET}_${SPLIT}
    local SPLITS="$DATA_DIR/${SPLIT}_splits.json"

    if [[ ! -f "$SPLITS" ]]; then
        echo "Splits file not found: $SPLITS"
        echo "Generate it with: PYTHONPATH=src python -m preprocess.make_splits --dir $DATA_DIR$([[ $SPLIT == kfold ]] && echo ' --kfold 5')"
        [[ "$SKIP_MISSING" == yes ]] && return 0
        exit 1
    fi

    echo "=== $MODEL | dataset=$DATASET | split=$SPLIT | splits=$SPLITS | out=$OUT_DIR ==="

    # GCL recipe per dataset (CLARE: run_hpsearch.sh; CL-Drive: run_rebuttal_v3.sh).
    local DATASET_ARGS RUN_MODEL_ARGS
    case "$DATASET" in
        clare)
            DATASET_ARGS="--threshold_metric acc \
                          --epochs 100 --patience 15 --batch 64 \
                          --lr 3e-4 --wd 1e-4 --dropout 0.1"
            RUN_MODEL_ARGS=${MODEL_ARGS_CLARE-${MODEL_ARGS:-}}
            ;;
        cldrive)
            DATASET_ARGS="--threshold_metric acc --no-auto_flip \
                          --epochs 300 --patience 25 --batch 64 \
                          --lr 3e-4 --wd 0.01 --dropout 0.2 --val_frac 0.15"
            RUN_MODEL_ARGS=${MODEL_ARGS_CLDRIVE-${MODEL_ARGS:-}}
            ;;
    esac

    # Last flag wins: dataset recipe < model args < EXTRA_ARGS.
    python train.py \
        --model "$MODEL"  \
        --data  "$SPLITS" \
        --out   "$OUT_DIR" \
        $DATASET_ARGS $RUN_MODEL_ARGS ${EXTRA_ARGS:-}

    local EXPECTED
    EXPECTED=$(python -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$SPLITS")
    python aggregate.py --dir "$OUT_DIR" --expected "$EXPECTED"
}

if [[ $# -eq 0 ]]; then
    for DATASET in clare cldrive; do
        for SPLIT in loso kfold; do
            run_one "$DATASET" "$SPLIT" "" "" yes
        done
    done
elif [[ $# -eq 1 ]]; then
    case "$1" in
        loso|kfold)
            for DATASET in clare cldrive; do
                run_one "$DATASET" "$1" "" "" yes
            done ;;
        clare|cldrive)
            for SPLIT in loso kfold; do
                run_one "$1" "$SPLIT" "" "" yes
            done ;;
        *)
            echo "Unknown argument: '$1' (expected clare, cldrive, loso, or kfold)"
            echo "$USAGE"; exit 1 ;;
    esac
else
    run_one "$1" "$2" "${3:-}" "${4:-}" no
fi
