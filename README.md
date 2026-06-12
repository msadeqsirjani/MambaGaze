# S4DGaze

**S4DGaze: Diagonal State Space Models with Explicit Missing Data Modeling for Cognitive Load Assessment from Eye-Gaze Tracking Data**

This repository contains the full implementation for training, evaluating, and edge-deploying S4DGaze and all baseline models reported in the paper.

---

## Repository Layout

```
S4DGaze_new/
├── train.py                     # Unified training CLI (all models, LOSO / k-fold)
├── aggregate.py                 # Aggregate per-fold JSON -> summary
├── src/
│   ├── config.py                # Training hyperparameters — single source of truth,
│   │                            #   identical for all baselines (s4dgaze: paper schedule)
│   ├── models/                  # One sub-package per model, one class per file
│   │   ├── registry.py          #   Declares every model once (ModelSpec): class,
│   │   │                        #     input layout, hyperparameters, output shape.
│   │   │                        #     Train/benchmark scripts derive everything
│   │   │                        #     from it — add a new model here only.
│   │   ├── s4d/                 #   S4DGaze main architecture
│   │   │   ├── kernel.py        #     S4DKernel       — S4D-Lin convolution kernel
│   │   │   ├── layer.py         #     S4DLayer        — S4D-conv + GLU layer
│   │   │   ├── block_stack.py   #     S4DBlockStack   — pre-norm residual stack
│   │   │   ├── attention_pool.py#     AttentionPool   — additive attention pooling
│   │   │   └── model.py         #     S4DGaze         — bidirectional classifier
│   │   ├── common/              #   Shared baseline sub-modules
│   │   │   ├── attention_pool.py#     AttentionPool
│   │   │   ├── mamba_stack.py   #     MambaStack (uni/bi Mamba)
│   │   │   └── positional_encoding.py # PositionalEncoding (Transformer)
│   │   ├── s4/                  #   Unidirectional S4 baseline
│   │   │   ├── kernel.py        #     S4Kernel
│   │   │   ├── layer.py         #     S4Layer
│   │   │   ├── stack.py         #     S4Stack
│   │   │   └── classifier.py    #     S4Classifier
│   │   ├── bi_mamba/            #   Bidirectional Mamba baseline
│   │   │   └── classifier.py    #     BiMambaClassifier
│   │   ├── uni_mamba/           #   Unidirectional Mamba baseline
│   │   │   └── classifier.py    #     UniMambaClassifier
│   │   ├── cnn/                 #   VGG-style 1D CNN (large kernels)
│   │   │   └── model.py         #     CNNClassifier
│   │   ├── transformer/         #   Transformer encoder
│   │   │   └── model.py         #     TransformerClassifier
│   │   ├── resnet/              #   ResNet-style 1D CNN
│   │   │   ├── res_block.py     #     ResBlock1D
│   │   │   └── model.py         #     ResNetClassifier
│   │   └── vgg/                 #   VGG-style 1D CNN (small kernels)
│   │       └── model.py         #     VGGClassifier
│   ├── preprocess/              # One sub-package per dataset
│   │   ├── common/              #   Shared pipeline
│   │   │   ├── config.py        #     PreprocessConfig
│   │   │   ├── gaze_csv.py      #     CSV loading, timestamp normalization
│   │   │   ├── features.py      #     Feature extraction to 50 Hz grid
│   │   │   ├── windows.py       #     Windowing + mask/delta computation
│   │   │   ├── splits.py        #     LOSO / k-fold split generation
│   │   │   └── runner.py        #     Parallel task execution
│   │   ├── clare/               #   CLARE     (see Usage below)
│   │   │   ├── labels.py        #     Label loading (54 bins/experiment)
│   │   │   ├── pipeline.py      #     Per-experiment processing
│   │   │   └── cli.py           #     Command-line entry point
│   │   ├── cldrive/             #   CL-Drive  (see Usage below)
│   │   │   ├── labels.py        #     Label loading (18 bins/level)
│   │   │   ├── pipeline.py      #     Per-level processing
│   │   │   └── cli.py           #     Command-line entry point
│   │   └── make_splits.py       #   Standalone LOSO / k-fold split generation
│   └── benchmark/               # Edge benchmark (run with python -m benchmark)
│       ├── power_monitor.py     #   Jetson power rail reader (INA3221 / hwmon)
│       ├── detect.py            #   Checkpoint arch auto-detection + loading
│       ├── metrics.py           #   FLOPs / MACs / params / model size
│       ├── device.py            #   Host & GPU metadata
│       ├── runner.py            #   Latency / throughput / memory / power / energy
│       └── cli.py               #   Command-line entry point
├── datasets/                    # Raw + processed data (uniform layout per dataset)
│   ├── clare/
│   │   ├── gaze/<pid>/          #   gaze_data_experiment_*.csv, gaze_data_baseline_*.csv
│   │   ├── labels/<pid>.csv     #   level_0..level_3 ratings (54 bins each)
│   │   └── processed/           #   NPZ windows + *_splits.json (created by preprocessing)
│   └── cldrive/
│       ├── gaze/<pid>/          #   gaze_data_level_*.csv, gaze_baseline_level_*.csv
│       ├── labels/<pid>.csv     #   lvl_1..lvl_9 ratings (18 bins each)
│       └── processed/           #   NPZ windows + *_splits.json (created by preprocessing)
├── scripts/
│   ├── train/                   # Per-model training launchers
│   │   ├── _common.sh           #   Shared launcher logic
│   │   └── train_<model>.sh     #   one per model, see Usage below
│   ├── benchmark/               # Per-model edge-benchmark launchers
│   │   ├── _common.sh           #   Shared launcher logic
│   │   └── benchmark_<model>.sh #   one per model, see Usage below
│   ├── run_preprocess_clare.sh
│   ├── run_preprocess_cldrive.sh
│   └── run_edge_bench.sh        # Benchmark all models on one device
├── outputs/                     # Created at runtime
│   └── <run>/
│       ├── models/
│       ├── logs/
│       └── metrics/
├── environment.yml
└── requirements.txt
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate s4dgaze
```

---

## Reproducing Results

### 1. Preprocess

```bash
# CLARE dataset
bash scripts/run_preprocess_clare.sh datasets/clare datasets/clare/processed

# CL-Drive dataset
bash scripts/run_preprocess_cldrive.sh datasets/cldrive datasets/cldrive/processed

# or invoke the preprocessors directly:
PYTHONPATH=src python -m preprocess.clare   --root datasets/clare    --out datasets/clare/processed
PYTHONPATH=src python -m preprocess.cldrive --root datasets/cldrive --out datasets/cldrive/processed
```

### 2. Train (LOSO or k-fold)

Every model has a launcher in `scripts/train/`. Hyperparameters come from
`src/config.py` and are identical for all baselines. DATA_DIR and OUT_DIR
are optional (defaults: `datasets/{clare,cldrive}/processed` and
`outputs/<model>_<dataset>_<split>`):

```bash
# train_<model>.sh {clare|cldrive} {loso|kfold} [DATA_DIR] [OUT_DIR]
bash scripts/train/train_s4dgaze.sh   clare   loso  datasets/clare/processed
bash scripts/train/train_cnn.sh       clare   kfold datasets/clare/processed
bash scripts/train/train_s4.sh        cldrive loso  datasets/cldrive/processed
```

To train all baselines for one dataset, loop over the launchers:

```bash
for M in s4 bi_mamba uni_mamba cnn transformer resnet vgg; do
    bash scripts/train/train_$M.sh clare loso datasets/clare/processed
done
```

`kfold` expects `kfold_splits.json` in DATA_DIR — generate it with
`--kfold K` during preprocessing, or afterwards with:

```bash
PYTHONPATH=src python -m preprocess.make_splits --dir datasets/clare/processed --kfold 5
```

### 3. Aggregate Results

```bash
python aggregate.py --dir outputs/s4dgaze_clare_loso
python aggregate.py --dir outputs --recursive
```

### 4. Edge Benchmark (run on Jetson device)

Each model has a launcher in `scripts/benchmark/` taking the device name
(`jetson-orin-nano`, `jetson-orin-nx`, `jetson-agx-orin`) and the split mode
(`loso`, `kfold`) whose checkpoints to benchmark. Outputs land in
`outputs/benchmark/<device>/.../benchmark_<model>_<device>.json` and include
FLOPs/MACs, parameters, model size, latency percentiles, FPS / throughput /
achieved GFLOP/s, GPU + host memory, board power (idle / load / dynamic),
and energy (mJ per inference, inferences per joule, GFLOPS/W):

```bash
# benchmark_<model>.sh {device} {loso|kfold} [CKPT_DIR] [OUT_DIR]
bash scripts/benchmark/benchmark_s4dgaze.sh jetson-orin-nx   loso
bash scripts/benchmark/benchmark_cnn.sh     jetson-orin-nano kfold outputs/cnn_clare_kfold

# or everything at once:
bash scripts/run_edge_bench.sh jetson-orin-nx loso
```

---

## Single-Model Training (CLI)

All models share one training CLI. Hyperparameter defaults come from
`src/config.py` (identical for all baselines; s4dgaze uses the paper
schedule) and `--inputs` defaults to the model's registry setting (xmd for
the SSM family, x for CNN/Transformer). Any flag overrides the config:

```bash
# S4DGaze on CLARE, single held-out subject (for SLURM array)
python train.py \
    --model s4dgaze \
    --data datasets/clare/processed/loso_splits.json \
    --held_subject 1026 \
    --out  outputs/s4dgaze_clare_loso \
    --amp

# CNN baseline on CLARE, overriding the configured epoch count
python train.py \
    --model cnn \
    --data datasets/clare/processed/loso_splits.json \
    --out  outputs/cnn_clare_loso \
    --epochs 80 --amp
```

---

## XMD Input Encoding

S4DGaze and the S4/Mamba baselines use **XMD** input encoding:

```
Z = [X_imputed | M_grud | D_grud]   shape: (B, T, 3F)
```

- **X** — forward-filled feature values  (F=10 channels)
- **M** — binary observation masks       (1 = genuine sensor update)
- **D** — log-scaled time-deltas         (seconds since last observation)

CNN/Transformer baselines use raw `X_imputed` only (`--inputs x`).
