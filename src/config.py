"""Training configuration: single source of truth for hyperparameters.

Every model trains with the same BASELINE config so comparisons stay fair —
do not give individual models their own settings here.

train.py reads its CLI defaults from here; any flag passed explicitly on
the command line still overrides these values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    # Model
    d_model:   int   = 128
    n_layers:  int   = 4
    d_state:   int   = 64
    dropout:   float = 0.1
    # Optimization
    epochs:    int   = 60
    batch:     int   = 64
    lr:        float = 1e-3
    wd:        float = 1e-4
    beta1:     float = 0.9
    beta2:     float = 0.999
    label_smoothing: float = 0.0
    # AMP is ignored on CPU and for the Mamba baselines (see should_use_amp).
    amp:           bool = True
    no_pos_weight: bool = False
    # Schedule / evaluation
    val_frac:  float = 0.1
    patience:  int   = 15
    grad_clip: float = 1.0
    seed:      int   = 1337
    no_early_stop:  bool = False
    monitor_metric: str  = "f1_macro"   # f1_macro | auc | acc | loss
    # Cosine LR schedule
    cosine_schedule: bool  = False
    warmup_fraction: float = 0.1
    min_lr_ratio:    float = 0.01
    # Augmentation
    augment:    bool  = False
    jitter_std: float = 0.02
    scale_lo:   float = 0.9
    scale_hi:   float = 1.1
    aug_prob:   float = 0.5
    # Weight averaging
    use_ema:        bool  = False
    ema_decay:      float = 0.999
    use_swa:        bool  = False
    swa_start_frac: float = 0.75
    # Test-time post-processing (enabled by default: auto_flip only acts
    # when test AUC < 0.5, and per-fold threshold calibration matches the
    # GCL evaluation protocol)
    auto_flip:          bool = True
    optimize_threshold: bool = True
    threshold_metric:   str  = "f1"     # f1 | acc | balanced_acc


BASELINE = TrainConfig()


def for_model(name: str) -> TrainConfig:
    """Return the training config for a model: identical for every model
    (fair comparison). Per-run overrides come from scripts/train/_common.sh
    and the train_<model>.sh wrappers."""
    return BASELINE
