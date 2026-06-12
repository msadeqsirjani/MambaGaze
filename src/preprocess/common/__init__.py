"""Dataset-agnostic preprocessing pipeline shared by clare/ and cldrive/."""

from .config import PreprocessConfig
from .gaze_csv import detect_time_unit, to_seconds, read_gaze_csv
from .features import CANONICAL, FEATURE_ORDER, coalesce, prune_short_runs, extract_features
from .windows import build_mask_delta, build_windows
from .splits import write_loso_splits, write_kfold_splits
from .runner import run_parallel

__all__ = [
    "PreprocessConfig",
    "detect_time_unit",
    "to_seconds",
    "read_gaze_csv",
    "CANONICAL",
    "FEATURE_ORDER",
    "coalesce",
    "prune_short_runs",
    "extract_features",
    "build_mask_delta",
    "build_windows",
    "write_loso_splits",
    "write_kfold_splits",
    "run_parallel",
]
