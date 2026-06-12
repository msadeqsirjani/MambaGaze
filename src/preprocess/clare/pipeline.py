"""CLARE per-experiment processing: gaze CSV -> windowed NPZ."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import List

import numpy as np

from ..common import (
    FEATURE_ORDER,
    PreprocessConfig,
    build_windows,
    extract_features,
    read_gaze_csv,
)
from .labels import load_labels


def find_experiment_csvs(root: Path) -> List[Path]:
    csv_paths = sorted(root.glob("gaze/**/gaze_data_experiment_*.csv"))
    if not csv_paths:
        # also try cleaned variants
        csv_paths = sorted(root.glob("gaze/**/gaze_data_experiment_*_clean.csv"))
    return csv_paths


def process_experiment(
    csv_path: Path,
    labels_dir: Path,
    out_dir: Path,
    cfg: PreprocessConfig,
) -> Path:
    # Parse PID and experiment index from path
    pid = None
    for part in csv_path.parts:
        if part.isdigit() and len(part) >= 3:
            pid = part
            break
    m = re.search(r"experiment_(\d+)", csv_path.name)
    if pid is None or m is None:
        raise ValueError(f"Cannot parse pid/exp from {csv_path}")
    exp = int(m.group(1))

    df = read_gaze_csv(csv_path)
    subj_dir = csv_path.parent
    base_candidates = (
        list(subj_dir.glob("gaze_data_baseline_*_clean.csv")) +
        list(subj_dir.glob("gaze_data_baseline_*.csv"))
    )
    baseline_df = read_gaze_csv(base_candidates[0]) if base_candidates else None

    X_raw, t_grid = extract_features(df, baseline_df, cfg)
    y10            = load_labels(labels_dir, pid, exp)
    X_imp, M, D, y, t0 = build_windows(X_raw, t_grid, y10, cfg)

    if len(y) == 0:
        warnings.warn(f"No windows extracted from {csv_path}")

    meta = json.dumps({"pid": pid, "exp": exp, "src_path": str(csv_path)})
    out_subj = out_dir / pid
    out_subj.mkdir(parents=True, exist_ok=True)
    out_path = out_subj / f"{pid}_exp{exp}.npz"

    np.savez_compressed(
        out_path,
        X_imputed     = X_imp,
        M_grud        = M,
        D_grud        = D,
        y             = y,
        t0            = t0,
        feature_names = np.array(FEATURE_ORDER),
        meta          = meta,
    )
    return out_path


def process_one(args: tuple) -> str:
    csv_path, labels_dir, out_dir, cfg = args
    out = process_experiment(Path(csv_path), Path(labels_dir), Path(out_dir), cfg)
    return str(out)
