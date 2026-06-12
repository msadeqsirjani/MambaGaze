"""CL-Drive per-level processing: gaze CSV -> windowed NPZ."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..common import (
    FEATURE_ORDER,
    PreprocessConfig,
    build_windows,
    extract_features,
    read_gaze_csv,
)
from .labels import load_labels


def find_level_tasks(root: Path, labels_dir: Path, out_dir: Path,
                     cfg: PreprocessConfig) -> List[tuple]:
    """One task per (subject, level) gaze CSV under gaze/."""
    tasks = []
    for pid_dir in sorted((root / "gaze").iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        for level in range(1, 10):
            csv = pid_dir / f"gaze_data_level_{level}.csv"
            if not csv.exists():
                continue
            base = pid_dir / f"gaze_baseline_level_{level}.csv"
            tasks.append((
                str(csv),
                str(base) if base.exists() else None,
                str(labels_dir), str(out_dir),
                pid, level, cfg,
            ))
    return tasks


def process_level(
    csv_path: Path,
    baseline_path: Optional[Path],
    labels_dir: Path,
    out_dir: Path,
    pid: str,
    level: int,
    cfg: PreprocessConfig,
) -> Path:
    df          = read_gaze_csv(csv_path)
    baseline_df = read_gaze_csv(baseline_path) if baseline_path and baseline_path.exists() else None

    X_raw, t_grid = extract_features(df, baseline_df, cfg)
    y10            = load_labels(labels_dir, pid, level)
    X_imp, M, D, y, t0 = build_windows(X_raw, t_grid, y10, cfg)

    if len(y) == 0:
        warnings.warn(f"No windows extracted from {csv_path}")

    meta = json.dumps({"pid": pid, "level": level, "src_path": str(csv_path)})
    out_subj = out_dir / pid
    out_subj.mkdir(parents=True, exist_ok=True)
    out_path = out_subj / f"{pid}_lvl{level}.npz"

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
    csv_path, baseline_path, labels_dir, out_dir, pid, level, cfg = args
    out = process_level(
        Path(csv_path),
        Path(baseline_path) if baseline_path else None,
        Path(labels_dir),
        Path(out_dir),
        pid, level, cfg,
    )
    return str(out)
