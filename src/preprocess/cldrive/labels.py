"""CL-Drive label loading: per-level self-reported cognitive load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(labels_dir: Path, pid: str, level: int) -> np.ndarray:
    """Binary labels for one level: 18 ten-second bins, rating >= 5 -> 1."""
    f   = labels_dir / f"{pid}.csv"
    df  = pd.read_csv(f)
    col = f"lvl_{level}"
    if col not in df.columns:
        raise KeyError(f"Column {col!r} not in {f}")
    vals = pd.to_numeric(df[col], errors="coerce").values[:18].astype(float)
    return (vals >= 5).astype(np.int64)
