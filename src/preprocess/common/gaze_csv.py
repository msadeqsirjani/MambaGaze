"""Raw gaze CSV loading and timestamp normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def detect_time_unit(ts: pd.Series) -> float:
    """Infer multiplier to convert timestamps to seconds."""
    s = pd.to_numeric(ts.dropna(), errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return 1.0
    vmax = float(s.max())
    if vmax > 1e17: return 1e-9
    if vmax > 1e14: return 1e-6
    if vmax > 1e11: return 1e-3
    if vmax > 1e4:  return 1e-3
    return 1.0


def to_seconds(ts: pd.Series) -> pd.Series:
    mul = detect_time_unit(ts)
    s   = pd.to_numeric(ts, errors="coerce") * mul
    return s - s.iloc[0]


def read_gaze_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "Timestamp" not in df.columns:
        cands = [c for c in df.columns if "time" in c.lower()]
        if not cands:
            raise ValueError(f"No Timestamp column in {path}")
        df = df.rename(columns={cands[0]: "Timestamp"})
    df["t_sec"] = to_seconds(df["Timestamp"])
    df = df.sort_values("t_sec").reset_index(drop=True)
    # Coalesce duplicate timestamps per column (keep last non-NaN)
    if df["t_sec"].duplicated().any():
        df = df.groupby("t_sec", sort=True).last().reset_index()
    return df
