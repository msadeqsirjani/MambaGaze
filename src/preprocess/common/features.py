"""Feature extraction: raw gaze dataframe -> 50 Hz feature grid.

Both datasets share the same 10 canonical eye-tracking features
(FEATURE_ORDER); CANONICAL maps each one to the raw CSV column names it may
appear under, in priority order.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig

CANONICAL = {
    "pupil_left":  ["ET_PupilLeft",          "PupilLeft",   "Pupil Left"],
    "pupil_right": ["ET_PupilRight",          "PupilRight",  "Pupil Right"],
    "gaze_x":      ["Interpolated Gaze X",   "Gaze X",      "ET_GazeLeftx"],
    "gaze_y":      ["Interpolated Gaze Y",   "Gaze Y",      "ET_GazeLefty"],
    "gaze_v":      ["Gaze Velocity",          "ET_GazeVelocity"],
    "gaze_a":      ["Gaze Acceleration",      "ET_GazeAcceleration"],
    "blink":       ["Blink detected (binary)","Blink detected", "Blink"],
    "fix_idx":     ["Fixation Index",         "FixationIndex"],
    "sac_idx":     ["Saccade Index",          "SaccadeIndex"],
    "distance":    ["Interpolated Distance",  "Distance"],
}

FEATURE_ORDER = [
    "pupil_left", "pupil_right",
    "gaze_x", "gaze_y",
    "gaze_v", "gaze_a",
    "blink", "fix_flag", "sac_flag",
    "distance",
]   # F = 10


def coalesce(keys: List[str], df: pd.DataFrame) -> Optional[str]:
    for k in keys:
        if k in df.columns:
            return k
    return None


def prune_short_runs(active: np.ndarray, min_len: int) -> np.ndarray:
    out = active.copy()
    i = 0
    while i < len(out):
        if out[i]:
            j = i
            while j < len(out) and out[j]:
                j += 1
            if (j - i) < min_len:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def extract_features(df: pd.DataFrame,
                     baseline_df: Optional[pd.DataFrame],
                     cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X_raw   : [T_grid, 10]  values on 50 Hz grid (NaN where missing)
        t_grid  : [T_grid]      time axis in seconds
    """
    t_raw = df["t_sec"].values.astype(float)
    hz    = cfg.target_hz
    dt    = 1.0 / hz

    # Build uniform time grid
    t_end  = t_raw[-1]
    t_grid = np.arange(0.0, t_end + dt / 2, dt)
    T_grid = len(t_grid)

    def _resample(col_name: Optional[str]) -> np.ndarray:
        """Forward-fill a raw column to the 50 Hz grid."""
        if col_name is None:
            return np.full(T_grid, np.nan, dtype=np.float32)
        raw = pd.to_numeric(df[col_name], errors="coerce").values.astype(float)
        s = pd.Series(raw, index=t_raw)
        resampled = s.reindex(t_grid, method="nearest", tolerance=dt * 1.5)
        ffilled   = resampled.ffill().bfill()
        return ffilled.values.astype(np.float32)

    def _baseline_stats(col_name: Optional[str]) -> Tuple[float, float]:
        """Per-subject baseline statistics for z-scoring."""
        if col_name is None or baseline_df is None:
            return 0.0, 1.0
        if col_name not in baseline_df.columns:
            return 0.0, 1.0
        vals = pd.to_numeric(baseline_df[col_name], errors="coerce").dropna().values
        mu = float(np.nanmean(vals)) if len(vals) else 0.0
        sd = float(np.nanstd(vals))
        return mu, max(sd, 1e-6)

    c_pl   = coalesce(CANONICAL["pupil_left"],  df)
    c_pr   = coalesce(CANONICAL["pupil_right"], df)
    c_gx   = coalesce(CANONICAL["gaze_x"],      df)
    c_gy   = coalesce(CANONICAL["gaze_y"],      df)
    c_gv   = coalesce(CANONICAL["gaze_v"],      df)
    c_ga   = coalesce(CANONICAL["gaze_a"],      df)
    c_bl   = coalesce(CANONICAL["blink"],       df)
    c_fi   = coalesce(CANONICAL["fix_idx"],     df)
    c_si   = coalesce(CANONICAL["sac_idx"],     df)
    c_dist = coalesce(CANONICAL["distance"],    df)

    pl   = _resample(c_pl)
    pr   = _resample(c_pr)
    gx   = _resample(c_gx)
    gy   = _resample(c_gy)
    dist = _resample(c_dist)

    # Velocity / acceleration (prefer explicit columns)
    if c_gv:
        gv = _resample(c_gv)
    else:
        gv = np.sqrt(np.gradient(gx, dt)**2 + np.gradient(gy, dt)**2).astype(np.float32)
    if c_ga:
        ga = _resample(c_ga)
    else:
        ga = np.gradient(gv, dt).astype(np.float32)

    if c_bl:
        blink = (_resample(c_bl) > 0.5).astype(np.float32)
    else:
        blink = np.zeros(T_grid, dtype=np.float32)

    # Fixation / saccade flags from index columns
    if c_fi:
        fi_raw = pd.to_numeric(df[c_fi], errors="coerce").ffill().values
        fi_idx_grid = np.interp(t_grid, t_raw, fi_raw)
        fix_flag  = (fi_idx_grid > 0).astype(np.float32)
    else:
        # I-VT fallback: velocity threshold 80 units/s
        sacc_raw = gv >= 80.0
        sacc_raw = prune_short_runs(sacc_raw, min_len=3)
        fix_flag = prune_short_runs(~sacc_raw, min_len=5).astype(np.float32)

    if c_si:
        si_raw = pd.to_numeric(df[c_si], errors="coerce").ffill().values
        si_idx_grid = np.interp(t_grid, t_raw, si_raw)
        sac_flag = (si_idx_grid > 0).astype(np.float32)
    else:
        sac_flag = (1.0 - fix_flag)

    # Baseline z-score for pupil and distance
    mu_pl, sd_pl   = _baseline_stats(c_pl)
    mu_pr, sd_pr   = _baseline_stats(c_pr)
    mu_di, sd_di   = _baseline_stats(c_dist)
    pl   = (pl   - mu_pl) / sd_pl
    pr   = (pr   - mu_pr) / sd_pr
    dist = (dist - mu_di) / sd_di

    X_raw = np.stack([pl, pr, gx, gy, gv, ga, blink, fix_flag, sac_flag, dist],
                     axis=1).astype(np.float32)
    return X_raw, t_grid.astype(np.float32)
