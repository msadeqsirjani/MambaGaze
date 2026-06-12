"""Windowing: feature grid -> fixed-length windows with masks and deltas."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig


def build_mask_delta(arr: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized (mask, delta_sec) for a 2-D array [T, C].
    mask[t,c] = 1 iff arr[t,c] is not NaN (genuine observation).
    delta[t,c] = seconds since last observed value.
    """
    T, C  = arr.shape
    mask  = (~np.isnan(arr)).astype(np.float32)
    idx   = np.arange(T, dtype=np.int32)[:, None]
    obs   = np.where(mask > 0.5, idx, -1)
    last  = np.maximum.accumulate(obs, axis=0)
    delta = (idx - last).astype(np.float32) * dt
    delta[last < 0] = 1e3          # never observed -> large gap
    return mask, delta


def build_windows(
    X_raw: np.ndarray,
    t_grid: np.ndarray,
    y10: np.ndarray,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment X_raw into non-overlapping 10-second windows.

    Returns:
        X_imp : (N, T_win, F)   forward-filled values (NaN filled)
        M     : (N, T_win, F)   observation masks
        D     : (N, T_win, F)   log-scaled time-deltas
        y     : (N,)            binary labels
        t0    : (N,)            window start times
    """
    hz      = cfg.target_hz
    T_win   = int(round(cfg.window_sec * hz))    # 500
    dt      = 1.0 / hz
    n_bins  = len(y10)

    X_list, M_list, D_list, y_list, t0_list = [], [], [], [], []

    start = 0
    while start + T_win <= len(t_grid):
        end    = start + T_win
        t_mid  = float(t_grid[start + T_win // 2])
        bin_i  = int(t_mid // cfg.window_sec)

        if bin_i >= n_bins:
            break

        chunk = X_raw[start:end].copy()   # (T_win, F) — may contain NaN

        mask, delta = build_mask_delta(chunk, dt)
        chunk_imp   = pd.DataFrame(chunk).ffill().bfill().values.astype(np.float32)
        delta_log   = np.log1p(delta).astype(np.float32)

        X_list.append(chunk_imp)
        M_list.append(mask)
        D_list.append(delta_log)
        y_list.append(int(y10[bin_i]))
        t0_list.append(float(t_grid[start]))
        start = end

    if not X_list:
        return (np.empty((0, T_win, X_raw.shape[1]), dtype=np.float32),) * 3 + \
               (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))

    return (
        np.stack(X_list).astype(np.float32),
        np.stack(M_list).astype(np.float32),
        np.stack(D_list).astype(np.float32),
        np.array(y_list,  dtype=np.int64),
        np.array(t0_list, dtype=np.float32),
    )
