"""
Preprocessing for the CLARE eye-tracking dataset.

Produces one .npz per experiment with:
  X_imputed : (N_windows, 500, F=10)   forward-filled feature values
  M_grud    : (N_windows, 500, F)      observation masks  (1 = genuine update)
  D_grud    : (N_windows, 500, F)      time-deltas since last observation (s)
  y         : (N_windows,)             binary cognitive-load label (>=5 -> 1)
  t0        : (N_windows,)             window start time (s from experiment start)
  feature_names : list[str]            F canonical names
  meta      : JSON string              {pid, exp, src_path}

Dataset layout expected:
  <root>/gaze/<pid>/gaze_data_experiment_{0,1,2,3}.csv
  <root>/gaze/<pid>/gaze_data_baseline_0.csv
  <root>/labels/<pid>.csv   columns: level_0..level_3, 54 rows each

Usage:
  python -m preprocess.clare \
      --root /data/CLARE \
      --out  /data/CLARE/processed \
      --hz 50 --win 10 --workers 8
"""

from .labels import load_labels
from .pipeline import find_experiment_csvs, process_experiment, process_one

__all__ = ["load_labels", "find_experiment_csvs", "process_experiment", "process_one"]
