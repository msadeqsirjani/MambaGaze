"""
Preprocessing for the CL-Drive eye-tracking dataset.

Dataset layout:
  <root>/gaze/<pid>/gaze_data_level_{1..9}.csv
  <root>/gaze/<pid>/gaze_baseline_level_{1..9}.csv
  <root>/labels/<pid>.csv   columns: time, lvl_1..lvl_9  (18 bins each)

Outputs one .npz per level with the same schema as the CLARE preprocessor:
  X_imputed, M_grud, D_grud, y, t0, feature_names, meta

Usage:
  python -m preprocess.cldrive \
      --root /data/CL_Drive \
      --out  /data/CL_Drive/processed \
      --hz 50 --win 10 --workers 8
"""

from .labels import load_labels
from .pipeline import find_level_tasks, process_level, process_one

__all__ = ["load_labels", "find_level_tasks", "process_level", "process_one"]
