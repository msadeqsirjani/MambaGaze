"""Dataset preprocessing: raw eye-tracking CSVs -> windowed NPZ files.

One sub-package per dataset:
  clare/   — CLARE     (python -m preprocess.clare)
  cldrive/ — CL-Drive  (python -m preprocess.cldrive)
  common/  — shared pipeline (CSV loading, features, windowing, LOSO splits)
"""
