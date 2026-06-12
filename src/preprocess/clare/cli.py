"""CLI: python -m preprocess.clare --root /data/CLARE --out /data/CLARE/processed"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..common import PreprocessConfig, run_parallel, write_kfold_splits, write_loso_splits
from .pipeline import find_experiment_csvs, process_one


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess CLARE eye-tracking data into NPZ windows for S4DGaze."
    )
    ap.add_argument("--root",    required=True, help="Dataset root (contains gaze/ and labels/)")
    ap.add_argument("--out",     required=True, help="Output directory for .npz files")
    ap.add_argument("--hz",      type=float, default=50.0)
    ap.add_argument("--win",     type=float, default=10.0)
    ap.add_argument("--workers", type=int,   default=min(8, os.cpu_count() or 1))
    ap.add_argument("--no_splits", action="store_true",
                    help="Skip writing loso_splits.json")
    ap.add_argument("--kfold",   type=int, default=0,
                    help="Also write subject-grouped k-fold splits with this k")
    args = ap.parse_args()

    cfg        = PreprocessConfig(target_hz=args.hz, window_sec=args.win)
    root       = Path(args.root)
    out_dir    = Path(args.out)
    labels_dir = root / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_experiment_csvs(root)
    if not csv_paths:
        print("No experiment CSVs found under gaze/. Check --root.")
        return

    print(f"Found {len(csv_paths)} experiment files. Processing with {args.workers} workers...")
    tasks = [(str(p), str(labels_dir), str(out_dir), cfg) for p in csv_paths]
    saved = run_parallel(tasks, process_one, args.workers)
    print(f"\nSaved {len(saved)} files to {out_dir}")

    if not args.no_splits:
        splits_path = write_loso_splits(out_dir)
        print(f"LOSO splits -> {splits_path}")
    if args.kfold > 0:
        splits_path = write_kfold_splits(out_dir, k=args.kfold)
        print(f"k-fold splits -> {splits_path}")
