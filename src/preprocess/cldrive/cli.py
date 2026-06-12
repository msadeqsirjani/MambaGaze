"""CLI: python -m preprocess.cldrive --root /data/CL_Drive --out /data/CL_Drive/processed"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..common import PreprocessConfig, run_parallel, write_kfold_splits, write_loso_splits
from .pipeline import find_level_tasks, process_one


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess CL-Drive eye-tracking data into NPZ windows for S4DGaze."
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

    tasks = find_level_tasks(root, labels_dir, out_dir, cfg)
    if not tasks:
        print("No level CSVs found under gaze/. Check --root.")
        return

    print(f"Found {len(tasks)} level files. Processing with {args.workers} workers...")
    saved = run_parallel(tasks, process_one, args.workers)
    print(f"\nSaved {len(saved)} files to {out_dir}")

    if not args.no_splits:
        splits_path = write_loso_splits(out_dir)
        print(f"LOSO splits -> {splits_path}")
    if args.kfold > 0:
        splits_path = write_kfold_splits(out_dir, k=args.kfold)
        print(f"k-fold splits -> {splits_path}")
