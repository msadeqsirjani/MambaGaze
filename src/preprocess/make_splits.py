"""Generate cross-validation split files from an already-preprocessed directory.

Usage:
  python -m preprocess.make_splits --dir /data/CLARE/processed            # LOSO
  python -m preprocess.make_splits --dir /data/CLARE/processed --kfold 5  # k-fold
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import write_kfold_splits, write_loso_splits


def main():
    ap = argparse.ArgumentParser(
        description="Write loso_splits.json / kfold_splits.json from preprocessed NPZ files."
    )
    ap.add_argument("--dir",   required=True, help="Processed dataset directory (contains <pid>/*.npz)")
    ap.add_argument("--kfold", type=int, default=0,
                    help="Also write subject-grouped k-fold splits with this k (0 = LOSO only)")
    ap.add_argument("--seed",  type=int, default=1337)
    args = ap.parse_args()

    out_dir = Path(args.dir)
    print(f"LOSO splits  -> {write_loso_splits(out_dir)}")
    if args.kfold > 0:
        print(f"k-fold splits -> {write_kfold_splits(out_dir, k=args.kfold, seed=args.seed)}")


if __name__ == "__main__":
    main()
