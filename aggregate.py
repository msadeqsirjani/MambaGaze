#!/usr/bin/env python3
"""
Aggregate per-fold JSON results from LOSO runs into a summary table.

Works for both S4DGaze and baseline experiments — any directory that contains
fold_result_*.json files produced by train.py.

Usage:
  python aggregate.py --dir outputs/s4dgaze_clare
  python aggregate.py --dir outputs/cnn --expected 20
  python aggregate.py --dir outputs/ --recursive   # scan all subdirs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    SKLEARN = True
except ImportError:
    SKLEARN = False


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_dir(result_dir: Path, expected_folds: int = 20) -> Optional[Dict]:
    metrics_dir = result_dir / "metrics"
    fold_files = sorted(metrics_dir.glob("fold_result_*.json")) if metrics_dir.exists() else []
    if not fold_files:
        fold_files = sorted(result_dir.glob("fold_result_*.json"))
    if not fold_files:
        return None

    aucs, accs, f1s, aps = [], [], [], []
    all_probs, all_labels = [], []
    model_name = None

    for ff in fold_files:
        with open(ff) as f:
            data = json.load(f)
        m = data.get("test_metrics", {})
        if "auc" in m and m["auc"] is not None: aucs.append(m["auc"])
        if "acc" in m and m["acc"] is not None: accs.append(m["acc"])
        if "f1"  in m and m["f1"]  is not None: f1s.append(m["f1"])
        if "ap"  in m and m["ap"]  is not None: aps.append(m["ap"])
        all_probs.extend(data.get("test_probs",  []))
        all_labels.extend(data.get("test_labels", []))
        if model_name is None:
            model_name = data.get("model", "s4dgaze")

    summary: Dict = {
        "model":          model_name or str(result_dir.name),
        "result_dir":     str(result_dir),
        "n_folds":        len(fold_files),
        "expected_folds": expected_folds,
        "complete":       len(fold_files) >= expected_folds,
    }

    def _stat(vals, key):
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    _stat(aucs, "AUC")
    _stat(aps,  "AP")
    _stat(accs, "ACC")
    _stat(f1s,  "F1_MACRO")

    # Pooled AUC / AP (concatenate all fold predictions)
    if SKLEARN and all_probs:
        all_p = np.array(all_probs)
        all_l = np.array(all_labels)
        if np.unique(all_l).size == 2:
            try:
                summary["AUC_POOL"] = float(roc_auc_score(all_l, all_p))
                summary["AP_POOL"]  = float(average_precision_score(all_l, all_p))
            except Exception:
                pass

    return summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_summary(s: Dict):
    tag = f"{s['model']}  [{s['n_folds']}/{s['expected_folds']} folds]"
    if not s["complete"]:
        tag += "  *** INCOMPLETE ***"
    print(f"\n===== {tag} =====")
    for key in ("AUC", "AP", "ACC", "F1_MACRO"):
        if key in s:
            print(f"  {key:<12} mean={s[key]['mean']:.4f}  std={s[key]['std']:.4f}")
    for key in ("AUC_POOL", "AP_POOL"):
        if key in s:
            print(f"  {key:<12} {s[key]:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-fold LOSO results into a summary."
    )
    ap.add_argument("--dir",       required=True,
                    help="Directory containing fold_result_*.json files "
                         "(or parent directory if --recursive).")
    ap.add_argument("--expected",  type=int, default=20,
                    help="Expected number of folds (for completeness check).")
    ap.add_argument("--recursive", action="store_true",
                    help="Scan all subdirectories for fold results.")
    ap.add_argument("--json_out",  default=None,
                    help="Optional path to save the summary JSON.")
    args = ap.parse_args()

    root = Path(args.dir)

    if args.recursive:
        # Collect all unique directories that contain fold_result_*.json
        dirs = sorted({p.parent for p in root.rglob("fold_result_*.json")})
    else:
        dirs = [root]

    all_summaries = []
    for d in dirs:
        s = aggregate_dir(d, expected_folds=args.expected)
        if s is None:
            print(f"  [skip] {d}  — no fold results found")
            continue
        print_summary(s)
        all_summaries.append(s)

    if not all_summaries:
        print("No fold results found.")
        sys.exit(1)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(all_summaries if len(all_summaries) > 1 else all_summaries[0],
                      f, indent=2)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
