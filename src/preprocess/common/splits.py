"""Cross-validation split generation: LOSO and subject-grouped k-fold."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


def _paths_by_subject(out_dir: Path) -> Dict[str, List[str]]:
    pid_to_paths: Dict[str, List[str]] = {}
    for p in sorted(out_dir.rglob("*.npz")):
        pid = p.parent.name
        pid_to_paths.setdefault(pid, []).append(str(p))
    return pid_to_paths


def write_loso_splits(out_dir: Path) -> Path:
    pid_to_paths = _paths_by_subject(out_dir)

    splits = {}
    for held in sorted(pid_to_paths):
        train = sorted(p for pid, paths in pid_to_paths.items()
                       if pid != held for p in paths)
        test  = sorted(pid_to_paths[held])
        splits[held] = {"train": train, "test": test}

    splits_path = out_dir / "loso_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits_path


def write_kfold_splits(out_dir: Path, k: int = 5, seed: int = 1337) -> Path:
    """Subject-grouped k-fold: subjects are shuffled and partitioned into k
    folds, so all windows of one subject stay in the same fold (no leakage).
    Same JSON schema as LOSO: {fold_name: {"train": [...], "test": [...]}}."""
    pid_to_paths = _paths_by_subject(out_dir)
    pids = sorted(pid_to_paths)
    if len(pids) < k:
        raise ValueError(f"Need at least k={k} subjects, found {len(pids)}")

    rng = random.Random(seed)
    rng.shuffle(pids)
    folds = [sorted(pids[i::k]) for i in range(k)]

    splits = {}
    for i, fold_pids in enumerate(folds):
        held  = set(fold_pids)
        test  = sorted(p for pid in fold_pids for p in pid_to_paths[pid])
        train = sorted(p for pid in pids if pid not in held
                       for p in pid_to_paths[pid])
        splits[f"fold_{i}"] = {"train": train, "test": test}

    splits_path = out_dir / "kfold_splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits_path
