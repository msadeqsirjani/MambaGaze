"""Parallel execution of per-file preprocessing tasks."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Sequence

import numpy as np
from tqdm import tqdm


def run_parallel(tasks: Sequence[tuple], fn: Callable[[tuple], str],
                 workers: int) -> List[str]:
    """Run fn over tasks in a process pool. Each task tuple starts with the
    source CSV path (used for error reporting); fn returns the output path."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    saved: List[str] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            src = futs[fut]
            try:
                out = fut.result()
                saved.append(out)
                with np.load(out, allow_pickle=True) as z:
                    N, T, F = z["X_imputed"].shape
                print(f"  {out}  N={N} T={T} F={F}")
            except Exception as e:
                print(f"  FAILED: {src} -> {e}")
    return saved
