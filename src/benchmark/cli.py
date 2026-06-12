"""CLI: benchmark every checkpoint in a directory on the current device.

Usage (run on the target device, from the repo root):
  PYTHONPATH=src python -m benchmark \
      --data   outputs/bi_s4d_clare_loso \
      --arch   bi_s4d \
      --device jetson-orin-nx \
      --out    outputs/benchmark/jetson-orin-nx

Known devices: jetson-orin-nano, jetson-orin-nx, jetson-agx-orin.
Writes <out>/benchmark_<arch>_<device>.json with per-checkpoint metrics
(compute, latency, throughput, memory, power, energy) and a summary.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# expandable_segments breaks Jetson CUDA virtual-memory API; keep only GC threshold
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.6")

from models import MODELS

from .detect import load_model
from .device import device_info
from .runner import benchmark_model, clear_gpu

KNOWN_DEVICES = ("jetson-orin-nano", "jetson-orin-nx", "jetson-agx-orin")

_SUMMARY_KEYS = [
    ("latency_ms", "mean_ms"),
    ("throughput", "fps"),
    ("throughput", "perf_gflops"),
    ("energy",     "energy_per_inference_mj"),
    ("energy",     "inferences_per_joule"),
    ("energy",     "gflops_per_watt"),
]


def _summarize(results: list) -> dict:
    summary = {}
    for section, key in _SUMMARY_KEYS:
        vals = [r[section][key] for r in results
                if section in r and key in r.get(section, {})]
        if vals:
            summary[f"{section}.{key}"] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            }
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark model checkpoints on an edge device.")
    ap.add_argument("--data",    required=True,
                    help="Directory containing .pt checkpoint files.")
    ap.add_argument("--arch",    required=True, choices=sorted(MODELS),
                    help="Architecture (fallback when auto-detection fails).")
    ap.add_argument("--device",  required=True,
                    help=f"Device name for the report, e.g. {', '.join(KNOWN_DEVICES)}")
    ap.add_argument("--out",     default=None, help="Output directory.")
    ap.add_argument("--seq_len", type=int, default=500)
    ap.add_argument("--in_dim",  type=int, default=30)
    ap.add_argument("--batch",   type=int, default=1)
    ap.add_argument("--warmup",  type=int, default=20)
    ap.add_argument("--iters",   type=int, default=100)
    ap.add_argument("--no_power", action="store_true",
                    help="Disable power / energy measurement.")
    args = ap.parse_args()

    data_dir = Path(args.data)
    pt_files = sorted(data_dir.rglob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {data_dir}")
        return

    out_dir = Path(args.out) if args.out else Path("outputs/benchmark") / args.device
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    metrics_dir = out_dir / "metrics"
    models_dir = out_dir / "models"
    for d in (logs_dir, metrics_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking {len(pt_files)} checkpoints | arch={args.arch} | "
          f"device={args.device} | cuda={torch.cuda.is_available()}")
    print("=" * 60)

    report = {
        "config": {
            "arch": args.arch, "seq_len": args.seq_len,
            "in_dim": args.in_dim, "batch": args.batch,
            "iters": args.iters, "warmup": args.warmup,
            "timestamp": datetime.now().isoformat(),
        },
        "device_info": device_info(args.device),
        "checkpoints": [],
    }

    for i, pt in enumerate(pt_files):
        print(f"[{i+1}/{len(pt_files)}] {pt.name}")
        clear_gpu()
        model = None
        try:
            model, arch, in_dim = load_model(str(pt), args.arch, in_dim=args.in_dim)
            res = benchmark_model(model, arch, in_dim,
                                  seq_len=args.seq_len,
                                  batch_size=args.batch,
                                  warmup=args.warmup,
                                  iters=args.iters,
                                  monitor_power=not args.no_power)
            res["checkpoint"] = str(pt)
            report["checkpoints"].append(res)
            lat, thr = res["latency_ms"], res["throughput"]
            line = (f"  latency {lat['mean_ms']:.2f}±{lat['std_ms']:.2f} ms | "
                    f"fps={thr['fps']:.1f} | {thr['perf_gflops']:.2f} GFLOP/s | "
                    f"params={res['model_info']['parameters']:,}")
            if res["energy"]:
                line += (f" | {res['energy']['energy_per_inference_mj']:.1f} mJ/inf "
                         f"| {res['energy']['gflops_per_watt']:.2f} GFLOPS/W")
            print(line)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            report["checkpoints"].append({"checkpoint": str(pt), "error": str(e)})
        finally:
            if model is not None:
                del model
            clear_gpu()

    ok = [r for r in report["checkpoints"] if "error" not in r]
    report["summary"] = _summarize(ok)
    if ok:
        print(f"\n{len(ok)}/{len(pt_files)} succeeded | "
              f"avg latency={report['summary']['latency_ms.mean_ms']['mean']:.2f} ms | "
              f"avg fps={report['summary']['throughput.fps']['mean']:.1f}")

    out_path = metrics_dir / f"benchmark_{args.arch}_{args.device}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Output -> {out_path}")
