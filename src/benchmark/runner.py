"""Benchmark execution: latency, throughput, memory, power, and energy.

All paper-reportable device-level metrics for one model on one device:

  compute    flops/macs per inference, parameters, model size
  latency    mean/std/min/max/median/p90/p95/p99 (ms)
  throughput FPS (inferences/s), samples/s, achieved GFLOP/s
  memory     parameter/buffer memory, peak GPU allocated/reserved, host peak RSS
  power      idle, load mean/peak/min, dynamic (load - idle)  [Jetson rails]
  energy     energy per inference (mJ), inferences per joule,
             GFLOPs per watt, GFLOPs per joule, total measured energy (J)
"""

from __future__ import annotations

import gc
import resource
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from models import MODELS

from .metrics import compute_metrics, model_size_metrics
from .power_monitor import PowerMonitor


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()


def _latency_stats(times_ms: np.ndarray) -> Dict:
    return {
        "mean_ms":   float(times_ms.mean()),
        "std_ms":    float(times_ms.std()),
        "min_ms":    float(times_ms.min()),
        "max_ms":    float(times_ms.max()),
        "median_ms": float(np.percentile(times_ms, 50)),
        "p90_ms":    float(np.percentile(times_ms, 90)),
        "p95_ms":    float(np.percentile(times_ms, 95)),
        "p99_ms":    float(np.percentile(times_ms, 99)),
    }


def _measure_idle_power(duration_s: float = 3.0, interval_ms: int = 50) -> Dict:
    pm = PowerMonitor(interval_ms=interval_ms)
    pm.start()
    time.sleep(duration_s)
    return pm.stop()


def _energy_metrics(power: Dict, idle: Dict, latency_mean_ms: float,
                    gflops_per_inference: float, batch_size: int,
                    total_time_s: float) -> Dict:
    """Derive energy figures from measured power. Power stats are in mW."""
    load_w = power["total_power_mw"] / 1000.0
    if load_w <= 0:
        return {}
    lat_s          = latency_mean_ms / 1000.0
    energy_inf_j   = load_w * lat_s / batch_size        # per single inference
    out = {
        "energy_per_inference_mj":  energy_inf_j * 1000.0,
        "inferences_per_joule":     1.0 / energy_inf_j,
        "gflops_per_watt":          (gflops_per_inference * batch_size / lat_s) / load_w,
        "gflops_per_joule":         gflops_per_inference / energy_inf_j,
        "total_energy_j":           load_w * total_time_s,
    }
    if idle and idle.get("total_power_mw", 0) > 0:
        dyn_w = max(load_w - idle["total_power_mw"] / 1000.0, 0.0)
        out["idle_power_mw"]    = idle["total_power_mw"]
        out["dynamic_power_mw"] = dyn_w * 1000.0
        if dyn_w > 0:
            dyn_inf_j = dyn_w * lat_s / batch_size
            out["dynamic_energy_per_inference_mj"] = dyn_inf_j * 1000.0
            out["dynamic_inferences_per_joule"]    = 1.0 / dyn_inf_j
    return out


def benchmark_model(model: nn.Module, arch: str, in_dim: int,
                    seq_len: int = 500, batch_size: int = 1,
                    warmup: int = 20, iters: int = 100,
                    monitor_power: bool = True,
                    idle_power_s: float = 3.0) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()

    if MODELS[arch].channels_first:
        x = torch.randn(batch_size, in_dim, seq_len, device=device)
    else:
        x = torch.randn(batch_size, seq_len, in_dim, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    idle = _measure_idle_power(idle_power_s) if monitor_power else {}

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        pm = PowerMonitor(interval_ms=50) if monitor_power else None
        if pm:
            pm.start()

        times = []
        t_run0 = time.perf_counter()
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms
        total_time_s = time.perf_counter() - t_run0

    power = pm.stop() if pm else {}
    times = np.array(times)

    comp    = compute_metrics(arch, in_dim, seq_len)
    size    = model_size_metrics(model)
    latency = _latency_stats(times)

    lat_s = latency["mean_ms"] / 1000.0
    throughput = {
        "fps":             float(batch_size / lat_s),    # inferences per second
        "samples_per_s":   float(batch_size / lat_s),
        "batches_per_s":   float(1.0 / lat_s),
        "perf_gflops":     float(comp["gflops_per_inference"] * batch_size / lat_s),
    }

    memory = dict(size)
    memory["host_peak_rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    if device.type == "cuda":
        memory["peak_gpu_mem_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1024**2, 2)
        memory["peak_gpu_mem_reserved_mb"]  = round(
            torch.cuda.max_memory_reserved() / 1024**2, 2)

    energy = (_energy_metrics(power, idle, latency["mean_ms"],
                              comp["gflops_per_inference"], batch_size,
                              total_time_s)
              if power else {})

    return {
        "model_info": {
            "arch":       arch,
            "in_dim":     in_dim,
            "seq_len":    seq_len,
            "batch_size": batch_size,
            **size,
            **comp,
        },
        "latency_ms": latency,
        "throughput": throughput,
        "memory":     memory,
        "power":      {**({"idle": idle} if idle else {}), **({"load": power} if power else {})},
        "energy":     energy,
        "device":     str(device),
        "iters":      iters,
        "warmup":     warmup,
    }
