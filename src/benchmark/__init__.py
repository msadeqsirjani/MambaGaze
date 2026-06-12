"""Edge deployment benchmark for S4DGaze and all baseline models.

Measures and reports every device-level metric used in the paper:
compute (FLOPs / MACs / parameters / model size), latency percentiles,
throughput (FPS, achieved GFLOP/s), memory (GPU peak, host RSS), board
power (idle / load / dynamic, Jetson INA3221 rails), and energy
(mJ per inference, inferences per joule, GFLOPS/W).

Run on the target device:  PYTHONPATH=src python -m benchmark ...
(see cli.py, or the per-model wrappers in scripts/benchmark/)
"""

from .detect import detect_arch, load_model
from .device import device_info
from .metrics import compute_metrics, count_params, estimate_flops, model_size_metrics
from .power_monitor import PowerMonitor
from .runner import benchmark_model, clear_gpu

__all__ = [
    "detect_arch",
    "load_model",
    "device_info",
    "compute_metrics",
    "count_params",
    "estimate_flops",
    "model_size_metrics",
    "PowerMonitor",
    "benchmark_model",
    "clear_gpu",
]
