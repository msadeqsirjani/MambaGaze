"""Device / platform information for benchmark reports."""

from __future__ import annotations

import os
import platform
from typing import Dict

import torch


def device_info(device_name: str) -> Dict:
    """Collect host, accelerator, and software-stack metadata."""
    info: Dict = {
        "device_name":    device_name,      # e.g. jetson-orin-nx
        "hostname":       platform.node(),
        "platform":       platform.platform(),
        "python":         platform.python_version(),
        "torch":          torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count":      os.cpu_count(),
    }
    # Jetson boards expose their model string in the device tree
    dt_model = "/proc/device-tree/model"
    if os.path.exists(dt_model):
        with open(dt_model) as f:
            info["board_model"] = f.read().strip().rstrip("\x00")
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        info.update({
            "gpu_name":         torch.cuda.get_device_name(0),
            "cuda_version":     torch.version.cuda,
            "cudnn_version":    torch.backends.cudnn.version(),
            "gpu_total_mem_mb": round(total_b / 1024**2, 1),
        })
    return info
