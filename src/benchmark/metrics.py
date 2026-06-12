"""Analytic compute / size metrics: FLOPs, MACs, parameters, model size.

FLOP counts are analytic estimates per single inference (one sample) using
the architecture formulas from the paper; MACs = FLOPs / 2 by the usual
1 MAC = 2 FLOPs convention.
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn


def estimate_flops(arch: str, in_dim: int, seq_len: int = 500,
                   d_model: int = 128, n_layers: int = 4,
                   d_state: int = 64) -> float:
    """Estimated FLOPs for ONE inference (batch size 1)."""
    if arch in ("bi_s4d", "uni_s4d", "bi_s4", "uni_s4"):
        # Per S4/S4D layer: FFT conv O(T log T) per channel; two branches for
        # the bidirectional variants. The DPLR low-rank correction is O(1) extra.
        fft_per_layer = 2 * d_model * seq_len * (seq_len.bit_length() + 1)
        proj_flops    = 2 * in_dim * d_model * seq_len
        branches      = 2 if arch.startswith("bi_") else 1
        total = branches * (proj_flops + n_layers * fft_per_layer)
    elif arch in ("bi_mamba", "uni_mamba"):
        d_inner     = d_model * 2
        ssm_flops   = 6 * d_inner * d_state * seq_len
        mamba_flops = 4 * d_inner * d_model * seq_len + ssm_flops
        branches    = 2 if arch == "bi_mamba" else 1
        total = branches * (2 * in_dim * d_model * seq_len +
                            n_layers * mamba_flops)
    elif arch == "cnn":
        total = (2 * in_dim * 64 * seq_len * 64 +
                 2 * 64  * 128 * (seq_len // 4)  * 32 +
                 2 * 128 * 256 * (seq_len // 16) * 17 +
                 2 * 256 * 512 * (seq_len // 64) * 7 +
                 2 * 512 * 256 + 2 * 256 * 128 + 2 * 128 * 1)
    elif arch == "transformer":
        attn = 4 * (2 * d_model**2) + 2 * d_model**2 * seq_len
        ff   = 2 * d_model * 256 + 2 * 256 * d_model
        total = (2 * in_dim * d_model * seq_len +
                 n_layers * (attn + ff) * seq_len)
    elif arch == "resnet":
        total = (2 * in_dim * 64 * seq_len * 64 +
                 6 * 64  * 64  * (seq_len // 4)  * 3 +
                 6 * 64  * 128 * (seq_len // 8)  * 3 +
                 6 * 128 * 256 * (seq_len // 16) * 3 +
                 2 * 256 * 128 + 2 * 128 * 64 + 2 * 64 * 1)
    elif arch == "vgg":
        total = (2 * 2 * in_dim * 64 * seq_len * 3 +
                 2 * 2 * 64  * 128 * (seq_len // 2)  * 3 +
                 2 * 2 * 128 * 256 * (seq_len // 4)  * 3 +
                 2 * 256 * 128 + 2 * 128 * 1)
    else:
        total = 1.0
    return float(total)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_metrics(model: nn.Module) -> Dict:
    """Parameter / buffer counts and in-memory model size."""
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "parameters":      count_params(model),
        "param_mem_mb":    round(param_bytes / 1024**2, 3),
        "buffer_mem_mb":   round(buffer_bytes / 1024**2, 3),
        "model_size_mb":   round((param_bytes + buffer_bytes) / 1024**2, 3),
    }


def compute_metrics(arch: str, in_dim: int, seq_len: int) -> Dict:
    """Per-inference compute cost."""
    flops = estimate_flops(arch, in_dim, seq_len)
    return {
        "flops_per_inference": flops,
        "macs_per_inference":  flops / 2.0,
        "gflops_per_inference": flops / 1e9,
        "gmacs_per_inference":  flops / 2.0 / 1e9,
    }
