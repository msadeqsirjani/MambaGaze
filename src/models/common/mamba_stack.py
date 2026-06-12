"""Mamba stack shared by the uni- and bidirectional Mamba baselines.

Requires: pip install mamba-ssm  (CUDA GPU required)
"""

from __future__ import annotations
import torch
import torch.nn as nn


def import_mamba():
    try:
        from mamba_ssm import Mamba2
        return Mamba2, "mamba2"
    except ImportError:
        pass
    try:
        from mamba_ssm import Mamba
        return Mamba, "mamba"
    except ImportError as e:
        raise ImportError(
            "mamba-ssm not installed. Run: pip install mamba-ssm"
        ) from e


class MambaStack(nn.Module):
    """Stack of Mamba layers with pre-norm residual connections."""

    def __init__(self, in_dim: int, d_model: int, n_layers: int,
                 dropout: float = 0.0, ssm_kwargs: dict | None = None):
        super().__init__()
        MambaLayer, _ = import_mamba()
        ssm_kwargs = ssm_kwargs or {"d_state": 16, "d_conv": 4, "expand": 2}
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers  = nn.ModuleList([MambaLayer(d_model=d_model, **ssm_kwargs)
                                      for _ in range(n_layers)])
        self.norms   = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + self.drop(layer(norm(h)))
        return h
