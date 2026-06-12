"""Unidirectional S4 (DPLR) classifier (forward-only)."""

from __future__ import annotations
import torch
import torch.nn as nn

from ..common import AttentionPool
from .stack import S4DPLRStack


class UniS4(nn.Module):
    """Unidirectional S4 (DPLR): single forward stack + attention pooling."""

    def __init__(self, in_dim: int = 30, d_model: int = 128,
                 n_layers: int = 4, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.stack = S4DPLRStack(in_dim, d_model, n_layers, d_state, dropout)
        self.pool  = AttentionPool(d_model)
        self.head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        h     = self.stack(x)
        c, a  = self.pool(h)
        logit = self.head(c).squeeze(-1)
        return logit, (a,)
