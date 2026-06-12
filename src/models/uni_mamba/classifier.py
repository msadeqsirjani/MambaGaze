"""Unidirectional Mamba classifier."""

from __future__ import annotations
import torch
import torch.nn as nn

from ..common import AttentionPool, MambaStack


class UniMambaClassifier(nn.Module):
    """Unidirectional Mamba: single forward-pass stack."""

    def __init__(self, in_dim: int = 30, d_model: int = 128,
                 n_layers: int = 4, dropout: float = 0.1,
                 ssm_kwargs: dict | None = None):
        super().__init__()
        self.stack = MambaStack(in_dim, d_model, n_layers, dropout, ssm_kwargs)
        self.pool  = AttentionPool(d_model)
        self.head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        h     = self.stack(x)
        c, a  = self.pool(h)
        logit = self.head(c).squeeze(-1)
        return logit, (a,)
