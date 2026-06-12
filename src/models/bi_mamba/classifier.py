"""Bidirectional Mamba classifier."""

from __future__ import annotations
import torch
import torch.nn as nn

from ..common import AttentionPool, MambaStack


class BiMambaClassifier(nn.Module):
    """Bidirectional Mamba: forward + time-reversed branches with separate weights."""

    def __init__(self, in_dim: int = 30, d_model: int = 128,
                 n_layers: int = 4, dropout: float = 0.1,
                 ssm_kwargs: dict | None = None):
        super().__init__()
        self.fwd      = MambaStack(in_dim, d_model, n_layers, dropout, ssm_kwargs)
        self.bwd      = MambaStack(in_dim, d_model, n_layers, dropout, ssm_kwargs)
        self.pool_fwd = AttentionPool(d_model)
        self.pool_bwd = AttentionPool(d_model)
        self.head     = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 1))

    def forward(self, x: torch.Tensor):
        h_fwd = self.fwd(x)
        h_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        c_fwd, a_fwd = self.pool_fwd(h_fwd)
        c_bwd, a_bwd = self.pool_bwd(h_bwd)
        logit = self.head(torch.cat([c_fwd, c_bwd], dim=-1)).squeeze(-1)
        return logit, (a_fwd, a_bwd)
