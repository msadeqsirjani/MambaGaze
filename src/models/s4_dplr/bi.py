"""Bidirectional S4 (DPLR) classifier (forward + time-reversed branches)."""

from __future__ import annotations
import torch
import torch.nn as nn

from ..common import AttentionPool
from .stack import S4DPLRStack


class BiS4(nn.Module):
    """Bidirectional S4 (DPLR): forward + time-reversed stacks with separate
    weights, per-branch attention pooling, concatenated into a linear head."""

    def __init__(self, in_dim: int = 30, d_model: int = 128,
                 n_layers: int = 4, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.fwd      = S4DPLRStack(in_dim, d_model, n_layers, d_state, dropout)
        self.bwd      = S4DPLRStack(in_dim, d_model, n_layers, d_state, dropout)
        self.pool_fwd = AttentionPool(d_model)
        self.pool_bwd = AttentionPool(d_model)
        self.head     = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, z: torch.Tensor):
        h_fwd = self.fwd(z)
        h_bwd = self.bwd(torch.flip(z, dims=[1]))
        h_bwd = torch.flip(h_bwd, dims=[1])
        c_fwd, a_fwd = self.pool_fwd(h_fwd)
        c_bwd, a_bwd = self.pool_bwd(h_bwd)
        logit = self.head(torch.cat([c_fwd, c_bwd], dim=-1)).squeeze(-1)
        return logit, (a_fwd, a_bwd)
