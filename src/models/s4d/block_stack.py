"""Stack of S4D layers forming one directional branch."""

from __future__ import annotations
import torch
import torch.nn as nn

from .layer import S4DLayer


class S4DBlockStack(nn.Module):
    """Stack of S4D layers with pre-norm residual connections."""

    def __init__(self, in_dim: int, d_model: int, n_layers: int,
                 d_state: int = 64, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layers  = nn.ModuleList(
            [S4DLayer(d_model, d_state=d_state, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.norms   = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        self.drop    = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + self.drop(layer(norm(h)))
        return h
