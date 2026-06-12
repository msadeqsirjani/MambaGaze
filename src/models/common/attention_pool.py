"""Additive attention pooling shared across baseline models."""

from __future__ import annotations
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Additive attention pooling with a learned query. Returns (ctx, weights)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor):
        e = self.v(torch.tanh(self.W(h))).squeeze(-1)
        a = torch.softmax(e, dim=1)
        c = (h * a.unsqueeze(-1)).sum(dim=1)
        return c, a
