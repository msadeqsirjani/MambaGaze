"""Single S4 layer: S4D-conv + skip-D + GELU + GLU output projection."""

from __future__ import annotations
import torch
import torch.nn as nn

from .kernel import S4Kernel


class S4Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0):
        super().__init__()
        self.kernel   = S4Kernel(d_model, d_state=d_state)
        self.D        = nn.Parameter(torch.randn(d_model))
        self.act      = nn.GELU()
        self.drop     = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.out_proj = nn.Linear(d_model, 2 * d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_t = u.transpose(1, 2)
        L   = u_t.size(-1)
        k   = self.kernel(L)
        y   = torch.fft.irfft(
                  torch.fft.rfft(u_t, n=2 * L) * torch.fft.rfft(k, n=2 * L),
                  n=2 * L)[..., :L]
        y   = (y + u_t * self.D.unsqueeze(-1)).transpose(1, 2)
        y   = self.act(self.drop(y))
        y1, y2 = self.out_proj(y).chunk(2, dim=-1)
        return y1 * torch.sigmoid(y2)
