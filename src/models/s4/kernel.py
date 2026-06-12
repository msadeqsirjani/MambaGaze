"""S4D-Lin convolution kernel for the unidirectional S4 baseline."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class S4Kernel(nn.Module):
    """S4D-Lin convolution kernel (HiPPO-initialized diagonal SSM)."""

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        H, N = d_model, d_state // 2
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt     = nn.Parameter(log_dt)
        self.C          = nn.Parameter(torch.view_as_real(torch.randn(H, N, dtype=torch.cfloat) * 0.5))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        self.A_imag     = nn.Parameter(math.pi * torch.arange(N).float().unsqueeze(0).expand(H, -1))

    def forward(self, L: int) -> torch.Tensor:
        dt    = torch.exp(self.log_dt)
        C     = torch.view_as_complex(self.C)
        A     = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA   = A * dt.unsqueeze(-1)
        C_bar = C * (torch.exp(dtA) - 1.0) / A
        t     = torch.arange(L, device=dtA.device, dtype=torch.float32)
        return 2.0 * torch.einsum("hn,hnl->hl", C_bar,
                                  torch.exp(dtA.unsqueeze(-1) * t)).real
