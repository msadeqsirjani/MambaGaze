"""S4D convolution kernel (S4D-Lin parameterization)."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class S4DKernel(nn.Module):
    """
    Generates the global SSM convolution kernel for one model dimension.

    S4D-Lin parameterization:
      A = diag(-exp(log_A_real) + i * A_imag)   HiPPO-initialized
      dt = exp(log_dt)
      K[t] = 2 Re( C_bar * (dt*A)^t )  where  C_bar = C*(exp(dt*A)-1)/A
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        H = d_model
        N = d_state // 2

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        C = torch.randn(H, N, dtype=torch.cfloat) * 0.5
        self.C = nn.Parameter(torch.view_as_real(C))

        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        A_imag = math.pi * torch.arange(N).float().unsqueeze(0).expand(H, -1)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L: int) -> torch.Tensor:
        """Compute convolution kernel of length L. Returns (H, L)."""
        dt = torch.exp(self.log_dt)
        C  = torch.view_as_complex(self.C)
        A  = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA   = A * dt.unsqueeze(-1)
        C_bar = C * (torch.exp(dtA) - 1.0) / A

        t = torch.arange(L, device=dtA.device, dtype=torch.float32)
        K = torch.exp(dtA.unsqueeze(-1) * t)
        K = 2.0 * torch.einsum("hn,hnl->hl", C_bar, K).real
        return K
