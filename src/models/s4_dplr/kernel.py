"""True S4 (DPLR) convolution kernel.

Unlike the diagonal S4D kernel, this uses the original S4 diagonal-plus-low-rank
(DPLR) state matrix  A = Lambda - P P^*  with HiPPO-LegS (NPLR) initialization.
The kernel is computed by the canonical generating-function method: evaluate the
truncated SSM transfer function at the L-th roots of unity via a Cauchy kernel
with a rank-1 Woodbury correction, then inverse-FFT to the time domain.

Reference: Gu, Goel, Re. "Efficiently Modeling Long Sequences with Structured
State Spaces" (S4), ICLR 2022; and the Annotated S4.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


def _make_dplr_hippo(N: int):
    """HiPPO-LegS in NPLR form, diagonalized.

    Returns (Lambda, P, B), each a length-N complex vector, where the normal
    part is diagonal Lambda and the low-rank term is P (rank 1, Q = P)."""
    n = torch.arange(N, dtype=torch.float64)
    # HiPPO-LegS matrix A = -(lower-triangular of sqrt outer) + diag(n)
    r = torch.sqrt(1.0 + 2.0 * n)
    A = r.unsqueeze(1) * r.unsqueeze(0)
    A = -(torch.tril(A) - torch.diag(n))
    # NPLR low-rank P and input projection B
    P = torch.sqrt(n + 0.5)
    B = torch.sqrt(2.0 * n + 1.0)
    # Normal (skew-symmetric up to constant diagonal) part S = A + P P^T
    S = A + P.unsqueeze(1) * P.unsqueeze(0)
    diag = torch.diagonal(S)
    Lambda_real = diag.mean() * torch.ones(N, dtype=torch.float64)  # ~ -0.5
    S_skew = (S - torch.diag(diag)).to(torch.complex128)
    M = -1j * S_skew
    M = 0.5 * (M + M.conj().transpose(-1, -2))      # enforce Hermitian
    Lambda_imag, V = torch.linalg.eigh(M)
    Vh = V.conj().transpose(-1, -2)
    Pc = Vh @ P.to(torch.complex128)
    Bc = Vh @ B.to(torch.complex128)
    Lambda = Lambda_real.to(torch.complex128) + 1j * Lambda_imag
    return Lambda, Pc, Bc


class S4DPLRKernel(nn.Module):
    """Generates the S4 (DPLR) convolution kernel for d_model channels.

    Per-channel: diagonal Lambda (HiPPO-LegS init) and output C; the low-rank P
    and input B are HiPPO-initialized and shared across channels, so the model
    differs from the diagonal S4D kernel only by the rank-1 low-rank term."""

    def __init__(self, d_model: int, d_state: int = 64,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        H = d_model
        N = d_state // 2
        Lambda, P, B = _make_dplr_hippo(N)

        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # Per-channel diagonal part (broadcast HiPPO init), stored so Re(A) < 0.
        self.log_A_real = nn.Parameter(torch.log(-Lambda.real).float().unsqueeze(0).repeat(H, 1))
        self.A_imag     = nn.Parameter(Lambda.imag.float().unsqueeze(0).repeat(H, 1))
        # Per-channel output projection C (random init, as in S4D).
        C = torch.randn(H, N, dtype=torch.cfloat) * 0.5
        self.C = nn.Parameter(torch.view_as_real(C))
        # Shared low-rank P and input B (HiPPO init). Named "P" so checkpoints
        # are distinguishable from the diagonal S4D kernel.
        self.P = nn.Parameter(torch.view_as_real(P.to(torch.cfloat)))   # (N, 2)
        self.B = nn.Parameter(torch.view_as_real(B.to(torch.cfloat)))   # (N, 2)

    def forward(self, L: int) -> torch.Tensor:
        """Compute the length-L convolution kernel. Returns (H, L) real."""
        dt  = torch.exp(self.log_dt)                                   # (H,)
        Lam = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (H, N)
        C   = torch.view_as_complex(self.C)                            # (H, N)
        P   = torch.view_as_complex(self.P)                            # (N,)
        B   = torch.view_as_complex(self.B)                            # (N,)

        Omega = torch.exp(-2j * math.pi * torch.arange(L, device=dt.device) / L)  # (L,)
        one_p = 1.0 + Omega
        one_p = torch.where(one_p.abs() < 1e-9, one_p + 1e-9, one_p)   # avoid z=-1 pole
        g = (2.0 / dt).unsqueeze(1) * ((1.0 - Omega) / one_p).unsqueeze(0)   # (H, L)
        c = 2.0 / one_p                                                # (L,)

        denom = g.unsqueeze(-1) - Lam.unsqueeze(1)                     # (H, L, N)

        def cauchy(v):                          # v: (..., N) -> (H, L)
            return (v.unsqueeze(-2) / denom).sum(-1)

        Cc, Pc = C.conj(), P.conj()
        Pb, Bb = P.unsqueeze(0), B.unsqueeze(0)                        # (1, N)
        k00 = cauchy(Cc * Bb)
        k01 = cauchy(Cc * Pb)
        k10 = cauchy(Pc.unsqueeze(0) * Bb)
        k11 = cauchy(Pc.unsqueeze(0) * Pb)
        at_roots = c.unsqueeze(0) * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)   # (H, L)
        return torch.fft.ifft(at_roots, n=L, dim=-1).real             # (H, L)
