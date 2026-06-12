"""S4DGaze: full bidirectional classifier."""

from __future__ import annotations
import torch
import torch.nn as nn

from .block_stack import S4DBlockStack
from .attention_pool import AttentionPool


class S4DGaze(nn.Module):
    """
    Bidirectional S4DGaze classifier.

    Input: XMD-encoded tensor Z in R^{B x T x 3F}  (F=10 features, 3F=30 channels)
    Output: scalar logit per sample (use BCEWithLogitsLoss during training)

    The forward branch processes Z left-to-right; the backward branch processes
    the time-reversed Z right-to-left. Each branch has its own attention pooling.
    Context vectors are concatenated and fed to a LayerNorm+Linear head.
    """

    def __init__(
        self,
        in_dim:   int   = 30,
        d_model:  int   = 128,
        n_layers: int   = 4,
        d_state:  int   = 64,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.fwd      = S4DBlockStack(in_dim, d_model, n_layers, d_state, dropout)
        self.bwd      = S4DBlockStack(in_dim, d_model, n_layers, d_state, dropout)
        self.pool_fwd = AttentionPool(d_model)
        self.pool_bwd = AttentionPool(d_model)
        self.head     = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 1),
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, 3F)  XMD-encoded input

        Returns:
            logit: (B,)           raw logit for BCEWithLogitsLoss
            attn:  tuple(fwd_a, bwd_a)  attention weights (B, T) each
        """
        h_fwd = self.fwd(z)
        h_bwd = self.bwd(torch.flip(z, dims=[1]))
        h_bwd = torch.flip(h_bwd, dims=[1])

        c_fwd, a_fwd = self.pool_fwd(h_fwd)
        c_bwd, a_bwd = self.pool_bwd(h_bwd)

        c    = torch.cat([c_fwd, c_bwd], dim=-1)
        logit = self.head(c).squeeze(-1)
        return logit, (a_fwd, a_bwd)
