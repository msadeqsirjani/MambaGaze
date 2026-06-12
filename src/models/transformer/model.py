"""TransformerClassifier: Transformer encoder."""

from __future__ import annotations
import torch
import torch.nn as nn

from ..common import AttentionPool, PositionalEncoding


class TransformerClassifier(nn.Module):

    def __init__(self, in_ch: int = 10, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.5):
        super().__init__()
        self.input_proj  = nn.Linear(in_ch, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool    = AttentionPool(d_model)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,     128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,       1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != x.size(2):
            x = x.transpose(1, 2)
        z = self.pos_encoder(self.input_proj(x))
        h = self.encoder(z)
        c, _ = self.pool(h)
        return self.head(c).squeeze(-1)
