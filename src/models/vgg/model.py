"""VGGClassifier: VGG-style 1D CNN."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGClassifier(nn.Module):

    def __init__(self, in_ch: int = 10, fc_dropout: float = 0.25):
        super().__init__()

        def _block(ci: int, co: int, k: int = 3) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(),
                nn.Conv1d(co, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(),
                nn.MaxPool1d(2),
            )

        self.block1  = _block(in_ch, 64)
        self.block2  = _block(64,   128)
        self.block3  = _block(128,  256)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop    = nn.Dropout(fc_dropout)
        self.fc1     = nn.Linear(256, 128)
        self.fc2     = nn.Linear(128,   1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.avgpool(self.block3(self.block2(self.block1(x)))).squeeze(-1)
        z = F.relu(self.fc1(self.drop(z)))
        return self.fc2(z).squeeze(-1)
