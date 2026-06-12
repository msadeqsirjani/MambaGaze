"""ResNetClassifier: ResNet-style 1D CNN."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .res_block import ResBlock1D


class ResNetClassifier(nn.Module):

    def __init__(self, in_ch: int = 10, conv_dropout: float = 0.5,
                 fc_dropout: float = 0.25):
        super().__init__()
        self.conv1   = nn.Conv1d(in_ch, 64, kernel_size=64, stride=2, padding=32, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.pool    = nn.MaxPool1d(3, stride=2, padding=1)
        self.block1  = nn.Sequential(ResBlock1D(64,  64,  dropout=conv_dropout),
                                     ResBlock1D(64,  64,  dropout=conv_dropout))
        self.block2  = nn.Sequential(ResBlock1D(64,  128, stride=2, dropout=conv_dropout),
                                     ResBlock1D(128, 128, dropout=conv_dropout))
        self.block3  = nn.Sequential(ResBlock1D(128, 256, stride=2, dropout=conv_dropout),
                                     ResBlock1D(256, 256, dropout=conv_dropout))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.drop    = nn.Dropout(fc_dropout)
        self.fc1     = nn.Linear(256, 128)
        self.fc2     = nn.Linear(128,  64)
        self.fc3     = nn.Linear( 64,   1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.pool(z)
        z = self.block3(self.block2(self.block1(z)))
        z = self.avgpool(z).squeeze(-1)
        z = F.relu(self.fc1(self.drop(z)))
        z = F.relu(self.fc2(self.drop(z)))
        return self.fc3(z).squeeze(-1)
