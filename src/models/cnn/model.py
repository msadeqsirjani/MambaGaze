"""CNNClassifier: VGG-style 1D CNN."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):

    def __init__(self, in_ch: int = 10, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, 64, padding=32), nn.ReLU(),
            nn.Conv1d(64,    64, 64, padding=32), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(dropout / 2),
            nn.Conv1d(64,  128, 32, padding=16), nn.ReLU(),
            nn.Conv1d(128, 128, 32, padding=16), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(dropout / 2),
            nn.Conv1d(128, 256, 17, padding=8), nn.ReLU(),
            nn.Conv1d(256, 256, 17, padding=8), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(dropout / 2),
            nn.Conv1d(256, 512, 7, padding=3), nn.ReLU(),
            nn.Conv1d(512, 512, 7, padding=3), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Dropout(dropout),
        )
        self.fc1  = nn.Linear(512, 256)
        self.fc2  = nn.Linear(256, 128)
        self.fc3  = nn.Linear(128, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).squeeze(-1)
        z = F.relu(self.fc1(self.drop(z)))
        z = F.relu(self.fc2(self.drop(z)))
        return self.fc3(z).squeeze(-1)
