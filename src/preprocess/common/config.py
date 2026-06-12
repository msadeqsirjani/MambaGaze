"""Shared preprocessing configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    target_hz:      float = 50.0
    window_sec:     float = 10.0
    min_window_sec: float = 10.0
