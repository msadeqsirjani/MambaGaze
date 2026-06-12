"""
VGGClassifier: VGG-style 1D CNN baseline.

Input : (B, C, T)
Output: logit (B,)
"""

from .model import VGGClassifier

__all__ = ["VGGClassifier"]
