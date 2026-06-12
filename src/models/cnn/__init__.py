"""
CNNClassifier: VGG-style 1D CNN baseline.

Input : (B, C, T)   C = feature channels, T = sequence length (500)
Output: logit (B,)
"""

from .model import CNNClassifier

__all__ = ["CNNClassifier"]
