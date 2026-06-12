"""
ResNetClassifier: ResNet-style 1D CNN baseline.

Input : (B, C, T)
Output: logit (B,)
"""

from .res_block import ResBlock1D
from .model import ResNetClassifier

__all__ = ["ResBlock1D", "ResNetClassifier"]
