"""
TransformerClassifier: Transformer encoder baseline.

Input : (B, C, T)  or (B, T, C) — the model transposes if needed
Output: logit (B,)
"""

from .model import TransformerClassifier

__all__ = ["TransformerClassifier"]
