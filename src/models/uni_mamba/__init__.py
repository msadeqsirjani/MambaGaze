"""
Unidirectional Mamba classifier baseline (forward-only).

Requires: pip install mamba-ssm  (CUDA GPU required)

Interface:
    logit, (attn,) = model(x)    x: (B, T, C)
"""

from .classifier import UniMambaClassifier

__all__ = ["UniMambaClassifier"]
