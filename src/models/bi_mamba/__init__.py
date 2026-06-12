"""
Bidirectional Mamba classifier baseline.

Requires: pip install mamba-ssm  (CUDA GPU required)

Two separate Mamba stacks process the sequence forward and backward;
their attention-pooled contexts are concatenated for classification.

Interface:
    logit, (a_fwd, a_bwd) = model(x)    x: (B, T, C)
"""

from .classifier import BiMambaClassifier

__all__ = ["BiMambaClassifier"]
