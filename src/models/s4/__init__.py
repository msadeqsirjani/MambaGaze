"""
S4 (unidirectional) baseline classifier.

Uses the same S4D-Lin diagonal kernel as S4DGaze but processes sequences
in a single forward direction only (no backward branch).

Interface:
    logit, (attn,) = model(x)    x: (B, T, C)
"""

from .kernel import S4Kernel
from .layer import S4Layer
from .stack import S4Stack
from .classifier import S4Classifier

__all__ = ["S4Kernel", "S4Layer", "S4Stack", "S4Classifier"]
