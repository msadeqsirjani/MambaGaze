"""
S4DGaze: Bidirectional Diagonal State Space Model for Cognitive Load Assessment.

Architecture:
  XMD input (X + mask + delta, 3F channels)
  -> input projection
  -> L stacked S4D layers with pre-norm residual (forward branch)
  -> L stacked S4D layers with pre-norm residual (backward branch, time-reversed)
  -> attention pooling (separate per direction)
  -> concat [fwd_ctx, bwd_ctx]
  -> LayerNorm -> linear -> sigmoid (binary classification)

Reference: "On the Parameterization and Initialization of Diagonal State Space Models"
           Gu, Gupta, Goel, Re. NeurIPS 2022.
"""

from .kernel import S4DKernel
from .layer import S4DLayer
from .block_stack import S4DBlockStack
from .attention_pool import AttentionPool
from .model import S4DGaze

__all__ = [
    "S4DKernel",
    "S4DLayer",
    "S4DBlockStack",
    "AttentionPool",
    "S4DGaze",
]
