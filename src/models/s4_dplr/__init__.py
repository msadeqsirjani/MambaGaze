"""True S4 (DPLR) family: diagonal-plus-low-rank state matrix with HiPPO-LegS
NPLR initialization, in unidirectional (UniS4) and bidirectional (BiS4) forms.

Distinct from the diagonal S4D family (s4/ and s4d/): the kernel here keeps the
rank-1 low-rank correction of the original S4, computed via the generating
function (Cauchy kernel + Woodbury).
"""

from .kernel import S4DPLRKernel
from .layer import S4DPLRLayer
from .stack import S4DPLRStack
from .uni import UniS4
from .bi import BiS4

__all__ = ["S4DPLRKernel", "S4DPLRLayer", "S4DPLRStack", "UniS4", "BiS4"]
