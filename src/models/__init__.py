"""Model zoo: S4DGaze plus baselines, one sub-package per model.

Shared baseline sub-modules (AttentionPool, MambaStack, PositionalEncoding)
live in common/. registry.py declares every model once (class, input layout,
hyperparameters, output convention); scripts should build models through it.
"""

from .s4d import S4DGaze
from .s4 import S4Classifier
from .bi_mamba import BiMambaClassifier
from .uni_mamba import UniMambaClassifier
from .cnn import CNNClassifier
from .transformer import TransformerClassifier
from .resnet import ResNetClassifier
from .vgg import VGGClassifier
from .registry import MODELS, BASELINES, ModelSpec

__all__ = [
    "S4DGaze",
    "S4Classifier",
    "BiMambaClassifier",
    "UniMambaClassifier",
    "CNNClassifier",
    "TransformerClassifier",
    "ResNetClassifier",
    "VGGClassifier",
    "MODELS",
    "BASELINES",
    "ModelSpec",
]
