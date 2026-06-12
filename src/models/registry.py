"""Central model registry: every model is declared exactly once here.

Each ModelSpec records everything the training and benchmark scripts need:
which class to build, which constructor kwarg receives the input size,
which shared hyperparameters the constructor accepts, the expected input
layout, and the forward-pass return convention.

To add a new model:
  1. Create its sub-package under models/ (one class per file).
  2. Add one ModelSpec entry to MODELS below.

CLI choices, model construction, input transposing, and output unpacking in
train.py and edge/benchmark.py are all derived from this registry —
no other file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Type

import torch.nn as nn

from .s4d import S4DGaze
from .s4 import S4Classifier
from .s4_dplr import UniS4, BiS4
from .bi_mamba import BiMambaClassifier
from .uni_mamba import UniMambaClassifier
from .cnn import CNNClassifier
from .transformer import TransformerClassifier
from .resnet import ResNetClassifier
from .vgg import VGGClassifier


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of one model."""

    name: str
    cls: Type[nn.Module]
    hparams: Mapping[str, str] = field(default_factory=dict)
    in_key: str = "in_dim"
    channels_first: bool = False
    returns_attn: bool = False
    baseline: bool = True
    # Input encoding used when --inputs is not given: "xmd" ([X|M|D]) or "x".
    default_inputs: str = "x"

    def build(self, in_dim: int, **hp) -> nn.Module:
        """Instantiate the model. Unknown/missing hyperparameters fall back
        to the class defaults."""
        kwargs = {self.in_key: in_dim}
        for common, ctor_kwarg in self.hparams.items():
            if common in hp:
                kwargs[ctor_kwarg] = hp[common]
        return self.cls(**kwargs)

    def unpack(self, output):
        """Normalize a forward-pass result to a scalar logit per sample."""
        return output[0] if self.returns_attn else output


_SSM_HPARAMS   = {"d_model": "d_model", "n_layers": "n_layers",
                  "d_state": "d_state", "dropout": "dropout"}
_MAMBA_HPARAMS = {"d_model": "d_model", "n_layers": "n_layers",
                  "dropout": "dropout"}

MODELS: Dict[str, ModelSpec] = {spec.name: spec for spec in [
    # S4 family: {uni,bi} x {S4D diagonal, S4 DPLR}.
    ModelSpec("bi_s4d", S4DGaze, _SSM_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("uni_s4d", S4Classifier, _SSM_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("bi_s4", BiS4, _SSM_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("uni_s4", UniS4, _SSM_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("bi_mamba", BiMambaClassifier, _MAMBA_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("uni_mamba", UniMambaClassifier, _MAMBA_HPARAMS,
              returns_attn=True, default_inputs="xmd"),
    ModelSpec("cnn", CNNClassifier, {"dropout": "dropout"},
              in_key="in_ch", channels_first=True),
    ModelSpec("transformer", TransformerClassifier,
              {"d_model": "d_model", "n_layers": "num_layers",
               "dropout": "dropout"},
              in_key="in_ch", channels_first=True),
    ModelSpec("resnet", ResNetClassifier, {},
              in_key="in_ch", channels_first=True),
    ModelSpec("vgg", VGGClassifier, {},
              in_key="in_ch", channels_first=True),
]}

BASELINES: Dict[str, ModelSpec] = {
    name: spec for name, spec in MODELS.items() if spec.baseline
}
