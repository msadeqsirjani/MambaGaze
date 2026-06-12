"""Checkpoint loading with architecture auto-detection from state-dict keys."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from models import MODELS


def detect_arch(state_dict: dict, hint: str = "") -> str:
    keys = list(state_dict.keys())

    # DPLR S4 kernels carry a low-rank "P" parameter; diagonal S4D kernels do not.
    is_dplr = any(".kernel.P" in k for k in keys)

    # Bidirectional S4/S4D: has fwd.in_proj + bwd + pool_fwd
    if (any(k.startswith("fwd.in_proj") for k in keys) and
            any(k.startswith("pool_fwd") for k in keys) and
            any(k.startswith("bwd.") for k in keys)):
        # Check if S4/S4D (has kernel layers) vs Mamba (has in_proj in layers)
        if any("kernel" in k for k in keys):
            return "bi_s4" if is_dplr else "bi_s4d"
        return "bi_mamba"

    # Unidirectional S4/S4D: has stack.in_proj and stack.layers with kernel
    if (any(k.startswith("stack.in_proj") for k in keys) and
            any("kernel" in k for k in keys)):
        return "uni_s4" if is_dplr else "uni_s4d"

    # UniMamba: has stack.in_proj but no kernel
    if (any(k.startswith("stack.in_proj") for k in keys) and
            not any("kernel" in k for k in keys)):
        return "uni_mamba"

    # CNN: has encoder.0.weight (first Conv1d in Sequential)
    if "encoder.0.weight" in state_dict and "fc1.weight" in state_dict:
        return "cnn"

    # Transformer: has encoder + pos_encoder
    if (any(k.startswith("encoder.layers") for k in keys) and
            any(k.startswith("pos_encoder") for k in keys)):
        return "transformer"

    # ResNet: has block1/block2/block3 + conv1 with large kernel
    if (any(k.startswith("block1.") for k in keys) and
            "conv1.weight" in state_dict and
            state_dict["conv1.weight"].shape[2] >= 32):
        return "resnet"

    # VGG: has block1/block2/block3 but no large initial conv1
    if (any(k.startswith("block1.") for k in keys) and
            any(k.startswith("block2.") for k in keys)):
        return "vgg"

    return hint  # fall back to CLI hint


def load_model(pt_path: str, hint: str, in_dim: int = 30) -> Tuple[nn.Module, str, int]:
    """Load a checkpoint and return (model, arch, actual_in_dim).
    Construction is driven by the registry in src/models/registry.py."""
    sd   = torch.load(pt_path, map_location="cpu", weights_only=True)
    arch = detect_arch(sd, hint)

    # Try to infer in_dim from state-dict
    for probe in ["fwd.in_proj.weight", "stack.in_proj.weight",
                  "encoder.0.weight", "conv1.weight", "block1.0.weight",
                  "input_proj.weight"]:
        if probe in sd:
            in_dim = int(sd[probe].shape[1])
            break

    if arch not in MODELS:
        raise ValueError(f"Cannot load unknown arch: {arch!r}")
    model = MODELS[arch].build(in_dim)

    model.load_state_dict(sd)
    return model, arch, in_dim
