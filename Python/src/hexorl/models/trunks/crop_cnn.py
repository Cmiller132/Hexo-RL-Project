"""Crop CNN trunk and dense-CNN family builder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexorl.models.heads import build_heads_for_family
from hexorl.models.inference_contracts import crop_input_tensors
from hexorl.models.inputs import CropInputs

DENSE_CNN_TRUNK = "dense_cnn"


class HexConv2d(nn.Conv2d):
    """3x3 convolution constrained to the valid axial hex neighborhood."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kernel_size != (3, 3):
            raise ValueError("HexConv2d is only defined for 3x3 kernels")
        mask = torch.ones_like(self.weight)
        mask[:, :, 0, 0] = 0.0
        mask[:, :, 2, 2] = 0.0
        self.register_buffer("hex_mask", mask, persistent=False)
        self.apply_hex_mask_()

    @torch.no_grad()
    def apply_hex_mask_(self) -> None:
        self.weight.mul_(self.hex_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self.hex_mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class GatedResBlock(nn.Module):
    """Gated residual block with BatchNorm2d for training stability."""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.conv_gate = HexConv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.dropout(x)
        return x * torch.sigmoid(self.bn_gate(self.conv_gate(residual))) + residual


class CropCnnTrunk(nn.Module):
    feature_channels: int
    input_tensors = crop_input_tensors(candidates=True, pairs=True)

    def __init__(self, *, channels: int, blocks: int, dropout: float = 0.0):
        super().__init__()
        self.feature_channels = int(channels)
        self.conv_in = HexConv2d(13, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(GatedResBlock(channels, dropout=dropout) for _ in range(blocks))

    def forward(self, inputs: CropInputs) -> torch.Tensor:
        x = torch.relu(self.conv_in(inputs.tensor))
        for block in self.res_blocks:
            x = block(x)
        return x

    @torch.no_grad()
    def apply_hex_masks_(self) -> None:
        for module in self.modules():
            if isinstance(module, HexConv2d):
                module.apply_hex_mask_()


def resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_crop_model(trunk: nn.Module, spec, cfg, *, device: torch.device | None, inference: bool) -> nn.Module:
    from hexorl.models.composers import CropModel

    model = CropModel(trunk, build_heads_for_family(spec, cfg, trunk))
    model.n_bins = int(getattr(spec.params, "n_bins", 65))
    model.apply_hex_masks_()
    resolved = resolve_device(device)
    model = model.to(resolved)
    if inference and cfg.inference.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def build_dense_cnn_model(spec, cfg, *, device: torch.device | None = None, inference: bool = False) -> nn.Module:
    params = spec.params
    trunk = CropCnnTrunk(channels=params.channels, blocks=params.blocks, dropout=params.dropout)
    return build_crop_model(trunk, spec, cfg, device=device, inference=inference)


__all__ = ["DENSE_CNN_TRUNK", "CropCnnTrunk", "GatedResBlock", "HexConv2d", "build_dense_cnn_model", "resolve_device"]
