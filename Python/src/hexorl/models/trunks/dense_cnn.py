"""Dense CNN trunk construction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexorl.models.specs import ModelSpec

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
        gate = torch.sigmoid(self.bn_gate(self.conv_gate(residual)))
        x = x * gate
        return x + residual


def resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_crop_trunk_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    family_kind: str,
    device: torch.device | None,
    inference: bool,
) -> nn.Module:
    from hexorl.models.crop_network import HexNet

    model_cfg = cfg.model
    inference_cfg = cfg.inference
    heads = list(spec.params["heads"])
    if spec.params["sparse_policy"] and "sparse_policy" not in heads:
        heads.append("sparse_policy")
    model = HexNet(
        channels=model_cfg.channels,
        blocks=model_cfg.blocks,
        heads=heads,
        family_kind=family_kind,
        attention_positions=list(getattr(model_cfg, "attention_positions", [])),
        attention_heads=getattr(model_cfg, "attention_heads", 8),
        attention_mlp_ratio=getattr(model_cfg, "attention_mlp_ratio", 2.0),
        attention_dropout=getattr(model_cfg, "attention_dropout", 0.0),
        dropout=getattr(model_cfg, "dropout", 0.0),
        relative_bias=getattr(model_cfg, "relative_bias", False),
        graph_token_set=getattr(model_cfg, "graph_token_set", "graph512_turn_pair_prior"),
        graph_token_budget=getattr(model_cfg, "graph_token_budget", 512),
        graph_layers=getattr(model_cfg, "graph_layers", 3),
        sparse_policy=getattr(model_cfg, "sparse_policy", False),
    )
    resolved = resolve_device(device)
    model = model.to(resolved)
    if inference and inference_cfg.fp16 and resolved.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def build_dense_cnn_model(
    spec: ModelSpec,
    cfg: Any,
    *,
    device: torch.device | None = None,
    inference: bool = False,
) -> nn.Module:
    return build_crop_trunk_model(spec, cfg, family_kind=DENSE_CNN_TRUNK, device=device, inference=inference)


__all__ = [
    "DENSE_CNN_TRUNK",
    "GatedResBlock",
    "HexConv2d",
    "build_crop_trunk_model",
    "build_dense_cnn_model",
    "resolve_device",
]
