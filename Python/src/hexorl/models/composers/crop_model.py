"""Crop model composer."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.models.inputs import CropInputs
from hexorl.models.trunks import CropTrunk


class CropModel(nn.Module):
    def __init__(self, trunk: CropTrunk, heads: dict[str, nn.Module]):
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleDict(heads)
        self.head_names = tuple(heads)
        self.conv_in = getattr(trunk, "conv_in", None)
        self.res_blocks = getattr(trunk, "res_blocks", nn.ModuleList())
        self.n_bins = 65

    def forward(self, inputs: CropInputs) -> dict[str, torch.Tensor]:
        features = self.trunk(inputs)
        return {name: head(features, inputs) for name, head in self.heads.items()}

    @torch.no_grad()
    def apply_hex_masks_(self) -> None:
        if hasattr(self.trunk, "apply_hex_masks_"):
            self.trunk.apply_hex_masks_()

    def half(self) -> "CropModel":
        super().half()
        return self

    @torch.no_grad()
    def forward_batch(
        self,
        inputs: CropInputs,
        autocast: bool = False,
        requested_heads: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.forward(inputs)
        else:
            out = self.forward(inputs)
        return {k: v for k, v in out.items() if k in requested_heads} if requested_heads is not None else out


__all__ = ["CropModel"]
