"""Global graph model composer."""

from __future__ import annotations

import torch
import torch.nn as nn

from hexorl.models.inputs import GraphInputs
from hexorl.models.trunks import GlobalTrunk


class GlobalModel(nn.Module):
    def __init__(self, trunk: GlobalTrunk, heads: dict[str, nn.Module]):
        super().__init__()
        self.trunk = trunk
        self.heads = nn.ModuleDict(heads)
        self.head_names = tuple(heads)
        self.architecture_family = trunk.variant_name
        self.n_bins = 65

    def forward(self, inputs: GraphInputs | None = None, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs is None:
            inputs = GraphInputs(**kwargs)
        outs = self.trunk(inputs)
        return {name: head(outs, inputs) for name, head in self.heads.items()}

    def half(self) -> "GlobalModel":
        super().half()
        return self


__all__ = ["GlobalModel"]
