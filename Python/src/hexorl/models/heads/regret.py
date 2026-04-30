"""Regret/value-ranking head names and implementation."""

import torch
import torch.nn as nn

from hexorl.models.heads.value import ValueBinnedHead


class RegretRankHead(nn.Module):
    """Regret ranking head: global avg pool -> Linear -> ReLU -> Linear -> scalar phi(s)."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

REGRET_HEADS = ("regret_value", "regret_rank")


__all__ = ["REGRET_HEADS", "RegretRankHead", "ValueBinnedHead"]
