"""Tactical and auxiliary head declarations and implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hexorl.models.constants import BOARD_AREA
from hexorl.models.heads.pair_policy import GLOBAL_PAIR_HEADS
from hexorl.models.heads.policy import GLOBAL_PLACE_HEAD
from hexorl.models.heads.value import VALUE_HEAD


class AuxPolicyHead(nn.Module):
    """Auxiliary policy head - same structure as PolicyHead. Used for opp_policy."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * BOARD_AREA, BOARD_AREA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class AxisHead(nn.Module):
    """Axis classification head: global avg pool -> (B, 3) logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class AxisMapHead(nn.Module):
    """Dense axis-map regression head: (B, C, 33, 33) -> (B, 6, 33, 33)."""

    def __init__(self, channels: int, planes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, max(8, channels // 2), kernel_size=1)
        self.conv2 = nn.Conv2d(max(8, channels // 2), planes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        return F.softplus(self.conv2(x))


class MovesLeftHead(nn.Module):
    """Moves-left head: global avg pool -> Linear -> ReLU -> Linear -> softplus."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return F.softplus(self.fc2(x))


GLOBAL_GRAPH_OUTPUT_HEADS = (GLOBAL_PLACE_HEAD, *GLOBAL_PAIR_HEADS, VALUE_HEAD)


__all__ = [
    "AuxPolicyHead",
    "AxisHead",
    "AxisMapHead",
    "GLOBAL_GRAPH_OUTPUT_HEADS",
    "MovesLeftHead",
]
