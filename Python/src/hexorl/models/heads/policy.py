"""Place-policy head names and implementation."""

import torch
import torch.nn as nn

from hexorl.models.constants import BOARD_AREA


class PolicyHead(nn.Module):
    """Policy head: (B, C, 33, 33) -> (B, 1089) logits."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * BOARD_AREA, BOARD_AREA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

DENSE_POLICY_HEAD = "policy"
GLOBAL_PLACE_HEAD = "policy_place"
DENSE_POLICY_HEADS = (DENSE_POLICY_HEAD,)


__all__ = ["DENSE_POLICY_HEAD", "DENSE_POLICY_HEADS", "GLOBAL_PLACE_HEAD", "PolicyHead"]
