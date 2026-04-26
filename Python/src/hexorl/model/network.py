"""KataGo-style CNN for Hexo.

Phase 1 stub: correct input/output shapes, untrained.
Full implementation in Phase 4.
"""

import torch
import torch.nn as nn


class HexNet(nn.Module):
    """AlphaZero-style network for Hexo.

    Input: (B, 13, 33, 33) tensor
    Output: (policy_logits, value) tuple
    """

    def __init__(self, channels: int = 32, blocks: int = 4):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        # Initial convolution
        self.conv_in = nn.Conv2d(13, channels, kernel_size=3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            for _ in range(blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 33 * 33, 1089)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(33 * 33, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[1:] == (13, 33, 33), f"Expected (B,13,33,33), got {x.shape}"

        x = torch.relu(self.conv_in(x))

        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual

        # Policy
        p = torch.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = torch.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v
