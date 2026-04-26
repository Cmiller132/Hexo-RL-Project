"""KataGo-style CNN for Hexo.

Input: (B, 13, 33, 33) f32 tensor
Output: policy logits (B, 1089), value (B, 1) tanh-bounded

Phase 2: Correct shapes, FP16-ready, factory from config.
Full KataGo-style heads (lookahead, axis, regret) in Phase 4.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class HexNet(nn.Module):
    """AlphaZero-style network for Hexo.

    Input:  (B, 13, 33, 33) — 13-channel board tensor from Rust encoder.
    Output: policy (B, 1089) — raw logits over 33×33 spatial grid.
            value  (B, 1)   — tanh-bounded win probability [-1, 1].
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 16,
        policy_filters: int = 2,
        value_hidden: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.blocks = blocks

        self.conv_in = nn.Conv2d(13, channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList([
            _ResBlock(channels) for _ in range(blocks)
        ])

        self.policy_conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.policy_fc = nn.Linear(policy_filters * 33 * 33, 1089)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(33 * 33, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for Conv2d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 13, 33, 33) float tensor.

        Returns:
            (policy_logits, value): policy of shape (B, 1089), value of shape (B, 1).
        """
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
        assert x.shape[1:] == (13, 33, 33), f"Expected (B,13,33,33), got {x.shape}"

        x = torch.relu(self.conv_in(x))

        for block in self.res_blocks:
            x = block(x)

        p = torch.relu(self.policy_conv(x))
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)

        v = torch.relu(self.value_conv(x))
        v = v.reshape(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def half(self) -> "HexNet":
        """Convert to FP16 (chained, like nn.Module.half())."""
        super().half()
        return self

    @torch.no_grad()
    def forward_batch(
        self,
        x: torch.Tensor,
        autocast: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only forward with optional FP16 autocast."""
        if autocast and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return self.forward(x)
        return self.forward(x)


class _ResBlock(nn.Module):
    """Pre-activation residual block with two 3x3 convs."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x + residual


def from_config(cfg, device: Optional[torch.device] = None) -> HexNet:
    """Create a HexNet from a Config object.

    Args:
        cfg: hexorl.config.Config instance.
        device: Target device (cpu, cuda, mps). Defaults to best available.

    Returns:
        HexNet instance on the requested device, optionally in FP16.
    """
    model_cfg = cfg.model
    inference_cfg = cfg.inference

    model = HexNet(
        channels=model_cfg.channels,
        blocks=model_cfg.blocks,
    )

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)

    if inference_cfg.fp16 and device.type == "cuda":
        model = model.half()

    model.eval()
    return model
