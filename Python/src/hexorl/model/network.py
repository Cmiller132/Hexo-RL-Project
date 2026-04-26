"""KataGo-style CNN for Hexo with binned value heads and configurable multi-head architecture.

Input: (B, 13, 33, 33) f32 tensor
Output: dict of head_name → tensor

Supports heads: policy, value (binned), lookahead_* (binned), opp_policy,
axis (3-class), regret_rank (scalar), regret_value (binned), moves_left (scalar).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class GatedResBlock(nn.Module):
    """Gated residual block with BatchNorm2d for training stability."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv_gate = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        gate = torch.sigmoid(self.bn_gate(self.conv_gate(residual)))
        x = x * gate
        return x + residual


class PolicyHead(nn.Module):
    """Policy head: (B, C, 33, 33) → (B, 1089) logits."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * 33 * 33, 1089)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ValueBinnedHead(nn.Module):
    """Binned value head: (B, C, 33, 33) → (B, N_BINS) logits.

    Used for: value, lookahead_*, regret_value.
    """

    def __init__(self, channels: int, n_bins: int = 65, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc1 = nn.Linear(33 * 33, hidden)
        self.fc2 = nn.Linear(hidden, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class AuxPolicyHead(nn.Module):
    """Auxiliary policy head — same structure as PolicyHead. Used for opp_policy."""

    def __init__(self, channels: int, policy_filters: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, policy_filters, kernel_size=1)
        self.fc = nn.Linear(policy_filters * 33 * 33, 1089)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class AxisHead(nn.Module):
    """Axis classification head: global avg pool → (B, 3) logits."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        return self.fc(x)


class RegretRankHead(nn.Module):
    """Regret ranking head: global avg pool → Linear → ReLU → Linear → scalar φ(s)."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MovesLeftHead(nn.Module):
    """Moves-left head: global avg pool → Linear → ReLU → Linear → softplus."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=[2, 3])
        x = torch.relu(self.fc1(x))
        return F.softplus(self.fc2(x))


class HexNet(nn.Module):
    """KataGo-style network for Hex with configurable multi-head architecture.

    Input:  (B, 13, 33, 33)
    Output: dict of head_name → tensor

    Heads:
        policy      — (B, 1089) policy logits
        value       — (B, N_BINS) binned value logits
        lookahead_* — (B, N_BINS) binned lookahead value logits
        opp_policy  — (B, 1089) opponent policy logits
        axis        — (B, 3) hex axis classification logits
        regret_rank — (B, 1) ranking score scalar
        regret_value— (B, N_BINS) binned regret value logits
        moves_left  — (B, 1) moves-left scalar (softplus)
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 16,
        heads: Optional[List[str]] = None,
        n_bins: int = 65,
    ):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.n_bins = n_bins

        if heads is None:
            heads = ["policy", "value"]
        self.head_names = list(heads)

        self.conv_in = nn.Conv2d(13, channels, kernel_size=3, padding=1)

        self.res_blocks = nn.ModuleList(
            [GatedResBlock(channels) for _ in range(blocks)]
        )

        head_modules: Dict[str, nn.Module] = {}
        for name in self.head_names:
            if name == "policy":
                head_modules[name] = PolicyHead(channels)
            elif name == "opp_policy":
                head_modules[name] = AuxPolicyHead(channels)
            elif name == "value" or name == "regret_value" or name.startswith("lookahead_"):
                head_modules[name] = ValueBinnedHead(channels, n_bins)
            elif name == "axis":
                head_modules[name] = AxisHead(channels)
            elif name == "regret_rank":
                head_modules[name] = RegretRankHead(channels)
            elif name == "moves_left":
                head_modules[name] = MovesLeftHead(channels)
            else:
                raise ValueError(f"Unknown head: {name}")
        self.heads = nn.ModuleDict(head_modules)

        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 13, 33, 33) float tensor.

        Returns:
            Dict mapping head name to output tensor.
        """
        x = torch.relu(self.conv_in(x))

        for block in self.res_blocks:
            x = block(x)

        out: Dict[str, torch.Tensor] = {}
        for name in self.head_names:
            out[name] = self.heads[name](x)

        return out

    @staticmethod
    def value_to_bins(t: torch.Tensor, n_bins: int = 65) -> torch.Tensor:
        """Convert continuous value in [-1, 1] to binned soft target.

        Uses linear interpolation between the two nearest bins —
        mirrors KataGo's value head target projection exactly.

        Args:
            t: (...,) tensor of continuous values in [-1, 1].
            n_bins: Number of bins (default 65).

        Returns:
            (..., n_bins) tensor of target probabilities summing to 1.
        """
        bin_width = 2.0 / (n_bins - 1)
        idx = (t + 1.0) / bin_width

        lo = idx.floor().long()
        hi = lo + 1
        hi = hi.clamp(min=0, max=n_bins - 1)
        lo = lo.clamp(min=0, max=n_bins - 1)

        w_hi = idx - lo.float()
        w_lo = 1.0 - w_hi

        target = torch.zeros(*t.shape, n_bins, device=t.device, dtype=torch.float32)
        target.scatter_add_(-1, lo.unsqueeze(-1), w_lo.unsqueeze(-1))
        target.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))

        return target

    @staticmethod
    def bins_to_value(logits: torch.Tensor) -> torch.Tensor:
        """Convert bin logits to expected value in [-1, 1].

        Args:
            logits: (..., N_BINS) tensor of logits.

        Returns:
            (...,) tensor of expected values in [-1, 1].
        """
        n_bins = logits.shape[-1]
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(-1.0, 1.0, n_bins, device=logits.device, dtype=logits.dtype)
        return (probs * bin_centers).sum(dim=-1)

    def half(self) -> "HexNet":
        """Convert to FP16 (chained, like nn.Module.half())."""
        super().half()
        return self

    @torch.no_grad()
    def forward_batch(
        self,
        x: torch.Tensor,
        autocast: bool = False,
        requested_heads: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference-only forward with optional heads filter and autocast.

        Args:
            x: (B, 13, 33, 33) input tensor.
            autocast: If True and CUDA available, use FP16 autocast.
            requested_heads: Optional list of head names to compute (filters output).

        Returns:
            Dict of requested head outputs (all heads if requested_heads is None).
        """
        if autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.forward(x)
        else:
            out = self.forward(x)

        if requested_heads is not None:
            out = {k: v for k, v in out.items() if k in requested_heads}

        return out


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
        heads=model_cfg.heads,
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
