"""Value head names and implementation."""

import torch
import torch.nn as nn

from hexorl.models.constants import BOARD_AREA


class ValueBinnedHead(nn.Module):
    """Binned value head: (B, C, 33, 33) -> (B, N_BINS) logits.

    Used for: value, lookahead_*, regret_value.
    """

    def __init__(self, channels: int, n_bins: int = 65, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc1 = nn.Linear(BOARD_AREA, hidden)
        self.fc2 = nn.Linear(hidden, n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def value_to_bins(t: torch.Tensor, n_bins: int = 65) -> torch.Tensor:
    """Convert continuous values in [-1, 1] to interpolated binned targets."""
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


def bins_to_value(logits: torch.Tensor) -> torch.Tensor:
    """Convert value-bin logits to expected values in [-1, 1]."""
    n_bins = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(-1.0, 1.0, n_bins, device=logits.device, dtype=logits.dtype)
    return (probs * bin_centers).sum(dim=-1)


def bins_to_scalar(logits: torch.Tensor, *, min_value: float, max_value: float) -> torch.Tensor:
    """Convert binned scalar logits to expected values over a fixed range."""
    n_bins = logits.shape[-1]
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(
        float(min_value),
        float(max_value),
        n_bins,
        device=logits.device,
        dtype=logits.dtype,
    )
    return (probs * bin_centers).sum(dim=-1)


VALUE_HEAD = "value"


__all__ = ["VALUE_HEAD", "ValueBinnedHead", "bins_to_scalar", "bins_to_value", "value_to_bins"]
