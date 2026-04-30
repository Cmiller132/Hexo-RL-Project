"""Sparse row-policy head names and implementation."""

from typing import Optional

import torch
import torch.nn as nn

from hexorl.contracts.candidates import CANDIDATE_FEATURES


class SparsePolicyHead(nn.Module):
    """Candidate/action-keyed policy head.

    The head consumes reusable trunk features, candidate features, optional
    in-crop dense logits, and optional in-crop trunk samples. Invalid
    candidates are masked by the loss, not by this module.
    """

    def __init__(
        self,
        channels: int,
        candidate_feature_dim: int = CANDIDATE_FEATURES,
        hidden: int = 128,
    ):
        super().__init__()
        self.candidate_feature_dim = candidate_feature_dim
        self.net = nn.Sequential(
            nn.Linear(channels + candidate_feature_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = features.shape
        k = candidate_features.shape[1]
        flat_features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        idx = candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (idx >= 0) & (idx < h * w)
        idx_clamped = idx.clamp(0, h * w - 1)
        gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, c)
        sampled = flat_features.gather(1, gather_idx)
        sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)

        if dense_policy_logits is None:
            dense = torch.zeros(b, k, 1, device=features.device, dtype=features.dtype)
        else:
            dense_vals = dense_policy_logits.gather(1, idx_clamped)
            dense_vals = dense_vals * valid.to(dtype=dense_vals.dtype)
            dense = dense_vals.unsqueeze(-1)

        cand = candidate_features.to(device=features.device, dtype=features.dtype)
        x = torch.cat([sampled, cand, dense], dim=-1)
        return self.net(x).squeeze(-1)

SPARSE_POLICY_HEAD = "sparse_policy"
GRAPH_HYBRID_POLICY_HEADS = ("policy", SPARSE_POLICY_HEAD)


__all__ = ["GRAPH_HYBRID_POLICY_HEADS", "SPARSE_POLICY_HEAD", "SparsePolicyHead"]
