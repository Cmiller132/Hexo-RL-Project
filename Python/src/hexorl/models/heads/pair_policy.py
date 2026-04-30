"""Pair-policy head names and implementation."""

from typing import Optional

import torch
import torch.nn as nn

from hexorl.contracts.candidates import CANDIDATE_FEATURES


class PairPolicyHead(nn.Module):
    """Auxiliary pair-action head over selected candidate rows."""

    def __init__(
        self,
        channels: int,
        candidate_feature_dim: int = CANDIDATE_FEATURES,
        hidden: int = 128,
    ):
        super().__init__()
        self.base_dim = channels + candidate_feature_dim + 1
        self.net = nn.Sequential(
            nn.Linear(self.base_dim * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _candidate_embeddings(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = features.shape
        flat_features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
        idx = candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (idx >= 0) & (idx < h * w)
        idx_clamped = idx.clamp(0, h * w - 1)
        gather_idx = idx_clamped.unsqueeze(-1).expand(-1, -1, c)
        sampled = flat_features.gather(1, gather_idx)
        sampled = sampled * valid.unsqueeze(-1).to(dtype=sampled.dtype)
        if dense_policy_logits is None:
            dense = torch.zeros(
                b,
                candidate_features.shape[1],
                1,
                device=features.device,
                dtype=features.dtype,
            )
        else:
            dense_vals = dense_policy_logits.gather(1, idx_clamped)
            dense_vals = dense_vals * valid.to(dtype=dense_vals.dtype)
            dense = dense_vals.unsqueeze(-1)
        cand = candidate_features.to(device=features.device, dtype=features.dtype)
        return torch.cat([sampled, cand, dense], dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        dense_policy_logits: Optional[torch.Tensor],
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
        pair_candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        base = self._candidate_embeddings(
            features,
            dense_policy_logits,
            candidate_features,
            candidate_indices,
        )
        b, k, d = base.shape
        pair_idx = pair_candidate_indices.to(device=features.device, dtype=torch.long)
        valid = (pair_idx[..., 0] >= 0) & (pair_idx[..., 0] < k) & (pair_idx[..., 1] >= 0) & (pair_idx[..., 1] < k)
        clamped = pair_idx.clamp(0, max(k - 1, 0))
        first = base.gather(1, clamped[..., 0].unsqueeze(-1).expand(-1, -1, d))
        second = base.gather(1, clamped[..., 1].unsqueeze(-1).expand(-1, -1, d))
        x = torch.cat([first, second, (first - second).abs(), first * second], dim=-1)
        logits = self.net(x).squeeze(-1)
        return logits.masked_fill(~valid, -80.0)

CROP_PAIR_HEAD = "pair_policy"
GLOBAL_PAIR_HEADS = ("policy_pair_first", "policy_pair_second", "policy_pair_joint")


def pair_policy_heads(*, global_graph: bool) -> tuple[str, ...]:
    return GLOBAL_PAIR_HEADS if global_graph else (CROP_PAIR_HEAD,)


__all__ = ["CROP_PAIR_HEAD", "GLOBAL_PAIR_HEADS", "PairPolicyHead", "pair_policy_heads"]
