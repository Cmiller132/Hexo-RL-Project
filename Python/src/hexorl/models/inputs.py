"""Typed model input envelopes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CropInputs:
    tensor: torch.Tensor
    candidate_indices: torch.Tensor | None = None
    candidate_features: torch.Tensor | None = None
    candidate_mask: torch.Tensor | None = None
    pair_candidate_indices: torch.Tensor | None = None
    pair_candidate_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphInputs:
    token_features: torch.Tensor
    token_type: torch.Tensor
    token_qr: torch.Tensor
    token_mask: torch.Tensor
    legal_token_indices: torch.Tensor
    legal_mask: torch.Tensor
    opp_legal_qr: torch.Tensor | None = None
    opp_legal_mask: torch.Tensor | None = None
    pair_first_indices: torch.Tensor | None = None
    pair_second_indices: torch.Tensor | None = None
    pair_token_indices: torch.Tensor | None = None
    relation_type: torch.Tensor | None = None
    relation_bias: torch.Tensor | None = None
    crop_tensor: torch.Tensor | None = None


@dataclass(frozen=True)
class GlobalTrunkOutputs:
    token_states: torch.Tensor
    legal_states: torch.Tensor
    state_token: torch.Tensor
    legal_mask: torch.Tensor
    pair_states: torch.Tensor | None = None
    pair_mask: torch.Tensor | None = None
    opp_legal_states: torch.Tensor | None = None
    opp_legal_mask: torch.Tensor | None = None


__all__ = ["CropInputs", "GlobalTrunkOutputs", "GraphInputs"]
