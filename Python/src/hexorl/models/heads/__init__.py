"""Head registry and named head implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from hexorl.contracts.candidates import CANDIDATE_FEATURES
from hexorl.models.inference_contracts import (
    BOARD_AREA,
    DECODER_POLICY_LOGITS,
    DECODER_REGRET_BINS,
    DECODER_SCALAR,
    DECODER_VALUE_BINS,
    DIM_BATCH,
    DIM_CANDIDATE,
    DIM_GRAPH_PAIR,
    DIM_LEGAL,
    DIM_OPP_LEGAL,
    DIM_PAIR,
    CapacitySpec,
    HeadDecoderSpec,
    TensorSpec,
)
from hexorl.models.inputs import CropInputs, GlobalTrunkOutputs, GraphInputs
from hexorl.models.heads.pair_policy import CROP_PAIR_HEAD, GLOBAL_PAIR_HEADS, PairPolicyHead
from hexorl.models.heads.policy import DENSE_POLICY_HEADS, GLOBAL_PLACE_HEAD, PolicyHead
from hexorl.models.heads.regret import REGRET_HEADS, RegretRankHead
from hexorl.models.heads.sparse_policy import GRAPH_HYBRID_POLICY_HEADS, SPARSE_POLICY_HEAD, SparsePolicyHead
from hexorl.models.heads.tactical import AuxPolicyHead, AxisHead, AxisMapHead, MovesLeftHead
from hexorl.models.heads.value import VALUE_HEAD, ValueBinnedHead, bins_to_scalar, bins_to_value, value_to_bins


@dataclass(frozen=True)
class TrunkOutputSpec:
    kind: str
    feature_channels: int = 0
    feature_dim: int = 0
    n_bins: int = 65
    candidate_feature_dim: int = CANDIDATE_FEATURES


@dataclass(frozen=True)
class HeadSpec:
    factory: Callable[[TrunkOutputSpec, CapacitySpec], nn.Module]
    output_tensor: TensorSpec
    decoder: HeadDecoderSpec
    row_mapping: str


def _tensor(name: str, shape: tuple[int | str, ...], semantic: str) -> TensorSpec:
    return TensorSpec(name, "float32", shape, semantic, "pad_and_stack")


def _decoder(name: str, kind: str, **kwargs) -> HeadDecoderSpec:
    return HeadDecoderSpec(name, kind, clamp_min=-80.0, clamp_max=80.0, **kwargs)


class _CropPlainHead(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, features: torch.Tensor, inputs: CropInputs) -> torch.Tensor:
        del inputs
        return self.module(features)


class _CropSparseHead(nn.Module):
    def __init__(self, spec: TrunkOutputSpec):
        super().__init__()
        self.module = SparsePolicyHead(spec.feature_channels, spec.candidate_feature_dim, max(64, min(256, spec.feature_channels * 2)))

    def forward(self, features: torch.Tensor, inputs: CropInputs) -> torch.Tensor:
        if inputs.candidate_features is None or inputs.candidate_indices is None:
            return features.new_empty((features.shape[0], 0))
        sparse = self.module(features, None, inputs.candidate_features, inputs.candidate_indices)
        if inputs.candidate_mask is not None:
            sparse = sparse.masked_fill(~inputs.candidate_mask.to(device=sparse.device, dtype=torch.bool), -80.0)
        return sparse


class _CropPairHead(nn.Module):
    def __init__(self, spec: TrunkOutputSpec):
        super().__init__()
        self.module = PairPolicyHead(spec.feature_channels, spec.candidate_feature_dim, max(64, min(256, spec.feature_channels * 2)))

    def forward(self, features: torch.Tensor, inputs: CropInputs) -> torch.Tensor:
        if inputs.candidate_features is None or inputs.candidate_indices is None or inputs.pair_candidate_indices is None:
            return features.new_empty((features.shape[0], 0))
        pair = self.module(features, None, inputs.candidate_features, inputs.candidate_indices, inputs.pair_candidate_indices)
        if inputs.pair_candidate_mask is not None:
            pair = pair.masked_fill(~inputs.pair_candidate_mask.to(device=pair.device, dtype=torch.bool), -80.0)
        return pair


class _GlobalStateHead(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, outputs: GlobalTrunkOutputs, inputs: GraphInputs) -> torch.Tensor:
        del inputs
        return self.module(outputs.state_token)


class _GlobalLegalHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.module = nn.Linear(dim, 1)

    def forward(self, outputs: GlobalTrunkOutputs, inputs: GraphInputs) -> torch.Tensor:
        del inputs
        return self.module(outputs.legal_states).squeeze(-1).masked_fill(~outputs.legal_mask, -80.0)


class _GlobalOppHead(_GlobalLegalHead):
    def forward(self, outputs: GlobalTrunkOutputs, inputs: GraphInputs) -> torch.Tensor:
        del inputs
        if outputs.opp_legal_states is None or outputs.opp_legal_mask is None:
            return outputs.state_token.new_empty((outputs.state_token.shape[0], 0))
        return self.module(outputs.opp_legal_states).squeeze(-1).masked_fill(~outputs.opp_legal_mask, -80.0)


class _GlobalPairHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(dim * 4, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, outputs: GlobalTrunkOutputs, inputs: GraphInputs) -> torch.Tensor:
        del inputs
        if outputs.pair_states is None or outputs.pair_mask is None:
            raise ValueError("pair heads require graph pair rows")
        return self.module(outputs.pair_states).squeeze(-1).masked_fill(~outputs.pair_mask, -80.0)


def _crop_value(spec: TrunkOutputSpec, capacity: CapacitySpec) -> nn.Module:
    del capacity
    return _CropPlainHead(ValueBinnedHead(spec.feature_channels, spec.n_bins))


def _global_value(spec: TrunkOutputSpec, capacity: CapacitySpec) -> nn.Module:
    del capacity
    return _GlobalStateHead(nn.Sequential(nn.Linear(spec.feature_dim, spec.feature_dim), nn.SiLU(), nn.Linear(spec.feature_dim, spec.n_bins)))


HEAD_REGISTRY: dict[str, HeadSpec] = {
    "policy": HeadSpec(lambda s, c: _CropPlainHead(PolicyHead(s.feature_channels)), _tensor("policy", (DIM_BATCH, BOARD_AREA), DIM_BATCH), _decoder("policy", DECODER_POLICY_LOGITS), DIM_BATCH),
    "opp_policy": HeadSpec(lambda s, c: _CropPlainHead(AuxPolicyHead(s.feature_channels)) if s.kind == "crop" else _GlobalOppHead(s.feature_dim), _tensor("opp_policy", (DIM_OPP_LEGAL,), DIM_OPP_LEGAL), _decoder("opp_policy", DECODER_POLICY_LOGITS), DIM_OPP_LEGAL),
    "value": HeadSpec(lambda s, c: _crop_value(s, c) if s.kind == "crop" else _global_value(s, c), _tensor("value", (DIM_BATCH,), DIM_BATCH), _decoder("value", DECODER_VALUE_BINS), DIM_BATCH),
    "sparse_policy": HeadSpec(lambda s, c: _CropSparseHead(s), _tensor("sparse_policy", (DIM_BATCH, DIM_CANDIDATE), DIM_CANDIDATE), _decoder("sparse_policy", DECODER_POLICY_LOGITS), DIM_CANDIDATE),
    "pair_policy": HeadSpec(lambda s, c: _CropPairHead(s), _tensor("pair_policy", (DIM_BATCH, DIM_PAIR), DIM_PAIR), _decoder("pair_policy", DECODER_POLICY_LOGITS), DIM_PAIR),
    "policy_place": HeadSpec(lambda s, c: _GlobalLegalHead(s.feature_dim), _tensor("policy_place", (DIM_LEGAL,), DIM_LEGAL), _decoder("policy_place", DECODER_POLICY_LOGITS), DIM_LEGAL),
    "policy_pair_first": HeadSpec(lambda s, c: _GlobalLegalHead(s.feature_dim), _tensor("policy_pair_first", (DIM_LEGAL,), DIM_LEGAL), _decoder("policy_pair_first", DECODER_POLICY_LOGITS), DIM_LEGAL),
    "policy_pair_second": HeadSpec(lambda s, c: _GlobalPairHead(s.feature_dim), _tensor("policy_pair_second", (DIM_GRAPH_PAIR,), DIM_GRAPH_PAIR), _decoder("policy_pair_second", DECODER_POLICY_LOGITS), DIM_GRAPH_PAIR),
    "policy_pair_joint": HeadSpec(lambda s, c: _GlobalPairHead(s.feature_dim), _tensor("policy_pair_joint", (DIM_GRAPH_PAIR,), DIM_GRAPH_PAIR), _decoder("policy_pair_joint", DECODER_POLICY_LOGITS), DIM_GRAPH_PAIR),
    "regret_rank": HeadSpec(lambda s, c: _CropPlainHead(RegretRankHead(s.feature_channels)) if s.kind == "crop" else _GlobalStateHead(nn.Sequential(nn.Linear(s.feature_dim, s.feature_dim), nn.SiLU(), nn.Linear(s.feature_dim, 1))), _tensor("regret_rank", (DIM_BATCH,), DIM_BATCH), _decoder("regret_rank", DECODER_SCALAR), DIM_BATCH),
    "regret_value": HeadSpec(lambda s, c: _crop_value(s, c) if s.kind == "crop" else _global_value(s, c), _tensor("regret_value", (DIM_BATCH,), DIM_BATCH), _decoder("regret_value", DECODER_REGRET_BINS, min_value=0.0, max_value=4.0), DIM_BATCH),
    "axis": HeadSpec(lambda s, c: _CropPlainHead(AxisHead(s.feature_channels)), _tensor("axis", (DIM_BATCH, 3), DIM_BATCH), _decoder("axis", DECODER_SCALAR), DIM_BATCH),
    "axis_delta_norm": HeadSpec(lambda s, c: _CropPlainHead(AxisMapHead(s.feature_channels)), _tensor("axis_delta_norm", (DIM_BATCH, 6, 33, 33), DIM_BATCH), _decoder("axis_delta_norm", DECODER_SCALAR), DIM_BATCH),
    "moves_left": HeadSpec(lambda s, c: _CropPlainHead(MovesLeftHead(s.feature_channels)) if s.kind == "crop" else _GlobalStateHead(nn.Sequential(nn.Linear(s.feature_dim, s.feature_dim), nn.SiLU(), nn.Linear(s.feature_dim, 1), nn.Softplus())), _tensor("moves_left", (DIM_BATCH,), DIM_BATCH), _decoder("moves_left", DECODER_SCALAR), DIM_BATCH),
    "tactical": HeadSpec(lambda s, c: _GlobalStateHead(nn.Linear(s.feature_dim, 4)), _tensor("tactical", (DIM_BATCH, 4), DIM_BATCH), _decoder("tactical", DECODER_SCALAR), DIM_BATCH),
    "legal_token_quality": HeadSpec(lambda s, c: _GlobalLegalHead(s.feature_dim), _tensor("legal_token_quality", (DIM_LEGAL,), DIM_LEGAL), _decoder("legal_token_quality", DECODER_POLICY_LOGITS), DIM_LEGAL),
}

for _horizon in (4, 12, 36):
    _name = f"lookahead_{_horizon}"
    HEAD_REGISTRY[_name] = HeadSpec(lambda s, c: _crop_value(s, c) if s.kind == "crop" else _global_value(s, c), _tensor(_name, (DIM_BATCH,), DIM_BATCH), _decoder(_name, DECODER_VALUE_BINS), DIM_BATCH)


def build_heads_for_family(spec, cfg, trunk: nn.Module) -> dict[str, nn.Module]:
    del cfg
    output = TrunkOutputSpec(
        kind="global" if hasattr(trunk, "feature_dim") else "crop",
        feature_channels=int(getattr(trunk, "feature_channels", 0)),
        feature_dim=int(getattr(trunk, "feature_dim", 0)),
        n_bins=int(getattr(spec.params, "n_bins", 65)),
    )
    capacity = CapacitySpec(max_batch_size=1)
    heads: dict[str, nn.Module] = {}
    for name in spec.params.heads:
        heads[name] = HEAD_REGISTRY[name].factory(output, capacity)
    return heads


GLOBAL_GRAPH_OUTPUT_HEADS = (GLOBAL_PLACE_HEAD, *GLOBAL_PAIR_HEADS, VALUE_HEAD)

__all__ = [
    "AuxPolicyHead", "AxisHead", "AxisMapHead", "CROP_PAIR_HEAD", "DENSE_POLICY_HEADS",
    "GLOBAL_GRAPH_OUTPUT_HEADS", "GLOBAL_PAIR_HEADS", "GLOBAL_PLACE_HEAD", "GRAPH_HYBRID_POLICY_HEADS",
    "HEAD_REGISTRY", "HeadSpec", "MovesLeftHead", "PairPolicyHead", "PolicyHead", "REGRET_HEADS",
    "RegretRankHead", "SPARSE_POLICY_HEAD", "SparsePolicyHead", "TrunkOutputSpec", "VALUE_HEAD",
    "ValueBinnedHead", "bins_to_scalar", "bins_to_value", "build_heads_for_family", "value_to_bins",
]
