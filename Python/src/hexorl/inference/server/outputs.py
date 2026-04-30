"""Server-side model output validation and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from hexorl.inference.protocol import InferenceOutputValidationError
from hexorl.models.heads.value import bins_to_scalar, bins_to_value


@dataclass(frozen=True)
class DenseForwardOutputs:
    policies: np.ndarray
    values: np.ndarray
    sparse_logits: Optional[np.ndarray] = None
    pair_logits: Optional[np.ndarray] = None
    regret_rank: Optional[np.ndarray] = None
    regret_value: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GraphForwardOutputs:
    place_logits: np.ndarray
    values: np.ndarray
    opp_logits: Optional[np.ndarray] = None
    pair_first_logits: Optional[np.ndarray] = None
    pair_joint_logits: Optional[np.ndarray] = None
    pair_second_logits: Optional[np.ndarray] = None
    regret_rank: Optional[np.ndarray] = None
    regret_value: Optional[np.ndarray] = None


def assert_finite_tensor(tensor: torch.Tensor, *, head_name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise InferenceOutputValidationError(f"inference model output contains non-finite values: {head_name}")


def bounded_policy_logits(logits: torch.Tensor, *, head_name: str) -> torch.Tensor:
    out = logits.float()
    assert_finite_tensor(out, head_name=head_name)
    return out.clone().clamp(-80.0, 80.0)


def bounded_value_logits(logits: torch.Tensor, *, head_name: str) -> torch.Tensor:
    out = logits.float()
    assert_finite_tensor(out, head_name=head_name)
    return out.clone().clamp(-80.0, 80.0)


def decode_dense_outputs(out: dict[str, torch.Tensor], *, sparse_inputs: dict[str, torch.Tensor] | None) -> DenseForwardOutputs:
    p = bounded_policy_logits(out["policy"], head_name="policy")
    value_logits = bounded_value_logits(out["value"], head_name="value")
    v = bins_to_value(value_logits).float().clamp(-1.0, 1.0)
    assert_finite_tensor(v, head_name="value_scalar")

    sparse = None
    pair = None
    regret = None
    regret_value = None
    if sparse_inputs is not None and "sparse_policy" in out:
        sparse = bounded_policy_logits(out["sparse_policy"], head_name="sparse_policy").cpu().numpy()
    if sparse_inputs is not None and "pair_policy" in out:
        pair = bounded_policy_logits(out["pair_policy"], head_name="pair_policy").cpu().numpy()
    if "regret_rank" in out:
        regret_tensor = out["regret_rank"].detach().float().reshape(-1)
        assert_finite_tensor(regret_tensor, head_name="regret_rank")
        regret = regret_tensor.cpu().numpy()
    if "regret_value" in out:
        regret_value_logits = bounded_value_logits(out["regret_value"], head_name="regret_value")
        regret_value_tensor = bins_to_scalar(regret_value_logits, min_value=0.0, max_value=4.0).float().clamp(0.0, 4.0)
        assert_finite_tensor(regret_value_tensor, head_name="regret_value_scalar")
        regret_value = regret_value_tensor.cpu().numpy()
    return DenseForwardOutputs(
        policies=p.cpu().numpy(),
        values=v.cpu().numpy(),
        sparse_logits=sparse,
        pair_logits=pair,
        regret_rank=regret,
        regret_value=regret_value,
    )


def decode_graph_outputs(out: dict[str, torch.Tensor]) -> GraphForwardOutputs:
    if "policy_place" not in out:
        raise RuntimeError("global graph model did not return policy_place")
    place = bounded_policy_logits(out["policy_place"], head_name="policy_place")
    value_logits = bounded_value_logits(out["value"], head_name="value")
    values = bins_to_value(value_logits).float().clamp(-1.0, 1.0)
    assert_finite_tensor(values, head_name="value_scalar")
    opp = bounded_policy_logits(out["opp_policy"], head_name="opp_policy") if "opp_policy" in out else None
    pair_first = bounded_policy_logits(out["policy_pair_first"], head_name="policy_pair_first") if "policy_pair_first" in out else None
    pair_joint = bounded_policy_logits(out["policy_pair_joint"], head_name="policy_pair_joint") if "policy_pair_joint" in out else None
    pair_second = bounded_policy_logits(out["policy_pair_second"], head_name="policy_pair_second") if "policy_pair_second" in out else None
    regret = out.get("regret_rank")
    if regret is not None:
        assert_finite_tensor(regret, head_name="regret_rank")
    regret_value_logits = out.get("regret_value")
    regret_value = None
    if regret_value_logits is not None:
        bounded_regret_value = bounded_value_logits(regret_value_logits, head_name="regret_value")
        regret_value = bins_to_scalar(bounded_regret_value, min_value=0.0, max_value=4.0).float().clamp(0.0, 4.0)
        assert_finite_tensor(regret_value, head_name="regret_value_scalar")
    return GraphForwardOutputs(
        place_logits=place.cpu().numpy(),
        values=values.cpu().numpy(),
        opp_logits=opp.cpu().numpy() if opp is not None else None,
        pair_first_logits=pair_first.cpu().numpy() if pair_first is not None else None,
        pair_joint_logits=pair_joint.cpu().numpy() if pair_joint is not None else None,
        pair_second_logits=pair_second.cpu().numpy() if pair_second is not None else None,
        regret_rank=regret.detach().float().cpu().numpy().reshape(-1) if regret is not None else None,
        regret_value=regret_value.detach().float().cpu().numpy().reshape(-1) if regret_value is not None else None,
    )


__all__ = [
    "DenseForwardOutputs",
    "GraphForwardOutputs",
    "assert_finite_tensor",
    "bounded_policy_logits",
    "bounded_value_logits",
    "decode_dense_outputs",
    "decode_graph_outputs",
]
