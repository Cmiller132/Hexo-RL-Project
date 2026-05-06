"""Model-specific inference adapters.

The server executes models and delegates semantic decoding here.  The client
also delegates shared-memory response decoding here so transport layout is not
treated as output meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import torch

from hexorl.inference.protocol import (
    GRAPH_HEAD_OPP,
    GRAPH_HEAD_PAIR_FIRST,
    GRAPH_HEAD_PAIR_JOINT,
    GRAPH_HEAD_PAIR_SECOND,
    GRAPH_HEAD_REGRET,
    OutputMetadata,
    row_table_metadata,
    validate_row_arrays_match,
    value_decoder_metadata,
)
from hexorl.models.contracts import ValueDecoderContract


ValueDecoderFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DecodedDenseOutputs:
    policy: np.ndarray
    value: np.ndarray
    sparse_policy: np.ndarray | None = None
    pair_policy: np.ndarray | None = None
    regret_rank: np.ndarray | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class DecodedGraphOutputs:
    policy_place: np.ndarray
    value: np.ndarray
    opp_policy: np.ndarray | None = None
    pair_first: np.ndarray | None = None
    pair_joint: np.ndarray | None = None
    pair_second: np.ndarray | None = None
    regret_rank: np.ndarray | None = None
    metadata: dict[str, object] | None = None


def sanitize_policy_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    ).clamp_(-80.0, 80.0)


def sanitize_value_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        logits.float(),
        nan=0.0,
        posinf=80.0,
        neginf=-80.0,
    ).clamp_(-80.0, 80.0)


def _require_output(outputs: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if value is None:
        raise RuntimeError(f"inference adapter requires model output {key!r}")
    return value


def _require_2d(name: str, tensor: torch.Tensor, width: int | None = None) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{name} output must be rank-2, got shape {tuple(tensor.shape)}")
    if width is not None and int(tensor.shape[1]) < int(width):
        raise ValueError(f"{name} output has width {tensor.shape[1]} for {width} row-table rows")
    return tensor


def _finite_numpy(name: str, tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} decoded output contains non-finite values")
    return arr


def decode_dense_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    value_decoder: ValueDecoderFn,
    sparse_requested: bool,
    value_contract: ValueDecoderContract | None = None,
) -> DecodedDenseOutputs:
    policy = _require_2d("policy", sanitize_policy_logits(_require_output(outputs, "policy")))
    value_logits = sanitize_value_logits(_require_output(outputs, "value"))
    values = torch.nan_to_num(
        value_decoder(value_logits).float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp_(-1.0, 1.0)
    batch = int(policy.shape[0])
    if values.reshape(-1).shape[0] != batch:
        raise ValueError(
            f"value decoder returned {values.reshape(-1).shape[0]} values for batch {batch}"
        )

    sparse = None
    if sparse_requested and "sparse_policy" in outputs:
        sparse = _finite_numpy("sparse_policy", _require_2d("sparse_policy", sanitize_policy_logits(outputs["sparse_policy"])))
        if sparse.shape[0] != batch:
            raise ValueError("sparse_policy batch dimension does not match policy")

    pair = None
    if sparse_requested and "pair_policy" in outputs:
        pair = _finite_numpy("pair_policy", _require_2d("pair_policy", sanitize_policy_logits(outputs["pair_policy"])))
        if pair.shape[0] != batch:
            raise ValueError("pair_policy batch dimension does not match policy")

    regret = None
    if "regret_rank" in outputs:
        regret = torch.nan_to_num(
            outputs["regret_rank"].detach().float().reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).cpu().numpy()

    metadata = {
        "outputs": {
            "policy": OutputMetadata(
                "policy",
                "policy",
                row_table=row_table_metadata(
                    "dense_board",
                    np.arange(policy.shape[1], dtype=np.int16),
                    np.ones(policy.shape[1], dtype=np.bool_),
                    phase="any_position",
                    source="dense_inference_adapter",
                ),
            ).to_dict(),
            "value": OutputMetadata(
                "value",
                "value",
                value_decoder=value_decoder_metadata(value_contract),
            ).to_dict(),
        }
    }
    return DecodedDenseOutputs(
        policy=_finite_numpy("policy", policy),
        value=_finite_numpy("value", values.reshape(-1)),
        sparse_policy=sparse,
        pair_policy=pair,
        regret_rank=regret,
        metadata=metadata,
    )


def decode_global_graph_outputs(
    outputs: Mapping[str, torch.Tensor],
    graph_inputs: Mapping[str, torch.Tensor],
    *,
    value_decoder: ValueDecoderFn,
    value_contract: ValueDecoderContract | None = None,
) -> DecodedGraphOutputs:
    legal_mask = graph_inputs.get("legal_mask")
    if legal_mask is None or legal_mask.ndim != 2:
        raise ValueError("global graph inference requires rank-2 legal_mask input")
    batch, legal_width = int(legal_mask.shape[0]), int(legal_mask.shape[1])
    pair_first = graph_inputs.get("pair_first_indices")
    pair_width = int(pair_first.shape[1]) if pair_first is not None and pair_first.ndim == 2 else 0
    opp_rows = graph_inputs.get("opp_legal_qr")
    opp_width = int(opp_rows.shape[1]) if opp_rows is not None and opp_rows.ndim == 3 else 0

    place = _require_2d(
        "policy_place",
        sanitize_policy_logits(_require_output(outputs, "policy_place")),
        legal_width,
    )
    if int(place.shape[0]) != batch:
        raise ValueError("policy_place batch dimension does not match graph inputs")

    value_logits = sanitize_value_logits(_require_output(outputs, "value"))
    values = torch.nan_to_num(
        value_decoder(value_logits).float(),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).clamp_(-1.0, 1.0)
    if values.reshape(-1).shape[0] != batch:
        raise ValueError(
            f"value decoder returned {values.reshape(-1).shape[0]} values for graph batch {batch}"
        )

    opp = None
    if "opp_policy" in outputs:
        opp_t = _require_2d("opp_policy", sanitize_policy_logits(outputs["opp_policy"]), opp_width)
        if int(opp_t.shape[0]) != batch:
            raise ValueError("opp_policy batch dimension does not match graph inputs")
        opp = _finite_numpy("opp_policy", opp_t)

    pair_first_np = None
    if "policy_pair_first" in outputs:
        pfirst_t = _require_2d(
            "policy_pair_first",
            sanitize_policy_logits(outputs["policy_pair_first"]),
            legal_width,
        )
        if int(pfirst_t.shape[0]) != batch:
            raise ValueError("policy_pair_first batch dimension does not match graph inputs")
        pair_first_np = _finite_numpy("policy_pair_first", pfirst_t)

    pair_joint = None
    if "policy_pair_joint" in outputs:
        joint_t = _require_2d(
            "policy_pair_joint",
            sanitize_policy_logits(outputs["policy_pair_joint"]),
            pair_width,
        )
        if int(joint_t.shape[0]) != batch:
            raise ValueError("policy_pair_joint batch dimension does not match graph inputs")
        pair_joint = _finite_numpy("policy_pair_joint", joint_t)

    pair_second = None
    if "policy_pair_second" in outputs:
        second_t = _require_2d(
            "policy_pair_second",
            sanitize_policy_logits(outputs["policy_pair_second"]),
            pair_width,
        )
        if int(second_t.shape[0]) != batch:
            raise ValueError("policy_pair_second batch dimension does not match graph inputs")
        pair_second = _finite_numpy("policy_pair_second", second_t)

    regret = None
    if "regret_rank" in outputs:
        regret = torch.nan_to_num(
            outputs["regret_rank"].detach().float().reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).cpu().numpy()

    metadata = {
        "outputs": {
            "policy_place": OutputMetadata("policy_place", "policy").to_dict(),
            "value": OutputMetadata(
                "value",
                "value",
                value_decoder=value_decoder_metadata(value_contract),
            ).to_dict(),
        }
    }
    return DecodedGraphOutputs(
        policy_place=_finite_numpy("policy_place", place),
        value=_finite_numpy("value", values.reshape(-1)),
        opp_policy=opp,
        pair_first=pair_first_np,
        pair_joint=pair_joint,
        pair_second=pair_second,
        regret_rank=regret,
        metadata=metadata,
    )


def decode_graph_slot_response(
    slot,
    graph_batches,
    counts: list[tuple[int, int, int, int]],
    offsets: list[tuple[int, int, int, int]],
    *,
    head_flags: int,
) -> list[dict[str, np.ndarray | dict[str, object]]]:
    results: list[dict[str, np.ndarray | dict[str, object]]] = []
    for row, (graph_batch, count_row, offset_row) in enumerate(zip(graph_batches, counts, offsets)):
        token_count, legal_count, opp_count, pair_count = count_row
        _token_off, legal_off, opp_off, pair_off = offset_row
        observed_legal = np.array(
            slot.req_graph_legal_qr[legal_off : legal_off + legal_count],
            copy=True,
        )
        observed_legal_mask = np.array(
            slot.req_graph_legal_mask[legal_off : legal_off + legal_count].astype(bool),
            copy=True,
        )
        legal_meta = validate_row_arrays_match(
            np.asarray(graph_batch.legal_qr, dtype=np.int32),
            np.asarray(graph_batch.legal_mask, dtype=np.bool_),
            observed_legal,
            observed_legal_mask,
            family="legal",
            phase="any_position",
            context="graph inference response",
        )
        meta: dict[str, object] = {
            "schema_version": int(slot.res_graph_meta[0]),
            "relation_schema_version": int(slot.res_graph_meta[1]),
            "legal_count": int(legal_count),
            "opp_legal_count": int(opp_count),
            "pair_count": int(pair_count),
            "token_count": int(token_count),
            "head_flags": int(head_flags),
            "prior_source": "global_graph",
            "legal_qr": observed_legal,
            "legal_mask": observed_legal_mask,
            "row_tables": {"policy_place": legal_meta.to_dict()},
            "outputs": {
                "policy_place": OutputMetadata(
                    "policy_place",
                    "policy",
                    row_table=legal_meta,
                ).to_dict(),
                "value": OutputMetadata(
                    "value",
                    "value",
                    value_decoder=value_decoder_metadata(),
                ).to_dict(),
            },
        }
        result: dict[str, np.ndarray | dict[str, object]] = {
            "policy_place": np.array(slot.res_graph_place_logits[legal_off : legal_off + legal_count], copy=True),
            "value": np.array(slot.res_value[row : row + 1], copy=True),
            "metadata": meta,
        }
        if head_flags & GRAPH_HEAD_OPP:
            result["opp_policy"] = np.array(slot.res_graph_opp_logits[opp_off : opp_off + opp_count], copy=True)
        if head_flags & GRAPH_HEAD_PAIR_FIRST:
            result["policy_pair_first"] = np.array(
                slot.res_graph_pair_first_logits[legal_off : legal_off + legal_count],
                copy=True,
            )
        if head_flags & GRAPH_HEAD_PAIR_JOINT:
            result["policy_pair_joint"] = np.array(
                slot.res_graph_pair_logits[pair_off : pair_off + pair_count],
                copy=True,
            )
        if head_flags & GRAPH_HEAD_PAIR_SECOND:
            result["policy_pair_second"] = np.array(
                slot.res_graph_pair_second_logits[pair_off : pair_off + pair_count],
                copy=True,
            )
        if head_flags & GRAPH_HEAD_REGRET:
            result["regret_rank"] = np.array(
                getattr(slot, "res_graph_regret_rank")[row : row + 1],
                copy=True,
            )
        results.append(result)
    return results
