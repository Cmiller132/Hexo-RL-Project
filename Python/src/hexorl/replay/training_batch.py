"""Training batch conversion that binds replay data to target contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from hexorl.graph.batch import GraphBatch, collate_graph_batches
from hexorl.train.loss_plan import infer_row_tables


@dataclass(frozen=True)
class PreparedTrainingBatch:
    model_inputs: Mapping[str, torch.Tensor]
    targets: dict[str, object]
    row_tables: Mapping[str, object]
    batch_size: int


def prepare_dense_training_batch(
    *,
    tensors,
    policies,
    values,
    lookahead_list: Sequence[object],
    aux_targets: Mapping[str, object],
    lookahead_keys: Sequence[str],
    device: torch.device,
    channels_last: bool,
    train_policy_on_full_search_only: bool,
) -> PreparedTrainingBatch:
    input_tensor = _to_device(tensors, device)
    if channels_last:
        input_tensor = input_tensor.contiguous(memory_format=torch.channels_last)
    targets: dict[str, object] = {
        "policy": _to_device(policies, device),
        "value": _to_device(values, device),
    }
    _attach_lookahead_targets(targets, lookahead_keys, lookahead_list, device)
    _attach_aux_targets(targets, aux_targets, device)
    _ensure_sample_weights(targets, "policy_weight", targets["policy"], train_policy_on_full_search_only)
    _ensure_sample_weights(targets, "value_weight", targets["value"], True)
    _ensure_sample_weights(targets, "sparse_policy_weight", targets["value"], True)
    _ensure_sample_weights(targets, "pair_policy_weight", targets["value"], True)
    return PreparedTrainingBatch(
        model_inputs={"tensors": input_tensor},
        targets=targets,
        row_tables=infer_row_tables(targets),
        batch_size=int(input_tensor.shape[0]),
    )


def prepare_global_graph_training_batch(
    *,
    values,
    lookahead_list: Sequence[object],
    aux_targets: Mapping[str, object],
    lookahead_keys: Sequence[str],
    device: torch.device,
    train_policy_on_full_search_only: bool,
    graph_batches: Sequence[GraphBatch] | None = None,
) -> PreparedTrainingBatch:
    targets: dict[str, object] = {"value": _to_device(values, device)}
    _attach_lookahead_targets(targets, lookahead_keys, lookahead_list, device)
    _attach_aux_targets(targets, aux_targets, device, skip_keys={"_graph_batches"})
    if graph_batches is not None:
        graph_batch = collate_graph_batches(graph_batches)
        targets.update(_graph_batch_targets_for_device(graph_batch, device))
    else:
        _attach_graph_phase_targets(targets, device)
    _ensure_sample_weights(targets, "policy_weight", targets["value"], train_policy_on_full_search_only)
    _ensure_sample_weights(targets, "value_weight", targets["value"], True)
    _ensure_sample_weights(targets, "pair_policy_weight", targets["value"], True)
    _ensure_sample_weights(targets, "opp_policy_weight", targets["value"], True, default=0.0)
    model_inputs = {
        key: targets[key]
        for key in (
            "token_features",
            "token_type",
            "token_qr",
            "token_mask",
            "legal_token_indices",
            "legal_mask",
            "relation_type",
            "relation_bias",
        )
    }
    optional_inputs = (
        "opp_legal_qr",
        "opp_legal_mask",
        "pair_first_indices",
        "pair_second_indices",
        "pair_token_indices",
    )
    for key in optional_inputs:
        if key in targets:
            model_inputs[key] = targets[key]
    return PreparedTrainingBatch(
        model_inputs=model_inputs,
        targets=targets,
        row_tables=infer_row_tables(targets),
        batch_size=int(targets["value"].shape[0]),  # type: ignore[index]
    )


def _graph_batch_targets_for_device(graph_batch: GraphBatch, device: torch.device) -> dict[str, torch.Tensor]:
    values = {
        "token_features": graph_batch.token_features,
        "token_type": graph_batch.token_type,
        "token_qr": graph_batch.token_qr,
        "token_mask": graph_batch.token_mask,
        "legal_token_indices": graph_batch.legal_token_indices,
        "legal_qr": graph_batch.legal_qr,
        "legal_mask": graph_batch.legal_mask,
        "pair_token_indices": graph_batch.pair_token_indices,
        "pair_first_indices": graph_batch.pair_first_indices,
        "pair_second_indices": graph_batch.pair_second_indices,
        "relation_type": graph_batch.relation_type,
        "relation_bias": graph_batch.relation_bias,
        "policy_target": graph_batch.policy_target,
        "legal_token_quality_target": graph_batch.policy_target,
        "opp_legal_qr": graph_batch.opp_legal_qr,
        "opp_legal_mask": graph_batch.opp_legal_mask,
        "opp_policy_target": graph_batch.opp_policy_target,
        "pair_first_policy_target": graph_batch.pair_first_policy_target,
        "pair_policy_target": graph_batch.pair_policy_target,
        "pair_second_policy_target": graph_batch.pair_second_policy_target,
        "tactical_target": graph_batch.tactical_target,
    }
    out = {key: _to_device(value, device) for key, value in values.items()}
    placements = getattr(graph_batch, "placements_remaining_by_sample", None)
    if placements is None:
        batch_size = int(out["value"].shape[0]) if "value" in out else int(out["legal_mask"].shape[0])
        placements_tensor = torch.full((batch_size,), int(graph_batch.placements_remaining), dtype=torch.long, device=device)
    else:
        placements_tensor = _to_device(placements, device).long()
    out["placements_remaining"] = placements_tensor
    _attach_graph_phase_targets(out, device)
    return out


def _attach_graph_phase_targets(targets: dict[str, object], device: torch.device) -> None:
    if "pair_first_indices" in targets and "pair_second_indices" in targets:
        first = _to_device(targets["pair_first_indices"], device).long()
        second = _to_device(targets["pair_second_indices"], device).long()
        pair_row_mask = (first >= 0) & (second >= 0) & (first != second)
        targets["pair_row_mask"] = pair_row_mask
    if "placements_remaining" in targets:
        known_first = _to_device(targets["placements_remaining"], device).long() == 1
        unordered_first = _to_device(targets["placements_remaining"], device).long() >= 2
        targets["pair_first_unordered"] = unordered_first
        targets["pair_second_known_first"] = known_first
        if "pair_row_mask" in targets:
            targets["pair_second_row_mask"] = _to_device(targets["pair_row_mask"], device, dtype=torch.bool) & known_first.unsqueeze(1)


def _attach_lookahead_targets(
    targets: dict[str, object],
    lookahead_keys: Sequence[str],
    lookahead_list: Sequence[object],
    device: torch.device,
) -> None:
    if len(lookahead_list) < len(lookahead_keys):
        missing = list(lookahead_keys[len(lookahead_list):])
        if missing:
            raise ValueError(f"training batch is missing lookahead targets: {missing}")
    for key, value in zip(lookahead_keys, lookahead_list):
        targets[key] = _to_device(value, device)


def _attach_aux_targets(
    targets: dict[str, object],
    aux_targets: Mapping[str, object],
    device: torch.device,
    *,
    skip_keys: set[str] | None = None,
) -> None:
    skip_keys = skip_keys or set()
    for key, value in aux_targets.items():
        if key in skip_keys:
            continue
        if key in targets:
            raise ValueError(f"auxiliary target {key!r} would overwrite prepared training target")
        if key.startswith("_"):
            targets[key] = value
            continue
        targets[key] = _to_device(value, device)


def _ensure_sample_weights(
    targets: dict[str, object],
    key: str,
    reference,
    train_on_existing: bool,
    *,
    default: float = 1.0,
) -> None:
    ref = _to_device(reference, reference.device if isinstance(reference, torch.Tensor) else torch.device("cpu"))
    batch_size = int(ref.shape[0])
    if key not in targets or not train_on_existing:
        targets[key] = torch.full((batch_size,), float(default), device=ref.device, dtype=torch.float32)


def _to_device(value, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, non_blocking=True)
    return tensor.to(dtype=dtype) if dtype is not None else tensor
