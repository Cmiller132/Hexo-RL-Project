"""Registry-owned training adapters and target validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from hexorl.models.specs import ModelSpec
from hexorl.train.losses import compute_losses


@dataclass
class ProjectedTrainingBatch:
    inputs: tuple[Any, ...]
    kwargs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]


@dataclass
class TrainingDebugBundle:
    trace_id: str
    owner: str
    spec_kind: str
    input_keys: list[str]
    target_keys: list[str]
    mask_keys: list[str]
    output_keys: list[str]
    loss_keys: list[str]
    tensor_hashes: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "owner": self.owner,
            "spec_kind": self.spec_kind,
            "input_keys": self.input_keys,
            "target_keys": self.target_keys,
            "mask_keys": self.mask_keys,
            "output_keys": self.output_keys,
            "loss_keys": self.loss_keys,
            "tensor_hashes": self.tensor_hashes,
        }


@dataclass
class TrainAdapter:
    """Single trainer-facing projection path for every model family."""

    spec: ModelSpec
    cfg: Any
    model: nn.Module
    device: torch.device
    loss_plan: Any

    def project_batch(self, batch: Any, *, channels_last: bool = False) -> ProjectedTrainingBatch:
        batch = _require_projected_replay_batch(batch)
        aux_targets = dict(batch.aux_targets)
        tensors = self._to_device(batch.tensors)
        if channels_last and tensors.ndim == 4:
            tensors = tensors.contiguous(memory_format=torch.channels_last)
        targets = {
            "policy": self._to_device(batch.policies),
            "value": self._to_device(batch.values),
        }
        lookahead_keys = [f"lookahead_{h}" for h in getattr(self.cfg.buffer, "lookahead_horizons", [])]
        for key, value in zip(lookahead_keys, batch.lookahead):
            targets[key] = self._to_device(value)
        for key, value in aux_targets.items():
            targets[key] = self._to_device(value)
        if not getattr(self.cfg.selfplay, "train_policy_on_full_search_only", True):
            targets.pop("policy_weight", None)

        return self._project_targets_to_model_inputs(tensors, targets)

    def _project_targets_to_model_inputs(self, tensors: Any, targets: dict[str, torch.Tensor]) -> ProjectedTrainingBatch:
        if self.spec.is_global_graph:
            self.validate_graph_targets(targets)
            kwargs = {
                "token_features": targets["token_features"],
                "token_type": targets["token_type"],
                "token_qr": targets["token_qr"],
                "token_mask": targets["token_mask"],
                "legal_token_indices": targets["legal_token_indices"],
                "legal_mask": targets["legal_mask"],
                "opp_legal_qr": targets.get("opp_legal_qr"),
                "opp_legal_mask": targets.get("opp_legal_mask"),
                "pair_first_indices": targets.get("pair_first_indices"),
                "pair_second_indices": targets.get("pair_second_indices"),
                "pair_token_indices": targets.get("pair_token_indices"),
                "relation_type": targets["relation_type"],
                "relation_bias": targets["relation_bias"],
            }
            return ProjectedTrainingBatch(inputs=(), kwargs={k: v for k, v in kwargs.items() if v is not None}, targets=targets)

        self.validate_crop_targets(targets)
        kwargs = {
            "candidate_features": targets.get("candidate_features"),
            "candidate_indices": targets.get("candidate_indices"),
            "candidate_mask": targets.get("candidate_mask"),
            "pair_candidate_features": targets.get("pair_candidate_features"),
            "pair_candidate_row_indices": targets.get("pair_candidate_row_indices"),
            "pair_candidate_indices": targets.get("pair_candidate_indices"),
            "pair_candidate_mask": targets.get("pair_candidate_mask"),
        }
        return ProjectedTrainingBatch(inputs=(tensors,), kwargs={k: v for k, v in kwargs.items() if v is not None}, targets=targets)

    def forward(self, projected: ProjectedTrainingBatch) -> dict[str, torch.Tensor]:
        outputs = self.model(*projected.inputs, **projected.kwargs)
        self.validate_outputs(outputs, projected.targets)
        return outputs

    def losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        n_bins: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return compute_losses(outputs, targets, loss_weights=self.loss_plan.weights, n_bins=n_bins)

    def debug_bundle(
        self,
        projected: ProjectedTrainingBatch,
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        *,
        trace_id: str,
    ) -> TrainingDebugBundle:
        tensors = {**projected.targets, **projected.kwargs}
        return TrainingDebugBundle(
            trace_id=trace_id,
            owner="train_adapter",
            spec_kind=self.spec.kind,
            input_keys=sorted(projected.kwargs.keys()) or ["crop_tensor"],
            target_keys=sorted(projected.targets.keys()),
            mask_keys=sorted(key for key in projected.targets if key.endswith("mask")),
            output_keys=sorted(outputs.keys()),
            loss_keys=sorted(losses.keys()),
            tensor_hashes={
                key: _tensor_identity_hash(value)
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor)
            },
        )

    def validate_crop_targets(self, targets: dict[str, torch.Tensor]) -> None:
        _require_finite("policy", targets["policy"])
        _require_finite("value", targets["value"])
        if "candidate_mask" in targets:
            _require_shape_match("sparse_policy_target", targets.get("sparse_policy_target"), targets["candidate_mask"])
        if "pair_candidate_mask" in targets:
            _require_shape_match("pair_policy_target", targets.get("pair_policy_target"), targets["pair_candidate_mask"])

    def validate_graph_targets(self, targets: dict[str, torch.Tensor]) -> None:
        required = [
            "token_features",
            "token_type",
            "token_qr",
            "token_mask",
            "legal_token_indices",
            "legal_qr",
            "legal_mask",
            "relation_type",
            "relation_bias",
        ]
        missing = [name for name in required if name not in targets]
        if missing:
            raise ValueError(f"train adapter graph projection missing input contracts: {missing}")
        _require_shape_match("policy_target", targets.get("policy_target"), targets["legal_mask"])
        if "pair_policy_target" in targets:
            if "pair_token_indices" not in targets:
                raise ValueError("train adapter owner: pair target requires pair_token_indices")
            for name in ("pair_first_indices", "pair_second_indices", "pair_rows", "pair_table_mask", "pair_phase"):
                if name not in targets:
                    raise ValueError(f"train adapter owner: pair target requires canonical {name}")
            _require_shape_match("pair_policy_target", targets["pair_policy_target"], targets["pair_token_indices"])
            _validate_graph_pair_semantics(targets)
            if bool(getattr(self.cfg.model, "pair_strategy", "none") != "none") and not torch.any(targets["pair_token_indices"] >= 0):
                raise ValueError("train adapter owner: pair loss is invalid on opening positions without pair rows")
        for name in ("policy_target", "pair_policy_target", "legal_mask", "token_features", "relation_bias"):
            if name in targets:
                _require_finite(name, targets[name])

    def validate_outputs(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> None:
        for name, value in outputs.items():
            _require_finite(f"model_output:{name}", value)
        if "policy_place" in outputs and "legal_mask" in targets:
            _require_shape_match("model_output:policy_place", outputs["policy_place"], targets["legal_mask"])
        if "policy" in outputs and "policy" in targets:
            _require_shape_match("model_output:policy", outputs["policy"], targets["policy"])

    def _to_device(self, value: Any) -> Any:
        if hasattr(value, "__array__") and not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True).clone()
        return value


def _require_shape_match(name: str, left: torch.Tensor | None, right: torch.Tensor) -> None:
    if left is None:
        raise ValueError(f"train adapter owner: missing {name}")
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"train adapter owner: {name} shape {tuple(left.shape)} does not match {tuple(right.shape)}")


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"train adapter owner: non-finite values in {name}")


def _validate_graph_pair_semantics(targets: dict[str, torch.Tensor]) -> None:
    token_type = targets["token_type"].detach().cpu()
    token_qr = targets["token_qr"].detach().cpu()
    token_mask = targets["token_mask"].detach().cpu().bool()
    legal_tokens = targets["legal_token_indices"].detach().cpu()
    legal_qr = targets["legal_qr"].detach().cpu()
    legal_mask = targets["legal_mask"].detach().cpu().bool()
    pair_first = targets["pair_first_indices"].detach().cpu()
    pair_second = targets["pair_second_indices"].detach().cpu()
    pair_rows = targets["pair_rows"].detach().cpu()
    pair_mask = targets["pair_table_mask"].detach().cpu().bool()
    pair_phase = targets["pair_phase"].detach().cpu()
    known_first = targets.get("pair_known_first")
    known_first_mask = targets.get("pair_known_first_mask")
    known_first_cpu = None if known_first is None else known_first.detach().cpu()
    known_first_mask_cpu = None if known_first_mask is None else known_first_mask.detach().cpu().bool()

    if token_type.ndim == 1:
        token_type = token_type.unsqueeze(0)
        token_qr = token_qr.unsqueeze(0)
        token_mask = token_mask.unsqueeze(0)
        legal_tokens = legal_tokens.unsqueeze(0)
        legal_qr = legal_qr.unsqueeze(0)
        legal_mask = legal_mask.unsqueeze(0)
        pair_first = pair_first.unsqueeze(0)
        pair_second = pair_second.unsqueeze(0)
        pair_rows = pair_rows.unsqueeze(0)
        pair_mask = pair_mask.unsqueeze(0)
        pair_phase = pair_phase.reshape(1)
        if known_first_cpu is not None:
            known_first_cpu = known_first_cpu.unsqueeze(0)
        if known_first_mask_cpu is not None:
            known_first_mask_cpu = known_first_mask_cpu.reshape(1)

    batch_size = int(token_type.shape[0])
    if pair_rows.shape[:2] != pair_first.shape:
        raise ValueError("train adapter owner: pair_rows shape does not match pair references")
    if pair_mask.shape != pair_first.shape:
        raise ValueError("train adapter owner: pair_table_mask shape does not match pair references")
    if pair_phase.reshape(-1).shape[0] != batch_size:
        raise ValueError("train adapter owner: pair_phase does not match graph batch size")

    for row in range(batch_size):
        active_legal = [
            (int(legal_qr[row, idx, 0]), int(legal_qr[row, idx, 1]))
            for idx in range(int(legal_mask[row].numel()))
            if bool(legal_mask[row, idx])
        ]
        legal_set = set(active_legal)
        token_count = int(torch.count_nonzero(token_mask[row]).item())
        token_qrs = [
            (int(token_qr[row, idx, 0]), int(token_qr[row, idx, 1]))
            for idx in range(token_count)
        ]
        token_types = [int(token_type[row, idx]) for idx in range(token_count)]
        legal_token_to_row: dict[int, int] = {}
        for legal_idx, token_idx_raw in enumerate(legal_tokens[row].tolist()):
            token_idx = int(token_idx_raw)
            if legal_idx >= len(active_legal):
                continue
            if not (0 <= token_idx < token_count):
                raise ValueError("train adapter owner: legal_token_indices reference outside token table")
            if token_types[token_idx] != 4:
                raise ValueError("train adapter owner: legal_token_indices must reference LEGAL tokens")
            if token_qrs[token_idx] != active_legal[legal_idx]:
                raise ValueError("train adapter owner: legal token coordinate does not match legal_qr")
            legal_token_to_row[token_idx] = legal_idx

        phase = int(pair_phase.reshape(-1)[row])
        active_pair_indices = [idx for idx in range(pair_first.shape[1]) if bool(pair_mask[row, idx])]
        if phase == 0 and active_pair_indices:
            raise ValueError("train adapter owner: empty PairActionTable cannot carry active pair rows")
        if phase == 2:
            if known_first_cpu is None or known_first_mask_cpu is None or not bool(known_first_mask_cpu.reshape(-1)[row]):
                raise ValueError("train adapter owner: known-first pair phase requires pair_known_first")
            expected_known = (int(known_first_cpu[row, 0]), int(known_first_cpu[row, 1]))
        else:
            expected_known = None

        for pair_idx in active_pair_indices:
            first = (int(pair_rows[row, pair_idx, 0]), int(pair_rows[row, pair_idx, 1]))
            second = (int(pair_rows[row, pair_idx, 2]), int(pair_rows[row, pair_idx, 3]))
            first_tok = int(pair_first[row, pair_idx])
            second_tok = int(pair_second[row, pair_idx])
            if first == second:
                raise ValueError("train adapter owner: duplicate pair row coordinates")
            if not (0 <= first_tok < token_count and 0 <= second_tok < token_count):
                raise ValueError("train adapter owner: pair token reference outside token table")
            if phase == 1:
                if first > second:
                    raise ValueError("train adapter owner: first-placement pair row is not canonical unordered")
                if first not in legal_set or second not in legal_set:
                    raise ValueError("train adapter owner: first-placement pair row is outside legal rows")
                if first_tok not in legal_token_to_row or second_tok not in legal_token_to_row:
                    raise ValueError("train adapter owner: first-placement pair references must point at LEGAL tokens")
                if active_legal[legal_token_to_row[first_tok]] != first or active_legal[legal_token_to_row[second_tok]] != second:
                    raise ValueError("train adapter owner: first-placement pair references do not match PairActionTable rows")
            elif phase == 2:
                if expected_known is None or first != expected_known:
                    raise ValueError("train adapter owner: second-placement pair row does not preserve known_first")
                if token_types[first_tok] != 3 or token_qrs[first_tok] != expected_known:
                    raise ValueError("train adapter owner: known_first pair reference must point at the STONE token")
                if second_tok not in legal_token_to_row or active_legal[legal_token_to_row[second_tok]] != second:
                    raise ValueError("train adapter owner: known-first second pair reference does not match legal row")
            elif phase != 0:
                raise ValueError(f"train adapter owner: unsupported pair phase code {phase}")


def _require_projected_replay_batch(batch: Any) -> Any:
    from hexorl.replay.projector import ProjectedReplayBatch

    if not isinstance(batch, ProjectedReplayBatch):
        raise TypeError(
            "TrainAdapter consumes only ProjectedReplayBatch from replay/projector.py; "
            f"got {type(batch).__name__}"
        )
    if batch.source != "replay/projector.py":
        raise ValueError(f"TrainAdapter rejected non-projector batch source: {batch.source!r}")
    if int(batch.schema_version) != 1:
        raise ValueError(f"TrainAdapter rejected unsupported projected replay schema: {batch.schema_version}")
    return batch


def _tensor_identity_hash(tensor: torch.Tensor) -> int:
    view = tensor.detach()
    return hash((tuple(view.shape), str(view.dtype), float(view.float().sum().cpu()) if view.numel() else 0.0))
