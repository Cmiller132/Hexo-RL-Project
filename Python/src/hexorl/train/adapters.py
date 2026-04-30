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
        if getattr(batch, "source", "") == "replay/projector.py":
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

        aux_targets: dict[str, Any] = {}
        if len(batch) == 5:
            tensors, policies, values, lookahead_list, aux_targets = batch
        elif len(batch) == 4:
            tensors, policies, values, lookahead_list = batch
        else:
            tensors, policies, values = batch
            lookahead_list = []

        tensors = self._to_device(tensors)
        if channels_last and tensors.ndim == 4:
            tensors = tensors.contiguous(memory_format=torch.channels_last)
        targets = {
            "policy": self._to_device(policies),
            "value": self._to_device(values),
        }
        lookahead_keys = [f"lookahead_{h}" for h in getattr(self.cfg.buffer, "lookahead_horizons", [])]
        for key, value in zip(lookahead_keys, lookahead_list):
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
            _require_shape_match("pair_policy_target", targets["pair_policy_target"], targets["pair_token_indices"])
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


def _tensor_identity_hash(tensor: torch.Tensor) -> int:
    view = tensor.detach()
    return hash((tuple(view.shape), str(view.dtype), float(view.float().sum().cpu()) if view.numel() else 0.0))
