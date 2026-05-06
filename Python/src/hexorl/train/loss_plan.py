"""Contract-owned loss plans for training heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from hexorl.contracts import ROW_TABLE_DEFINITIONS, RowTableInstance
from hexorl.models.specs import ResolvedArchitectureSpec
from hexorl.train import losses as primitive


class LossContractError(ValueError):
    """Raised when predictions, targets, masks, weights, or phases violate a loss contract."""


@dataclass(frozen=True)
class TargetContract:
    head_name: str
    target_key: str
    loss_kind: str
    row_family: str | None = None
    mask_key: str | None = None
    weight_key: str | None = None
    phase_key: str | None = None
    phase_value: bool | None = None
    require_weight: bool = False
    require_positive_mass: bool = False


@dataclass(frozen=True)
class LossPlanEntry:
    head_name: str
    contract: TargetContract
    weight: float
    output_kind: str


@dataclass(frozen=True)
class LossPlan:
    entries: Mapping[str, LossPlanEntry]
    output_names: frozenset[str]
    entropy_weight: float | None = None

    def compute(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, object],
        *,
        n_bins: int = 65,
        row_tables: Mapping[str, RowTableInstance] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row_tables = row_tables or infer_row_tables(targets)
        per_head: dict[str, torch.Tensor] = {}
        unexpected = sorted(set(predictions) - self.output_names)
        if unexpected:
            raise LossContractError(f"predictions contain outputs without contracts: {unexpected}")

        for head_name, entry in self.entries.items():
            pred = predictions.get(head_name)
            if pred is None:
                raise LossContractError(f"trainable head {head_name!r} is missing from model predictions")
            _validate_entry(entry, pred, targets, row_tables)
            handler = LOSS_HANDLERS[entry.contract.loss_kind]
            loss = handler(entry, pred, targets, n_bins)
            per_head[head_name] = float(entry.weight) * loss

        if self.entropy_weight is not None:
            entropy_head = "policy" if "policy" in predictions else "policy_place" if "policy_place" in predictions else None
            if entropy_head is None:
                raise LossContractError("entropy loss requires policy or policy_place predictions")
            per_head["entropy"] = float(self.entropy_weight) * primitive.entropy_loss(predictions[entropy_head])

        if not per_head:
            raise LossContractError("loss plan contains no trainable heads")
        total = sum(per_head.values())
        return total, per_head


def build_loss_plan(
    resolved: ResolvedArchitectureSpec | Sequence[str],
    loss_weights: Mapping[str, float],
) -> LossPlan:
    if isinstance(resolved, ResolvedArchitectureSpec):
        outputs = tuple(resolved.outputs)
        contracts = resolved.output_contracts
    else:
        outputs = tuple(str(name) for name in resolved)
        contracts = {}

    entries: dict[str, LossPlanEntry] = {}
    for head_name in outputs:
        if head_name == "entropy":
            continue
        template = _template_for_head(head_name)
        if template is None:
            continue
        output_contract = contracts.get(head_name)
        output_kind = output_contract.kind if output_contract is not None else template.loss_kind
        if head_name not in loss_weights:
            raise LossContractError(f"trainable head {head_name!r} is missing train.loss_weights entry")
        weight = float(loss_weights[head_name])
        if weight <= 0.0:
            raise LossContractError(f"trainable head {head_name!r} has inactive loss weight {weight}")
        row_family = output_contract.row_family if output_contract is not None else template.row_family
        target_key = template.target_key
        mask_key = template.mask_key
        if head_name == "opp_policy" and row_family == "dense_board":
            target_key = "opp_policy"
            mask_key = None
        elif head_name == "opp_policy" and row_family == "opponent_legal":
            target_key = "opp_policy_target"
            mask_key = "opp_legal_mask"
        contract = TargetContract(
            head_name=template.head_name,
            target_key=target_key,
            loss_kind=template.loss_kind,
            row_family=row_family,
            mask_key=mask_key,
            weight_key=template.weight_key,
            phase_key=template.phase_key,
            phase_value=template.phase_value,
            require_weight=template.require_weight,
            require_positive_mass=template.require_positive_mass,
        )
        entries[head_name] = LossPlanEntry(
            head_name=head_name,
            contract=contract,
            weight=weight,
            output_kind=output_kind,
        )

    entropy_weight = loss_weights.get("entropy")
    if entropy_weight is not None and float(entropy_weight) <= 0.0:
        entropy_weight = None
    return LossPlan(entries=entries, output_names=frozenset(outputs), entropy_weight=entropy_weight)


def infer_row_tables(targets: Mapping[str, object]) -> dict[str, RowTableInstance]:
    tables: dict[str, RowTableInstance] = {}
    dense_target = targets.get("policy", targets.get("opp_policy"))
    if dense_target is not None:
        rows = np.arange(int(_shape(dense_target)[-1]), dtype=np.int16).reshape(-1, 1)
        tables["dense_board"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["dense_board"],
            rows=rows,
            mask=np.ones(rows.shape[0], dtype=np.bool_),
            phase="dense_policy",
            source="training_batch",
        )
    if "candidate_mask" in targets:
        rows = _rows_from_optional(targets.get("candidate_qr"), targets.get("candidate_indices"))
        tables["candidate"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["candidate"],
            rows=rows,
            mask=_numpy_bool(targets["candidate_mask"]),
            phase="candidate_policy",
            source="training_batch",
        )
    if "legal_mask" in targets:
        rows = _numpy_array(targets.get("legal_qr", np.zeros((*_shape(targets["legal_mask"]), 2), dtype=np.int32)))
        tables["legal"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["legal"],
            rows=rows,
            mask=_numpy_bool(targets["legal_mask"]),
            phase="legal_policy",
            source="training_batch",
        )
    if "opp_legal_mask" in targets:
        rows = _numpy_array(targets.get("opp_legal_qr", np.zeros((*_shape(targets["opp_legal_mask"]), 2), dtype=np.int32)))
        tables["opponent_legal"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["opponent_legal"],
            rows=rows,
            mask=_numpy_bool(targets["opp_legal_mask"]),
            phase="opponent_policy",
            source="training_batch",
        )
    if "pair_candidate_mask" in targets:
        tables["pair_joint"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["pair_joint"],
            rows=_numpy_array(targets.get("pair_candidate_indices", np.zeros((*_shape(targets["pair_candidate_mask"]), 2), dtype=np.int64))),
            mask=_numpy_bool(targets["pair_candidate_mask"]),
            phase="crop_pair_policy",
            source="training_batch",
        )
    if "pair_row_mask" in targets:
        pair_rows = _pair_rows_from_graph_targets(targets)
        pair_mask = _numpy_bool(targets["pair_row_mask"])
        tables["pair_joint"] = RowTableInstance(
            definition=ROW_TABLE_DEFINITIONS["pair_joint"],
            rows=pair_rows,
            mask=pair_mask,
            phase="pair_joint",
            source="training_batch",
        )
        if "pair_second_row_mask" in targets:
            tables["known_first_pair"] = RowTableInstance(
                definition=ROW_TABLE_DEFINITIONS["known_first_pair"],
                rows=pair_rows,
                mask=_numpy_bool(targets["pair_second_row_mask"]),
                phase="known_first",
                source="training_batch",
            )
    return tables


def _validate_entry(
    entry: LossPlanEntry,
    pred: torch.Tensor,
    targets: Mapping[str, object],
    row_tables: Mapping[str, RowTableInstance],
) -> None:
    contract = entry.contract
    if contract.target_key not in targets:
        raise LossContractError(f"{entry.head_name} requires target {contract.target_key!r}")
    if contract.mask_key is not None and contract.mask_key not in targets:
        raise LossContractError(f"{entry.head_name} requires mask {contract.mask_key!r}")
    if contract.require_weight and contract.weight_key is not None and contract.weight_key not in targets:
        raise LossContractError(f"{entry.head_name} requires weight {contract.weight_key!r}")
    if contract.phase_key is not None and contract.phase_key not in targets:
        raise LossContractError(f"{entry.head_name} requires phase {contract.phase_key!r}")
    if contract.row_family is not None and contract.row_family not in row_tables:
        raise LossContractError(f"{entry.head_name} requires row table {contract.row_family!r}")
    if contract.row_family is not None:
        _validate_row_table(entry.head_name, row_tables[contract.row_family], pred)
    target = _tensor(targets[contract.target_key], pred.device)
    if target.shape[0] != pred.shape[0]:
        raise LossContractError(
            f"{entry.head_name} target batch mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if contract.mask_key is not None:
        mask = _tensor(targets[contract.mask_key], pred.device, dtype=torch.bool)
        if mask.shape != pred.shape:
            raise LossContractError(
                f"{entry.head_name} mask shape mismatch: pred={tuple(pred.shape)} mask={tuple(mask.shape)}"
            )
    if contract.phase_key is not None:
        phase = _tensor(targets[contract.phase_key], pred.device, dtype=torch.bool)
        expected = bool(contract.phase_value)
        if phase.ndim != 1 or phase.shape[0] != pred.shape[0]:
            raise LossContractError(f"{entry.head_name} phase {contract.phase_key!r} must be a batch vector")
        phase_ok = phase == expected
        target_mass = _target_mass(target, targets, contract, pred.device)
        wrong_phase = (~phase_ok) & (target_mass > 0)
        if torch.any(wrong_phase):
            raise LossContractError(
                f"{entry.head_name} has positive target mass outside required phase {contract.phase_key}={expected}"
            )
    if contract.require_positive_mass:
        target_mass = _target_mass(target, targets, contract, pred.device)
        active = torch.ones_like(target_mass, dtype=torch.bool)
        if contract.weight_key is not None and contract.weight_key in targets:
            weight = _tensor(targets[contract.weight_key], pred.device, dtype=target_mass.dtype)
            active &= weight > 0
        if contract.mask_key is not None and contract.mask_key in targets:
            mask = _tensor(targets[contract.mask_key], pred.device, dtype=torch.bool)
            active &= mask.reshape(mask.shape[0], -1).any(dim=1)
        if contract.phase_key is not None:
            phase = _tensor(targets[contract.phase_key], pred.device, dtype=torch.bool)
            active &= phase == bool(contract.phase_value)
        if torch.any(active & (target_mass <= 0)):
            raise LossContractError(f"{entry.head_name} requires positive target mass for active rows")


def _target_mass(
    target: torch.Tensor,
    targets: Mapping[str, object],
    contract: TargetContract,
    device: torch.device,
) -> torch.Tensor:
    values = target.float()
    if contract.mask_key is not None and contract.mask_key in targets:
        mask = _tensor(targets[contract.mask_key], device, dtype=torch.bool)
        values = values * mask.to(dtype=values.dtype)
    if values.ndim == 1:
        return values.abs()
    return values.abs().sum(dim=tuple(range(1, values.ndim)))


def _validate_row_table(
    head_name: str,
    row_table: RowTableInstance,
    pred: torch.Tensor,
) -> None:
    rows = np.asarray(row_table.rows)
    mask = np.asarray(row_table.mask).astype(np.bool_, copy=False)
    if mask.ndim == 0:
        raise LossContractError(f"{head_name} row table {row_table.definition.family!r} mask must not be scalar")
    if rows.ndim == mask.ndim:
        rows = rows[..., None]
    elif rows.ndim != mask.ndim + 1:
        raise LossContractError(
            f"{head_name} row table {row_table.definition.family!r} rows/mask rank mismatch: "
            f"rows={rows.shape} mask={mask.shape}"
        )
    if rows.shape[: mask.ndim] != mask.shape:
        raise LossContractError(
            f"{head_name} row table {row_table.definition.family!r} rows/mask shape mismatch: "
            f"rows={rows.shape} mask={mask.shape}"
        )
    if pred.ndim >= 2 and int(pred.shape[1]) != int(mask.shape[-1]):
        raise LossContractError(
            f"{head_name} row table {row_table.definition.family!r} row count mismatch: "
            f"pred={tuple(pred.shape)} mask={mask.shape}"
        )
    batch_shape = mask.shape[:-1]
    if not batch_shape:
        _validate_active_rows_unique(head_name, row_table.definition.family, rows, mask)
        return
    for batch_index in np.ndindex(batch_shape):
        _validate_active_rows_unique(
            head_name,
            row_table.definition.family,
            rows[batch_index],
            mask[batch_index],
        )


def _validate_active_rows_unique(
    head_name: str,
    row_family: str,
    rows: np.ndarray,
    mask: np.ndarray,
) -> None:
    active_rows = np.ascontiguousarray(rows[mask])
    if active_rows.shape[0] <= 1:
        return
    flat = active_rows.reshape(active_rows.shape[0], -1)
    unique = np.unique(flat, axis=0)
    if unique.shape[0] != flat.shape[0]:
        raise LossContractError(f"{head_name} row table {row_family!r} contains duplicate active rows")


def _template_for_head(head_name: str) -> TargetContract | None:
    if head_name.startswith("lookahead_"):
        return TargetContract(
            head_name=head_name,
            target_key=head_name,
            loss_kind="value",
            require_positive_mass=False,
        )
    return HEAD_TARGETS.get(head_name)


def _policy(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.policy_loss(pred, _tensor(targets[c.target_key], pred.device), _maybe_tensor(targets, c.weight_key, pred.device))


def _masked_policy(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.sparse_policy_loss(
        pred,
        _tensor(targets[c.target_key], pred.device),
        _tensor(targets[c.mask_key], pred.device, dtype=torch.bool),  # type: ignore[arg-type]
        _maybe_tensor(targets, c.weight_key, pred.device),
    )


def _opp_policy(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    target = _tensor(targets[c.target_key], pred.device)
    weight = _maybe_tensor(targets, c.weight_key, pred.device)
    if c.mask_key is not None and c.mask_key in targets:
        return primitive.graph_policy_loss(pred, target, _tensor(targets[c.mask_key], pred.device, dtype=torch.bool), weight)
    return primitive.opp_policy_loss(pred, target, weight)


def _value(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.binned_value_loss(pred, _tensor(targets[c.target_key], pred.device), n_bins, _maybe_tensor(targets, c.weight_key, pred.device))


def _regret_rank(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.regret_rank_loss(pred.squeeze(-1), _tensor(targets[c.target_key], pred.device), _maybe_tensor(targets, c.weight_key, pred.device))


def _regret_value(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.regret_value_loss(pred, _tensor(targets[c.target_key], pred.device), n_bins, _maybe_tensor(targets, c.weight_key, pred.device))


def _axis(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    return primitive.axis_loss(pred, _tensor(targets[entry.contract.target_key], pred.device, dtype=torch.long))


def _axis_map(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    return primitive.axis_map_loss(pred, _tensor(targets[entry.contract.target_key], pred.device))


def _moves_left(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.moves_left_loss(pred, _tensor(targets[c.target_key], pred.device), _maybe_tensor(targets, c.weight_key, pred.device))


def _tactical(entry: LossPlanEntry, pred: torch.Tensor, targets: Mapping[str, object], _n_bins: int) -> torch.Tensor:
    c = entry.contract
    return primitive.tactical_loss(pred, _tensor(targets[c.target_key], pred.device), _maybe_tensor(targets, c.weight_key, pred.device))


HEAD_TARGETS: Mapping[str, TargetContract] = {
    "policy": TargetContract("policy", "policy", "policy", row_family="dense_board", weight_key="policy_weight", require_weight=True, require_positive_mass=True),
    "sparse_policy": TargetContract("sparse_policy", "sparse_policy_target", "masked_policy", row_family="candidate", mask_key="candidate_mask", weight_key="sparse_policy_weight", require_weight=True, require_positive_mass=True),
    "pair_policy": TargetContract("pair_policy", "pair_policy_target", "masked_policy", row_family="pair_joint", mask_key="pair_candidate_mask", weight_key="pair_policy_weight", require_weight=True, require_positive_mass=True),
    "policy_place": TargetContract("policy_place", "policy_target", "masked_policy", row_family="legal", mask_key="legal_mask", weight_key="policy_weight", require_weight=True, require_positive_mass=True),
    "legal_token_quality": TargetContract("legal_token_quality", "legal_token_quality_target", "masked_policy", row_family="legal", mask_key="legal_mask", weight_key="policy_weight", require_weight=True, require_positive_mass=True),
    "policy_pair_first": TargetContract("policy_pair_first", "pair_first_policy_target", "masked_policy", row_family="legal", mask_key="legal_mask", weight_key="pair_policy_weight", phase_key="pair_first_unordered", phase_value=True, require_weight=True, require_positive_mass=True),
    "policy_pair_joint": TargetContract("policy_pair_joint", "pair_policy_target", "masked_policy", row_family="pair_joint", mask_key="pair_row_mask", weight_key="pair_policy_weight", require_weight=True, require_positive_mass=True),
    "policy_pair_second": TargetContract("policy_pair_second", "pair_second_policy_target", "masked_policy", row_family="known_first_pair", mask_key="pair_second_row_mask", weight_key="pair_policy_weight", phase_key="pair_second_known_first", phase_value=True, require_weight=True, require_positive_mass=True),
    "opp_policy": TargetContract("opp_policy", "opp_policy", "opp_policy", row_family="dense_board", weight_key="opp_policy_weight", require_weight=True, require_positive_mass=True),
    "value": TargetContract("value", "value", "value", weight_key="value_weight", require_weight=True),
    "regret_rank": TargetContract("regret_rank", "regret_rank", "regret_rank", weight_key="regret_weight", require_weight=True),
    "regret_value": TargetContract("regret_value", "regret_value", "regret_value", weight_key="regret_weight", require_weight=True),
    "axis": TargetContract("axis", "axis", "axis"),
    "axis_delta_norm": TargetContract("axis_delta_norm", "axis_delta_norm", "axis_map"),
    "moves_left": TargetContract("moves_left", "moves_left", "moves_left", weight_key="moves_left_weight", require_weight=True),
    "tactical": TargetContract("tactical", "tactical_target", "tactical", weight_key="policy_weight", require_weight=True),
}

LOSS_HANDLERS: Mapping[str, Callable[[LossPlanEntry, torch.Tensor, Mapping[str, object], int], torch.Tensor]] = {
    "policy": _policy,
    "masked_policy": _masked_policy,
    "opp_policy": _opp_policy,
    "value": _value,
    "regret_rank": _regret_rank,
    "regret_value": _regret_value,
    "axis": _axis,
    "axis_map": _axis_map,
    "moves_left": _moves_left,
    "tactical": _tactical,
}


def _tensor(value: object, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _maybe_tensor(targets: Mapping[str, object], key: str | None, device: torch.device) -> torch.Tensor | None:
    if key is None or key not in targets:
        return None
    return _tensor(targets[key], device)


def _numpy_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _numpy_bool(value: object) -> np.ndarray:
    return _numpy_array(value).astype(np.bool_, copy=False)


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    return tuple(int(dim) for dim in np.asarray(value).shape)


def _rows_from_optional(primary: object | None, fallback: object | None) -> np.ndarray:
    if primary is not None:
        return _numpy_array(primary)
    if fallback is not None:
        return _numpy_array(fallback)
    return np.zeros((0, 1), dtype=np.int32)


def _pair_rows_from_graph_targets(targets: Mapping[str, object]) -> np.ndarray:
    first = _numpy_array(targets.get("pair_first_indices"))
    second = _numpy_array(targets.get("pair_second_indices"))
    if first.size == 0 or second.size == 0:
        return np.zeros((*_shape(targets["pair_row_mask"]), 4), dtype=np.int32)
    token_qr = targets.get("token_qr")
    if token_qr is None:
        return np.stack([first, second], axis=-1).astype(np.int64, copy=False)
    qr = _numpy_array(token_qr)
    rows = np.zeros((*first.shape, 4), dtype=np.int32)
    for batch_idx in range(first.shape[0]):
        for row_idx in range(first.shape[1]):
            a = int(first[batch_idx, row_idx])
            b = int(second[batch_idx, row_idx])
            if a < 0 or b < 0:
                continue
            rows[batch_idx, row_idx, 0:2] = qr[batch_idx, a]
            rows[batch_idx, row_idx, 2:4] = qr[batch_idx, b]
    return rows
