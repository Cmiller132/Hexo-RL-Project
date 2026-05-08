"""V1 sampled joint-pair target construction.

The V1 replay schema logs an admitted support set plus explicit support flags.
This module converts that schema into training arrays without ever inferring
that an omitted legal pair is a negative example.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from hexorl.selfplay.records import PairKey, V1SearchPairMetadata


V1_PAIR_TARGET_VERSION = 1


@dataclass(frozen=True)
class V1PairTrainingTargets:
    """Dense arrays over the admitted V1 candidate pair support."""

    candidate_pair_qr: np.ndarray
    legal_row_ids: np.ndarray
    cell_marginal_target: np.ndarray
    pair_joint_target: np.ndarray
    pair_joint_mask: np.ndarray
    pair_softened_target: np.ndarray
    pair_completion_target: np.ndarray
    pair_completion_mask: np.ndarray
    pair_proposal_score_target: np.ndarray
    pair_ranking_target: np.ndarray
    pair_ranking_mask: np.ndarray
    conditional_pair_indices: np.ndarray
    conditional_first_legal_row_ids: np.ndarray
    conditional_second_legal_row_ids: np.ndarray
    conditional_target: np.ndarray
    conditional_mask: np.ndarray
    completed_q_target: np.ndarray
    completed_q_mask: np.ndarray
    q_value_target: np.ndarray
    q_value_mask: np.ndarray
    terminal_tactical_target: np.ndarray
    explicit_negative_mask: np.ndarray
    sampled_negative_mask: np.ndarray
    unsampled_mask: np.ndarray
    terminal_exact_mask: np.ndarray
    terminal_equivalent_mask: np.ndarray
    forced_mask: np.ndarray
    ranking_positive_indices: np.ndarray
    ranking_negative_indices: np.ndarray
    selected_pair_index: int
    support_type_id: int
    schema_version: int = V1_PAIR_TARGET_VERSION

    def as_dict(self) -> dict[str, np.ndarray | int]:
        return {
            "v1_pair_schema_version": int(self.schema_version),
            "v1_pair_target_schema_version": int(self.schema_version),
            "v1_candidate_pair_qr": self.candidate_pair_qr,
            "v1_pair_legal_row_ids": self.legal_row_ids,
            "v1_cell_marginal_target": self.cell_marginal_target,
            "v1_pair_joint_target": self.pair_joint_target,
            "v1_pair_joint_mask": self.pair_joint_mask,
            "v1_pair_softened_target": self.pair_softened_target,
            "v1_pair_completion_target": self.pair_completion_target,
            "v1_pair_completion_mask": self.pair_completion_mask,
            "v1_pair_proposal_score_target": self.pair_proposal_score_target,
            "v1_pair_ranking_target": self.pair_ranking_target,
            "v1_pair_ranking_mask": self.pair_ranking_mask,
            "v1_conditional_pair_indices": self.conditional_pair_indices,
            "v1_conditional_first_legal_row_ids": self.conditional_first_legal_row_ids,
            "v1_conditional_second_legal_row_ids": self.conditional_second_legal_row_ids,
            "v1_pair_conditional_target": self.conditional_target,
            "v1_pair_conditional_mask": self.conditional_mask,
            "v1_completed_q_target": self.completed_q_target,
            "v1_completed_q_mask": self.completed_q_mask,
            "v1_pair_q_value_target": self.q_value_target,
            "v1_pair_q_value_mask": self.q_value_mask,
            "v1_terminal_tactical_target": self.terminal_tactical_target,
            "v1_explicit_negative_mask": self.explicit_negative_mask,
            "v1_sampled_negative_mask": self.sampled_negative_mask,
            "v1_unsampled_pair_mask": self.unsampled_mask,
            "v1_terminal_exact_mask": self.terminal_exact_mask,
            "v1_terminal_equivalent_mask": self.terminal_equivalent_mask,
            "v1_forced_pair_mask": self.forced_mask,
            "v1_ranking_positive_indices": self.ranking_positive_indices,
            "v1_ranking_negative_indices": self.ranking_negative_indices,
            "v1_selected_pair_index": int(self.selected_pair_index),
            "v1_support_type_id": int(self.support_type_id),
        }


_SUPPORT_TYPE_IDS: Mapping[str, int] = {
    "exhaustive_legal_pair_table": 1,
    "admitted_candidate_set_without_explicit_negatives": 2,
    "admitted_candidate_set_with_explicit_negatives": 3,
    "completed_q_candidate_posterior": 4,
}


def build_v1_pair_training_targets(
    metadata: V1SearchPairMetadata,
    *,
    legal_row_count: int | None = None,
    legal_row_index_by_qr: Mapping[tuple[int, int], int] | None = None,
    pair_key_transform: Callable[[tuple[int, int]], tuple[int, int]] | None = None,
    posterior_temperature: float = 1.0,
    softening_alpha: float = 0.05,
    terminal_equivalent_mass: str = "collapse_uniform",
) -> V1PairTrainingTargets:
    """Build V1 pair targets over logged support rows.

    ``metadata.legal_pair_count`` is used only for validation and audit. The
    target support is exactly ``metadata.candidate_pairs``. Unsampled rows are
    masked out of policy, Q, completed-Q, and ranking losses.
    """

    if float(posterior_temperature) <= 0.0 or not np.isfinite(posterior_temperature):
        raise ValueError("posterior_temperature must be finite and positive")
    if terminal_equivalent_mass not in {"collapse_uniform", "omit_policy"}:
        raise ValueError(
            "terminal_equivalent_mass must be 'collapse_uniform' or 'omit_policy'"
        )
    if not 0.0 <= float(softening_alpha) <= 1.0 or not np.isfinite(softening_alpha):
        raise ValueError("softening_alpha must be finite and in [0, 1]")

    n = len(metadata.candidate_pairs)
    pair_qr = np.zeros((n, 4), dtype=np.int32)
    legal_row_ids = np.zeros((n, 2), dtype=np.int64)
    source_score = np.zeros(n, dtype=np.float32)
    masks = _support_masks(metadata)
    transformed_pair_keys: list[PairKey] = []

    for idx, candidate in enumerate(metadata.candidate_pairs):
        pair_key = _transform_pair_key(candidate.pair_key, pair_key_transform)
        transformed_pair_keys.append(pair_key)
        pair_qr[idx] = [
            int(pair_key[0][0]),
            int(pair_key[0][1]),
            int(pair_key[1][0]),
            int(pair_key[1][1]),
        ]
        legal_row_ids[idx] = _legal_row_ids_for_pair(candidate, pair_key, legal_row_index_by_qr)
        source_score[idx] = _proposal_score(candidate.proposal_propensity_metadata.to_dict())

    selected_pair = None
    if metadata.selected_pair is not None:
        selected_pair = _transform_pair_key(metadata.selected_pair, pair_key_transform)
    selected_index = _selected_pair_index(selected_pair, transformed_pair_keys)
    inferred_legal_count = int(legal_row_ids.max()) + 1 if legal_row_ids.size else 0
    legal_count = int(legal_row_count) if legal_row_count is not None else inferred_legal_count
    if legal_count < inferred_legal_count:
        raise ValueError(
            f"legal_row_count={legal_count} cannot cover V1 legal row ids up to {inferred_legal_count - 1}"
        )

    pair_joint_mask = masks["admitted"] | masks["forced"] | masks["terminal_exact"] | masks["terminal_equivalent"]
    pair_joint_mask &= ~masks["unsampled"]
    base_pair_target = _posterior_target(
        metadata,
        trainable_mask=pair_joint_mask,
        posterior_temperature=float(posterior_temperature),
    )
    softened_target = _softened_target(
        metadata,
        base_target=base_pair_target,
        trainable_mask=pair_joint_mask,
        posterior_temperature=float(posterior_temperature),
        alpha=float(softening_alpha),
    )
    pair_joint_target = softened_target.copy()

    terminal_mask = masks["terminal_exact"] | masks["terminal_equivalent"]
    if terminal_mask.any():
        if terminal_equivalent_mass == "omit_policy":
            pair_joint_target[terminal_mask] = 0.0
            pair_joint_mask[terminal_mask] = False
            pair_joint_target = _renormalize(pair_joint_target, pair_joint_mask)
        else:
            terminal_mass = float(pair_joint_target[terminal_mask].sum())
            if terminal_mass <= 0.0 and not pair_joint_target.any():
                terminal_mass = 1.0
            if terminal_mass > 0.0:
                pair_joint_target[terminal_mask] = terminal_mass / float(terminal_mask.sum())
                pair_joint_target[~terminal_mask] *= max(0.0, 1.0 - terminal_mass)
                pair_joint_target = _renormalize(pair_joint_target, pair_joint_mask)
        softened_target = pair_joint_target.copy()

    completed_q = np.asarray(metadata.completed_q_values, dtype=np.float32)
    q_values = np.asarray(metadata.q_values, dtype=np.float32)
    explicit_negative = masks["explicit_negative"]
    sampled_negative = masks["sampled_negative"]
    negative_mask = (explicit_negative | sampled_negative) & ~masks["unsampled"]
    pair_completion_mask = (pair_joint_mask | negative_mask) & ~masks["unsampled"]
    completed_q_mask = pair_completion_mask.copy()
    q_value_mask = pair_completion_mask.copy()

    ranking_positive = np.flatnonzero(pair_joint_target > 0.0).astype(np.int64)
    ranking_negative = np.flatnonzero(negative_mask).astype(np.int64)
    ranking_target = np.zeros(n, dtype=np.float32)
    ranking_mask = np.zeros(n, dtype=np.bool_)
    ranking_target[ranking_positive] = 1.0
    ranking_mask[ranking_positive] = True
    ranking_mask[ranking_negative] = True

    if masks["unsampled"].any() and (explicit_negative & masks["unsampled"]).any():
        raise ValueError("unsampled V1 legal pairs cannot be explicit negatives")
    if masks["unsampled"].any() and (ranking_mask & masks["unsampled"]).any():
        raise ValueError("unsampled V1 legal pairs cannot enter ranking losses")
    cell_target = _cell_marginal_target(legal_row_ids, pair_joint_target, legal_count)
    conditional = _conditional_targets(legal_row_ids, pair_joint_target)

    return V1PairTrainingTargets(
        candidate_pair_qr=pair_qr,
        legal_row_ids=legal_row_ids,
        cell_marginal_target=cell_target.astype(np.float32, copy=False),
        pair_joint_target=pair_joint_target.astype(np.float32, copy=False),
        pair_joint_mask=pair_joint_mask.astype(np.bool_, copy=False),
        pair_softened_target=softened_target.astype(np.float32, copy=False),
        pair_completion_target=terminal_mask.astype(np.float32, copy=False),
        pair_completion_mask=pair_completion_mask.astype(np.bool_, copy=False),
        pair_proposal_score_target=source_score.astype(np.float32, copy=False),
        pair_ranking_target=ranking_target.astype(np.float32, copy=False),
        pair_ranking_mask=ranking_mask.astype(np.bool_, copy=False),
        conditional_pair_indices=conditional["pair_indices"].astype(np.int64, copy=False),
        conditional_first_legal_row_ids=conditional["first_ids"].astype(np.int64, copy=False),
        conditional_second_legal_row_ids=conditional["second_ids"].astype(np.int64, copy=False),
        conditional_target=conditional["target"].astype(np.float32, copy=False),
        conditional_mask=conditional["mask"].astype(np.bool_, copy=False),
        completed_q_target=completed_q,
        completed_q_mask=completed_q_mask.astype(np.bool_, copy=False),
        q_value_target=q_values,
        q_value_mask=q_value_mask.astype(np.bool_, copy=False),
        terminal_tactical_target=_terminal_tactical_target(masks, selected_index),
        explicit_negative_mask=explicit_negative.astype(np.bool_, copy=False),
        sampled_negative_mask=sampled_negative.astype(np.bool_, copy=False),
        unsampled_mask=masks["unsampled"].astype(np.bool_, copy=False),
        terminal_exact_mask=masks["terminal_exact"].astype(np.bool_, copy=False),
        terminal_equivalent_mask=masks["terminal_equivalent"].astype(np.bool_, copy=False),
        forced_mask=masks["forced"].astype(np.bool_, copy=False),
        ranking_positive_indices=ranking_positive,
        ranking_negative_indices=ranking_negative,
        selected_pair_index=selected_index,
        support_type_id=int(_SUPPORT_TYPE_IDS[metadata.support_type]),
    )


def collate_v1_pair_training_targets(
    targets: Sequence[V1PairTrainingTargets | None],
    *,
    legal_width: int,
    pair_width: int,
) -> dict[str, np.ndarray]:
    """Pad per-record V1 targets into graph-training batch arrays."""

    batch_size = len(targets)
    condition_width = max(
        (int(target.conditional_pair_indices.shape[0]) for target in targets if target is not None),
        default=0,
    )

    arrays: dict[str, np.ndarray] = {
        "v1_pair_schema_version": np.zeros(batch_size, dtype=np.int16),
        "v1_pair_target_schema_version": np.zeros(batch_size, dtype=np.int16),
        "v1_support_type_id": np.zeros(batch_size, dtype=np.int16),
        "v1_pair_weight": np.zeros(batch_size, dtype=np.float32),
        "v1_candidate_pair_qr": np.zeros((batch_size, pair_width, 4), dtype=np.int32),
        "v1_pair_legal_row_ids": np.full((batch_size, pair_width, 2), -1, dtype=np.int64),
        "v1_cell_marginal_target": np.zeros((batch_size, legal_width), dtype=np.float32),
        "v1_pair_joint_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_joint_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_pair_softened_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_completion_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_completion_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_pair_proposal_score_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_ranking_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_ranking_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_conditional_pair_indices": np.full((batch_size, condition_width), -1, dtype=np.int64),
        "v1_conditional_first_legal_row_ids": np.full((batch_size, condition_width), -1, dtype=np.int64),
        "v1_conditional_second_legal_row_ids": np.full((batch_size, condition_width), -1, dtype=np.int64),
        "v1_pair_conditional_target": np.zeros((batch_size, condition_width), dtype=np.float32),
        "v1_pair_conditional_mask": np.zeros((batch_size, condition_width), dtype=np.bool_),
        "v1_completed_q_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_completed_q_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_pair_q_value_target": np.zeros((batch_size, pair_width), dtype=np.float32),
        "v1_pair_q_value_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_terminal_tactical_target": np.zeros((batch_size, 8), dtype=np.float32),
        "v1_explicit_negative_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_sampled_negative_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_unsampled_pair_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_terminal_exact_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_terminal_equivalent_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_forced_pair_mask": np.zeros((batch_size, pair_width), dtype=np.bool_),
        "v1_ranking_positive_indices": np.full((batch_size, pair_width), -1, dtype=np.int64),
        "v1_ranking_negative_indices": np.full((batch_size, pair_width), -1, dtype=np.int64),
        "v1_selected_pair_index": np.full(batch_size, -1, dtype=np.int64),
    }
    for row, target in enumerate(targets):
        if target is None:
            continue
        pair_count = int(target.candidate_pair_qr.shape[0])
        if pair_count > pair_width:
            raise ValueError(f"V1 target pair_count={pair_count} exceeds pair_width={pair_width}")
        legal_count = int(target.cell_marginal_target.shape[0])
        if legal_count > legal_width:
            raise ValueError(f"V1 target legal_count={legal_count} exceeds legal_width={legal_width}")
        cond_count = int(target.conditional_pair_indices.shape[0])
        arrays["v1_pair_schema_version"][row] = target.schema_version
        arrays["v1_pair_target_schema_version"][row] = target.schema_version
        arrays["v1_support_type_id"][row] = target.support_type_id
        arrays["v1_pair_weight"][row] = 1.0
        arrays["v1_selected_pair_index"][row] = int(target.selected_pair_index)
        _copy(arrays["v1_candidate_pair_qr"][row], target.candidate_pair_qr, pair_count)
        _copy(arrays["v1_pair_legal_row_ids"][row], target.legal_row_ids, pair_count)
        _copy(arrays["v1_cell_marginal_target"][row], target.cell_marginal_target, legal_count)
        for key, source in (
            ("v1_pair_joint_target", target.pair_joint_target),
            ("v1_pair_joint_mask", target.pair_joint_mask),
            ("v1_pair_softened_target", target.pair_softened_target),
            ("v1_pair_completion_target", target.pair_completion_target),
            ("v1_pair_completion_mask", target.pair_completion_mask),
            ("v1_pair_proposal_score_target", target.pair_proposal_score_target),
            ("v1_pair_ranking_target", target.pair_ranking_target),
            ("v1_pair_ranking_mask", target.pair_ranking_mask),
            ("v1_completed_q_target", target.completed_q_target),
            ("v1_completed_q_mask", target.completed_q_mask),
            ("v1_pair_q_value_target", target.q_value_target),
            ("v1_pair_q_value_mask", target.q_value_mask),
            ("v1_explicit_negative_mask", target.explicit_negative_mask),
            ("v1_sampled_negative_mask", target.sampled_negative_mask),
            ("v1_unsampled_pair_mask", target.unsampled_mask),
            ("v1_terminal_exact_mask", target.terminal_exact_mask),
            ("v1_terminal_equivalent_mask", target.terminal_equivalent_mask),
            ("v1_forced_pair_mask", target.forced_mask),
        ):
            _copy(arrays[key][row], source, pair_count)
        for key, source in (
            ("v1_conditional_pair_indices", target.conditional_pair_indices),
            ("v1_conditional_first_legal_row_ids", target.conditional_first_legal_row_ids),
            ("v1_conditional_second_legal_row_ids", target.conditional_second_legal_row_ids),
            ("v1_pair_conditional_target", target.conditional_target),
            ("v1_pair_conditional_mask", target.conditional_mask),
        ):
            _copy(arrays[key][row], source, cond_count)
        _copy(arrays["v1_terminal_tactical_target"][row], target.terminal_tactical_target, 8)
        _copy(arrays["v1_ranking_positive_indices"][row], target.ranking_positive_indices, min(pair_width, target.ranking_positive_indices.shape[0]))
        _copy(arrays["v1_ranking_negative_indices"][row], target.ranking_negative_indices, min(pair_width, target.ranking_negative_indices.shape[0]))
    return arrays


def _support_masks(metadata: V1SearchPairMetadata) -> dict[str, np.ndarray]:
    names = (
        "admitted",
        "explicit_negative",
        "forced",
        "sampled_negative",
        "terminal_exact",
        "terminal_equivalent",
        "unsampled",
    )
    masks = {name: np.zeros(len(metadata.candidate_pairs), dtype=np.bool_) for name in names}
    for idx, flags in enumerate(metadata.target_support_flags):
        flag_set = set(flags)
        for name in names:
            masks[name][idx] = name in flag_set
    return masks


def _posterior_target(
    metadata: V1SearchPairMetadata,
    *,
    trainable_mask: np.ndarray,
    posterior_temperature: float,
) -> np.ndarray:
    target = np.zeros(len(metadata.candidate_pairs), dtype=np.float32)
    if not bool(trainable_mask.any()):
        return target
    visits = np.asarray(metadata.visit_counts, dtype=np.float64)
    visits = np.where(trainable_mask, np.maximum(visits, 0.0), 0.0)
    if float(visits.sum()) > 0.0:
        target = (visits / float(visits.sum())).astype(np.float32)
        return target

    completed_q = np.asarray(metadata.completed_q_values, dtype=np.float64)
    logits = np.where(trainable_mask, completed_q / float(posterior_temperature), -np.inf)
    finite = np.isfinite(logits)
    if not finite.any():
        target[trainable_mask] = 1.0 / float(trainable_mask.sum())
        return target
    max_logit = float(np.max(logits[finite]))
    exp = np.zeros_like(logits, dtype=np.float64)
    exp[finite] = np.exp(logits[finite] - max_logit)
    denom = float(exp.sum())
    if denom <= 0.0:
        target[trainable_mask] = 1.0 / float(trainable_mask.sum())
    else:
        target = (exp / denom).astype(np.float32)
    return target


def _softened_target(
    metadata: V1SearchPairMetadata,
    *,
    base_target: np.ndarray,
    trainable_mask: np.ndarray,
    posterior_temperature: float,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0.0 or int(trainable_mask.sum()) <= 1:
        return _renormalize(base_target, trainable_mask)
    completed_target = _completed_q_soft_target(
        metadata,
        trainable_mask=trainable_mask,
        posterior_temperature=posterior_temperature,
    )
    mixed = ((1.0 - alpha) * base_target) + (alpha * completed_target)
    return _renormalize(mixed.astype(np.float32, copy=False), trainable_mask)


def _completed_q_soft_target(
    metadata: V1SearchPairMetadata,
    *,
    trainable_mask: np.ndarray,
    posterior_temperature: float,
) -> np.ndarray:
    target = np.zeros(len(metadata.candidate_pairs), dtype=np.float32)
    if not bool(trainable_mask.any()):
        return target
    completed_q = np.asarray(metadata.completed_q_values, dtype=np.float64)
    logits = np.where(trainable_mask, completed_q / float(posterior_temperature), -np.inf)
    finite = np.isfinite(logits)
    if not finite.any():
        target[trainable_mask] = 1.0 / float(trainable_mask.sum())
        return target
    max_logit = float(np.max(logits[finite]))
    exp = np.zeros_like(logits, dtype=np.float64)
    exp[finite] = np.exp(logits[finite] - max_logit)
    denom = float(exp.sum())
    if denom <= 0.0:
        target[trainable_mask] = 1.0 / float(trainable_mask.sum())
    else:
        target = (exp / denom).astype(np.float32)
    return target


def _renormalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, values, 0.0).astype(np.float32, copy=False)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _cell_marginal_target(
    legal_row_ids: np.ndarray,
    pair_joint_target: np.ndarray,
    legal_row_count: int,
) -> np.ndarray:
    out = np.zeros(max(0, int(legal_row_count)), dtype=np.float32)
    if out.size == 0:
        return out
    for row_ids, mass in zip(legal_row_ids, pair_joint_target):
        value = float(mass)
        if value <= 0.0:
            continue
        first = int(row_ids[0])
        second = int(row_ids[1])
        if first < 0 or second < 0 or first >= out.shape[0] or second >= out.shape[0]:
            raise ValueError(f"V1 pair legal row ids out of range for marginal target: {(first, second)}")
        out[first] += 0.5 * value
        out[second] += 0.5 * value
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _conditional_targets(
    legal_row_ids: np.ndarray,
    pair_joint_target: np.ndarray,
) -> dict[str, np.ndarray]:
    marginal: dict[int, float] = {}
    for row_ids, mass in zip(legal_row_ids, pair_joint_target):
        value = float(mass)
        if value <= 0.0:
            continue
        first = int(row_ids[0])
        second = int(row_ids[1])
        if first < 0 or second < 0:
            continue
        marginal[first] = marginal.get(first, 0.0) + value
        marginal[second] = marginal.get(second, 0.0) + value
    pair_indices: list[int] = []
    first_ids: list[int] = []
    second_ids: list[int] = []
    targets: list[float] = []
    for pair_idx, (row_ids, mass) in enumerate(zip(legal_row_ids, pair_joint_target)):
        value = float(mass)
        if value <= 0.0:
            continue
        first = int(row_ids[0])
        second = int(row_ids[1])
        for cond_first, cond_second in ((first, second), (second, first)):
            denom = marginal.get(cond_first, 0.0)
            if denom <= 0.0:
                continue
            pair_indices.append(pair_idx)
            first_ids.append(cond_first)
            second_ids.append(cond_second)
            targets.append(value / denom)
    return {
        "pair_indices": np.asarray(pair_indices, dtype=np.int64),
        "first_ids": np.asarray(first_ids, dtype=np.int64),
        "second_ids": np.asarray(second_ids, dtype=np.int64),
        "target": np.asarray(targets, dtype=np.float32),
        "mask": np.ones(len(targets), dtype=np.bool_),
    }


def _terminal_tactical_target(masks: Mapping[str, np.ndarray], selected_index: int) -> np.ndarray:
    out = np.zeros(8, dtype=np.float32)
    out[0] = float(bool(masks["terminal_exact"].any()))
    out[1] = float(bool(masks["terminal_equivalent"].any()))
    out[2] = float(bool(masks["forced"].any()))
    out[3] = float(bool(masks["terminal_exact"].any() or masks["terminal_equivalent"].any()))
    if selected_index >= 0:
        out[4] = float(bool(masks["terminal_exact"][selected_index]))
        out[5] = float(bool(masks["terminal_equivalent"][selected_index]))
    out[6] = float(bool((masks["explicit_negative"] | masks["sampled_negative"]).any()))
    out[7] = float(bool(masks["unsampled"].any()))
    return out


def _selected_pair_index(selected_pair: PairKey | None, pair_keys: Sequence[PairKey]) -> int:
    if selected_pair is None:
        return -1
    for idx, pair_key in enumerate(pair_keys):
        if pair_key == selected_pair:
            return idx
    raise ValueError("selected_pair must be present in candidate_pairs")


def _transform_pair_key(
    pair_key: PairKey,
    transform: Callable[[tuple[int, int]], tuple[int, int]] | None,
) -> PairKey:
    first, second = pair_key
    a = (int(first[0]), int(first[1]))
    b = (int(second[0]), int(second[1]))
    if transform is not None:
        a = transform(a)
        b = transform(b)
    if a == b:
        raise ValueError(f"duplicate coordinates are illegal for V1 pair metadata: {a}")
    return (a, b) if a <= b else (b, a)


def _legal_row_ids_for_pair(
    candidate,
    pair_key: PairKey,
    legal_row_index_by_qr: Mapping[tuple[int, int], int] | None,
) -> tuple[int, int]:
    if legal_row_index_by_qr is None:
        return (int(candidate.first_legal_row_id), int(candidate.second_legal_row_id))
    first = legal_row_index_by_qr.get(pair_key[0])
    second = legal_row_index_by_qr.get(pair_key[1])
    if first is None or second is None:
        raise ValueError(f"V1 candidate pair references non-LEGAL cells after transform: {pair_key}")
    if first == second:
        raise ValueError(f"V1 candidate pair legal row ids must be distinct: {pair_key}")
    return (int(first), int(second))


def _proposal_score(proposal: Mapping[str, object]) -> float:
    log_prob = proposal.get("log_proposal_probability")
    if log_prob is not None:
        value = float(log_prob)
        if np.isfinite(value):
            return value
    prob = proposal.get("total_proposal_probability")
    if prob is not None:
        value = float(prob)
        if value > 0.0 and np.isfinite(value):
            return float(np.log(value))
    return 0.0


def _copy(destination: np.ndarray, source: np.ndarray, count: int) -> None:
    if int(count) <= 0:
        return
    destination[: int(count)] = source[: int(count)]
