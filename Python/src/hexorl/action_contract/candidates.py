"""Candidate/action-keyed policy construction.

Phase 1 keeps this deliberately conservative: MCTS target actions are always
included, legal moves fill the remaining budget deterministically, and sparse
targets report any mass not represented by the candidate list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hexorl.selfplay.records import BOARD_AREA, BOARD_SIZE, PolicyTargetV2, action_to_board_index


CANDIDATE_FEATURES = 12


@dataclass(frozen=True)
class CandidateBatch:
    qr: np.ndarray
    indices: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    missing_mass: float
    recall_top1: float
    recall_top4: float
    recall_top8: float
    recall_winning_move: float
    recall_forced_block: float
    recall_two_placement_cover: float


@dataclass(frozen=True)
class PairCandidateBatch:
    pair_indices: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    missing_mass: float


def _unique_qr(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in items:
        qr = (int(q), int(r))
        if qr not in seen:
            seen.add(qr)
            out.append(qr)
    return out


def build_candidate_set(
    legal_moves: Sequence[tuple[int, int]],
    policy_target_v2: PolicyTargetV2,
    budget: int,
    *,
    winning_moves: Sequence[tuple[int, int]] = (),
    forced_block_moves: Sequence[tuple[int, int]] = (),
    cover_cells: Sequence[tuple[int, int]] = (),
    critical_actions: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Build a deterministic candidate set.

    Target and tactical actions are critical and can exceed the nominal budget;
    the budget controls only the non-critical fill.
    """
    budget = max(1, int(budget))
    target_actions = _unique_qr((q, r) for q, r, prob in policy_target_v2 if prob > 0.0)
    legal_set = set(_unique_qr((int(move[0]), int(move[1])) for move in legal_moves))
    tactical_actions = _unique_qr(
        list(winning_moves) + list(forced_block_moves) + list(cover_cells) + list(critical_actions)
    )
    tactical_actions = [qr for qr in tactical_actions if qr in legal_set]
    protected = _unique_qr(target_actions + tactical_actions)
    fill_pool = sorted(
        legal_set - set(protected),
        key=lambda qr: (max(abs(qr[0]), abs(qr[1]), abs(qr[0] + qr[1])), qr[0], qr[1]),
    )
    slots = max(0, budget - len(protected))
    return protected + fill_pool[:slots]


def build_candidate_batch(
    legal_moves: Sequence[tuple[int, int]],
    policy_target_v2: PolicyTargetV2,
    *,
    offset_q: int,
    offset_r: int,
    budget: int,
    winning_moves: Sequence[tuple[int, int]] = (),
    forced_block_moves: Sequence[tuple[int, int]] = (),
    cover_cells: Sequence[tuple[int, int]] = (),
    critical_actions: Sequence[tuple[int, int]] = (),
) -> CandidateBatch:
    winning_set = set(_unique_qr(winning_moves))
    forced_set = set(_unique_qr(forced_block_moves))
    cover_set = set(_unique_qr(cover_cells))
    critical_set = set(_unique_qr(critical_actions)) | winning_set | forced_set | cover_set
    candidates = build_candidate_set(
        legal_moves,
        policy_target_v2,
        budget,
        winning_moves=winning_moves,
        forced_block_moves=forced_block_moves,
        cover_cells=cover_cells,
        critical_actions=critical_actions,
    )
    target_map: dict[tuple[int, int], float] = {
        (int(q), int(r)): float(prob) for q, r, prob in policy_target_v2 if prob > 0.0
    }
    k = max(len(candidates), int(budget))
    qr_arr = np.zeros((k, 2), dtype=np.int32)
    indices = np.full(k, -1, dtype=np.int64)
    features = np.zeros((k, CANDIDATE_FEATURES), dtype=np.float32)
    mask = np.zeros(k, dtype=np.bool_)
    target = np.zeros(k, dtype=np.float32)

    legal_list = _unique_qr((int(move[0]), int(move[1])) for move in legal_moves)
    legal_rank = {qr: i for i, qr in enumerate(sorted(set(legal_list)))}
    target_sorted = sorted(policy_target_v2, key=lambda item: -float(item[2]))
    target_top1 = {(int(q), int(r)) for q, r, _ in target_sorted[:1]}
    target_top4 = {(int(q), int(r)) for q, r, _ in target_sorted[:4]}
    target_top8 = {(int(q), int(r)) for q, r, _ in target_sorted[:8]}
    candidate_set = set(candidates)

    represented_mass = 0.0
    half = BOARD_SIZE // 2
    max_coord = float(max(half, 1))
    max_legal_rank = float(max(len(legal_rank) - 1, 1))
    for i, (q, r) in enumerate(candidates):
        if i >= k:
            break
        idx = action_to_board_index(q, r, offset_q, offset_r)
        prob = target_map.get((q, r), 0.0)
        represented_mass += prob
        qr_arr[i] = (q, r)
        indices[i] = idx
        mask[i] = True
        target[i] = prob
        gi = q - offset_q
        gj = r - offset_r
        in_crop = 1.0 if idx >= 0 else 0.0
        dist = max(abs(q), abs(r), abs(q + r))
        features[i] = np.array(
            [
                q / max_coord,
                r / max_coord,
                (q + r) / max_coord,
                dist / max_coord,
                in_crop,
                ((gi - half) / max_coord) if idx >= 0 else 0.0,
                ((gj - half) / max_coord) if idx >= 0 else 0.0,
                legal_rank.get((q, r), 0) / max_legal_rank,
                1.0 if (q, r) in winning_set else 0.0,
                1.0 if (q, r) in forced_set else 0.0,
                1.0 - in_crop,
                1.0 if (q, r) in critical_set else 0.0,
            ],
            dtype=np.float32,
        )
    if represented_mass > 0:
        target /= represented_mass
    missing_mass = max(0.0, 1.0 - represented_mass)

    def recall(top: set[tuple[int, int]]) -> float:
        if not top:
            return 1.0
        return len(top & candidate_set) / float(len(top))

    return CandidateBatch(
        qr=qr_arr,
        indices=indices,
        features=features,
        mask=mask,
        target=target,
        missing_mass=missing_mass,
        recall_top1=recall(target_top1),
        recall_top4=recall(target_top4),
        recall_top8=recall(target_top8),
        recall_winning_move=recall(winning_set),
        recall_forced_block=recall(forced_set),
        recall_two_placement_cover=recall(cover_set),
    )


def _canonical_pair(
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    return (a, b) if a <= b else (b, a)


def build_pair_candidate_batch(
    candidate_qr: Sequence[tuple[int, int]],
    pair_policy_target_v2: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    budget: int,
    candidate_mask: Sequence[bool] | None = None,
) -> PairCandidateBatch:
    """Build a bounded pair-action target over candidate row indices."""
    budget = max(1, int(budget))
    if candidate_mask is None:
        mask_iter = [True] * len(candidate_qr)
    else:
        mask_iter = [bool(x) for x in candidate_mask]
    candidate_list: list[tuple[int, int]] = []
    candidate_index: dict[tuple[int, int], int] = {}
    for row, (qr_raw, keep) in enumerate(zip(candidate_qr, mask_iter)):
        if not keep:
            continue
        qr = (int(qr_raw[0]), int(qr_raw[1]))
        if qr in candidate_index:
            continue
        candidate_index[qr] = row
        candidate_list.append(qr)

    target_map: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    protected: list[tuple[tuple[int, int], tuple[int, int]]] = []
    total_target_mass = 0.0
    for a, b, prob in pair_policy_target_v2:
        if prob <= 0.0:
            continue
        key = _canonical_pair(a, b)
        total_target_mass += float(prob)
        target_map[key] = target_map.get(key, 0.0) + float(prob)
        if key[0] in candidate_index and key[1] in candidate_index and key[0] != key[1]:
            protected.append(key)
    protected = list(dict.fromkeys(protected))

    fill: list[tuple[tuple[int, int], tuple[int, int]]] = []
    n = min(len(candidate_list), 64)
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_pair(candidate_list[i], candidate_list[j])
            if key not in target_map:
                fill.append(key)
    fill.sort(
        key=lambda pair: (
            max(abs(pair[0][0]), abs(pair[0][1]), abs(pair[0][0] + pair[0][1]))
            + max(abs(pair[1][0]), abs(pair[1][1]), abs(pair[1][0] + pair[1][1])),
            pair,
        )
    )
    pairs = protected + fill[: max(0, budget - len(protected))]
    width = max(len(pairs), budget)
    pair_indices = np.full((width, 2), -1, dtype=np.int64)
    mask = np.zeros(width, dtype=np.bool_)
    target = np.zeros(width, dtype=np.float32)
    represented_mass = 0.0

    for row, pair in enumerate(pairs[:width]):
        first = candidate_index.get(pair[0], -1)
        second = candidate_index.get(pair[1], -1)
        if first < 0 or second < 0 or first == second:
            continue
        pair_indices[row] = (first, second)
        mask[row] = True
        prob = target_map.get(pair, 0.0)
        target[row] = prob
        represented_mass += prob
    if represented_mass > 0.0:
        target /= represented_mass
    missing_mass = max(0.0, total_target_mass - represented_mass)
    return PairCandidateBatch(pair_indices=pair_indices, mask=mask, target=target, missing_mass=missing_mass)
