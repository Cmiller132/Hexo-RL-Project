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
) -> list[tuple[int, int]]:
    """Build a deterministic candidate set.

    Target actions are critical and can exceed the nominal budget; the budget
    controls only the non-critical fill.
    """
    budget = max(1, int(budget))
    target_actions = _unique_qr((q, r) for q, r, prob in policy_target_v2 if prob > 0.0)
    legal_set = set(_unique_qr((int(move[0]), int(move[1])) for move in legal_moves))
    fill_pool = sorted(
        legal_set - set(target_actions),
        key=lambda qr: (max(abs(qr[0]), abs(qr[1]), abs(qr[0] + qr[1])), qr[0], qr[1]),
    )
    slots = max(0, budget - len(target_actions))
    return target_actions + fill_pool[:slots]


def build_candidate_batch(
    legal_moves: Sequence[tuple[int, int]],
    policy_target_v2: PolicyTargetV2,
    *,
    offset_q: int,
    offset_r: int,
    budget: int,
) -> CandidateBatch:
    candidates = build_candidate_set(legal_moves, policy_target_v2, budget)
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
                prob,
                1.0 if prob > 0.0 else 0.0,
                legal_rank.get((q, r), 0) / max_legal_rank,
                1.0 - in_crop,
                1.0,
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
    )
