"""Target computation for self-play training data.

Runs CPU-side on completed game records. Computes:
  - Value targets (outcome-based, with optional EMA lookahead).
  - Policy targets (sparse from MCTS visits, top-K renormalized).
  - Lookahead value targets (KataGo-style multi-horizon).
  - Auxiliary targets used by the full training head set:
    opponent policy, regret rank/value, dominant axis, and moves-left.
"""

from bisect import bisect_right
import numpy as np
from typing import List, Tuple, Optional
from hexorl.selfplay.records import GameRecord, PositionRecord, pair_policy_v2_from_place_target


def compute_value_targets(
    positions: List[PositionRecord],
    outcome: float,
    lookahead_horizons: Optional[List[int]] = None,
) -> None:
    """Assign value targets to each position in a game.

    Outcome assignment:
      - Player 0: value = outcome
      - Player 1: value = -outcome

    Args:
        positions: List of positions from one game (in order).
        outcome: Final game outcome from P0's perspective (1.0, -1.0).
        lookahead_horizons: EMA lookahead horizons in turn boundaries.
    """
    for pos in positions:
        pos.outcome = outcome


def _turn_boundary_indices(positions: List[PositionRecord]) -> List[int]:
    """Return indices of positions that are turn boundaries.

    In Hexo, the opening turn has one placement and later turns have two
    placements by the same player. Boundaries are therefore player-run starts,
    e.g. players [0, 1, 1, 0, 0] -> indices [0, 1, 3].
    """
    return [
        i
        for i, pos in enumerate(positions)
        if i == 0 or pos.player != positions[i - 1].player
    ]


def compute_ema_lookahead(
    positions: List[PositionRecord],
    horizon: int,
    lambda_: float = 0.75,
) -> np.ndarray:
    """Compute EMA-weighted lookahead value targets at a given horizon.

    Uses MCTS root values (not final outcomes) and counts horizons
    in turn boundaries, matching the KataGo-adapted design.

    KataGo-style: for each position at index i, the lookahead target is:
      target_i = (1 - λ) * root_value_i + λ * target_{i + horizon_boundaries}

    where root_value_i is the MCTS root value from that position's
    perspective, and the horizon is counted in turn boundaries.

    Args:
        positions: List of positions from one game.
        horizon: Number of turn boundaries to look ahead.
        lambda_: EMA decay factor (λ ∈ [0, 1)).

    Returns:
        np.ndarray of shape (num_positions,) with EMA lookahead values.
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.float32)

    boundaries = _turn_boundary_indices(positions)
    mcts_values = np.array([pos.root_value for pos in positions], dtype=np.float32)
    result = np.copy(mcts_values)

    for i in range(n - 1, -1, -1):
        target_bi = bisect_right(boundaries, i) + horizon - 1
        if target_bi < len(boundaries):
            j = boundaries[target_bi]
            future = result[j]
            if positions[j].player != positions[i].player:
                future = -future
            result[i] = (1.0 - lambda_) * mcts_values[i] + lambda_ * future
        else:
            result[i] = mcts_values[i]

    return result


def compute_policy_targets(
    visit_counts: np.ndarray,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert MCTS visit counts to a normalized sparse policy target.

    Args:
        visit_counts: Dense array of shape (BOARD_AREA,) with visit counts.
            Zero for illegal/unvisited moves.
        top_k: Number of top entries to keep in sparse representation.

    Returns:
        (dense_policy, sparse_indices, sparse_probs):
            dense_policy: (BOARD_AREA,) float32 normalized visit distribution.
            sparse_indices: (top_k,) int array of action indices.
            sparse_probs: (top_k,) float32 array of probabilities.
    """
    counts = visit_counts.astype(np.float32)
    total = counts.sum()
    if total > 0:
        dense = counts / total
    else:
        dense = np.ones_like(counts) / len(counts)

    # Top-K
    n = min(top_k, len(dense))
    indices = np.argpartition(-dense, n - 1)[:n]
    probs = dense[indices]
    probs = probs / probs.sum()

    return dense, indices.astype(np.int32), probs.astype(np.float32)


def process_game_record(
    record: GameRecord,
    lookahead_horizons: Optional[List[int]] = None,
    lookahead_lambdas: Optional[List[float]] = None,
) -> List[PositionRecord]:
    """Process a complete game record into training-ready PositionRecords.

    Steps:
      1. Assign value targets (outcome-based).
      2. Optionally compute EMA lookahead targets at each horizon.
      3. Populate auxiliary targets for the configured model heads.

    Returns:
        List of PositionRecords ready for buffer insertion.
    """
    record.assign_outcomes()

    # Compute lookahead targets if requested
    lookahead_targets = {}
    if lookahead_horizons and lookahead_lambdas:
        for h, lam in zip(lookahead_horizons, lookahead_lambdas):
            lookahead_targets[h] = compute_ema_lookahead(
                record.positions, horizon=h, lambda_=lam
            )

    # Assign lookahead values to each position
    for i, pos in enumerate(record.positions):
        pos.lookahead_values = [
            float(lookahead_targets[h][i]) for h in lookahead_targets
        ]

    _assign_auxiliary_targets(record)

    return record.positions


def _assign_auxiliary_targets(record: GameRecord) -> None:
    """Populate auxiliary targets that can be computed once the game is complete."""
    positions = record.positions
    total = len(positions)
    if total == 0:
        return

    for i, pos in enumerate(positions):
        opp_idx = _next_full_search_opponent_turn_start(positions, i)
        if opp_idx is not None:
            opp = positions[opp_idx]
            pos.opp_policy_target = dict(opp.policy_target)
            pos.opp_policy_target_v2 = list(getattr(opp, "policy_target_v2", []))
            pos.opp_policy_weight = 1.0
        else:
            pos.opp_policy_target = {}
            pos.opp_policy_target_v2 = []
            pos.opp_policy_weight = 0.0
        if not pos.pair_policy_target_v2 and pos.policy_target_v2:
            pos.pair_policy_target_v2 = pair_policy_v2_from_place_target(pos.policy_target_v2)
        perspective_outcome = record.outcome if pos.player == 0 else -record.outcome
        tail = positions[i:]
        regret = sum(
            (
                (
                    p.selected_action_value
                    if p.selected_action_value is not None
                    else p.root_value
                )
                - (record.outcome if p.player == 0 else -record.outcome)
            ) ** 2
            for p in tail
        ) / max(len(tail), 1)
        pos.regret_rank = float(regret)
        pos.regret_value = float(regret)
        pos.moves_left = float(max(total - pos.turn_index, 0))
        if pos.outcome is None:
            pos.outcome = perspective_outcome if pos.player == 0 else -perspective_outcome

    winner = 0 if record.outcome > 0 else 1 if record.outcome < 0 else None
    axis_history = record.final_move_history or positions[-1].move_history
    axis = _dominant_axis_label(axis_history, winner)
    for pos in positions:
        pos.axis_label = axis


def _next_full_search_opponent_turn_start(
    positions: List[PositionRecord],
    i: int,
) -> int | None:
    """Return the next opponent turn-start index with a full-search target."""
    player = positions[i].player
    for j in range(i + 1, len(positions)):
        if positions[j].player == player:
            continue
        if j > 0 and positions[j - 1].player == positions[j].player:
            continue
        if positions[j].is_full_search:
            return j
    return None


def _dominant_axis_label(history_bytes: bytes, winner: int | None) -> int:
    """Return the longest-run axis for the winner, or -1 when unknown."""
    if winner is None or not history_bytes:
        return -1

    stones = set()
    stride = 12
    for offset in range(0, len(history_bytes) - stride + 1, stride):
        player = int.from_bytes(history_bytes[offset:offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4:offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8:offset + 12], "little", signed=True)
        if player == winner:
            stones.add((q, r))

    if not stones:
        return -1

    axes = [(1, 0), (0, 1), (1, -1)]
    best_axis = -1
    best_len = 0
    for axis_idx, (dq, dr) in enumerate(axes):
        for q, r in stones:
            prev = (q - dq, r - dr)
            if prev in stones:
                continue
            run = 1
            nq, nr = q + dq, r + dr
            while (nq, nr) in stones:
                run += 1
                nq += dq
                nr += dr
            if run > best_len:
                best_len = run
                best_axis = axis_idx

    return best_axis if best_len > 0 else -1
