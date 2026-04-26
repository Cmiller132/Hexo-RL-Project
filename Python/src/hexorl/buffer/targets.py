"""Target computation for self-play training data.

Runs CPU-side on completed game records. Computes:
  - Value targets (outcome-based, with optional EMA lookahead).
  - Policy targets (sparse from MCTS visits, top-K renormalized).
  - Lookahead value targets (KataGo-style multi-horizon).

Phase 3: Simple outcome assignment. Phase 4: Full EMA lookahead.
"""

import numpy as np
from typing import List, Tuple, Optional
from hexorl.selfplay.records import GameRecord, PositionRecord, sparsify_policy


def compute_value_targets(
    positions: List[PositionRecord],
    outcome: float,
    lookahead_horizons: Optional[List[int]] = None,
) -> None:
    """Assign value targets to each position in a game.

    Simple outcome assignment (Phase 3):
      - Player 0: value = outcome
      - Player 1: value = -outcome

    Phase 4 will add KataGo-style EMA lookahead at specified horizons.

    Args:
        positions: List of positions from one game (in order).
        outcome: Final game outcome from P0's perspective (1.0, -1.0).
        lookahead_horizons: Future: EMA lookahead horizons in turn boundaries.
    """
    for pos in positions:
        pos.outcome = outcome


def compute_ema_lookahead(
    positions: List[PositionRecord],
    horizon: int,
    lambda_: float = 0.75,
) -> np.ndarray:
    """Compute EMA-weighted lookahead value targets at a given horizon.

    KataGo-style: for each position at index i, the lookahead target is:
      target_i = (1 - λ) * outcome_i + λ * target_{i + horizon}

    where outcome_i is the position player's value target.

    This provides a smoother training signal that accounts for
    intermediate who-is-winning transitions.

    Args:
        positions: List of positions from one game.
        horizon: Number of future positions to look ahead (in turn boundaries).
        lambda_: EMA decay factor (λ ∈ [0, 1)).

    Returns:
        np.ndarray of shape (num_positions,) with EMA lookahead values.
    """
    n = len(positions)
    if n == 0:
        return np.array([], dtype=np.float32)

    outcomes = np.array([pos.to_value_target() for pos in positions], dtype=np.float32)
    result = np.copy(outcomes)

    # Backward EMA: result[i] = λ * result[i+horizon] + (1-λ) * outcomes[i]
    for i in range(n - 1, -1, -1):
        j = i + horizon
        if j < n:
            result[i] = (1.0 - lambda_) * outcomes[i] + lambda_ * result[j]
        else:
            result[i] = outcomes[i]

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
      3. Sparsify policy targets to top-K.

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

    return record.positions
