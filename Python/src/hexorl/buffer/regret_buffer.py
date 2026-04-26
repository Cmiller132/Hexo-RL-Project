"""Prioritized Regret Buffer (PRB) — RGSC §3.3.

Stores a fixed-capacity set of high-regret states for search control.
During self-play, with probability β, the agent starts from a PRB-sampled
state instead of the empty board.

Key mechanisms:
  - Fixed capacity K (default 100)
  - Insertion: if not full, add state. If full, add only if regret > min in buffer.
  - Sampling: softmax over regret^(1/τ), τ=0.1 (favor high-regret states).
  - EMA update: R_new = (1-α)*R_old + α*R_current, α=0.5.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class PRBEntry:
    """One entry in the prioritized regret buffer."""
    move_history: bytes
    regret: float
    rank_score: float = 0.0
    game_id: int = 0


class PrioritizedRegretBuffer:
    """Fixed-capacity buffer of high-regret states for search control.

    RGSC §3.3: maintains K states with highest regret. Sampled
    with softmax over regret^(1/τ) as restart positions.
    """

    def __init__(
        self,
        capacity: int = 100,
        ema_alpha: float = 0.5,
        sampling_temperature: float = 0.1,
    ):
        self.capacity = capacity
        self.ema_alpha = ema_alpha
        self.sampling_temperature = sampling_temperature
        self._entries: List[PRBEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def is_full(self) -> bool:
        return len(self._entries) >= self.capacity

    def add(
        self,
        move_history: bytes,
        regret: float,
        rank_score: float = 0.0,
        game_id: int = 0,
    ) -> bool:
        """Try to add a state to the PRB.

        Returns True if the state was added (or updated), False otherwise.
        """
        entry = PRBEntry(
            move_history=move_history,
            regret=regret,
            rank_score=rank_score,
            game_id=game_id,
        )

        if not self.is_full:
            self._entries.append(entry)
            return True

        min_idx = min(range(len(self._entries)), key=lambda i: self._entries[i].regret)
        if regret > self._entries[min_idx].regret:
            self._entries[min_idx] = entry
            return True

        return False

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Optional[PRBEntry]:
        """Sample a state from the PRB using softmax over regret^(1/τ).

        Higher-regret states are more likely to be sampled.
        Returns None if the buffer is empty.
        """
        if not self._entries:
            return None

        regrets = np.array([e.regret for e in self._entries], dtype=np.float64)
        regrets = np.maximum(regrets, 1e-8)
        inv_temp = 1.0 / max(self.sampling_temperature, 1e-8)
        weights = regrets ** inv_temp
        probs = weights / weights.sum()

        if rng is None:
            rng = np.random.RandomState()

        idx = rng.choice(len(self._entries), p=probs)
        return self._entries[idx]

    def update_regret(self, entry_index: int, new_regret: float):
        """Update an entry's regret via EMA.

        R_new = (1-α)*R_old + α*R_new_observed
        """
        if 0 <= entry_index < len(self._entries):
            old = self._entries[entry_index].regret
            self._entries[entry_index].regret = (
                (1.0 - self.ema_alpha) * old + self.ema_alpha * new_regret
            )

    def get_entries(self) -> List[PRBEntry]:
        return list(self._entries)

    def clear(self):
        self._entries.clear()


def compute_regret(
    positions: List,
    outcome: float,
) -> List[float]:
    """Compute regret R(st) per Equation 2 of RGSC paper.

    R(st) = (1/(T-t)) * Σ_{i=t}^{T} (V_selected(si) - z)^2

    where:
      - V_selected(si) = MCTS root value at state si (from that player's perspective)
      - z = game outcome from P0's perspective
      - T = total number of positions

    For each state, the regret is the average squared discrepancy between
    the MCTS evaluation and the final outcome, accumulated from that state
    to the end of the game.
    """
    T = len(positions)
    if T == 0:
        return []

    regrets = []
    for t in range(T):
        sum_sq = 0.0
        for i in range(t, T):
            v_selected = positions[i].root_value
            player = positions[i].player
            z = outcome if player == 0 else -outcome
            sum_sq += (v_selected - z) ** 2
        regret = sum_sq / max(T - t, 1)
        regrets.append(regret)

    return regrets
