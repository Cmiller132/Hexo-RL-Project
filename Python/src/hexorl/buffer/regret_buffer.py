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
    entry_id: int = 0
    observed_regret: float = 0.0
    refresh_count: int = 0
    inserted_step: int = 0
    last_sampled_step: int = 0
    last_updated_step: int = 0
    source: str = "trajectory_observed_regret"
    checkpoint_step: int = 0
    eviction_reason: str = ""


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
        self._next_entry_id = 1
        self._step = 0

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
        source: str = "trajectory_observed_regret",
    ) -> bool:
        """Try to add a state to the PRB.

        Returns True if the state was added (or updated), False otherwise.
        """
        self._step += 1
        entry = PRBEntry(
            move_history=move_history,
            regret=regret,
            rank_score=rank_score,
            game_id=game_id,
            entry_id=self._next_entry_id,
            observed_regret=regret,
            inserted_step=self._step,
            last_updated_step=self._step,
            source=source,
        )

        if not self.is_full:
            self._entries.append(entry)
            self._next_entry_id += 1
            return True

        min_idx = min(
            range(len(self._entries)),
            key=lambda i: (self._entries[i].regret, self._entries[i].last_sampled_step),
        )
        if regret > self._entries[min_idx].regret:
            entry.eviction_reason = "replaced_lower_ema_regret"
            self._entries[min_idx] = entry
            self._next_entry_id += 1
            return True

        return False

    def sample_with_index(
        self,
        rng: Optional[np.random.RandomState] = None,
    ) -> Optional[tuple[int, PRBEntry]]:
        """Sample a state and return its physical PRB index with the entry."""
        if not self._entries:
            return None

        ranks = np.array(
            [e.rank_score if e.rank_score > 0.0 else e.regret for e in self._entries],
            dtype=np.float64,
        )
        ranks = np.maximum(ranks, 1e-8)
        inv_temp = 1.0 / max(self.sampling_temperature, 1e-8)
        weights = ranks ** inv_temp
        probs = weights / weights.sum()

        if rng is None:
            rng = np.random.RandomState()

        idx = int(rng.choice(len(self._entries), p=probs))
        self._step += 1
        self._entries[idx].last_sampled_step = self._step
        return idx, self._entries[idx]

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Optional[PRBEntry]:
        """Sample a state from the PRB using softmax over regret^(1/τ).

        Higher-regret states are more likely to be sampled.
        Returns None if the buffer is empty.
        """
        sampled = self.sample_with_index(rng)
        return sampled[1] if sampled is not None else None

    def update_regret(self, entry_index: int, new_regret: float) -> float:
        """Update an entry's regret via EMA.

        R_new = (1-α)*R_old + α*R_new_observed
        """
        if 0 <= entry_index < len(self._entries):
            self._step += 1
            old = self._entries[entry_index].regret
            self._entries[entry_index].regret = (
                (1.0 - self.ema_alpha) * old + self.ema_alpha * new_regret
            )
            self._entries[entry_index].observed_regret = new_regret
            self._entries[entry_index].refresh_count += 1
            self._entries[entry_index].last_updated_step = self._step
            return float(self._entries[entry_index].regret - old)
        return 0.0

    def get_entries(self) -> List[PRBEntry]:
        return list(self._entries)

    def clear(self):
        self._entries.clear()
        self._next_entry_id = 1
        self._step = 0


def compute_regret(
    positions: List,
    outcome: float,
    *,
    allow_root_value_fallback: bool = False,
) -> List[float]:
    """Compute regret R(st) per Equation 2 of RGSC paper.

    R(st) = (1/(T-t)) * Σ_{i=t}^{T} (V_selected(si) - z)^2

    where:
      - V_selected(si) = MCTS value of the selected action from that player's perspective
      - z = game outcome from P0's perspective
      - T = total number of positions

    For each state, the regret is the average squared discrepancy between
    the MCTS evaluation and the final outcome, accumulated from that state
    to the end of the game.
    """
    T = len(positions)
    if T == 0:
        return []

    missing = [
        i for i, pos in enumerate(positions)
        if getattr(pos, "selected_action_value", None) is None
    ]
    if missing and not allow_root_value_fallback:
        raise ValueError(
            "selected_action_value is required for RGSC regret targets; "
            f"missing at positions {missing[:8]}"
        )

    regrets = []
    for t in range(T):
        sum_sq = 0.0
        for i in range(t, T):
            v_selected = (
                positions[i].selected_action_value
                if getattr(positions[i], "selected_action_value", None) is not None
                else positions[i].root_value
            )
            player = positions[i].player
            z = outcome if player == 0 else -outcome
            sum_sq += (v_selected - z) ** 2
        regret = sum_sq / max(T - t, 1)
        regrets.append(regret)

    return regrets
