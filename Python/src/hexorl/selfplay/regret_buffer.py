"""Prioritized regret restart buffer for self-play RGSC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PRBEntry:
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
    def __init__(
        self,
        capacity: int = 100,
        ema_alpha: float = 0.5,
        sampling_temperature: float = 0.1,
    ):
        self.capacity = int(capacity)
        self.ema_alpha = float(ema_alpha)
        self.sampling_temperature = float(sampling_temperature)
        self._entries: list[PRBEntry] = []
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
        self._step += 1
        entry = PRBEntry(
            move_history=bytes(move_history),
            regret=float(regret),
            rank_score=float(rank_score),
            game_id=int(game_id),
            entry_id=self._next_entry_id,
            observed_regret=float(regret),
            inserted_step=self._step,
            last_updated_step=self._step,
            source=str(source),
        )
        if not self.is_full:
            self._entries.append(entry)
            self._next_entry_id += 1
            return True
        min_idx = min(range(len(self._entries)), key=lambda i: (self._entries[i].regret, self._entries[i].last_sampled_step))
        if float(regret) > self._entries[min_idx].regret:
            entry.eviction_reason = "replaced_lower_ema_regret"
            self._entries[min_idx] = entry
            self._next_entry_id += 1
            return True
        return False

    def sample_with_index(self, rng: Optional[np.random.RandomState] = None) -> Optional[tuple[int, PRBEntry]]:
        if not self._entries:
            return None
        ranks = np.array([e.rank_score if e.rank_score > 0.0 else e.regret for e in self._entries], dtype=np.float64)
        ranks = np.maximum(ranks, 1e-8)
        weights = ranks ** (1.0 / max(self.sampling_temperature, 1e-8))
        probs = weights / weights.sum()
        rng = rng or np.random.RandomState()
        idx = int(rng.choice(len(self._entries), p=probs))
        self._step += 1
        self._entries[idx].last_sampled_step = self._step
        return idx, self._entries[idx]

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Optional[PRBEntry]:
        sampled = self.sample_with_index(rng)
        return sampled[1] if sampled is not None else None

    def update_regret(self, entry_index: int, new_regret: float) -> float:
        if 0 <= int(entry_index) < len(self._entries):
            self._step += 1
            entry = self._entries[int(entry_index)]
            old = entry.regret
            entry.regret = (1.0 - self.ema_alpha) * old + self.ema_alpha * float(new_regret)
            entry.observed_regret = float(new_regret)
            entry.refresh_count += 1
            entry.last_updated_step = self._step
            return float(entry.regret - old)
        return 0.0

    def get_entries(self) -> list[PRBEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._next_entry_id = 1
        self._step = 0


def compute_regret(positions: list, outcome: float, *, allow_root_value_fallback: bool = False) -> list[float]:
    if not positions:
        return []
    missing = [i for i, pos in enumerate(positions) if getattr(pos, "selected_action_value", None) is None]
    if missing and not allow_root_value_fallback:
        raise ValueError(f"selected_action_value is required for RGSC regret targets; missing at positions {missing[:8]}")
    regrets: list[float] = []
    for start in range(len(positions)):
        total = 0.0
        for pos in positions[start:]:
            selected = getattr(pos, "selected_action_value", None)
            if selected is None:
                selected = pos.root_value
            z = float(outcome) if int(pos.player) == 0 else -float(outcome)
            total += (float(selected) - z) ** 2
        regrets.append(float(total / max(len(positions) - start, 1)))
    return regrets
