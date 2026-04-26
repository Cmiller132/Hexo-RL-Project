"""KataGo-style continuous ring buffer for experience replay.

Stores compact game records (§7.1 of SYSTEM_DESIGN.md). Features:
  - Fixed capacity circular buffer with oldest-first eviction.
  - Struct-of-arrays storage for efficient numpy access.
  - Recency-weighted random sampling.
  - PCR quality gating (full-search vs low-sim weights).
  - Thread-safe append and read.

Memory budget: ~700 MB at 2M capacity (avg 350 bytes/sample).
"""

import threading
import numpy as np
from typing import List, Dict, Optional
from hexorl.selfplay.records import PositionRecord


class RingBuffer:
    """Fixed-capacity circular buffer for experience replay."""

    def __init__(
        self,
        capacity: int,
        max_policy_entries: int = 20,
        recency_decay: float = 0.99,
    ):
        self.capacity = capacity
        self.max_policy_entries = max_policy_entries
        self.recency_decay = recency_decay

        # Storage arrays — struct of arrays
        self._histories: List[Optional[bytes]] = [None] * capacity
        self._policies = np.zeros((capacity, max_policy_entries), dtype=np.uint16)
        self._policy_probs = np.zeros((capacity, max_policy_entries), dtype=np.float32)
        self._policy_counts = np.zeros(capacity, dtype=np.uint16)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._game_ids = np.zeros(capacity, dtype=np.uint32)
        self._is_full = np.zeros(capacity, dtype=np.bool_)
        self._players = np.zeros(capacity, dtype=np.uint8)

        # Ring pointers
        self._head: int = 0
        self._tail: int = 0
        self._size: int = 0
        self._max_game_id: int = 0

        # Thread safety
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._size

    @property
    def max_game_id(self) -> int:
        return self._max_game_id

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def append(self, record: PositionRecord):
        """Append one position record. Thread-safe."""
        with self._lock:
            idx = self._head

            self._histories[idx] = record.move_history

            entries = list(record.policy_target.items())
            n = min(len(entries), self.max_policy_entries)
            self._policy_counts[idx] = n
            for j, (action_idx, prob) in enumerate(entries[:n]):
                self._policies[idx, j] = action_idx
                self._policy_probs[idx, j] = prob

            self._values[idx] = record.to_value_target()
            self._game_ids[idx] = record.game_id
            self._is_full[idx] = record.is_full_search
            self._players[idx] = record.player

            self._head = (self._head + 1) % self.capacity
            if self._size == self.capacity:
                self._tail = (self._tail + 1) % self.capacity
            else:
                self._size += 1

            self._max_game_id = max(self._max_game_id, record.game_id)

    def extend(self, records: List[PositionRecord]):
        """Append multiple records (single lock acquisition)."""
        with self._lock:
            for record in records:
                self._append_unlocked(record)

    def _append_unlocked(self, record: PositionRecord):
        """Internal append — caller holds self._lock."""
        idx = self._head
        self._histories[idx] = record.move_history
        entries = list(record.policy_target.items())
        n = min(len(entries), self.max_policy_entries)
        self._policy_counts[idx] = n
        for j, (action_idx, prob) in enumerate(entries[:n]):
            self._policies[idx, j] = action_idx
            self._policy_probs[idx, j] = prob
        self._values[idx] = record.to_value_target()
        self._game_ids[idx] = record.game_id
        self._is_full[idx] = record.is_full_search
        self._players[idx] = record.player
        self._head = (self._head + 1) % self.capacity
        if self._size == self.capacity:
            self._tail = (self._tail + 1) % self.capacity
        else:
            self._size += 1
        self._max_game_id = max(self._max_game_id, record.game_id)

    def sample_indices(
        self,
        n: int,
        recency_decay: Optional[float] = None,
        pcr_weight: float = 0.25,
    ) -> np.ndarray:
        """Sample n indices with recency-weighted probability.

        Probability ∝ decay^(max_game_id - game_id) × (4.0 if full-search else pcr_weight).
        """
        if n <= 0 or self._size == 0:
            return np.array([], dtype=np.int64)

        decay = recency_decay if recency_decay is not None else self.recency_decay

        with self._lock:
            weights = np.ones(self._size, dtype=np.float64)
            for i in range(self._size):
                idx = (self._tail + i) % self.capacity
                game_age = self._max_game_id - int(self._game_ids[idx])
                recency_w = decay ** max(game_age, 0)
                quality_w = 4.0 if self._is_full[idx] else pcr_weight
                weights[i] = recency_w * quality_w

        total = weights.sum()
        if total <= 0:
            return np.array([], dtype=np.int64)

        probs = weights / total
        logical = np.random.choice(self._size, size=n, p=probs, replace=True)
        physical = (self._tail + logical) % self.capacity
        return physical.astype(np.int64)

    def __getitem__(self, idx: int) -> Optional[PositionRecord]:
        """Retrieve a single position record by physical index."""
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"Index {idx} out of range [0, {self.capacity})")
        if self._histories[idx] is None:
            return None

        policy = {}
        n = int(self._policy_counts[idx])
        for j in range(n):
            action_idx = int(self._policies[idx, j])
            prob = float(self._policy_probs[idx, j])
            if prob > 0:
                policy[action_idx] = prob

        stored_value = float(self._values[idx])
        player = int(self._players[idx])
        outcome = stored_value if player == 0 else -stored_value

        return PositionRecord(
            move_history=self._histories[idx],
            policy_target=policy,
            root_value=0.0,
            player=player,
            game_id=int(self._game_ids[idx]),
            is_full_search=bool(self._is_full[idx]),
            outcome=outcome,
        )

    def get_batch(self, indices: np.ndarray) -> List[PositionRecord]:
        """Retrieve multiple records by physical indices."""
        records = []
        for idx in indices:
            rec = self[idx]
            if rec is not None:
                records.append(rec)
        return records

    @property
    def stats(self) -> dict:
        """Return buffer statistics."""
        with self._lock:
            if self._size == 0:
                return {"size": 0, "capacity": self.capacity, "max_game_id": 0}

            full_count = 0
            for i in range(self._size):
                idx = (self._tail + i) % self.capacity
                if self._is_full[idx]:
                    full_count += 1

            return {
                "size": self._size,
                "capacity": self.capacity,
                "max_game_id": self._max_game_id,
                "full_search_pct": full_count / self._size * 100.0 if self._size > 0 else 0.0,
            }

    def clear(self):
        """Reset the buffer to empty."""
        with self._lock:
            self._histories = [None] * self.capacity
            self._policies.fill(0)
            self._policy_probs.fill(0.0)
            self._policy_counts.fill(0)
            self._values.fill(0.0)
            self._game_ids.fill(0)
            self._is_full.fill(False)
            self._players.fill(0)
            self._head = 0
            self._tail = 0
            self._size = 0
            self._max_game_id = 0
