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
from typing import List, Optional
from hexorl.selfplay.records import PositionRecord


class RingBuffer:
    """Fixed-capacity circular buffer for experience replay."""

    def __init__(
        self,
        capacity: int,
        max_policy_entries: int = 20,
        recency_decay: float = 0.99,
        num_lookahead: int = 0,
        max_policy_v2_entries: int | None = None,
        store_opp_policy: bool = True,
        store_pair_policy: bool = True,
        store_sparse_diagnostics: bool = True,
    ):
        if capacity <= 0:
            raise ValueError("RingBuffer capacity must be positive")
        if max_policy_entries <= 0:
            raise ValueError("RingBuffer max_policy_entries must be positive")
        if num_lookahead < 0:
            raise ValueError("RingBuffer num_lookahead cannot be negative")

        self.capacity = capacity
        self.max_policy_entries = max_policy_entries
        self.max_policy_v2_entries = int(max_policy_v2_entries or max_policy_entries)
        self.recency_decay = recency_decay
        self.num_lookahead = num_lookahead
        self.store_opp_policy = bool(store_opp_policy)
        self.store_pair_policy = bool(store_pair_policy)
        self.store_sparse_diagnostics = bool(store_sparse_diagnostics)

        # Storage arrays — struct of arrays
        self._histories: List[Optional[bytes]] = [None] * capacity
        self._policies = np.zeros((capacity, max_policy_entries), dtype=np.uint16)
        self._policy_probs = np.zeros((capacity, max_policy_entries), dtype=np.float32)
        self._policy_counts = np.zeros(capacity, dtype=np.uint16)
        self._policy_v2_q = np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
        self._policy_v2_r = np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
        self._policy_v2_probs = np.zeros((capacity, self.max_policy_v2_entries), dtype=np.float32)
        self._policy_v2_counts = np.zeros(capacity, dtype=np.uint16)
        self._opp_policy_v2_q = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_v2_r = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_v2_probs = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.float32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_v2_counts = np.zeros(capacity, dtype=np.uint16)
        self._opp_policy_legal_v2_q = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_legal_v2_r = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_legal_v2_counts = np.zeros(capacity, dtype=np.uint16)
        self._pair_policy_v2_q1 = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_pair_policy
            else None
        )
        self._pair_policy_v2_r1 = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_pair_policy
            else None
        )
        self._pair_policy_v2_q2 = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_pair_policy
            else None
        )
        self._pair_policy_v2_r2 = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.int32)
            if self.store_pair_policy
            else None
        )
        self._pair_policy_v2_probs = (
            np.zeros((capacity, self.max_policy_v2_entries), dtype=np.float32)
            if self.store_pair_policy
            else None
        )
        self._pair_policy_v2_counts = np.zeros(capacity, dtype=np.uint16)
        self._policy_v2_exact: List[List[tuple[int, int, float]]] = [[] for _ in range(capacity)]
        self._opp_policy_v2_exact: List[List[tuple[int, int, float]]] = [[] for _ in range(capacity)]
        self._opp_policy_legal_v2_exact: List[List[tuple[int, int]]] = [[] for _ in range(capacity)]
        self._pair_policy_v2_exact: List[List[tuple[tuple[int, int], tuple[int, int], float]]] = [
            [] for _ in range(capacity)
        ]
        self._outside_policy_mass = np.zeros(capacity, dtype=np.float32)
        self._missing_policy_mass = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_top1 = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_top4 = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_top8 = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_winning = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_forced_block = np.zeros(capacity, dtype=np.float32)
        self._candidate_recall_cover = np.zeros(capacity, dtype=np.float32)
        self._candidate_discovery_top1 = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_top4 = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_top8 = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_winning = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_forced_block = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_cover = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_open_four = np.ones(capacity, dtype=np.float32)
        self._candidate_discovery_open_five = np.ones(capacity, dtype=np.float32)
        self._candidate_critical_count = np.zeros(capacity, dtype=np.float32)
        self._candidate_critical_overflow_count = np.zeros(capacity, dtype=np.float32)
        self._candidate_critical_overflow_examples = np.zeros((capacity, 8, 2), dtype=np.int32)
        self._candidate_critical_overflow_example_counts = np.zeros(capacity, dtype=np.uint8)
        self._sparse_prior_stage = np.zeros(capacity, dtype=np.uint8)
        self._sparse_prior_root_candidate_count = np.zeros(capacity, dtype=np.float32)
        self._sparse_prior_leaf_candidate_count = np.zeros(capacity, dtype=np.float32)
        self._sparse_prior_root_hit_frac = np.zeros(capacity, dtype=np.float32)
        self._sparse_prior_leaf_hit_frac = np.zeros(capacity, dtype=np.float32)
        self._fallback_prior_use = np.zeros(capacity, dtype=np.float32)
        self._fallback_prior_use_top1 = np.zeros(capacity, dtype=np.float32)
        self._fallback_prior_use_top4 = np.zeros(capacity, dtype=np.float32)
        self._fallback_prior_use_top8 = np.zeros(capacity, dtype=np.float32)
        self._sparse_vs_dense_disagreement = np.zeros(capacity, dtype=np.float32)
        self._sparse_prior_forward_ms = np.zeros(capacity, dtype=np.float32)
        self._sparse_prior_candidate_build_ms = np.zeros(capacity, dtype=np.float32)
        self._pair_prior_candidate_count = np.zeros(capacity, dtype=np.float32)
        self._pair_prior_hit_frac = np.zeros(capacity, dtype=np.float32)
        self._pair_fallback_prior_use = np.zeros(capacity, dtype=np.float32)
        self._pair_fallback_prior_use_top1 = np.zeros(capacity, dtype=np.float32)
        self._pair_fallback_prior_use_top4 = np.zeros(capacity, dtype=np.float32)
        self._pair_fallback_prior_use_top8 = np.zeros(capacity, dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._selected_action_values = np.zeros(capacity, dtype=np.float32)
        self._selected_action_present = np.zeros(capacity, dtype=np.bool_)
        self._value_weights = np.ones(capacity, dtype=np.float32)
        self._regret_rank = np.zeros(capacity, dtype=np.float32)
        self._regret_value = np.zeros(capacity, dtype=np.float32)
        self._regret_weights = np.zeros(capacity, dtype=np.float32)
        self._axis = np.full(capacity, -1, dtype=np.int16)
        self._moves_left = np.zeros(capacity, dtype=np.float32)
        self._opp_policies = (
            np.zeros((capacity, max_policy_entries), dtype=np.uint16)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_probs = (
            np.zeros((capacity, max_policy_entries), dtype=np.float32)
            if self.store_opp_policy
            else None
        )
        self._opp_policy_counts = np.zeros(capacity, dtype=np.uint16)
        self._opp_policy_weights = np.zeros(capacity, dtype=np.float32)
        self._game_ids = np.zeros(capacity, dtype=np.uint32)
        self._is_full = np.zeros(capacity, dtype=np.bool_)
        self._players = np.zeros(capacity, dtype=np.uint8)
        # Per-horizon lookahead value targets — shape (capacity, num_lookahead)
        if num_lookahead > 0:
            self._lookahead = np.zeros((capacity, num_lookahead), dtype=np.float32)
        else:
            self._lookahead = None

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

    def memory_estimate(self) -> dict:
        """Return an allocation estimate for replay telemetry."""
        numpy_bytes = 0
        arrays: dict[str, int] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                arrays[name] = int(value.nbytes)
                numpy_bytes += int(value.nbytes)
        python_list_bytes = (
            self.capacity * 8 * 5
            + len(self._histories) * 8
        )
        return {
            "capacity": int(self.capacity),
            "max_policy_entries": int(self.max_policy_entries),
            "max_policy_v2_entries": int(self.max_policy_v2_entries),
            "feature_groups": {
                "opp_policy": bool(self.store_opp_policy),
                "pair_policy": bool(self.store_pair_policy),
                "sparse_diagnostics": bool(self.store_sparse_diagnostics),
            },
            "allocated_numpy_mib": round(numpy_bytes / (1024.0 * 1024.0), 3),
            "estimated_python_list_mib": round(python_list_bytes / (1024.0 * 1024.0), 3),
            "estimated_total_mib": round((numpy_bytes + python_list_bytes) / (1024.0 * 1024.0), 3),
            "largest_arrays_mib": {
                name: round(size / (1024.0 * 1024.0), 3)
                for name, size in sorted(arrays.items(), key=lambda item: item[1], reverse=True)[:8]
            },
        }

    def append(self, record: PositionRecord):
        """Append one position record. Thread-safe."""
        with self._lock:
            idx = self._head

            self._histories[idx] = record.move_history

            entries = list(record.policy_target.items())
            n = min(len(entries), self.max_policy_entries)
            self._policy_counts[idx] = n
            self._policies[idx].fill(0)
            self._policy_probs[idx].fill(0.0)
            for j, (action_idx, prob) in enumerate(entries[:n]):
                self._policies[idx, j] = action_idx
                self._policy_probs[idx, j] = prob

            self._values[idx] = record.to_value_target()
            self._selected_action_present[idx] = record.selected_action_value is not None
            self._selected_action_values[idx] = (
                0.0 if record.selected_action_value is None else float(record.selected_action_value)
            )
            self._value_weights[idx] = record.value_weight
            self._write_aux_targets(idx, record)
            self._write_v2_targets(idx, record)
            self._game_ids[idx] = record.game_id
            self._is_full[idx] = record.is_full_search
            self._players[idx] = record.player
            if self._lookahead is not None:
                lv = record.lookahead_values
                k = min(len(lv), self.num_lookahead)
                self._lookahead[idx, :k] = lv[:k]
                if k < self.num_lookahead:
                    self._lookahead[idx, k:] = self._values[idx]

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
        self._policies[idx].fill(0)
        self._policy_probs[idx].fill(0.0)
        for j, (action_idx, prob) in enumerate(entries[:n]):
            self._policies[idx, j] = action_idx
            self._policy_probs[idx, j] = prob
        self._values[idx] = record.to_value_target()
        self._selected_action_present[idx] = record.selected_action_value is not None
        self._selected_action_values[idx] = (
            0.0 if record.selected_action_value is None else float(record.selected_action_value)
        )
        self._value_weights[idx] = record.value_weight
        self._write_aux_targets(idx, record)
        self._write_v2_targets(idx, record)
        self._game_ids[idx] = record.game_id
        self._is_full[idx] = record.is_full_search
        self._players[idx] = record.player
        if self._lookahead is not None:
            lv = record.lookahead_values
            k = min(len(lv), self.num_lookahead)
            self._lookahead[idx, :k] = lv[:k]
            if k < self.num_lookahead:
                self._lookahead[idx, k:] = self._values[idx]
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

    def sample_regret_indices(
        self,
        n: int,
        temperature: float = 0.1,
    ) -> np.ndarray:
        """Sample physical indices biased toward high-regret positions."""
        if n <= 0 or self._size == 0:
            return np.array([], dtype=np.int64)

        with self._lock:
            regrets = np.zeros(self._size, dtype=np.float64)
            for i in range(self._size):
                idx = (self._tail + i) % self.capacity
                if self._regret_weights[idx] > 0.0:
                    regrets[i] = max(float(self._regret_rank[idx]), 1e-8)

        inv_temp = 1.0 / max(temperature, 1e-6)
        weights = regrets ** inv_temp
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return self.sample_indices(n)

        probs = weights / total
        logical = np.random.choice(self._size, size=n, p=probs, replace=True)
        physical = (self._tail + logical) % self.capacity
        return physical.astype(np.int64)

    def __getitem__(self, idx: int) -> Optional[PositionRecord]:
        """Retrieve a single position record by physical index. Thread-safe."""
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"Index {idx} out of range [0, {self.capacity})")
        with self._lock:
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

            lv = []
            if self._lookahead is not None:
                lv = self._lookahead[idx].tolist()
            opp_policy = {}
            if self.store_opp_policy and self._opp_policies is not None and self._opp_policy_probs is not None:
                n_opp = int(self._opp_policy_counts[idx])
                for j in range(n_opp):
                    action_idx = int(self._opp_policies[idx, j])
                    prob = float(self._opp_policy_probs[idx, j])
                    if prob > 0:
                        opp_policy[action_idx] = prob
            policy_v2 = list(self._policy_v2_exact[idx])
            opp_policy_v2 = list(self._opp_policy_v2_exact[idx]) if self.store_opp_policy else []
            opp_policy_legal_v2 = list(self._opp_policy_legal_v2_exact[idx]) if self.store_opp_policy else []
            pair_policy_v2 = list(self._pair_policy_v2_exact[idx]) if self.store_pair_policy else []

            return PositionRecord(
                move_history=self._histories[idx],
                policy_target=policy,
                root_value=0.0,
                player=player,
                selected_action_value=(
                    float(self._selected_action_values[idx])
                    if self._selected_action_present[idx]
                    else None
                ),
                game_id=int(self._game_ids[idx]),
                is_full_search=bool(self._is_full[idx]),
                outcome=outcome,
                lookahead_values=lv,
                opp_policy_target=opp_policy,
                opp_policy_weight=float(self._opp_policy_weights[idx]) if self.store_opp_policy else 0.0,
                policy_target_v2=policy_v2,
                opp_policy_target_v2=opp_policy_v2,
                opp_policy_legal_v2=opp_policy_legal_v2,
                pair_policy_target_v2=pair_policy_v2,
                target_policy_mass_outside_window=float(self._outside_policy_mass[idx]),
                missing_target_policy_mass=float(self._missing_policy_mass[idx]),
                candidate_recall_mcts_top1=float(self._candidate_recall_top1[idx]),
                candidate_recall_mcts_top4=float(self._candidate_recall_top4[idx]),
                candidate_recall_mcts_top8=float(self._candidate_recall_top8[idx]),
                candidate_recall_winning_move=float(self._candidate_recall_winning[idx]),
                candidate_recall_forced_block=float(self._candidate_recall_forced_block[idx]),
                candidate_recall_two_placement_cover=float(self._candidate_recall_cover[idx]),
                candidate_discovery_top1=float(self._candidate_discovery_top1[idx]),
                candidate_discovery_top4=float(self._candidate_discovery_top4[idx]),
                candidate_discovery_top8=float(self._candidate_discovery_top8[idx]),
                candidate_discovery_winning_move=float(self._candidate_discovery_winning[idx]),
                candidate_discovery_forced_block=float(self._candidate_discovery_forced_block[idx]),
                candidate_discovery_two_placement_cover=float(self._candidate_discovery_cover[idx]),
                candidate_discovery_open_four=float(self._candidate_discovery_open_four[idx]),
                candidate_discovery_open_five=float(self._candidate_discovery_open_five[idx]),
                candidate_critical_count=int(self._candidate_critical_count[idx]),
                candidate_critical_overflow_count=int(self._candidate_critical_overflow_count[idx]),
                candidate_critical_overflow_examples=tuple(
                    (int(q), int(r))
                    for q, r in self._candidate_critical_overflow_examples[
                        idx, : int(self._candidate_critical_overflow_example_counts[idx])
                    ]
                ),
                sparse_prior_stage=int(self._sparse_prior_stage[idx]),
                sparse_prior_root_candidate_count=int(self._sparse_prior_root_candidate_count[idx]),
                sparse_prior_leaf_candidate_count=float(self._sparse_prior_leaf_candidate_count[idx]),
                sparse_prior_root_hit_frac=float(self._sparse_prior_root_hit_frac[idx]),
                sparse_prior_leaf_hit_frac=float(self._sparse_prior_leaf_hit_frac[idx]),
                fallback_prior_use=float(self._fallback_prior_use[idx]),
                fallback_prior_use_on_mcts_top1=float(self._fallback_prior_use_top1[idx]),
                fallback_prior_use_on_mcts_top4=float(self._fallback_prior_use_top4[idx]),
                fallback_prior_use_on_mcts_top8=float(self._fallback_prior_use_top8[idx]),
                sparse_vs_dense_disagreement=float(self._sparse_vs_dense_disagreement[idx]),
                sparse_prior_forward_ms=float(self._sparse_prior_forward_ms[idx]),
                sparse_prior_candidate_build_ms=float(self._sparse_prior_candidate_build_ms[idx]),
                pair_prior_candidate_count=int(self._pair_prior_candidate_count[idx]),
                pair_prior_hit_frac=float(self._pair_prior_hit_frac[idx]),
                pair_fallback_prior_use=float(self._pair_fallback_prior_use[idx]),
                pair_fallback_prior_use_on_mcts_top1=float(self._pair_fallback_prior_use_top1[idx]),
                pair_fallback_prior_use_on_mcts_top4=float(self._pair_fallback_prior_use_top4[idx]),
                pair_fallback_prior_use_on_mcts_top8=float(self._pair_fallback_prior_use_top8[idx]),
                regret_rank=float(self._regret_rank[idx]),
                regret_value=float(self._regret_value[idx]),
                regret_weight=float(self._regret_weights[idx]),
                axis_label=int(self._axis[idx]),
                moves_left=float(self._moves_left[idx]),
                value_weight=float(self._value_weights[idx]),
            )

    def get_batch(self, indices: np.ndarray) -> List[PositionRecord]:
        """Retrieve multiple records by physical indices."""
        records = []
        for idx in indices:
            rec = self[idx]
            if rec is not None:
                records.append(rec)
        return records

    def records(self) -> List[PositionRecord]:
        """Return all records in oldest-to-newest logical order."""
        with self._lock:
            indices = [(self._tail + i) % self.capacity for i in range(self._size)]
        return [rec for rec in (self[idx] for idx in indices) if rec is not None]

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
                "avg_target_policy_mass_outside_window": float(
                    self._outside_policy_mass[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_missing_target_policy_mass": float(
                    self._missing_policy_mass[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_mcts_top1": float(
                    self._candidate_recall_top1[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_mcts_top4": float(
                    self._candidate_recall_top4[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_mcts_top8": float(
                    self._candidate_recall_top8[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_winning_move": float(
                    self._candidate_recall_winning[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_forced_block": float(
                    self._candidate_recall_forced_block[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "avg_candidate_recall_two_placement_cover": float(
                    self._candidate_recall_cover[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_top1": float(
                    self._candidate_discovery_top1[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_top4": float(
                    self._candidate_discovery_top4[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_top8": float(
                    self._candidate_discovery_top8[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_winning_move": float(
                    self._candidate_discovery_winning[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_forced_block": float(
                    self._candidate_discovery_forced_block[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_two_placement_cover": float(
                    self._candidate_discovery_cover[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_open_four": float(
                    self._candidate_discovery_open_four[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "candidate_discovery_open_five": float(
                    self._candidate_discovery_open_five[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "critical_count": float(
                    self._candidate_critical_count[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "critical_overflow_count": float(
                    self._candidate_critical_overflow_count[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].sum()
                ) if self._size > 0 else 0.0,
                "fallback_prior_use": float(
                    self._fallback_prior_use[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "fallback_prior_use_on_mcts_top1": float(
                    self._fallback_prior_use_top1[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "fallback_prior_use_on_mcts_top4": float(
                    self._fallback_prior_use_top4[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "fallback_prior_use_on_mcts_top8": float(
                    self._fallback_prior_use_top8[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "fallback_prior_use_on_mcts_topk": float(
                    self._fallback_prior_use_top4[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "sparse_prior_root_hit_frac": float(
                    self._sparse_prior_root_hit_frac[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "sparse_prior_leaf_hit_frac": float(
                    self._sparse_prior_leaf_hit_frac[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "sparse_vs_dense_disagreement": float(
                    self._sparse_vs_dense_disagreement[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_fallback_prior_use": float(
                    self._pair_fallback_prior_use[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_prior_hit_frac": float(
                    self._pair_prior_hit_frac[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_prior_candidate_count": float(
                    self._pair_prior_candidate_count[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_fallback_prior_use_on_mcts_top1": float(
                    self._pair_fallback_prior_use_top1[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_fallback_prior_use_on_mcts_top4": float(
                    self._pair_fallback_prior_use_top4[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
                "pair_fallback_prior_use_on_mcts_top8": float(
                    self._pair_fallback_prior_use_top8[
                        [(self._tail + i) % self.capacity for i in range(self._size)]
                    ].mean()
                ) if self._size > 0 else 0.0,
            }

    def clear(self):
        """Reset the buffer to empty."""
        with self._lock:
            self._histories = [None] * self.capacity
            self._policies.fill(0)
            self._policy_probs.fill(0.0)
            self._policy_counts.fill(0)
            self._policy_v2_q.fill(0)
            self._policy_v2_r.fill(0)
            self._policy_v2_probs.fill(0.0)
            self._policy_v2_counts.fill(0)
            if self._opp_policy_v2_q is not None:
                self._opp_policy_v2_q.fill(0)
            if self._opp_policy_v2_r is not None:
                self._opp_policy_v2_r.fill(0)
            if self._opp_policy_v2_probs is not None:
                self._opp_policy_v2_probs.fill(0.0)
            self._opp_policy_v2_counts.fill(0)
            if self._opp_policy_legal_v2_q is not None:
                self._opp_policy_legal_v2_q.fill(0)
            if self._opp_policy_legal_v2_r is not None:
                self._opp_policy_legal_v2_r.fill(0)
            self._opp_policy_legal_v2_counts.fill(0)
            if self._pair_policy_v2_q1 is not None:
                self._pair_policy_v2_q1.fill(0)
            if self._pair_policy_v2_r1 is not None:
                self._pair_policy_v2_r1.fill(0)
            if self._pair_policy_v2_q2 is not None:
                self._pair_policy_v2_q2.fill(0)
            if self._pair_policy_v2_r2 is not None:
                self._pair_policy_v2_r2.fill(0)
            if self._pair_policy_v2_probs is not None:
                self._pair_policy_v2_probs.fill(0.0)
            self._pair_policy_v2_counts.fill(0)
            for store in (
                self._policy_v2_exact,
                self._opp_policy_v2_exact,
                self._opp_policy_legal_v2_exact,
                self._pair_policy_v2_exact,
            ):
                for idx in range(self.capacity):
                    store[idx].clear()
            self._outside_policy_mass.fill(0.0)
            self._missing_policy_mass.fill(0.0)
            self._candidate_recall_top1.fill(0.0)
            self._candidate_recall_top4.fill(0.0)
            self._candidate_recall_top8.fill(0.0)
            self._candidate_recall_winning.fill(0.0)
            self._candidate_recall_forced_block.fill(0.0)
            self._candidate_recall_cover.fill(0.0)
            self._candidate_discovery_top1.fill(1.0)
            self._candidate_discovery_top4.fill(1.0)
            self._candidate_discovery_top8.fill(1.0)
            self._candidate_discovery_winning.fill(1.0)
            self._candidate_discovery_forced_block.fill(1.0)
            self._candidate_discovery_cover.fill(1.0)
            self._candidate_discovery_open_four.fill(1.0)
            self._candidate_discovery_open_five.fill(1.0)
            self._candidate_critical_count.fill(0.0)
            self._candidate_critical_overflow_count.fill(0.0)
            self._candidate_critical_overflow_examples.fill(0)
            self._candidate_critical_overflow_example_counts.fill(0)
            self._sparse_prior_stage.fill(0)
            self._sparse_prior_root_candidate_count.fill(0.0)
            self._sparse_prior_leaf_candidate_count.fill(0.0)
            self._sparse_prior_root_hit_frac.fill(0.0)
            self._sparse_prior_leaf_hit_frac.fill(0.0)
            self._fallback_prior_use.fill(0.0)
            self._fallback_prior_use_top1.fill(0.0)
            self._fallback_prior_use_top4.fill(0.0)
            self._fallback_prior_use_top8.fill(0.0)
            self._sparse_vs_dense_disagreement.fill(0.0)
            self._sparse_prior_forward_ms.fill(0.0)
            self._sparse_prior_candidate_build_ms.fill(0.0)
            self._pair_prior_candidate_count.fill(0.0)
            self._pair_prior_hit_frac.fill(0.0)
            self._pair_fallback_prior_use.fill(0.0)
            self._pair_fallback_prior_use_top1.fill(0.0)
            self._pair_fallback_prior_use_top4.fill(0.0)
            self._pair_fallback_prior_use_top8.fill(0.0)
            self._values.fill(0.0)
            self._selected_action_values.fill(0.0)
            self._selected_action_present.fill(False)
            self._value_weights.fill(1.0)
            self._regret_rank.fill(0.0)
            self._regret_value.fill(0.0)
            self._regret_weights.fill(0.0)
            self._axis.fill(-1)
            self._moves_left.fill(0.0)
            if self._opp_policies is not None:
                self._opp_policies.fill(0)
            if self._opp_policy_probs is not None:
                self._opp_policy_probs.fill(0.0)
            self._opp_policy_counts.fill(0)
            self._opp_policy_weights.fill(0.0)
            self._game_ids.fill(0)
            self._is_full.fill(False)
            self._players.fill(0)
            self._head = 0
            self._tail = 0
            self._size = 0
            self._max_game_id = 0
            if self._lookahead is not None:
                self._lookahead.fill(0.0)

    def _write_aux_targets(self, idx: int, record: PositionRecord):
        """Write optional auxiliary targets into struct-of-arrays storage."""
        self._regret_rank[idx] = record.regret_rank
        self._regret_value[idx] = record.regret_value
        self._regret_weights[idx] = record.regret_weight
        self._axis[idx] = record.axis_label
        self._moves_left[idx] = record.moves_left
        self._opp_policy_weights[idx] = record.opp_policy_weight if self.store_opp_policy else 0.0

        if not self.store_opp_policy or self._opp_policies is None or self._opp_policy_probs is None:
            self._opp_policy_counts[idx] = 0
            return
        opp_entries = list(record.opp_policy_target.items())
        n_opp = min(len(opp_entries), self.max_policy_entries)
        self._opp_policy_counts[idx] = n_opp
        self._opp_policies[idx].fill(0)
        self._opp_policy_probs[idx].fill(0.0)
        for j, (action_idx, prob) in enumerate(opp_entries[:n_opp]):
            self._opp_policies[idx, j] = action_idx
            self._opp_policy_probs[idx, j] = prob

    def _write_v2_targets(self, idx: int, record: PositionRecord):
        """Write action-keyed global policy targets and diagnostics."""
        entries = list(record.policy_target_v2)
        self._policy_v2_exact[idx] = [
            (int(q), int(r), float(prob))
            for q, r, prob in entries
            if float(prob) > 0.0
        ]
        n = min(len(entries), self.max_policy_v2_entries)
        self._policy_v2_counts[idx] = n
        self._policy_v2_q[idx].fill(0)
        self._policy_v2_r[idx].fill(0)
        self._policy_v2_probs[idx].fill(0.0)
        for j, (q, r, prob) in enumerate(entries[:n]):
            self._policy_v2_q[idx, j] = int(q)
            self._policy_v2_r[idx, j] = int(r)
            self._policy_v2_probs[idx, j] = float(prob)

        if self.store_opp_policy:
            opp_entries = list(record.opp_policy_target_v2)
            self._opp_policy_v2_exact[idx] = [
                (int(q), int(r), float(prob))
                for q, r, prob in opp_entries
                if float(prob) > 0.0
            ]
            n_opp = min(len(opp_entries), self.max_policy_v2_entries)
            self._opp_policy_v2_counts[idx] = n_opp
            if self._opp_policy_v2_q is not None:
                self._opp_policy_v2_q[idx].fill(0)
            if self._opp_policy_v2_r is not None:
                self._opp_policy_v2_r[idx].fill(0)
            if self._opp_policy_v2_probs is not None:
                self._opp_policy_v2_probs[idx].fill(0.0)
            for j, (q, r, prob) in enumerate(opp_entries[:n_opp]):
                if self._opp_policy_v2_q is not None:
                    self._opp_policy_v2_q[idx, j] = int(q)
                if self._opp_policy_v2_r is not None:
                    self._opp_policy_v2_r[idx, j] = int(r)
                if self._opp_policy_v2_probs is not None:
                    self._opp_policy_v2_probs[idx, j] = float(prob)

            opp_legal_entries = list(record.opp_policy_legal_v2)
            self._opp_policy_legal_v2_exact[idx] = [
                (int(q), int(r))
                for q, r in opp_legal_entries
            ]
            n_opp_legal = min(len(opp_legal_entries), self.max_policy_v2_entries)
            self._opp_policy_legal_v2_counts[idx] = n_opp_legal
            if self._opp_policy_legal_v2_q is not None:
                self._opp_policy_legal_v2_q[idx].fill(0)
            if self._opp_policy_legal_v2_r is not None:
                self._opp_policy_legal_v2_r[idx].fill(0)
            for j, (q, r) in enumerate(opp_legal_entries[:n_opp_legal]):
                if self._opp_policy_legal_v2_q is not None:
                    self._opp_policy_legal_v2_q[idx, j] = int(q)
                if self._opp_policy_legal_v2_r is not None:
                    self._opp_policy_legal_v2_r[idx, j] = int(r)
        else:
            self._opp_policy_v2_exact[idx] = []
            self._opp_policy_legal_v2_exact[idx] = []
            self._opp_policy_v2_counts[idx] = 0
            self._opp_policy_legal_v2_counts[idx] = 0

        if self.store_pair_policy:
            pair_entries = list(record.pair_policy_target_v2)
            self._pair_policy_v2_exact[idx] = [
                ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])), float(prob))
                for first, second, prob in pair_entries
                if float(prob) > 0.0
            ]
            n_pair = min(len(pair_entries), self.max_policy_v2_entries)
            self._pair_policy_v2_counts[idx] = n_pair
            if self._pair_policy_v2_q1 is not None:
                self._pair_policy_v2_q1[idx].fill(0)
            if self._pair_policy_v2_r1 is not None:
                self._pair_policy_v2_r1[idx].fill(0)
            if self._pair_policy_v2_q2 is not None:
                self._pair_policy_v2_q2[idx].fill(0)
            if self._pair_policy_v2_r2 is not None:
                self._pair_policy_v2_r2[idx].fill(0)
            if self._pair_policy_v2_probs is not None:
                self._pair_policy_v2_probs[idx].fill(0.0)
            for j, (first, second, prob) in enumerate(pair_entries[:n_pair]):
                q1, r1 = first
                q2, r2 = second
                if self._pair_policy_v2_q1 is not None:
                    self._pair_policy_v2_q1[idx, j] = int(q1)
                if self._pair_policy_v2_r1 is not None:
                    self._pair_policy_v2_r1[idx, j] = int(r1)
                if self._pair_policy_v2_q2 is not None:
                    self._pair_policy_v2_q2[idx, j] = int(q2)
                if self._pair_policy_v2_r2 is not None:
                    self._pair_policy_v2_r2[idx, j] = int(r2)
                if self._pair_policy_v2_probs is not None:
                    self._pair_policy_v2_probs[idx, j] = float(prob)
        else:
            self._pair_policy_v2_exact[idx] = []
            self._pair_policy_v2_counts[idx] = 0

        dropped_mass = sum(float(prob) for _q, _r, prob in entries[n:])
        self._outside_policy_mass[idx] = float(record.target_policy_mass_outside_window)
        self._missing_policy_mass[idx] = float(record.missing_target_policy_mass) + float(dropped_mass)
        self._candidate_recall_top1[idx] = float(record.candidate_recall_mcts_top1)
        self._candidate_recall_top4[idx] = float(record.candidate_recall_mcts_top4)
        self._candidate_recall_top8[idx] = float(record.candidate_recall_mcts_top8)
        self._candidate_recall_winning[idx] = float(record.candidate_recall_winning_move)
        self._candidate_recall_forced_block[idx] = float(record.candidate_recall_forced_block)
        self._candidate_recall_cover[idx] = float(record.candidate_recall_two_placement_cover)
        self._candidate_discovery_top1[idx] = float(record.candidate_discovery_top1)
        self._candidate_discovery_top4[idx] = float(record.candidate_discovery_top4)
        self._candidate_discovery_top8[idx] = float(record.candidate_discovery_top8)
        self._candidate_discovery_winning[idx] = float(record.candidate_discovery_winning_move)
        self._candidate_discovery_forced_block[idx] = float(record.candidate_discovery_forced_block)
        self._candidate_discovery_cover[idx] = float(record.candidate_discovery_two_placement_cover)
        self._candidate_discovery_open_four[idx] = float(record.candidate_discovery_open_four)
        self._candidate_discovery_open_five[idx] = float(record.candidate_discovery_open_five)
        self._candidate_critical_count[idx] = float(record.candidate_critical_count)
        self._candidate_critical_overflow_count[idx] = float(record.candidate_critical_overflow_count)
        self._candidate_critical_overflow_examples[idx].fill(0)
        example_count = min(len(record.candidate_critical_overflow_examples), 8)
        self._candidate_critical_overflow_example_counts[idx] = example_count
        for row, (q, r) in enumerate(record.candidate_critical_overflow_examples[:example_count]):
            self._candidate_critical_overflow_examples[idx, row] = (int(q), int(r))
        self._sparse_prior_stage[idx] = int(record.sparse_prior_stage)
        self._sparse_prior_root_candidate_count[idx] = float(record.sparse_prior_root_candidate_count)
        self._sparse_prior_leaf_candidate_count[idx] = float(record.sparse_prior_leaf_candidate_count)
        self._sparse_prior_root_hit_frac[idx] = float(record.sparse_prior_root_hit_frac)
        self._sparse_prior_leaf_hit_frac[idx] = float(record.sparse_prior_leaf_hit_frac)
        self._fallback_prior_use[idx] = float(record.fallback_prior_use)
        self._fallback_prior_use_top1[idx] = float(record.fallback_prior_use_on_mcts_top1)
        self._fallback_prior_use_top4[idx] = float(record.fallback_prior_use_on_mcts_top4)
        self._fallback_prior_use_top8[idx] = float(record.fallback_prior_use_on_mcts_top8)
        self._sparse_vs_dense_disagreement[idx] = float(record.sparse_vs_dense_disagreement)
        self._sparse_prior_forward_ms[idx] = float(record.sparse_prior_forward_ms)
        self._sparse_prior_candidate_build_ms[idx] = float(record.sparse_prior_candidate_build_ms)
        self._pair_prior_candidate_count[idx] = float(record.pair_prior_candidate_count)
        self._pair_prior_hit_frac[idx] = float(record.pair_prior_hit_frac)
        self._pair_fallback_prior_use[idx] = float(record.pair_fallback_prior_use)
        self._pair_fallback_prior_use_top1[idx] = float(record.pair_fallback_prior_use_on_mcts_top1)
        self._pair_fallback_prior_use_top4[idx] = float(record.pair_fallback_prior_use_on_mcts_top4)
        self._pair_fallback_prior_use_top8[idx] = float(record.pair_fallback_prior_use_on_mcts_top8)
