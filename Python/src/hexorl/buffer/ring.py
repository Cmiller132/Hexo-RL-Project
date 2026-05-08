"""Compact CPU-decoded replay buffer.

The trainer consumes :class:`PositionRecord` objects and already rebuilds board
tensors and global graph batches from compact move history. This buffer keeps
that public contract while storing replay rows in compact arrays/blobs instead
of wide dense target tables.
"""

from __future__ import annotations

import threading
from typing import Iterable, List, Optional

import numpy as np

from hexorl.selfplay.records import (
    PositionRecord,
    v1_search_metadata_from_json_bytes,
    v1_search_metadata_to_json_bytes,
)
from hexorl.models.registry import replay_uses_sparse_diagnostics


HISTORY_STRIDE = 12
REPLAY_POLICY_WIDTH_CAP = 512
PAIR_POLICY_HEADS = {"policy_pair_first", "policy_pair_second", "policy_pair_joint"}

_POLICY_BLOB_DTYPE = np.dtype([("action", "<u2"), ("prob", "<f2")])
_V2_BLOB_DTYPE = np.dtype([("q", "<i2"), ("r", "<i2"), ("prob", "<f2")])
_LEGAL_BLOB_DTYPE = np.dtype([("q", "<i2"), ("r", "<i2")])
_PAIR_BLOB_DTYPE = np.dtype(
    [("q1", "<i2"), ("r1", "<i2"), ("q2", "<i2"), ("r2", "<i2"), ("prob", "<f2")]
)


class _DefaultArray:
    """Zero-allocation array facade used when optional diagnostics are disabled."""

    nbytes = 0

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype | type, default: float | int = 0):
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = np.dtype(dtype)
        self.default = default

    def _view(self) -> np.ndarray:
        scalar = np.asarray(self.default, dtype=self.dtype)
        return np.broadcast_to(scalar, self.shape)

    def __getitem__(self, key):
        value = self._view()[key]
        return value.copy() if isinstance(value, np.ndarray) else value

    def __setitem__(self, key, value) -> None:
        return None

    def fill(self, value) -> None:
        return None


def _optional_array(
    enabled: bool,
    shape: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    default: float | int = 0,
) -> np.ndarray | _DefaultArray:
    if enabled:
        return np.full(shape, default, dtype=dtype)
    return _DefaultArray(shape, dtype, default)


def replay_feature_flags(
    heads: Iterable[str],
    *,
    architecture: str = "cnn",
    sparse_policy: bool = False,
    graph: bool = False,
) -> dict[str, bool]:
    """Return compact replay feature groups needed by a model/trial."""
    head_set = {str(head) for head in heads}
    store_pair = bool(head_set & PAIR_POLICY_HEADS)
    sparse_diagnostics = replay_uses_sparse_diagnostics(
        head_set,
        architecture=architecture,
        sparse_policy=sparse_policy,
        graph=graph or store_pair,
    )
    return {
        "store_opp_policy": "opp_policy" in head_set,
        "store_pair_policy": store_pair,
        "store_sparse_diagnostics": sparse_diagnostics,
    }


def _as_int_game_id(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _history_prefix_plies(history: bytes) -> int:
    if len(history) % HISTORY_STRIDE != 0:
        raise ValueError("move history length must be a multiple of 12 bytes")
    return len(history) // HISTORY_STRIDE


def _compatible_history(a: bytes, b: bytes) -> bool:
    if len(a) <= len(b):
        return b.startswith(a)
    return a.startswith(b)


def _normalize_policy(policy: dict[int, float]) -> dict[int, float]:
    total = float(sum(policy.values()))
    if total <= 0.0:
        return policy
    return {action: prob / total for action, prob in policy.items() if prob > 0.0}


def _pack_policy_blob(entries: Iterable[tuple[int, float]]) -> bytes:
    rows = [(int(action), float(prob)) for action, prob in entries if float(prob) > 0.0]
    if not rows:
        return b""
    arr = np.zeros(len(rows), dtype=_POLICY_BLOB_DTYPE)
    arr["action"] = [max(0, min(65535, action)) for action, _prob in rows]
    arr["prob"] = [prob for _action, prob in rows]
    return arr.tobytes()


def _unpack_policy_blob(blob: bytes | None) -> dict[int, float]:
    if not blob:
        return {}
    arr = np.frombuffer(blob, dtype=_POLICY_BLOB_DTYPE)
    policy = {
        int(row["action"]): float(row["prob"])
        for row in arr
        if float(row["prob"]) > 0.0
    }
    return _normalize_policy(policy)


def _pack_v2_blob(entries: Iterable[tuple[int, int, float]]) -> bytes:
    rows = [(int(q), int(r), float(prob)) for q, r, prob in entries if float(prob) > 0.0]
    if not rows:
        return b""
    arr = np.zeros(len(rows), dtype=_V2_BLOB_DTYPE)
    arr["q"] = [q for q, _r, _prob in rows]
    arr["r"] = [r for _q, r, _prob in rows]
    arr["prob"] = [prob for _q, _r, prob in rows]
    return arr.tobytes()


def _unpack_v2_blob(blob: bytes | None) -> list[tuple[int, int, float]]:
    if not blob:
        return []
    arr = np.frombuffer(blob, dtype=_V2_BLOB_DTYPE)
    return [
        (int(row["q"]), int(row["r"]), float(row["prob"]))
        for row in arr
        if float(row["prob"]) > 0.0
    ]


def _pack_legal_blob(entries: Iterable[tuple[int, int]]) -> bytes:
    rows = [(int(q), int(r)) for q, r in entries]
    if not rows:
        return b""
    arr = np.zeros(len(rows), dtype=_LEGAL_BLOB_DTYPE)
    arr["q"] = [q for q, _r in rows]
    arr["r"] = [r for _q, r in rows]
    return arr.tobytes()


def _unpack_legal_blob(blob: bytes | None) -> list[tuple[int, int]]:
    if not blob:
        return []
    arr = np.frombuffer(blob, dtype=_LEGAL_BLOB_DTYPE)
    return [(int(row["q"]), int(row["r"])) for row in arr]


def _pack_pair_blob(entries: Iterable[tuple[tuple[int, int], tuple[int, int], float]]) -> bytes:
    rows = [
        ((int(first[0]), int(first[1])), (int(second[0]), int(second[1])), float(prob))
        for first, second, prob in entries
        if float(prob) > 0.0
    ]
    if not rows:
        return b""
    arr = np.zeros(len(rows), dtype=_PAIR_BLOB_DTYPE)
    arr["q1"] = [first[0] for first, _second, _prob in rows]
    arr["r1"] = [first[1] for first, _second, _prob in rows]
    arr["q2"] = [second[0] for _first, second, _prob in rows]
    arr["r2"] = [second[1] for _first, second, _prob in rows]
    arr["prob"] = [prob for _first, _second, prob in rows]
    return arr.tobytes()


def _unpack_pair_blob(blob: bytes | None) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    if not blob:
        return []
    arr = np.frombuffer(blob, dtype=_PAIR_BLOB_DTYPE)
    return [
        (
            (int(row["q1"]), int(row["r1"])),
            (int(row["q2"]), int(row["r2"])),
            float(row["prob"]),
        )
        for row in arr
        if float(row["prob"]) > 0.0
    ]


class RingBuffer:
    """Fixed-capacity compact replay buffer.

    The class name is retained because callers import ``RingBuffer`` directly,
    but the storage is compact and CPU-decoded. Histories are shared by game
    where possible and target rows are stored as small fixed arrays or binary
    blobs instead of Python tuple lists.
    """

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
        if max_policy_v2_entries is not None and int(max_policy_v2_entries) <= 0:
            raise ValueError("RingBuffer max_policy_v2_entries must be positive")

        self.capacity = int(capacity)
        self.max_policy_entries = int(max_policy_entries)
        requested_v2 = int(max_policy_v2_entries or max_policy_entries)
        self.max_policy_v2_entries = min(max(1, requested_v2), REPLAY_POLICY_WIDTH_CAP)
        self.recency_decay = float(recency_decay)
        self.num_lookahead = int(num_lookahead)
        self.store_opp_policy = bool(store_opp_policy)
        self.store_pair_policy = bool(store_pair_policy)
        self.store_sparse_diagnostics = bool(store_sparse_diagnostics)

        self._game_slots = np.full(self.capacity, -1, dtype=np.int32)
        self._prefix_plies = np.zeros(self.capacity, dtype=np.uint16)
        self._policy_actions = np.zeros((self.capacity, self.max_policy_entries), dtype=np.uint16)
        self._policy_probs = np.zeros((self.capacity, self.max_policy_entries), dtype=np.float16)
        self._policy_counts = np.zeros(self.capacity, dtype=np.uint16)
        self._policy_v2_q = np.zeros((self.capacity, self.max_policy_v2_entries), dtype=np.int16)
        self._policy_v2_r = np.zeros((self.capacity, self.max_policy_v2_entries), dtype=np.int16)
        self._policy_v2_probs = np.zeros((self.capacity, self.max_policy_v2_entries), dtype=np.float16)
        self._policy_v2_counts = np.zeros(self.capacity, dtype=np.uint16)

        self._outside_policy_mass = np.zeros(self.capacity, dtype=np.float32)
        self._missing_policy_mass = np.zeros(self.capacity, dtype=np.float32)
        diag = self.store_sparse_diagnostics
        self._candidate_recall_top1 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_recall_top4 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_recall_top8 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_recall_winning = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_recall_forced_block = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_recall_cover = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_top1 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_top4 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_top8 = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_winning = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_forced_block = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_cover = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_open_four = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_discovery_open_five = _optional_array(diag, (self.capacity,), np.float32, default=1.0)
        self._candidate_critical_count = _optional_array(diag, (self.capacity,), np.float32)
        self._candidate_critical_overflow_count = _optional_array(diag, (self.capacity,), np.float32)
        self._candidate_critical_overflow_examples = _optional_array(diag, (self.capacity, 8, 2), np.int16)
        self._candidate_critical_overflow_example_counts = _optional_array(diag, (self.capacity,), np.uint8)
        self._sparse_prior_stage = _optional_array(diag, (self.capacity,), np.uint8)
        self._sparse_prior_root_candidate_count = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_prior_leaf_candidate_count = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_prior_root_hit_frac = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_prior_leaf_hit_frac = _optional_array(diag, (self.capacity,), np.float32)
        self._fallback_prior_use = _optional_array(diag, (self.capacity,), np.float32)
        self._fallback_prior_use_top1 = _optional_array(diag, (self.capacity,), np.float32)
        self._fallback_prior_use_top4 = _optional_array(diag, (self.capacity,), np.float32)
        self._fallback_prior_use_top8 = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_vs_dense_disagreement = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_prior_forward_ms = _optional_array(diag, (self.capacity,), np.float32)
        self._sparse_prior_candidate_build_ms = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_prior_candidate_count = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_prior_hit_frac = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_fallback_prior_use = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_fallback_prior_use_top1 = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_fallback_prior_use_top4 = _optional_array(diag, (self.capacity,), np.float32)
        self._pair_fallback_prior_use_top8 = _optional_array(diag, (self.capacity,), np.float32)
        self._values = np.zeros(self.capacity, dtype=np.float32)
        self._selected_action_values = np.zeros(self.capacity, dtype=np.float32)
        self._selected_action_present = np.zeros(self.capacity, dtype=np.bool_)
        self._value_weights = np.ones(self.capacity, dtype=np.float32)
        self._regret_rank = np.zeros(self.capacity, dtype=np.float32)
        self._regret_value = np.zeros(self.capacity, dtype=np.float32)
        self._regret_weights = np.zeros(self.capacity, dtype=np.float32)
        self._axis = np.full(self.capacity, -1, dtype=np.int16)
        self._moves_left = np.zeros(self.capacity, dtype=np.float32)
        self._opp_policy_weights = np.zeros(self.capacity, dtype=np.float32)
        self._game_ids = np.zeros(self.capacity, dtype=np.uint32)
        self._is_full = np.zeros(self.capacity, dtype=np.bool_)
        self._players = np.zeros(self.capacity, dtype=np.uint8)
        self._lookahead = (
            np.zeros((self.capacity, self.num_lookahead), dtype=np.float32)
            if self.num_lookahead > 0
            else None
        )
        self._lookahead_counts = (
            np.zeros(self.capacity, dtype=np.uint16)
            if self.num_lookahead > 0
            else None
        )

        self._opp_policy_blobs: list[bytes | None] | None = [None] * self.capacity if self.store_opp_policy else None
        self._opp_policy_v2_blobs: list[bytes | None] | None = [None] * self.capacity if self.store_opp_policy else None
        self._opp_legal_v2_blobs: list[bytes | None] | None = [None] * self.capacity if self.store_opp_policy else None
        self._pair_policy_v2_blobs: list[bytes | None] | None = [None] * self.capacity if self.store_pair_policy else None
        self._v1_search_metadata_blobs: list[bytes | None] = [None] * self.capacity
        self._pair_policy_complete = _optional_array(self.store_pair_policy, (self.capacity,), np.bool_, default=False)

        self._game_histories: list[bytes | None] = []
        self._game_refcounts: list[int] = []
        self._game_ids_by_slot: list[int] = []
        self._game_slots_by_id: dict[int, list[int]] = {}
        self._free_game_slots: list[int] = []

        self._head = 0
        self._tail = 0
        self._size = 0
        self._max_game_id = 0
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
        arrays: dict[str, int] = {}
        numpy_bytes = 0
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                arrays[name] = int(value.nbytes)
                numpy_bytes += int(value.nbytes)

        optional_blob_bytes = 0
        optional_blob_refs = 0
        for blobs in (
            self._opp_policy_blobs,
            self._opp_policy_v2_blobs,
            self._opp_legal_v2_blobs,
            self._pair_policy_v2_blobs,
            self._v1_search_metadata_blobs,
        ):
            if blobs is None:
                continue
            optional_blob_refs += len(blobs) * 8
            optional_blob_bytes += sum(len(blob) for blob in blobs if blob)

        history_bytes = sum(len(history) for history in self._game_histories if history)
        history_refs = len(self._game_histories) * 8
        total = numpy_bytes + optional_blob_bytes + optional_blob_refs + history_bytes + history_refs

        return {
            "capacity": int(self.capacity),
            "max_policy_entries": int(self.max_policy_entries),
            "max_policy_v2_entries": int(self.max_policy_v2_entries),
            "policy_width_cap": int(REPLAY_POLICY_WIDTH_CAP),
            "feature_groups": {
                "opp_policy": bool(self.store_opp_policy),
                "pair_policy": bool(self.store_pair_policy),
                "sparse_diagnostics": bool(self.store_sparse_diagnostics),
            },
            "allocated_numpy_mib": round(numpy_bytes / (1024.0 * 1024.0), 3),
            "history_mib": round((history_bytes + history_refs) / (1024.0 * 1024.0), 3),
            "optional_target_blob_mib": round((optional_blob_bytes + optional_blob_refs) / (1024.0 * 1024.0), 3),
            "estimated_total_mib": round(total / (1024.0 * 1024.0), 3),
            "active_game_history_slots": int(sum(1 for item in self._game_histories if item is not None)),
            "largest_arrays_mib": {
                name: round(size / (1024.0 * 1024.0), 3)
                for name, size in sorted(arrays.items(), key=lambda item: item[1], reverse=True)[:8]
            },
        }

    def append(self, record: PositionRecord):
        with self._lock:
            self._append_unlocked(record)

    def extend(self, records: List[PositionRecord]):
        with self._lock:
            for record in records:
                self._append_unlocked(record)

    def _append_unlocked(self, record: PositionRecord):
        idx = self._head
        self._release_row(idx)

        history = bytes(record.move_history or b"")
        prefix_plies = _history_prefix_plies(history)
        game_id = _as_int_game_id(record.game_id)
        game_slot = self._find_or_create_game_slot(game_id, history)
        self._game_refcounts[game_slot] += 1

        self._game_slots[idx] = game_slot
        self._prefix_plies[idx] = min(prefix_plies, np.iinfo(np.uint16).max)
        self._write_policy(idx, record)
        self._write_aux_targets(idx, record)
        self._write_v2_targets(idx, record)

        self._values[idx] = record.to_value_target()
        self._selected_action_present[idx] = record.selected_action_value is not None
        self._selected_action_values[idx] = (
            0.0 if record.selected_action_value is None else float(record.selected_action_value)
        )
        self._value_weights[idx] = float(record.value_weight)
        self._game_ids[idx] = max(0, min(game_id, np.iinfo(np.uint32).max))
        self._is_full[idx] = bool(record.is_full_search)
        self._players[idx] = int(record.player) & 0xFF
        if self._lookahead is not None:
            values = list(record.lookahead_values)
            k = min(len(values), self.num_lookahead)
            self._lookahead_counts[idx] = k
            self._lookahead[idx].fill(0.0)
            if k:
                self._lookahead[idx, :k] = values[:k]

        self._head = (self._head + 1) % self.capacity
        if self._size == self.capacity:
            self._tail = (self._tail + 1) % self.capacity
        else:
            self._size += 1
        self._max_game_id = max(self._max_game_id, game_id)

    def _release_row(self, idx: int):
        slot = int(self._game_slots[idx])
        if slot >= 0 and slot < len(self._game_refcounts):
            self._game_refcounts[slot] = max(0, self._game_refcounts[slot] - 1)
            if self._game_refcounts[slot] == 0:
                game_id = self._game_ids_by_slot[slot]
                slots = self._game_slots_by_id.get(game_id, [])
                if slot in slots:
                    slots.remove(slot)
                if slots:
                    self._game_slots_by_id[game_id] = slots
                else:
                    self._game_slots_by_id.pop(game_id, None)
                self._game_histories[slot] = None
                self._free_game_slots.append(slot)
        self._game_slots[idx] = -1
        self._prefix_plies[idx] = 0
        if self._lookahead_counts is not None:
            self._lookahead_counts[idx] = 0
        if self._opp_policy_blobs is not None:
            self._opp_policy_blobs[idx] = None
            self._opp_policy_v2_blobs[idx] = None
            self._opp_legal_v2_blobs[idx] = None
        if self._pair_policy_v2_blobs is not None:
            self._pair_policy_v2_blobs[idx] = None
        self._v1_search_metadata_blobs[idx] = None
        self._pair_policy_complete[idx] = False

    def _find_or_create_game_slot(self, game_id: int, history: bytes) -> int:
        for slot in list(self._game_slots_by_id.get(game_id, [])):
            stream = self._game_histories[slot]
            if stream is None:
                continue
            if _compatible_history(stream, history):
                if len(history) > len(stream):
                    self._game_histories[slot] = history
                return slot

        if self._free_game_slots:
            slot = self._free_game_slots.pop()
            self._game_histories[slot] = history
            self._game_refcounts[slot] = 0
            self._game_ids_by_slot[slot] = game_id
        else:
            slot = len(self._game_histories)
            self._game_histories.append(history)
            self._game_refcounts.append(0)
            self._game_ids_by_slot.append(game_id)
        self._game_slots_by_id.setdefault(game_id, []).append(slot)
        return slot

    def _write_policy(self, idx: int, record: PositionRecord):
        entries = [(int(action), float(prob)) for action, prob in record.policy_target.items() if float(prob) > 0.0]
        n = min(len(entries), self.max_policy_entries)
        self._policy_counts[idx] = n
        self._policy_actions[idx].fill(0)
        self._policy_probs[idx].fill(0.0)
        for row, (action, prob) in enumerate(entries[:n]):
            self._policy_actions[idx, row] = max(0, min(65535, action))
            self._policy_probs[idx, row] = prob

    def _write_aux_targets(self, idx: int, record: PositionRecord):
        self._regret_rank[idx] = float(record.regret_rank)
        self._regret_value[idx] = float(record.regret_value)
        self._regret_weights[idx] = float(record.regret_weight)
        self._axis[idx] = int(record.axis_label)
        self._moves_left[idx] = float(record.moves_left)
        self._opp_policy_weights[idx] = float(record.opp_policy_weight) if self.store_opp_policy else 0.0
        if self._opp_policy_blobs is not None:
            self._opp_policy_blobs[idx] = _pack_policy_blob(record.opp_policy_target.items())

    def _write_v2_targets(self, idx: int, record: PositionRecord):
        entries = [(int(q), int(r), float(prob)) for q, r, prob in record.policy_target_v2 if float(prob) > 0.0]
        n = min(len(entries), self.max_policy_v2_entries)
        self._policy_v2_counts[idx] = n
        self._policy_v2_q[idx].fill(0)
        self._policy_v2_r[idx].fill(0)
        self._policy_v2_probs[idx].fill(0.0)
        for row, (q, r, prob) in enumerate(entries[:n]):
            self._policy_v2_q[idx, row] = int(q)
            self._policy_v2_r[idx, row] = int(r)
            self._policy_v2_probs[idx, row] = float(prob)

        if self._opp_policy_v2_blobs is not None:
            self._opp_policy_v2_blobs[idx] = _pack_v2_blob(record.opp_policy_target_v2)
            self._opp_legal_v2_blobs[idx] = _pack_legal_blob(record.opp_policy_legal_v2)
        if self._pair_policy_v2_blobs is not None:
            if record.v1_search_metadata is None:
                self._pair_policy_v2_blobs[idx] = _pack_pair_blob(record.pair_policy_target_v2)
                self._pair_policy_complete[idx] = bool(record.pair_policy_complete)
            else:
                self._pair_policy_v2_blobs[idx] = b""
                self._pair_policy_complete[idx] = False
        else:
            self._pair_policy_complete[idx] = False
        self._v1_search_metadata_blobs[idx] = v1_search_metadata_to_json_bytes(record.v1_search_metadata)

        dropped_mass = sum(prob for _q, _r, prob in entries[n:])
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

    def sample_indices(
        self,
        n: int,
        recency_decay: Optional[float] = None,
        pcr_weight: float = 0.25,
    ) -> np.ndarray:
        if n <= 0 or self._size == 0:
            return np.array([], dtype=np.int64)
        decay = float(recency_decay if recency_decay is not None else self.recency_decay)
        with self._lock:
            indices = self._active_indices_unlocked()
            game_age = np.maximum(self._max_game_id - self._game_ids[indices].astype(np.int64), 0)
            recency_w = np.power(decay, game_age, dtype=np.float64)
            quality_w = np.where(self._is_full[indices], 4.0, float(pcr_weight))
            weights = recency_w * quality_w
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return np.array([], dtype=np.int64)
        logical = np.random.choice(len(indices), size=int(n), p=weights / total, replace=True)
        return indices[logical].astype(np.int64)

    def sample_regret_indices(self, n: int, temperature: float = 0.1) -> np.ndarray:
        if n <= 0 or self._size == 0:
            return np.array([], dtype=np.int64)
        with self._lock:
            indices = self._active_indices_unlocked()
            weights = np.zeros(len(indices), dtype=np.float64)
            active = self._regret_weights[indices] > 0.0
            weights[active] = np.maximum(self._regret_rank[indices][active].astype(np.float64), 1e-8)
        inv_temp = 1.0 / max(float(temperature), 1e-6)
        weights = weights ** inv_temp
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return self.sample_indices(n)
        logical = np.random.choice(len(indices), size=int(n), p=weights / total, replace=True)
        return indices[logical].astype(np.int64)

    def __getitem__(self, idx: int) -> Optional[PositionRecord]:
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"Index {idx} out of range [0, {self.capacity})")
        with self._lock:
            if int(self._game_slots[idx]) < 0:
                return None
            return self._record_unlocked(idx)

    def _record_unlocked(self, idx: int) -> PositionRecord:
        game_slot = int(self._game_slots[idx])
        history = self._game_histories[game_slot] or b""
        prefix_bytes = int(self._prefix_plies[idx]) * HISTORY_STRIDE
        move_history = history[:prefix_bytes]

        policy = {}
        for row in range(int(self._policy_counts[idx])):
            prob = float(self._policy_probs[idx, row])
            if prob > 0.0:
                policy[int(self._policy_actions[idx, row])] = prob
        policy = _normalize_policy(policy)

        policy_v2 = [
            (int(self._policy_v2_q[idx, row]), int(self._policy_v2_r[idx, row]), float(self._policy_v2_probs[idx, row]))
            for row in range(int(self._policy_v2_counts[idx]))
            if float(self._policy_v2_probs[idx, row]) > 0.0
        ]
        if policy_v2 and float(self._missing_policy_mass[idx]) <= 1e-6:
            total_v2 = float(sum(prob for _q, _r, prob in policy_v2))
            if 0.99 <= total_v2 <= 1.01:
                policy_v2 = [(q, r, prob / total_v2) for q, r, prob in policy_v2]
        opp_policy = _unpack_policy_blob(self._opp_policy_blobs[idx] if self._opp_policy_blobs is not None else None)
        opp_policy_v2 = _unpack_v2_blob(
            self._opp_policy_v2_blobs[idx] if self._opp_policy_v2_blobs is not None else None
        )
        opp_legal_v2 = _unpack_legal_blob(
            self._opp_legal_v2_blobs[idx] if self._opp_legal_v2_blobs is not None else None
        )
        pair_policy_v2 = _unpack_pair_blob(
            self._pair_policy_v2_blobs[idx] if self._pair_policy_v2_blobs is not None else None
        )
        v1_search_metadata = v1_search_metadata_from_json_bytes(self._v1_search_metadata_blobs[idx])

        stored_value = float(self._values[idx])
        player = int(self._players[idx])
        outcome = stored_value if player == 0 else -stored_value
        if self._lookahead is not None and self._lookahead_counts is not None:
            lookahead_count = min(int(self._lookahead_counts[idx]), self.num_lookahead)
            lookahead = self._lookahead[idx, :lookahead_count].tolist()
        else:
            lookahead = []

        return PositionRecord(
            move_history=move_history,
            policy_target=policy,
            root_value=0.0,
            player=player,
            selected_action_value=(
                float(self._selected_action_values[idx])
                if bool(self._selected_action_present[idx])
                else None
            ),
            game_id=int(self._game_ids[idx]),
            is_full_search=bool(self._is_full[idx]),
            outcome=outcome,
            lookahead_values=lookahead,
            opp_policy_target=opp_policy,
            opp_policy_weight=float(self._opp_policy_weights[idx]) if self.store_opp_policy else 0.0,
            policy_target_v2=policy_v2,
            opp_policy_target_v2=opp_policy_v2,
            opp_policy_legal_v2=opp_legal_v2,
            pair_policy_target_v2=pair_policy_v2,
            pair_policy_complete=bool(self._pair_policy_complete[idx]),
            v1_search_metadata=v1_search_metadata,
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
        records = []
        for idx in indices:
            rec = self[int(idx)]
            if rec is not None:
                records.append(rec)
        return records

    def records(self) -> List[PositionRecord]:
        with self._lock:
            return [self._record_unlocked(int(idx)) for idx in self._active_indices_unlocked()]

    @property
    def stats(self) -> dict:
        with self._lock:
            if self._size == 0:
                return {"size": 0, "capacity": self.capacity, "max_game_id": 0}
            indices = self._active_indices_unlocked()
            full_count = int(self._is_full[indices].sum())

            def mean(arr: np.ndarray) -> float:
                return float(arr[indices].mean()) if len(indices) else 0.0

            def total(arr: np.ndarray) -> float:
                return float(arr[indices].sum()) if len(indices) else 0.0

            return {
                "size": int(self._size),
                "capacity": int(self.capacity),
                "max_game_id": int(self._max_game_id),
                "full_search_pct": full_count / self._size * 100.0 if self._size else 0.0,
                "avg_target_policy_mass_outside_window": mean(self._outside_policy_mass),
                "avg_missing_target_policy_mass": mean(self._missing_policy_mass),
                "avg_candidate_recall_mcts_top1": mean(self._candidate_recall_top1),
                "avg_candidate_recall_mcts_top4": mean(self._candidate_recall_top4),
                "avg_candidate_recall_mcts_top8": mean(self._candidate_recall_top8),
                "avg_candidate_recall_winning_move": mean(self._candidate_recall_winning),
                "avg_candidate_recall_forced_block": mean(self._candidate_recall_forced_block),
                "avg_candidate_recall_two_placement_cover": mean(self._candidate_recall_cover),
                "candidate_discovery_top1": mean(self._candidate_discovery_top1),
                "candidate_discovery_top4": mean(self._candidate_discovery_top4),
                "candidate_discovery_top8": mean(self._candidate_discovery_top8),
                "candidate_discovery_winning_move": mean(self._candidate_discovery_winning),
                "candidate_discovery_forced_block": mean(self._candidate_discovery_forced_block),
                "candidate_discovery_two_placement_cover": mean(self._candidate_discovery_cover),
                "candidate_discovery_open_four": mean(self._candidate_discovery_open_four),
                "candidate_discovery_open_five": mean(self._candidate_discovery_open_five),
                "critical_count": mean(self._candidate_critical_count),
                "critical_overflow_count": total(self._candidate_critical_overflow_count),
                "fallback_prior_use": mean(self._fallback_prior_use),
                "fallback_prior_use_on_mcts_top1": mean(self._fallback_prior_use_top1),
                "fallback_prior_use_on_mcts_top4": mean(self._fallback_prior_use_top4),
                "fallback_prior_use_on_mcts_top8": mean(self._fallback_prior_use_top8),
                "fallback_prior_use_on_mcts_topk": mean(self._fallback_prior_use_top4),
                "sparse_prior_root_hit_frac": mean(self._sparse_prior_root_hit_frac),
                "sparse_prior_leaf_hit_frac": mean(self._sparse_prior_leaf_hit_frac),
                "sparse_vs_dense_disagreement": mean(self._sparse_vs_dense_disagreement),
                "pair_fallback_prior_use": mean(self._pair_fallback_prior_use),
                "pair_prior_hit_frac": mean(self._pair_prior_hit_frac),
                "pair_prior_candidate_count": mean(self._pair_prior_candidate_count),
                "pair_fallback_prior_use_on_mcts_top1": mean(self._pair_fallback_prior_use_top1),
                "pair_fallback_prior_use_on_mcts_top4": mean(self._pair_fallback_prior_use_top4),
                "pair_fallback_prior_use_on_mcts_top8": mean(self._pair_fallback_prior_use_top8),
            }

    def clear(self):
        with self._lock:
            for idx in range(self.capacity):
                self._release_row(idx)
            self._policy_actions.fill(0)
            self._policy_probs.fill(0.0)
            self._policy_counts.fill(0)
            self._policy_v2_q.fill(0)
            self._policy_v2_r.fill(0)
            self._policy_v2_probs.fill(0.0)
            self._policy_v2_counts.fill(0)
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
            self._opp_policy_weights.fill(0.0)
            self._game_ids.fill(0)
            self._is_full.fill(False)
            self._players.fill(0)
            if self._lookahead is not None:
                self._lookahead.fill(0.0)
            if self._lookahead_counts is not None:
                self._lookahead_counts.fill(0)
            if self._opp_policy_blobs is not None:
                self._opp_policy_blobs = [None] * self.capacity
                self._opp_policy_v2_blobs = [None] * self.capacity
                self._opp_legal_v2_blobs = [None] * self.capacity
            if self._pair_policy_v2_blobs is not None:
                self._pair_policy_v2_blobs = [None] * self.capacity
            self._v1_search_metadata_blobs = [None] * self.capacity
            self._pair_policy_complete.fill(False)
            self._game_histories.clear()
            self._game_refcounts.clear()
            self._game_ids_by_slot.clear()
            self._game_slots_by_id.clear()
            self._free_game_slots.clear()
            self._head = 0
            self._tail = 0
            self._size = 0
            self._max_game_id = 0

    def _active_indices_unlocked(self) -> np.ndarray:
        return np.asarray(
            [(self._tail + i) % self.capacity for i in range(self._size)],
            dtype=np.int64,
        )
