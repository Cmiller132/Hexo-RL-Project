"""Bounded canonical replay storage."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from hexorl.replay.codec import ReplayCodecError, ReplayGameRecord, ReplayPositionRecord


@dataclass
class ReplayStorageStats:
    capacity: int
    size: int = 0
    games_written: int = 0
    positions_written: int = 0
    rejected_records: int = 0
    write_backpressure_events: int = 0
    read_samples: int = 0
    write_samples_per_sec: float = 0.0
    read_samples_per_sec: float = 0.0
    high_watermark: int = 0
    memory_high_watermark_bytes: int = 0
    last_error_owner: str = ""
    last_error: str = ""

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "games_written": self.games_written,
            "positions_written": self.positions_written,
            "rejected_records": self.rejected_records,
            "write_backpressure_events": self.write_backpressure_events,
            "read_samples": self.read_samples,
            "write_samples_per_sec": self.write_samples_per_sec,
            "read_samples_per_sec": self.read_samples_per_sec,
            "high_watermark": self.high_watermark,
            "memory_high_watermark_bytes": self.memory_high_watermark_bytes,
            "last_error_owner": self.last_error_owner,
            "last_error": self.last_error,
        }


class ReplayStorage:
    """Thread-safe bounded storage for new replay records only."""

    def __init__(self, capacity: int, *, prefetch_records: int = 0) -> None:
        if int(capacity) <= 0:
            raise ValueError("ReplayStorage capacity must be positive")
        self.capacity = int(capacity)
        self.prefetch_records = max(0, int(prefetch_records))
        self._records: deque[ReplayPositionRecord] = deque(maxlen=self.capacity)
        self._games: deque[ReplayGameRecord] = deque(maxlen=self.capacity)
        self._lock = threading.RLock()
        self._created = time.monotonic()
        self._stats = ReplayStorageStats(capacity=self.capacity)
        self.max_game_id = -1

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def stats(self) -> dict:
        with self._lock:
            elapsed = max(time.monotonic() - self._created, 1e-6)
            records = list(self._records)
            self._stats.size = len(self._records)
            self._stats.write_samples_per_sec = self._stats.positions_written / elapsed
            self._stats.read_samples_per_sec = self._stats.read_samples / elapsed
            self._stats.high_watermark = max(self._stats.high_watermark, len(self._records))
            self._stats.memory_high_watermark_bytes = max(
                self._stats.memory_high_watermark_bytes,
                sum(len(rec.move_history) + len(rec.legal_rows) * 8 for rec in records),
            )
            out = self._stats.to_dict()
            out.update(
                {
                    "avg_missing_target_policy_mass": _mean(getattr(rec, "missing_target_policy_mass", 0.0) for rec in records),
                    "avg_target_policy_mass_outside_window": _mean(getattr(rec, "target_policy_mass_outside_window", 0.0) for rec in records),
                    "critical_count": _mean(getattr(rec, "candidate_critical_count", 0.0) for rec in records),
                    "critical_overflow_count": sum(float(getattr(rec, "candidate_critical_overflow_count", 0.0)) for rec in records),
                }
            )
            return out

    def append_game(self, game: ReplayGameRecord) -> None:
        if not isinstance(game, ReplayGameRecord):
            self._reject("ReplayStorage accepts only ReplayGameRecord", owner="replay.storage")
        with self._lock:
            overflow = max(0, len(self._records) + len(game.positions) - self.capacity)
            if overflow:
                self._stats.write_backpressure_events += 1
            for rec in game.positions:
                self._records.append(rec)
            self._games.append(game)
            self._stats.games_written += 1
            self._stats.positions_written += len(game.positions)
            self._stats.size = len(self._records)
            self._stats.high_watermark = max(self._stats.high_watermark, len(self._records))
            self.max_game_id = max(self.max_game_id, int(game.game_id))

    def extend_games(self, games: Iterable[ReplayGameRecord]) -> None:
        for game in games:
            self.append_game(game)

    def sample_records(
        self,
        count: int,
        *,
        recency_decay: float = 0.99,
        pcr_weight: float = 0.25,
        rng: random.Random | None = None,
    ) -> list[ReplayPositionRecord]:
        n = int(count)
        if n <= 0:
            return []
        with self._lock:
            if len(self._records) < n:
                return []
            records = list(self._records)
        weights = np.ones(len(records), dtype=np.float64)
        if 0.0 < float(recency_decay) < 1.0:
            age = np.arange(len(records) - 1, -1, -1, dtype=np.float64)
            weights *= np.power(float(recency_decay), age)
        for idx, rec in enumerate(records):
            if not rec.is_full_search:
                weights[idx] *= float(pcr_weight)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(weights).all():
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total
        choice_rng = rng or random
        indices = choice_rng.choices(range(len(records)), weights=weights.tolist(), k=n)
        with self._lock:
            self._stats.read_samples += n
        return [records[idx] for idx in indices]

    def records(self) -> list[ReplayPositionRecord]:
        with self._lock:
            return list(self._records)

    def games(self) -> list[ReplayGameRecord]:
        with self._lock:
            return list(self._games)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._games.clear()
            self.max_game_id = -1
            self._stats.size = 0

    def _reject(self, message: str, *, owner: str) -> None:
        with self._lock:
            self._stats.rejected_records += 1
            self._stats.last_error_owner = owner
            self._stats.last_error = message
        raise ReplayCodecError(message, owner=owner)


def _mean(values) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0
