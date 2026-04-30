"""Inference batching and backpressure helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceBatchingProfile:
    max_batch_size: int
    max_wait_us: int
    ready_workers: int
    total_positions: int
    backpressure_policy: str = "bounded_max_batch_skip_overflow"

    def to_markdown(self) -> str:
        return (
            f"- max_batch_size: {self.max_batch_size}\n"
            f"- max_wait_us: {self.max_wait_us}\n"
            f"- ready_workers: {self.ready_workers}\n"
            f"- total_positions: {self.total_positions}\n"
            f"- backpressure_policy: {self.backpressure_policy}\n"
        )


def would_exceed_batch(total: int, count: int, max_batch_size: int) -> bool:
    return int(total) + int(count) > int(max_batch_size)
