"""Server-side inference counters and timing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServerMetrics:
    n_batches: int = 0
    n_positions: int = 0
    total_build_ms: float = 0.0
    total_forward_ms: float = 0.0
    total_model_ms: float = 0.0
    total_postprocess_ms: float = 0.0
    total_download_ms: float = 0.0
    total_scatter_ms: float = 0.0
    min_batch: int = 0
    max_batch_seen: int = 0

    def record_batch(self, total_count: int) -> None:
        self.n_batches += 1
        self.n_positions += int(total_count)
        self.min_batch = total_count if self.min_batch == 0 else min(self.min_batch, total_count)
        self.max_batch_seen = max(self.max_batch_seen, int(total_count))

    @property
    def positions_per_sec(self) -> float:
        if self.total_forward_ms <= 0:
            return 0.0
        return self.n_positions / (self.total_forward_ms / 1000.0)

    def summary(self) -> tuple[str, str]:
        avg_batch = self.n_positions / max(self.n_batches, 1)
        counts = (
            "[inference-server] Shutting down. "
            f"Batches: {self.n_batches}, Positions: {self.n_positions}, "
            f"Avg batch: {avg_batch:.1f}, Min batch: {self.min_batch}, "
            f"Max batch: {self.max_batch_seen}"
        )
        timing = (
            "[inference-server] Timing ms total: "
            f"build={self.total_build_ms:.1f}, "
            f"forward={self.total_forward_ms:.1f}, "
            f"model={self.total_model_ms:.1f}, "
            f"postprocess={self.total_postprocess_ms:.1f}, "
            f"download={self.total_download_ms:.1f}, "
            f"scatter={self.total_scatter_ms:.1f}"
        )
        return counts, timing


__all__ = ["ServerMetrics"]
