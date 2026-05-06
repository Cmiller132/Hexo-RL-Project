"""Engine adapter validation between decoded inference and Rust MCTS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EngineAdapter:
    """Validation helpers for Rust MCTS calls.

    The adapter is intentionally small at the start of Phase 4: it owns the
    semantic checks that had been embedded in self-play before model outputs
    reach MCTS expansion or pair-prior hooks.
    """

    value_min: float = -1.0
    value_max: float = 1.0

    def validate_value(self, value: float, *, context: str) -> float:
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"{context}: value is not finite")
        if out < self.value_min or out > self.value_max:
            raise ValueError(
                f"{context}: value {out:g} outside [{self.value_min:g}, {self.value_max:g}]"
            )
        return out

    @staticmethod
    def align_global_logits_to_rust_legal(
        graph_legal: np.ndarray,
        rust_legal: np.ndarray,
        logits: np.ndarray,
        *,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return legal rows/logits in the exact order Rust MCTS expects."""
        graph_legal = np.asarray(graph_legal, dtype=np.int32).reshape(-1, 2)
        rust_legal = np.asarray(rust_legal, dtype=np.int32).reshape(-1, 2)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits.shape[0] < graph_legal.shape[0]:
            raise ValueError(
                f"{context}: logits have {logits.shape[0]} rows for "
                f"{graph_legal.shape[0]} graph legal moves"
            )
        if graph_legal.shape != rust_legal.shape:
            raise ValueError(
                f"{context}: legal row count mismatch graph={graph_legal.shape[0]} "
                f"rust={rust_legal.shape[0]}"
            )
        if np.array_equal(graph_legal, rust_legal):
            return rust_legal, logits[: rust_legal.shape[0]]

        graph_index: dict[tuple[int, int], int] = {}
        duplicate_graph_rows: list[tuple[int, int]] = []
        for idx, qr in enumerate(graph_legal.tolist()):
            key = (int(qr[0]), int(qr[1]))
            if key in graph_index:
                duplicate_graph_rows.append(key)
            graph_index[key] = idx
        rust_keys = [(int(q), int(r)) for q, r in rust_legal.tolist()]
        missing = [key for key in rust_keys if key not in graph_index]
        extras = sorted(set(graph_index) - set(rust_keys))
        if duplicate_graph_rows or missing or extras:
            raise ValueError(
                f"{context}: legal_qr set mismatch "
                f"duplicates={duplicate_graph_rows[:5]} missing={missing[:5]} extra={extras[:5]}"
            )
        order = np.asarray([graph_index[key] for key in rust_keys], dtype=np.int64)
        return rust_legal, logits[order]

    @staticmethod
    def validate_global_logits_legal_subset(
        graph_legal: np.ndarray,
        rust_legal: np.ndarray,
        logits: np.ndarray,
        *,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return graph rows/logits after proving every row is Rust-legal."""
        graph_legal = np.asarray(graph_legal, dtype=np.int32).reshape(-1, 2)
        rust_legal = np.asarray(rust_legal, dtype=np.int32).reshape(-1, 2)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        if logits.shape[0] < graph_legal.shape[0]:
            raise ValueError(
                f"{context}: policy_place has {logits.shape[0]} rows for "
                f"{graph_legal.shape[0]} graph legal moves"
            )
        seen: set[tuple[int, int]] = set()
        duplicates: list[tuple[int, int]] = []
        for q, r in graph_legal.tolist():
            key = (int(q), int(r))
            if key in seen:
                duplicates.append(key)
            seen.add(key)
        rust_set = {(int(q), int(r)) for q, r in rust_legal.tolist()}
        extras = sorted(seen - rust_set)
        if duplicates or extras:
            raise ValueError(
                f"{context}: graph legal rows are not a Rust-legal subset "
                f"duplicates={duplicates[:5]} extras={extras[:5]}"
            )
        return graph_legal, logits[: graph_legal.shape[0]]

