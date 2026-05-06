"""Engine adapter validation between decoded inference and Rust MCTS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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

    @staticmethod
    def validate_search_phase(
        placements_remaining: int,
        *,
        root: bool,
        context: str,
    ) -> int:
        phase = int(placements_remaining)
        if phase not in (1, 2):
            location = "root" if root else "leaf"
            raise ValueError(
                f"{context}: {location} search phase must have placements_remaining 1 or 2, got {phase}"
            )
        return phase

    @staticmethod
    def validate_batch_generation(
        observed: int,
        expected: int,
        *,
        context: str,
    ) -> int:
        observed_i = int(observed)
        expected_i = int(expected)
        if observed_i < 0 or expected_i < 0:
            raise ValueError(f"{context}: batch generation must be non-negative")
        if observed_i != expected_i:
            raise ValueError(
                f"{context}: stale MCTS batch generation observed={observed_i} expected={expected_i}"
            )
        return observed_i

    def validate_value(self, value: float, *, context: str) -> float:
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"{context}: value is not finite")
        if out < self.value_min or out > self.value_max:
            raise ValueError(
                f"{context}: value {out:g} outside [{self.value_min:g}, {self.value_max:g}]"
            )
        return out

    def validate_value_perspective(
        self,
        metadata: Mapping[str, object] | None,
        *,
        expected: str = "current_player",
        context: str,
    ) -> None:
        if metadata is None:
            raise ValueError(f"{context}: value metadata is missing")
        outputs = metadata.get("outputs") if isinstance(metadata, Mapping) else None
        value_meta = outputs.get("value") if isinstance(outputs, Mapping) else None
        decoder_meta = value_meta.get("value_decoder") if isinstance(value_meta, Mapping) else None
        if not isinstance(decoder_meta, Mapping):
            decoder_meta = metadata.get("value_decoder") if isinstance(metadata, Mapping) else None
        if not isinstance(decoder_meta, Mapping):
            raise ValueError(f"{context}: value decoder metadata is missing")
        observed = str(decoder_meta.get("perspective", ""))
        if observed != str(expected):
            raise ValueError(
                f"{context}: value perspective {observed!r} does not match expected {expected!r}"
            )

    @staticmethod
    def validate_legal_bytes_alignment(
        rust_legal: np.ndarray,
        legal_bytes: bytes | bytearray | memoryview,
        *,
        context: str,
    ) -> np.ndarray:
        expected = np.asarray(rust_legal, dtype=np.int32).reshape(-1, 2)
        observed = np.frombuffer(bytes(legal_bytes), dtype=np.int32)
        if observed.size % 2 != 0:
            raise ValueError(f"{context}: legal byte buffer does not contain q/r pairs")
        observed = observed.reshape(-1, 2)
        if expected.shape != observed.shape or not np.array_equal(expected, observed):
            raise ValueError(
                f"{context}: Rust legal rows changed before model consumption "
                f"expected={expected.shape[0]} observed={observed.shape[0]}"
            )
        return observed

    @staticmethod
    def validate_dense_offset_mapping(
        legal: np.ndarray,
        offset_q: int,
        offset_r: int,
        *,
        board_size: int = 33,
        context: str,
    ) -> np.ndarray:
        rows = np.asarray(legal, dtype=np.int32).reshape(-1, 2)
        q = rows[:, 0] - int(offset_q)
        r = rows[:, 1] - int(offset_r)
        size = int(board_size)
        in_bounds = (q >= 0) & (q < size) & (r >= 0) & (r < size)
        if not bool(np.all(in_bounds)):
            bad = rows[~in_bounds][:5].tolist()
            raise ValueError(
                f"{context}: dense policy offset does not cover legal rows bad_examples={bad}"
            )
        return (q * size + r).astype(np.int64)

    @staticmethod
    def validate_pair_phase(
        pair_qr: np.ndarray,
        *,
        placements_remaining: int,
        first_qr: tuple[int, int] | None = None,
        context: str,
    ) -> np.ndarray:
        phase = EngineAdapter.validate_search_phase(
            placements_remaining,
            root=True,
            context=context,
        )
        rows = np.asarray(pair_qr, dtype=np.int32).reshape(-1, 4)
        if rows.shape[0] == 0:
            return rows
        duplicate = np.all(rows[:, :2] == rows[:, 2:], axis=1)
        if bool(np.any(duplicate)):
            bad = rows[duplicate][:5].tolist()
            raise ValueError(f"{context}: pair rows contain duplicate first/second cells {bad}")
        if phase == 1:
            if first_qr is None:
                raise ValueError(f"{context}: second-placement pair rows require first_qr")
            first = np.asarray(first_qr, dtype=np.int32).reshape(1, 2)
            mismatch = ~np.all(rows[:, :2] == first, axis=1)
            if bool(np.any(mismatch)):
                bad = rows[mismatch][:5].tolist()
                raise ValueError(
                    f"{context}: known-first pair rows do not match first_qr {tuple(first.reshape(-1))} "
                    f"bad_examples={bad}"
                )
        return rows

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
