"""Canonical legal action table contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hexorl.contracts.coordinates import PLACEMENT_RADIUS, dense_index
from hexorl.contracts.identity import ContractIdentity, ndarray_digest, readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source


LEGAL_TABLE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LegalActionTable:
    rows: np.ndarray
    dense_indices: np.ndarray
    source: str
    radius: int
    occupied_count: int
    current_player: int
    placements_remaining: int
    history_hash: str
    schema_version: int = LEGAL_TABLE_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="LegalActionTable")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 2), dtype=np.int32)
        dense_indices = readonly_array(np.asarray(self.dense_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
        if rows.shape[0] != dense_indices.shape[0]:
            raise ContractValidationError("legal rows and dense indices length mismatch", owner="LegalActionTable", source=source)
        seen: set[tuple[int, int]] = set()
        for q_raw, r_raw in rows.tolist():
            qr = (int(q_raw), int(r_raw))
            if qr in seen:
                raise ContractValidationError(f"duplicate legal row {qr}", owner="LegalActionTable", source=source)
            seen.add(qr)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "dense_indices", dense_indices)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "radius", int(self.radius))
        object.__setattr__(self, "occupied_count", int(self.occupied_count))
        object.__setattr__(self, "current_player", int(self.current_player))
        object.__setattr__(self, "placements_remaining", int(self.placements_remaining))

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[int, int]],
        *,
        source: str,
        radius: int = PLACEMENT_RADIUS,
        occupied_count: int = 0,
        current_player: int = 0,
        placements_remaining: int = 1,
        history_hash: str = "",
        allow_fixture: bool = False,
    ) -> "LegalActionTable":
        arr = np.asarray([(int(q), int(r)) for q, r in rows], dtype=np.int32).reshape(-1, 2)
        dense = np.asarray([dense_index(int(q), int(r)) for q, r in arr.tolist()], dtype=np.int64)
        return cls(
            rows=arr,
            dense_indices=dense,
            source=source,
            radius=radius,
            occupied_count=occupied_count,
            current_player=current_player,
            placements_remaining=placements_remaining,
            history_hash=history_hash,
            allow_fixture=allow_fixture,
        )

    @property
    def table_hash(self) -> str:
        return stable_digest(
            (
                "LegalActionTable",
                self.schema_version,
                self.source,
                self.radius,
                self.occupied_count,
                self.current_player,
                self.placements_remaining,
                self.history_hash,
                ndarray_digest(self.rows, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.dense_indices, schema_version=self.schema_version, source=self.source),
            )
        )

    @property
    def identity(self) -> ContractIdentity:
        return ContractIdentity("LegalActionTable", self.schema_version, self.source, self.table_hash)

    def assert_semantic_consistency(self, *, occupied: set[tuple[int, int]] | None = None, terminal: bool = False) -> None:
        if terminal and self.rows.shape[0] != 0:
            raise ContractValidationError("terminal state cannot expose legal rows", owner="LegalActionTable", source=self.source)
        if occupied is not None:
            for q_raw, r_raw in self.rows.tolist():
                qr = (int(q_raw), int(r_raw))
                if qr in occupied:
                    raise ContractValidationError(f"legal row is occupied: {qr}", owner="LegalActionTable", source=self.source)

    def debug_payload(self) -> dict[str, object]:
        return {
            "contract": "LegalActionTable",
            "schema_version": self.schema_version,
            "source": self.source,
            "table_hash": self.table_hash,
            "radius": self.radius,
            "occupied_count": self.occupied_count,
            "current_player": self.current_player,
            "placements_remaining": self.placements_remaining,
            "history_hash": self.history_hash,
            "legal_count": int(self.rows.shape[0]),
            "rows": self.rows.tolist(),
        }
