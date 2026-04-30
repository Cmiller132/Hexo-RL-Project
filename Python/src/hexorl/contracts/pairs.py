"""Pair action table contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexorl.contracts.identity import readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source


PAIR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairActionTable:
    rows: np.ndarray
    source: str = "rust"
    known_first: tuple[int, int] | None = None
    schema_version: int = PAIR_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="PairActionTable")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 4), dtype=np.int32)
        seen: set[tuple[int, int, int, int]] = set()
        for row in rows.tolist():
            key = tuple(int(x) for x in row)
            if key in seen:
                raise ContractValidationError(f"duplicate pair row {key}", owner="PairActionTable", source=source)
            seen.add(key)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "source", source)

    @property
    def table_hash(self) -> str:
        known = "" if self.known_first is None else f"{self.known_first[0]},{self.known_first[1]}"
        return stable_digest(("PairActionTable", self.schema_version, self.source, known, self.rows.tobytes()))
