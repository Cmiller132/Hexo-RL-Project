"""Candidate table contract shell for Phase 01 verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexorl.contracts.identity import readonly_array, stable_digest
from hexorl.contracts.validation import validate_source


CANDIDATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CandidateTable:
    rows: np.ndarray
    source: str = "rust"
    schema_version: int = CANDIDATE_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="CandidateTable")
        object.__setattr__(self, "rows", readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 2), dtype=np.int32))
        object.__setattr__(self, "source", source)

    @property
    def table_hash(self) -> str:
        return stable_digest(("CandidateTable", self.schema_version, self.source, self.rows.tobytes()))


@dataclass(frozen=True)
class CandidateDiagnostics:
    missing_mass: float = 0.0
    truncated: bool = False
