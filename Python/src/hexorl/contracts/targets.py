"""Policy target value objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexorl.contracts.identity import readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source


TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicyTarget:
    rows: np.ndarray
    probabilities: np.ndarray
    source: str = "rust"
    schema_version: int = TARGET_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="PolicyTarget")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 2), dtype=np.int32)
        probs = readonly_array(np.asarray(self.probabilities, dtype=np.float32).reshape(-1), dtype=np.float32)
        if rows.shape[0] != probs.shape[0]:
            raise ContractValidationError("policy target row/probability length mismatch", owner="PolicyTarget", source=source)
        if not np.isfinite(probs).all():
            raise ContractValidationError("policy target probabilities must be finite", owner="PolicyTarget", source=source)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "source", source)

    @property
    def target_hash(self) -> str:
        return stable_digest(("PolicyTarget", self.schema_version, self.source, self.rows.tobytes(), self.probabilities.tobytes()))


@dataclass(frozen=True)
class PairPolicyTarget:
    rows: np.ndarray
    probabilities: np.ndarray
    source: str = "rust"
    schema_version: int = TARGET_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="PairPolicyTarget")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 4), dtype=np.int32)
        probs = readonly_array(np.asarray(self.probabilities, dtype=np.float32).reshape(-1), dtype=np.float32)
        if rows.shape[0] != probs.shape[0]:
            raise ContractValidationError("pair target row/probability length mismatch", owner="PairPolicyTarget", source=source)
        if not np.isfinite(probs).all():
            raise ContractValidationError("pair target probabilities must be finite", owner="PairPolicyTarget", source=source)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "probabilities", probs)
        object.__setattr__(self, "source", source)

    @property
    def target_hash(self) -> str:
        return stable_digest(("PairPolicyTarget", self.schema_version, self.source, self.rows.tobytes(), self.probabilities.tobytes()))
