"""Row-mapped search prior contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from hexorl.contracts.identity import ndarray_digest, readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError
from hexorl.search.context import SearchContext


PRIOR_SOURCE_DENSE = 2
PRIOR_SOURCE_SPARSE = 1
PRIOR_SOURCE_DEFAULT = 3
PRIOR_SOURCE_PAIR = 4
PRIOR_SOURCE_GLOBAL = 5


@dataclass(frozen=True)
class SearchEvaluation:
    context: SearchContext
    value: float
    legal_row_ids: np.ndarray
    legal_dense_indices: np.ndarray
    row_priors: np.ndarray
    prior_source: np.ndarray
    policy_provider: str
    model_family: str
    model_spec_version: str
    inference_protocol: str
    warnings: tuple[str, ...] = ()
    timings: Mapping[str, float] = field(default_factory=dict)
    raw_metadata: Mapping[str, object] = field(default_factory=dict)
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        legal_rows = self.context.legal_table.rows
        width = int(legal_rows.shape[0])
        row_ids = readonly_array(np.asarray(self.legal_row_ids, dtype=np.int64).reshape(-1), dtype=np.int64)
        dense = readonly_array(np.asarray(self.legal_dense_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
        priors = readonly_array(np.asarray(self.row_priors, dtype=np.float32).reshape(-1), dtype=np.float32)
        sources = readonly_array(np.asarray(self.prior_source, dtype=np.uint8).reshape(-1), dtype=np.uint8)
        if not (row_ids.shape[0] == dense.shape[0] == priors.shape[0] == sources.shape[0] == width):
            raise ContractValidationError("SearchEvaluation prior length must equal legal row count", owner="SearchEvaluation")
        if width == 0:
            raise ContractValidationError("SearchEvaluation requires at least one legal row", owner="SearchEvaluation")
        if not np.array_equal(row_ids, np.arange(width, dtype=np.int64)):
            raise ContractValidationError("SearchEvaluation row ids must map one-to-one to LegalActionTable rows", owner="SearchEvaluation")
        if not np.array_equal(dense, self.context.legal_table.dense_indices):
            raise ContractValidationError("SearchEvaluation dense indices do not match legal table", owner="SearchEvaluation")
        if not np.isfinite(priors).all() or not np.isfinite(float(self.value)):
            raise ContractValidationError("SearchEvaluation contains non-finite value or priors", owner="SearchEvaluation")
        if np.any(priors < -1e-7):
            raise ContractValidationError("SearchEvaluation priors contain negative mass", owner="SearchEvaluation")
        mass = float(np.sum(priors))
        if mass <= 1e-12 and not self.fallback_reason:
            raise ContractValidationError("SearchEvaluation all-zero priors require explicit fallback reason", owner="SearchEvaluation")
        if mass > 0.0:
            priors = readonly_array(priors / mass, dtype=np.float32)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "legal_row_ids", row_ids)
        object.__setattr__(self, "legal_dense_indices", dense)
        object.__setattr__(self, "row_priors", priors)
        object.__setattr__(self, "prior_source", sources)
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "raw_metadata", MappingProxyType(dict(self.raw_metadata)))

    @property
    def evaluation_hash(self) -> str:
        return stable_digest(
            (
                "SearchEvaluation",
                self.context.position_hash,
                self.policy_provider,
                self.inference_protocol,
                ndarray_digest(self.row_priors, schema_version=1, source="search"),
                ndarray_digest(self.prior_source, schema_version=1, source="search"),
            )
        )

    def dense_policy(self, board_area: int = 1089) -> np.ndarray:
        dense = np.zeros(int(board_area), dtype=np.float32)
        dense[self.legal_dense_indices] = self.row_priors
        return dense

    def sparse_payload(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qr = self.context.legal_table.rows.reshape(1, -1, 2)
        logits = self.row_priors.reshape(1, -1)
        counts = np.asarray([self.row_priors.shape[0]], dtype=np.uint16)
        return qr, logits, counts


def priors_from_logits(logits: np.ndarray, *, fallback_reason: str | None = None) -> tuple[np.ndarray, str | None]:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if not np.isfinite(arr).all():
        raise ContractValidationError("logits contain non-finite values", owner="SearchEvaluation")
    if arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), fallback_reason
    shifted = arr - float(np.max(arr))
    exp = np.exp(shifted).astype(np.float32)
    mass = float(np.sum(exp))
    if mass <= 1e-12:
        if fallback_reason is None:
            fallback_reason = "zero_logit_mass_uniform"
        exp = np.ones_like(exp, dtype=np.float32)
        mass = float(exp.sum())
    return (exp / mass).astype(np.float32), fallback_reason
