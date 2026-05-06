"""Inference protocol metadata and row-contract validation.

Shared memory remains the transport.  This module owns the semantic metadata
that makes transported arrays safe to consume by search and evaluation code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from hexorl.contracts import (
    ROW_TABLE_DEFINITIONS,
    RowTableInstance,
    ValueDecoderContract,
)


INFERENCE_PROTOCOL_VERSION = 1

GRAPH_HEAD_OPP = 1 << 0
GRAPH_HEAD_PAIR_FIRST = 1 << 1
GRAPH_HEAD_PAIR_JOINT = 1 << 2
GRAPH_HEAD_PAIR_SECOND = 1 << 3
GRAPH_HEAD_REGRET = 1 << 4


@dataclass(frozen=True)
class RowTableMetadata:
    family: str
    schema_version: int
    row_count: int
    active_count: int
    identity_hash: str
    phase: str
    source: str
    feature_schema_version: int | None = None
    relation_schema_version: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ValueDecoderMetadata:
    name: str
    logits_key: str
    n_bins: int
    output_range: tuple[float, float]
    perspective: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OutputMetadata:
    name: str
    kind: str
    row_table: RowTableMetadata | None = None
    value_decoder: ValueDecoderMetadata | None = None
    protocol_version: int = INFERENCE_PROTOCOL_VERSION

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "protocol_version": int(self.protocol_version),
        }
        if self.row_table is not None:
            out["row_table"] = self.row_table.to_dict()
        if self.value_decoder is not None:
            out["value_decoder"] = self.value_decoder.to_dict()
        return out


def value_decoder_metadata(
    contract: ValueDecoderContract | None = None,
) -> ValueDecoderMetadata:
    contract = contract or ValueDecoderContract()
    return ValueDecoderMetadata(
        name=contract.name,
        logits_key=contract.logits_key,
        n_bins=int(contract.n_bins),
        output_range=tuple(float(x) for x in contract.output_range),
        perspective=contract.perspective,
    )


def row_table_metadata(
    family: str,
    rows: np.ndarray,
    mask: np.ndarray,
    *,
    phase: str,
    source: str,
    feature_schema_version: int | None = None,
    relation_schema_version: int | None = None,
) -> RowTableMetadata:
    if family not in ROW_TABLE_DEFINITIONS:
        raise ValueError(f"unknown row-table family {family!r}")
    row_arr = np.asarray(rows)
    mask_arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if row_arr.shape[0] != mask_arr.shape[0]:
        raise ValueError(
            f"row table {family!r} has {row_arr.shape[0]} rows but "
            f"{mask_arr.shape[0]} mask entries"
        )
    instance = RowTableInstance(
        definition=ROW_TABLE_DEFINITIONS[family],
        rows=row_arr,
        mask=mask_arr,
        phase=str(phase),
        source=str(source),
        feature_schema_version=feature_schema_version,
        relation_schema_version=relation_schema_version,
    )
    return RowTableMetadata(
        family=family,
        schema_version=instance.definition.schema_version,
        row_count=int(row_arr.shape[0]),
        active_count=int(mask_arr.sum()),
        identity_hash=instance.identity_hash,
        phase=str(phase),
        source=str(source),
        feature_schema_version=feature_schema_version,
        relation_schema_version=relation_schema_version,
    )


def validate_same_row_table(
    expected: RowTableMetadata,
    observed: RowTableMetadata,
    *,
    context: str,
) -> None:
    if expected.identity_hash != observed.identity_hash:
        raise ValueError(
            f"{context}: row-table identity mismatch for {expected.family!r}: "
            f"expected {expected.identity_hash}, observed {observed.identity_hash}"
        )


def validate_row_arrays_match(
    expected_rows: np.ndarray,
    expected_mask: np.ndarray,
    observed_rows: np.ndarray,
    observed_mask: np.ndarray,
    *,
    family: str,
    phase: str,
    context: str,
) -> RowTableMetadata:
    expected = row_table_metadata(
        family,
        expected_rows,
        expected_mask,
        phase=phase,
        source=f"{context}:expected",
    )
    observed = row_table_metadata(
        family,
        observed_rows,
        observed_mask,
        phase=phase,
        source=f"{context}:observed",
    )
    validate_same_row_table(expected, observed, context=context)
    return observed


def graph_head_flags(outputs: Mapping[str, object]) -> int:
    flags = 0
    if "opp_policy" in outputs:
        flags |= GRAPH_HEAD_OPP
    if "policy_pair_first" in outputs:
        flags |= GRAPH_HEAD_PAIR_FIRST
    if "policy_pair_joint" in outputs:
        flags |= GRAPH_HEAD_PAIR_JOINT
    if "policy_pair_second" in outputs:
        flags |= GRAPH_HEAD_PAIR_SECOND
    if "regret_rank" in outputs:
        flags |= GRAPH_HEAD_REGRET
    return flags

