"""Architecture boundary contracts for model modularity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RowTableDefinition:
    family: str
    schema_version: int
    payload_schema: tuple[str, ...]
    ordering_rule: str
    mask_semantics: str = "true means active row"

    @property
    def id(self) -> str:
        return f"{self.family}:v{self.schema_version}"


@dataclass(frozen=True)
class RowTableInstance:
    definition: RowTableDefinition
    rows: np.ndarray
    mask: np.ndarray
    phase: str
    source: str
    feature_schema_version: int | None = None
    relation_schema_version: int | None = None

    @property
    def identity_hash(self) -> str:
        rows = np.ascontiguousarray(self.rows)
        mask = np.ascontiguousarray(self.mask.astype(np.bool_))
        h = hashlib.sha256()
        h.update(self.definition.id.encode("utf-8"))
        h.update(self.phase.encode("utf-8"))
        h.update(str(self.feature_schema_version or "").encode("ascii"))
        h.update(str(self.relation_schema_version or "").encode("ascii"))
        h.update(rows.dtype.str.encode("ascii"))
        h.update(np.asarray(rows.shape, dtype=np.int64).tobytes())
        h.update(rows.tobytes())
        h.update(mask.tobytes())
        return "sha256:" + h.hexdigest()


@dataclass(frozen=True)
class OutputContract:
    name: str
    kind: str
    prediction_key: str
    row_family: str | None = None
    state_row: str | None = None
    mask_semantics: str = ""
    trainable: bool = True
    runtime_consumed: bool = False
    required_for_selfplay: bool = False
    optional: bool = False
    diagnostic_only: bool = False


@dataclass(frozen=True)
class ValueDecoderContract:
    name: str = "binned_expected_value_65"
    logits_key: str = "value"
    n_bins: int = 65
    output_range: tuple[float, float] = (-1.0, 1.0)
    perspective: str = "current_player"
    clamp_non_finite: bool = True


ROW_TABLE_DEFINITIONS: Mapping[str, RowTableDefinition] = {
    "dense_board": RowTableDefinition(
        family="dense_board",
        schema_version=1,
        payload_schema=("action:int16",),
        ordering_rule="fixed board indices 0..1088",
    ),
    "candidate": RowTableDefinition(
        family="candidate",
        schema_version=1,
        payload_schema=("q:int32", "r:int32", "action:int16", "features:f32[*]"),
        ordering_rule="candidate builder order",
    ),
    "legal": RowTableDefinition(
        family="legal",
        schema_version=1,
        payload_schema=("q:int32", "r:int32"),
        ordering_rule="rust legal order or validated legal subset order",
    ),
    "opponent_legal": RowTableDefinition(
        family="opponent_legal",
        schema_version=1,
        payload_schema=("q:int32", "r:int32"),
        ordering_rule="opponent legal builder order",
    ),
    "pair_joint": RowTableDefinition(
        family="pair_joint",
        schema_version=1,
        payload_schema=("q1:int32", "r1:int32", "q2:int32", "r2:int32"),
        ordering_rule="unordered canonical legal pair order",
    ),
    "known_first_pair": RowTableDefinition(
        family="known_first_pair",
        schema_version=1,
        payload_schema=("known_q:int32", "known_r:int32", "second_q:int32", "second_r:int32"),
        ordering_rule="known first, legal second order",
    ),
    "graph_token": RowTableDefinition(
        family="graph_token",
        schema_version=1,
        payload_schema=("token_type:int16", "q:int32", "r:int32", "features:f32[*]"),
        ordering_rule="graph token builder order",
    ),
}


def row_table_definitions_for(families: Sequence[str]) -> Mapping[str, RowTableDefinition]:
    return {family: ROW_TABLE_DEFINITIONS[family] for family in families}
