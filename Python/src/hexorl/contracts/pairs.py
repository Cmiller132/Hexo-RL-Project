"""Canonical pair-action table contract and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from hexorl.contracts.candidates import CandidateTable
from hexorl.contracts.identity import ContractIdentity, ndarray_digest, readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source


PAIR_SCHEMA_VERSION = 2
PairPhase = Literal["empty", "first_placement", "second_placement_known_first"]
PairGenerationMode = Literal["none", "selected", "capped_fill", "full_capped"]


@dataclass(frozen=True)
class PairStrategy:
    """Phase 02 cap/selection request for pair construction.

    Final search consumption remains owned by Phase 05. This object only makes
    row generation mode and caps explicit enough to prevent implicit full-pair
    enumeration in candidate, graph, and replay projections.
    """

    mode: PairGenerationMode = "capped_fill"
    max_pairs: int = 0
    allow_full: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"none", "selected", "capped_fill", "full_capped"}:
            raise ContractValidationError(f"unsupported pair generation mode: {self.mode!r}", owner="PairStrategy")
        object.__setattr__(self, "max_pairs", int(self.max_pairs))
        object.__setattr__(self, "allow_full", bool(self.allow_full))
        if self.mode == "none" and self.max_pairs != 0:
            raise ContractValidationError("no-pair strategy must have max_pairs=0", owner="PairStrategy")
        if self.mode != "none" and self.max_pairs <= 0:
            raise ContractValidationError("pair strategy requires a positive max_pairs cap", owner="PairStrategy")
        if self.mode == "full_capped" and not self.allow_full:
            raise ContractValidationError("full pair generation requires allow_full=True", owner="PairStrategy")


@dataclass(frozen=True)
class PairActionTable:
    rows: np.ndarray
    first_candidate_rows: np.ndarray
    second_candidate_rows: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    phase: PairPhase
    first_policy_target: np.ndarray | None = None
    source: str = "rust"
    known_first: tuple[int, int] | None = None
    generation_mode: PairGenerationMode = "capped_fill"
    possible_pair_count: int = 0
    selected_pair_count: int = 0
    missing_mass: float = 0.0
    candidate_table_hash: str = ""
    schema_version: int = PAIR_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="PairActionTable")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 4), dtype=np.int32)
        first_refs = readonly_array(np.asarray(self.first_candidate_rows, dtype=np.int64).reshape(-1), dtype=np.int64)
        second_refs = readonly_array(np.asarray(self.second_candidate_rows, dtype=np.int64).reshape(-1), dtype=np.int64)
        mask = readonly_array(np.asarray(self.mask, dtype=np.bool_).reshape(-1), dtype=np.bool_)
        target = readonly_array(np.asarray(self.target, dtype=np.float32).reshape(-1), dtype=np.float32)
        first_policy_target = readonly_array(
            np.zeros(0, dtype=np.float32) if self.first_policy_target is None else np.asarray(self.first_policy_target, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        width = int(rows.shape[0])
        if first_refs.shape[0] != width or second_refs.shape[0] != width or mask.shape[0] != width or target.shape[0] != width:
            raise ContractValidationError("pair rows, refs, mask, and target length mismatch", owner="PairActionTable", source=source)
        if self.phase not in {"empty", "first_placement", "second_placement_known_first"}:
            raise ContractValidationError(f"unsupported pair phase: {self.phase!r}", owner="PairActionTable", source=source)
        if self.generation_mode not in {"none", "selected", "capped_fill", "full_capped"}:
            raise ContractValidationError(f"unsupported pair generation mode: {self.generation_mode!r}", owner="PairActionTable", source=source)
        if self.phase == "second_placement_known_first" and self.known_first is None:
            raise ContractValidationError("second-placement pair table requires known_first", owner="PairActionTable", source=source)
        seen: set[tuple[int, int, int, int]] = set()
        for idx, row in enumerate(rows.tolist()):
            if not bool(mask[idx]):
                continue
            key = tuple(int(x) for x in row)
            if key in seen:
                raise ContractValidationError(f"duplicate pair row {key}", owner="PairActionTable", source=source)
            seen.add(key)
            a = (key[0], key[1])
            b = (key[2], key[3])
            if a == b:
                raise ContractValidationError(f"duplicate coordinates are illegal for pair row: {a}", owner="PairActionTable", source=source)
            if self.phase == "first_placement" and a > b:
                raise ContractValidationError(f"first-placement pair row is not canonical unordered identity: {key}", owner="PairActionTable", source=source)
            if self.phase == "second_placement_known_first" and a != self.known_first:
                raise ContractValidationError(f"second-placement pair row does not start with known_first: {key}", owner="PairActionTable", source=source)
        if float(np.sum(target[~mask])) != 0.0:
            raise ContractValidationError("inactive pair target rows must be zero", owner="PairActionTable", source=source)
        if np.any(target < -1e-7):
            raise ContractValidationError("pair target cannot contain negative mass", owner="PairActionTable", source=source)
        if np.any(first_policy_target < -1e-7):
            raise ContractValidationError("pair first-policy target cannot contain negative mass", owner="PairActionTable", source=source)
        selected = int(np.count_nonzero(mask))
        if int(self.selected_pair_count) not in {0, selected}:
            raise ContractValidationError("selected_pair_count does not match active pair mask", owner="PairActionTable", source=source)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "first_candidate_rows", first_refs)
        object.__setattr__(self, "second_candidate_rows", second_refs)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "first_policy_target", first_policy_target)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "known_first", None if self.known_first is None else (int(self.known_first[0]), int(self.known_first[1])))
        object.__setattr__(self, "possible_pair_count", int(self.possible_pair_count))
        object.__setattr__(self, "selected_pair_count", selected)
        object.__setattr__(self, "missing_mass", float(self.missing_mass))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def pair_indices(self) -> np.ndarray:
        return np.stack([self.first_candidate_rows, self.second_candidate_rows], axis=1)

    @property
    def table_hash(self) -> str:
        known = "" if self.known_first is None else f"{self.known_first[0]},{self.known_first[1]}"
        return stable_digest(
            (
                "PairActionTable",
                self.schema_version,
                self.source,
                self.phase,
                self.generation_mode,
                known,
                self.possible_pair_count,
                self.selected_pair_count,
                round(self.missing_mass, 12),
                self.candidate_table_hash,
                ndarray_digest(self.rows, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.first_candidate_rows, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.second_candidate_rows, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.mask, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.target, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.first_policy_target, schema_version=self.schema_version, source=self.source),
            )
        )

    @property
    def identity(self) -> ContractIdentity:
        return ContractIdentity("PairActionTable", self.schema_version, self.source, self.table_hash)

    def debug_payload(self) -> dict[str, object]:
        return {
            "contract": "PairActionTable",
            "schema_version": self.schema_version,
            "source": self.source,
            "table_hash": self.table_hash,
            "candidate_table_hash": self.candidate_table_hash,
            "phase": self.phase,
            "known_first": self.known_first,
            "generation_mode": self.generation_mode,
            "possible_pair_count": self.possible_pair_count,
            "selected_pair_count": self.selected_pair_count,
            "target_mass": float(np.sum(self.target)),
            "missing_mass": self.missing_mass,
        }


class PairActionTableBuilder:
    def build(
        self,
        candidate_table: CandidateTable,
        pair_policy_target_v2: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
        *,
        strategy: PairStrategy,
        legal_moves: Sequence[tuple[int, int]] | None = None,
        known_first: tuple[int, int] | None = None,
        source: str | None = None,
        allow_fixture: bool | None = None,
    ) -> PairActionTable:
        source_value = candidate_table.source if source is None else source
        allow_fixture_value = candidate_table.allow_fixture if allow_fixture is None else bool(allow_fixture)
        source_value = validate_source(source_value, allow_fixture=allow_fixture_value, owner="PairActionTableBuilder")
        if strategy.mode == "none":
            return self._empty(candidate_table, source=source_value, allow_fixture=allow_fixture_value, generation_mode="none")
        active_rows = [
            (idx, (int(q), int(r)))
            for idx, (q, r) in enumerate(candidate_table.rows.tolist())
            if bool(candidate_table.mask[idx])
        ]
        candidate_index = {qr: idx for idx, qr in active_rows}
        if len(candidate_index) != len(active_rows):
            raise ContractValidationError("candidate table has duplicate active rows", owner="PairActionTableBuilder", source=source_value)
        legal_set = set(_unique_qr(_as_qr(move) for move in legal_moves)) if legal_moves is not None else None
        known = None if known_first is None else _as_qr(known_first)
        phase: PairPhase = "second_placement_known_first" if known is not None else "first_placement"
        target_map: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        first_target_mass = np.zeros(int(candidate_table.rows.shape[0]), dtype=np.float32)
        protected: list[tuple[tuple[int, int], tuple[int, int]]] = []
        total_target_mass = 0.0
        for first_raw, second_raw, prob_raw in pair_policy_target_v2:
            prob = float(prob_raw)
            if prob <= 0.0:
                continue
            first = _as_qr(first_raw)
            second = _as_qr(second_raw)
            if (
                phase == "second_placement_known_first"
                and known is not None
                and first != known
                and legal_set is not None
                and first in legal_set
                and second in legal_set
            ):
                total_target_mass += prob
                continue
            key = self._canonical_phase_pair(first, second, phase=phase, known_first=known, legal_set=legal_set, source=source_value)
            total_target_mass += prob
            target_map[key] = target_map.get(key, 0.0) + prob
            if phase == "first_placement" and first in candidate_index:
                first_target_mass[int(candidate_index[first])] += prob
            if key[0] in candidate_index and key[1] in candidate_index:
                protected.append(key)
        protected = list(dict.fromkeys(protected))
        fill = self._fill_pairs(active_rows, phase=phase, known_first=known, target_map=target_map)
        possible = self._possible_pair_count(active_rows, phase=phase, known_first=known)
        if strategy.mode == "full_capped":
            if strategy.max_pairs < possible:
                raise ContractValidationError("full pair strategy cap is smaller than possible pair count", owner="PairActionTableBuilder", source=source_value)
            selected_pairs = protected + [pair for pair in fill if pair not in set(protected)]
        elif strategy.mode == "selected":
            selected_pairs = protected
        else:
            protected_set = set(protected)
            selected_pairs = protected + [pair for pair in fill if pair not in protected_set]
        if len(selected_pairs) > strategy.max_pairs:
            if strategy.mode == "selected":
                raise ContractValidationError("selected pair targets exceed pair strategy cap", owner="PairActionTableBuilder", source=source_value)
            selected_pairs = selected_pairs[: strategy.max_pairs]
        width = max(strategy.max_pairs, len(selected_pairs), 1)
        rows = np.zeros((width, 4), dtype=np.int32)
        first_refs = np.full(width, -1, dtype=np.int64)
        second_refs = np.full(width, -1, dtype=np.int64)
        mask = np.zeros(width, dtype=np.bool_)
        target = np.zeros(width, dtype=np.float32)
        represented_mass = 0.0
        for row_idx, pair in enumerate(selected_pairs[:width]):
            rows[row_idx] = (pair[0][0], pair[0][1], pair[1][0], pair[1][1])
            first_refs[row_idx] = candidate_index.get(pair[0], -1)
            second_refs[row_idx] = candidate_index.get(pair[1], -1)
            if first_refs[row_idx] < 0 or second_refs[row_idx] < 0 or first_refs[row_idx] == second_refs[row_idx]:
                continue
            mask[row_idx] = True
            prob = target_map.get(pair, 0.0)
            target[row_idx] = prob
            represented_mass += prob
        if represented_mass > 0.0:
            target /= represented_mass
        first_total = float(first_target_mass.sum())
        if first_total > 0.0:
            first_target_mass /= first_total
        return PairActionTable(
            rows=rows,
            first_candidate_rows=first_refs,
            second_candidate_rows=second_refs,
            mask=mask,
            target=target,
            first_policy_target=first_target_mass,
            phase=phase,
            source=source_value,
            known_first=known,
            generation_mode=strategy.mode,
            possible_pair_count=possible,
            selected_pair_count=int(np.count_nonzero(mask)),
            missing_mass=max(0.0, total_target_mass - represented_mass),
            candidate_table_hash=candidate_table.table_hash,
            allow_fixture=allow_fixture_value,
        )

    def _empty(
        self,
        candidate_table: CandidateTable,
        *,
        source: str,
        allow_fixture: bool,
        generation_mode: PairGenerationMode,
    ) -> PairActionTable:
        return PairActionTable(
            rows=np.zeros((0, 4), dtype=np.int32),
            first_candidate_rows=np.zeros(0, dtype=np.int64),
            second_candidate_rows=np.zeros(0, dtype=np.int64),
            mask=np.zeros(0, dtype=np.bool_),
            target=np.zeros(0, dtype=np.float32),
            first_policy_target=np.zeros(int(candidate_table.rows.shape[0]), dtype=np.float32),
            phase="empty",
            source=source,
            generation_mode=generation_mode,
            possible_pair_count=0,
            selected_pair_count=0,
            candidate_table_hash=candidate_table.table_hash,
            allow_fixture=allow_fixture,
        )

    def _canonical_phase_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        *,
        phase: PairPhase,
        known_first: tuple[int, int] | None,
        legal_set: set[tuple[int, int]] | None,
        source: str,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if first == second:
            raise ContractValidationError(f"duplicate coordinates are illegal for pair policy: {first}", owner="PairActionTableBuilder", source=source)
        if phase == "second_placement_known_first":
            if known_first is None:
                raise ContractValidationError("known_first is required for second-placement pair rows", owner="PairActionTableBuilder", source=source)
            if first != known_first:
                raise ContractValidationError(f"pair target first action {first} does not match known_first {known_first}", owner="PairActionTableBuilder", source=source)
            if legal_set is not None and second not in legal_set:
                raise ContractValidationError(f"pair target contains illegal second action: {second}", owner="PairActionTableBuilder", source=source)
            return (first, second)
        if legal_set is not None and (first not in legal_set or second not in legal_set):
            raise ContractValidationError(f"pair target contains illegal action pair: {(first, second)}", owner="PairActionTableBuilder", source=source)
        return _canonical_unordered_pair(first, second)

    def _fill_pairs(
        self,
        active_rows: Sequence[tuple[int, tuple[int, int]]],
        *,
        phase: PairPhase,
        known_first: tuple[int, int] | None,
        target_map: MappingPair,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        candidates = [qr for _idx, qr in active_rows]
        if phase == "second_placement_known_first":
            if known_first is None or known_first not in candidates:
                return []
            return [(known_first, second) for second in candidates if second != known_first and (known_first, second) not in target_map]
        fill: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                key = _canonical_unordered_pair(candidates[i], candidates[j])
                if key not in target_map:
                    fill.append(key)
        fill.sort(key=lambda pair: (_pair_distance(pair), pair))
        return fill

    def _possible_pair_count(
        self,
        active_rows: Sequence[tuple[int, tuple[int, int]]],
        *,
        phase: PairPhase,
        known_first: tuple[int, int] | None,
    ) -> int:
        count = len(active_rows)
        if phase == "second_placement_known_first":
            if known_first is None:
                return 0
            candidates = {qr for _idx, qr in active_rows}
            return max(0, count - 1) if known_first in candidates else 0
        return count * (count - 1) // 2


MappingPair = dict[tuple[tuple[int, int], tuple[int, int]], float]


def _pair_distance(pair: tuple[tuple[int, int], tuple[int, int]]) -> int:
    return _hex_distance(pair[0]) + _hex_distance(pair[1])


def _hex_distance(qr: tuple[int, int]) -> int:
    return max(abs(int(qr[0])), abs(int(qr[1])), abs(int(qr[0]) + int(qr[1])))


def _canonical_unordered_pair(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    a = _as_qr(first)
    b = _as_qr(second)
    if a == b:
        raise ContractValidationError(f"duplicate coordinates are illegal for pair policy: {a}", owner="PairActionTableBuilder")
    return (a, b) if a <= b else (b, a)


def _as_qr(item: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
    return (int(item[0]), int(item[1]))


def _unique_qr(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in items:
        qr = (int(q), int(r))
        if qr not in seen:
            seen.add(qr)
            out.append(qr)
    return out
