"""Canonical candidate table contract and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from hexorl.contracts.identity import ContractIdentity, ndarray_digest, readonly_array, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source
from hexorl.selfplay.records import BOARD_SIZE, PolicyTargetV2, action_to_board_index


CANDIDATE_SCHEMA_VERSION = 2
CANDIDATE_FEATURE_VERSION = 2
CANDIDATE_FEATURE_NAMES = (
    "q_norm",
    "r_norm",
    "s_norm",
    "hex_distance_norm",
    "inside_crop",
    "crop_q_norm",
    "crop_r_norm",
    "legal_rank_norm",
    "winning_cell",
    "forced_block",
    "outside_crop",
    "critical_cell",
)
CANDIDATE_FEATURES = len(CANDIDATE_FEATURE_NAMES)

CandidateMode = str
CandidateFeatureBlock = Callable[["CandidateFeatureContext"], Mapping[str, np.ndarray]]


@dataclass(frozen=True)
class CandidateDiagnostics:
    missing_mass: float = 0.0
    truncated: bool = False
    recall_top1: float = 1.0
    recall_top4: float = 1.0
    recall_top8: float = 1.0
    recall_winning_move: float = 1.0
    recall_forced_block: float = 1.0
    recall_two_placement_cover: float = 1.0
    discovery_top1: float = 1.0
    discovery_top4: float = 1.0
    discovery_top8: float = 1.0
    discovery_winning_move: float = 1.0
    discovery_forced_block: float = 1.0
    discovery_two_placement_cover: float = 1.0
    discovery_open_four: float = 1.0
    discovery_open_five: float = 1.0
    recall_open_four: float = 1.0
    recall_open_five: float = 1.0
    critical_count: int = 0
    critical_overflow_count: int = 0
    critical_overflow_examples: tuple[tuple[int, int], ...] = ()

    def hash_parts(self) -> tuple[object, ...]:
        return (
            round(float(self.missing_mass), 12),
            int(self.truncated),
            round(float(self.recall_top1), 12),
            round(float(self.recall_top4), 12),
            round(float(self.recall_top8), 12),
            round(float(self.recall_winning_move), 12),
            round(float(self.recall_forced_block), 12),
            round(float(self.recall_two_placement_cover), 12),
            round(float(self.discovery_top1), 12),
            round(float(self.discovery_top4), 12),
            round(float(self.discovery_top8), 12),
            round(float(self.discovery_winning_move), 12),
            round(float(self.discovery_forced_block), 12),
            round(float(self.discovery_two_placement_cover), 12),
            round(float(self.discovery_open_four), 12),
            round(float(self.discovery_open_five), 12),
            round(float(self.recall_open_four), 12),
            round(float(self.recall_open_five), 12),
            int(self.critical_count),
            int(self.critical_overflow_count),
            str(tuple((int(q), int(r)) for q, r in self.critical_overflow_examples)),
        )


@dataclass(frozen=True)
class CandidateFeatureContext:
    rows: np.ndarray
    dense_indices: np.ndarray
    mask: np.ndarray
    legal_rank: Mapping[tuple[int, int], int]
    winning_moves: frozenset[tuple[int, int]]
    forced_block_moves: frozenset[tuple[int, int]]
    cover_cells: frozenset[tuple[int, int]]
    open_four_cells: frozenset[tuple[int, int]]
    open_five_cells: frozenset[tuple[int, int]]
    critical_actions: frozenset[tuple[int, int]]
    offset_q: int
    offset_r: int


@dataclass(frozen=True)
class CandidateTable:
    rows: np.ndarray
    dense_indices: np.ndarray
    features: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    diagnostics: CandidateDiagnostics
    source: str = "rust"
    schema_version: int = CANDIDATE_SCHEMA_VERSION
    feature_version: int = CANDIDATE_FEATURE_VERSION
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="CandidateTable")
        rows = readonly_array(np.asarray(self.rows, dtype=np.int32).reshape(-1, 2), dtype=np.int32)
        dense_indices = readonly_array(np.asarray(self.dense_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
        features = readonly_array(np.asarray(self.features, dtype=np.float32), dtype=np.float32)
        mask = readonly_array(np.asarray(self.mask, dtype=np.bool_).reshape(-1), dtype=np.bool_)
        target = readonly_array(np.asarray(self.target, dtype=np.float32).reshape(-1), dtype=np.float32)
        width = int(rows.shape[0])
        if dense_indices.shape[0] != width or mask.shape[0] != width or target.shape[0] != width:
            raise ContractValidationError("candidate rows, dense indices, mask, and target length mismatch", owner="CandidateTable", source=source)
        if features.ndim != 2 or features.shape[0] != width:
            raise ContractValidationError("candidate feature rows mismatch candidate width", owner="CandidateTable", source=source)
        if features.shape[1] != len(self.feature_names):
            raise ContractValidationError("candidate feature width mismatch feature_names", owner="CandidateTable", source=source)
        active_seen: set[tuple[int, int]] = set()
        for row_idx, (q_raw, r_raw) in enumerate(rows.tolist()):
            if not bool(mask[row_idx]):
                continue
            qr = (int(q_raw), int(r_raw))
            if qr in active_seen:
                raise ContractValidationError(f"duplicate active candidate row {qr}", owner="CandidateTable", source=source)
            active_seen.add(qr)
            if int(dense_indices[row_idx]) < -1:
                raise ContractValidationError("candidate dense index must be -1 or non-negative", owner="CandidateTable", source=source)
        if float(np.sum(target[~mask])) != 0.0:
            raise ContractValidationError("inactive candidate target rows must be zero", owner="CandidateTable", source=source)
        if np.any(target < -1e-7):
            raise ContractValidationError("candidate target cannot contain negative mass", owner="CandidateTable", source=source)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "dense_indices", dense_indices)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "feature_version", int(self.feature_version))

    @property
    def qr(self) -> np.ndarray:
        return self.rows

    @property
    def indices(self) -> np.ndarray:
        return self.dense_indices

    @property
    def missing_mass(self) -> float:
        return float(self.diagnostics.missing_mass)

    @property
    def recall_top1(self) -> float:
        return float(self.diagnostics.recall_top1)

    @property
    def recall_top4(self) -> float:
        return float(self.diagnostics.recall_top4)

    @property
    def recall_top8(self) -> float:
        return float(self.diagnostics.recall_top8)

    @property
    def recall_winning_move(self) -> float:
        return float(self.diagnostics.recall_winning_move)

    @property
    def recall_forced_block(self) -> float:
        return float(self.diagnostics.recall_forced_block)

    @property
    def recall_two_placement_cover(self) -> float:
        return float(self.diagnostics.recall_two_placement_cover)

    @property
    def discovery_top1(self) -> float:
        return float(self.diagnostics.discovery_top1)

    @property
    def discovery_top4(self) -> float:
        return float(self.diagnostics.discovery_top4)

    @property
    def discovery_top8(self) -> float:
        return float(self.diagnostics.discovery_top8)

    @property
    def discovery_winning_move(self) -> float:
        return float(self.diagnostics.discovery_winning_move)

    @property
    def discovery_forced_block(self) -> float:
        return float(self.diagnostics.discovery_forced_block)

    @property
    def discovery_two_placement_cover(self) -> float:
        return float(self.diagnostics.discovery_two_placement_cover)

    @property
    def discovery_open_four(self) -> float:
        return float(self.diagnostics.discovery_open_four)

    @property
    def discovery_open_five(self) -> float:
        return float(self.diagnostics.discovery_open_five)

    @property
    def recall_open_four(self) -> float:
        return float(self.diagnostics.recall_open_four)

    @property
    def recall_open_five(self) -> float:
        return float(self.diagnostics.recall_open_five)

    @property
    def critical_count(self) -> int:
        return int(self.diagnostics.critical_count)

    @property
    def critical_overflow_count(self) -> int:
        return int(self.diagnostics.critical_overflow_count)

    @property
    def critical_overflow_examples(self) -> tuple[tuple[int, int], ...]:
        return self.diagnostics.critical_overflow_examples

    @property
    def table_hash(self) -> str:
        return stable_digest(
            (
                "CandidateTable",
                self.schema_version,
                self.feature_version,
                self.source,
                ",".join(self.feature_names),
                ndarray_digest(self.rows, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.dense_indices, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.features, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.mask, schema_version=self.schema_version, source=self.source),
                ndarray_digest(self.target, schema_version=self.schema_version, source=self.source),
                str(self.diagnostics.hash_parts()),
            )
        )

    @property
    def identity(self) -> ContractIdentity:
        return ContractIdentity("CandidateTable", self.schema_version, self.source, self.table_hash)

    def debug_payload(self) -> dict[str, object]:
        return {
            "contract": "CandidateTable",
            "schema_version": self.schema_version,
            "feature_version": self.feature_version,
            "source": self.source,
            "table_hash": self.table_hash,
            "candidate_count": int(np.count_nonzero(self.mask)),
            "storage_width": int(self.rows.shape[0]),
            "target_mass": float(np.sum(self.target)),
            "missing_mass": self.missing_mass,
            "critical_count": self.critical_count,
            "critical_overflow_count": self.critical_overflow_count,
            "critical_overflow_examples": list(self.critical_overflow_examples),
        }


class CandidateContractBuilder:
    def __init__(self, *, feature_blocks: Sequence[CandidateFeatureBlock] = ()) -> None:
        self._feature_blocks = tuple(feature_blocks)

    def build(
        self,
        legal_moves: Sequence[tuple[int, int]],
        policy_target_v2: PolicyTargetV2,
        *,
        offset_q: int,
        offset_r: int,
        budget: int,
        winning_moves: Sequence[tuple[int, int]] = (),
        forced_block_moves: Sequence[tuple[int, int]] = (),
        cover_cells: Sequence[tuple[int, int]] = (),
        open_four_cells: Sequence[tuple[int, int]] = (),
        open_five_cells: Sequence[tuple[int, int]] = (),
        critical_actions: Sequence[tuple[int, int]] = (),
        mode: CandidateMode = "protected",
        storage_width: int | None = None,
        source: str = "rust",
        allow_fixture: bool = False,
    ) -> CandidateTable:
        source = validate_source(source, allow_fixture=allow_fixture, owner="CandidateContractBuilder")
        winning_set = frozenset(_unique_qr(winning_moves))
        forced_set = frozenset(_unique_qr(forced_block_moves))
        cover_set = frozenset(_unique_qr(cover_cells))
        open_four_set = frozenset(_unique_qr(open_four_cells))
        open_five_set = frozenset(_unique_qr(open_five_cells))
        legal_list = _unique_qr((int(move[0]), int(move[1])) for move in legal_moves)
        legal_set = set(legal_list)
        target_map: dict[tuple[int, int], float] = {
            (int(q), int(r)): float(prob) for q, r, prob in policy_target_v2 if prob > 0.0
        }
        tactical_critical_set = (
            set(_unique_qr(critical_actions))
            | set(winning_set)
            | set(forced_set)
            | set(cover_set)
            | set(open_four_set)
            | set(open_five_set)
        )
        diagnostic_critical_set = set(target_map) | tactical_critical_set
        candidates = self.build_candidate_set(
            legal_list,
            policy_target_v2,
            budget,
            winning_moves=winning_set,
            forced_block_moves=forced_set,
            cover_cells=cover_set,
            open_four_cells=open_four_set,
            open_five_cells=open_five_set,
            critical_actions=critical_actions,
            mode=mode,
        )
        discovery_candidates = self.build_candidate_set(
            legal_list,
            policy_target_v2,
            budget,
            winning_moves=winning_set,
            forced_block_moves=forced_set,
            cover_cells=cover_set,
            open_four_cells=open_four_set,
            open_five_cells=open_five_set,
            critical_actions=critical_actions,
            mode="discovery",
        )
        critical_count = len(diagnostic_critical_set & legal_set)
        width = max(1, max(len(candidates), int(budget)) if storage_width is None else int(storage_width))
        overflowed_critical = [qr for qr in candidates[width:] if qr in diagnostic_critical_set]
        rows = np.zeros((width, 2), dtype=np.int32)
        dense_indices = np.full(width, -1, dtype=np.int64)
        features = np.zeros((width, CANDIDATE_FEATURES), dtype=np.float32)
        mask = np.zeros(width, dtype=np.bool_)
        target = np.zeros(width, dtype=np.float32)
        legal_rank = {qr: i for i, qr in enumerate(sorted(set(legal_list)))}
        target_sorted = sorted(policy_target_v2, key=lambda item: -float(item[2]))
        target_top1 = {(int(q), int(r)) for q, r, _ in target_sorted[:1]}
        target_top4 = {(int(q), int(r)) for q, r, _ in target_sorted[:4]}
        target_top8 = {(int(q), int(r)) for q, r, _ in target_sorted[:8]}
        candidate_set = set(candidates[:width])
        discovery_candidate_set = set(discovery_candidates)
        represented_mass = 0.0
        half = BOARD_SIZE // 2
        max_coord = float(max(half, 1))
        max_legal_rank = float(max(len(legal_rank) - 1, 1))
        for row, (q, r) in enumerate(candidates[:width]):
            dense = action_to_board_index(q, r, offset_q, offset_r)
            prob = target_map.get((q, r), 0.0)
            represented_mass += prob
            rows[row] = (q, r)
            dense_indices[row] = dense
            mask[row] = True
            target[row] = prob
            gi = q - int(offset_q)
            gj = r - int(offset_r)
            in_crop = 1.0 if dense >= 0 else 0.0
            dist = _hex_distance((q, r))
            features[row] = np.array(
                [
                    q / max_coord,
                    r / max_coord,
                    (q + r) / max_coord,
                    dist / max_coord,
                    in_crop,
                    ((gi - half) / max_coord) if dense >= 0 else 0.0,
                    ((gj - half) / max_coord) if dense >= 0 else 0.0,
                    legal_rank.get((q, r), 0) / max_legal_rank,
                    1.0 if (q, r) in winning_set else 0.0,
                    1.0 if (q, r) in forced_set else 0.0,
                    1.0 - in_crop,
                    1.0 if (q, r) in tactical_critical_set else 0.0,
                ],
                dtype=np.float32,
            )
        if represented_mass > 0.0:
            target /= represented_mass
        context = CandidateFeatureContext(
            rows=rows,
            dense_indices=dense_indices,
            mask=mask,
            legal_rank=legal_rank,
            winning_moves=winning_set,
            forced_block_moves=forced_set,
            cover_cells=cover_set,
            open_four_cells=open_four_set,
            open_five_cells=open_five_set,
            critical_actions=frozenset(_unique_qr(critical_actions)),
            offset_q=int(offset_q),
            offset_r=int(offset_r),
        )
        feature_names = list(CANDIDATE_FEATURE_NAMES)
        if self._feature_blocks:
            extensions: list[np.ndarray] = []
            for block in self._feature_blocks:
                block_result = dict(block(context))
                for name, values in sorted(block_result.items()):
                    arr = np.asarray(values, dtype=np.float32).reshape(width, -1)
                    extensions.append(arr)
                    if arr.shape[1] == 1:
                        feature_names.append(str(name))
                    else:
                        feature_names.extend(f"{name}_{idx}" for idx in range(arr.shape[1]))
            if extensions:
                features = np.concatenate([features, *extensions], axis=1)

        def recall(top: set[tuple[int, int]]) -> float:
            if not top:
                return 1.0
            return len(top & candidate_set) / float(len(top))

        def discovery_recall(top: set[tuple[int, int]]) -> float:
            if not top:
                return 1.0
            return len(top & discovery_candidate_set) / float(len(top))

        diagnostics = CandidateDiagnostics(
            missing_mass=max(0.0, 1.0 - represented_mass),
            truncated=len(candidates) > width,
            recall_top1=recall(target_top1),
            recall_top4=recall(target_top4),
            recall_top8=recall(target_top8),
            recall_winning_move=recall(set(winning_set)),
            recall_forced_block=recall(set(forced_set)),
            recall_two_placement_cover=recall(set(cover_set)),
            discovery_top1=discovery_recall(target_top1),
            discovery_top4=discovery_recall(target_top4),
            discovery_top8=discovery_recall(target_top8),
            discovery_winning_move=discovery_recall(set(winning_set)),
            discovery_forced_block=discovery_recall(set(forced_set)),
            discovery_two_placement_cover=discovery_recall(set(cover_set)),
            discovery_open_four=discovery_recall(set(open_four_set)),
            discovery_open_five=discovery_recall(set(open_five_set)),
            recall_open_four=recall(set(open_four_set)),
            recall_open_five=recall(set(open_five_set)),
            critical_count=critical_count,
            critical_overflow_count=len(overflowed_critical),
            critical_overflow_examples=tuple(overflowed_critical[:8]),
        )
        return CandidateTable(
            rows=rows,
            dense_indices=dense_indices,
            features=features,
            mask=mask,
            target=target,
            diagnostics=diagnostics,
            source=source,
            feature_names=tuple(feature_names),
            allow_fixture=allow_fixture,
        )

    def build_candidate_set(
        self,
        legal_moves: Sequence[tuple[int, int]],
        policy_target_v2: PolicyTargetV2,
        budget: int,
        *,
        winning_moves: Sequence[tuple[int, int]] = (),
        forced_block_moves: Sequence[tuple[int, int]] = (),
        cover_cells: Sequence[tuple[int, int]] = (),
        open_four_cells: Sequence[tuple[int, int]] = (),
        open_five_cells: Sequence[tuple[int, int]] = (),
        critical_actions: Sequence[tuple[int, int]] = (),
        mode: CandidateMode = "protected",
    ) -> list[tuple[int, int]]:
        budget = max(1, int(budget))
        if mode not in {"protected", "discovery"}:
            raise ContractValidationError(f"unsupported candidate mode: {mode!r}", owner="CandidateContractBuilder")
        target_actions = (
            _unique_qr((q, r) for q, r, prob in policy_target_v2 if prob > 0.0)
            if mode == "protected"
            else []
        )
        legal_set = set(_unique_qr((int(move[0]), int(move[1])) for move in legal_moves))
        tactical_actions = _unique_qr(
            list(winning_moves)
            + list(forced_block_moves)
            + list(cover_cells)
            + list(open_four_cells)
            + list(open_five_cells)
            + list(critical_actions)
        )
        tactical_actions = [qr for qr in tactical_actions if qr in legal_set]
        protected = _unique_qr(target_actions + tactical_actions)
        fill_pool = sorted(
            legal_set - set(protected),
            key=lambda qr: (_hex_distance(qr), qr[0], qr[1]),
        )
        slots = max(0, budget - len(protected))
        return protected + fill_pool[:slots]


def _hex_distance(qr: tuple[int, int]) -> int:
    return max(abs(int(qr[0])), abs(int(qr[1])), abs(int(qr[0]) + int(qr[1])))


def _unique_qr(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in items:
        qr = (int(q), int(r))
        if qr not in seen:
            seen.add(qr)
            out.append(qr)
    return out


def build_candidate_set(
    legal_moves: Sequence[tuple[int, int]],
    policy_target_v2: PolicyTargetV2,
    budget: int,
    **kwargs,
) -> list[tuple[int, int]]:
    return CandidateContractBuilder().build_candidate_set(legal_moves, policy_target_v2, budget, **kwargs)
