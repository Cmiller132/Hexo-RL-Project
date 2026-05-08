"""V1 pair-candidate scoring primitives.

This module owns only bounded candidate scoring helpers.  It deliberately does
not project pair scores back to single-cell action logits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


PAIR_SCORER_VERSION_V1 = "pair_scorer_v1"

PairCoord = tuple[int, int]
PairKey = tuple[PairCoord, PairCoord]


def _coord(value: Any, *, field_name: str = "cell") -> PairCoord:
    if isinstance(value, Mapping):
        if "cell" in value:
            return _coord(value["cell"], field_name=field_name)
        return (int(value["q"]), int(value["r"]))
    return (int(value[0]), int(value[1]))


def canonical_pair_key(first: Any, second: Any) -> PairKey:
    """Return the unordered canonical coordinate key for a two-cell V1 pair."""

    a = _coord(first, field_name="first")
    b = _coord(second, field_name="second")
    if a == b:
        raise ValueError(f"duplicate cells are illegal for V1 pair candidates: {a}")
    return (a, b) if a <= b else (b, a)


def hex_distance(first: PairCoord, second: PairCoord) -> int:
    dq = int(first[0]) - int(second[0])
    dr = int(first[1]) - int(second[1])
    return int(max(abs(dq), abs(dr), abs(dq + dr)))


def same_hex_line(first: PairCoord, second: PairCoord) -> bool:
    return (
        int(first[0]) == int(second[0])
        or int(first[1]) == int(second[1])
        or int(first[0]) + int(first[1]) == int(second[0]) + int(second[1])
    )


def same_origin_axis(first: PairCoord, second: PairCoord) -> bool:
    return (
        int(first[0]) == 0 and int(second[0]) == 0
        or int(first[1]) == 0 and int(second[1]) == 0
        or int(first[0]) + int(first[1]) == 0 and int(second[0]) + int(second[1]) == 0
    )


@dataclass(frozen=True)
class LegalRowIdentityV1:
    """Rust V1 legal-row identity as consumed by Python candidate builders."""

    row_id: int
    q: int
    r: int

    def __post_init__(self) -> None:
        row_id = int(self.row_id)
        if row_id < 0:
            raise ValueError("V1 legal row_id cannot be negative")
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "q", int(self.q))
        object.__setattr__(self, "r", int(self.r))

    @property
    def cell(self) -> PairCoord:
        return (self.q, self.r)


@dataclass(frozen=True)
class PairRowIdentityV1:
    """Canonical V1 pair row referencing start-of-turn legal-row IDs."""

    first_legal_row_id: int
    second_legal_row_id: int
    first: PairCoord
    second: PairCoord
    row_id: int | None = None
    pair_row_key: int | None = None

    def __post_init__(self) -> None:
        first_id = int(self.first_legal_row_id)
        second_id = int(self.second_legal_row_id)
        if first_id < 0 or second_id < 0:
            raise ValueError("V1 pair row legal IDs cannot be negative")
        if first_id == second_id:
            raise ValueError("V1 pair row legal IDs must be distinct")
        first = _coord(self.first, field_name="first")
        second = _coord(self.second, field_name="second")
        if first == second:
            raise ValueError(f"duplicate cells are illegal for V1 pair rows: {first}")
        if first_id > second_id:
            first_id, second_id = second_id, first_id
            first, second = second, first
        row_id = None if self.row_id is None or int(self.row_id) < 0 else int(self.row_id)
        pair_row_key = None if self.pair_row_key is None else int(self.pair_row_key)
        object.__setattr__(self, "first_legal_row_id", first_id)
        object.__setattr__(self, "second_legal_row_id", second_id)
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "pair_row_key", pair_row_key)

    @property
    def row_id_pair(self) -> tuple[int, int]:
        return (self.first_legal_row_id, self.second_legal_row_id)

    @property
    def pair_key(self) -> PairKey:
        return canonical_pair_key(self.first, self.second)


@dataclass(frozen=True)
class CheapPairFeatureWeightsV1:
    axial_distance: float = 0.0
    same_line: float = 0.0
    same_axis: float = 0.0


@dataclass(frozen=True)
class DirectPairRetrievalCandidateV1:
    identity: PairRowIdentityV1
    score: float
    source_rank: int

    def __post_init__(self) -> None:
        score = float(self.score)
        if not np.isfinite(score):
            raise ValueError("direct pair retrieval score must be finite")
        rank = int(self.source_rank)
        if rank < 0:
            raise ValueError("direct pair retrieval source_rank cannot be negative")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "source_rank", rank)

    @property
    def pair_key(self) -> PairKey:
        return self.identity.pair_key

    @property
    def first_legal_row_id(self) -> int:
        return self.identity.first_legal_row_id

    @property
    def second_legal_row_id(self) -> int:
        return self.identity.second_legal_row_id


@dataclass(frozen=True)
class DirectPairRetrievalResultV1:
    candidates: tuple[DirectPairRetrievalCandidateV1, ...]
    scored_pair_count: int
    block_size: int


def parse_legal_rows_v1(legal_rows: Sequence[Any] | Mapping[str, Any]) -> tuple[LegalRowIdentityV1, ...]:
    rows_raw: Any
    if isinstance(legal_rows, Mapping) and "rows" in legal_rows:
        rows_raw = legal_rows["rows"]
    else:
        rows_raw = legal_rows
    rows: list[LegalRowIdentityV1] = []
    seen_ids: set[int] = set()
    seen_cells: set[PairCoord] = set()
    for row in rows_raw:
        parsed = _parse_one_legal_row(row)
        if parsed.row_id in seen_ids:
            raise ValueError(f"duplicate V1 legal row_id: {parsed.row_id}")
        if parsed.cell in seen_cells:
            raise ValueError(f"duplicate V1 legal cell: {parsed.cell}")
        seen_ids.add(parsed.row_id)
        seen_cells.add(parsed.cell)
        rows.append(parsed)
    return tuple(rows)


def parse_pair_rows_v1(pair_rows: Sequence[Any] | Mapping[str, Any] | None) -> tuple[PairRowIdentityV1, ...]:
    if pair_rows is None:
        return ()
    rows_raw: Any
    if isinstance(pair_rows, Mapping) and "rows" in pair_rows:
        rows_raw = pair_rows["rows"]
    else:
        rows_raw = pair_rows
    rows: list[PairRowIdentityV1] = []
    seen: set[tuple[int, int]] = set()
    for row in rows_raw:
        parsed = _parse_one_pair_row(row)
        if parsed.row_id_pair in seen:
            raise ValueError(f"duplicate V1 pair row identity: {parsed.row_id_pair}")
        seen.add(parsed.row_id_pair)
        rows.append(parsed)
    return tuple(rows)


def pair_identity_from_legal_row_ids(
    first_legal_row_id: int,
    second_legal_row_id: int,
    legal_rows_by_id: Mapping[int, LegalRowIdentityV1],
    *,
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1] | None = None,
) -> PairRowIdentityV1:
    first_id = int(first_legal_row_id)
    second_id = int(second_legal_row_id)
    if first_id == second_id:
        raise ValueError("V1 pair identity requires distinct legal-row IDs")
    if first_id > second_id:
        first_id, second_id = second_id, first_id
    key = (first_id, second_id)
    if pair_rows_by_id_pair is not None and key in pair_rows_by_id_pair:
        return pair_rows_by_id_pair[key]
    try:
        first = legal_rows_by_id[first_id]
        second = legal_rows_by_id[second_id]
    except KeyError as exc:
        raise ValueError(f"V1 pair references unknown legal-row ID: {exc.args[0]}") from exc
    return PairRowIdentityV1(
        first_legal_row_id=first_id,
        second_legal_row_id=second_id,
        first=first.cell,
        second=second.cell,
    )


def direct_pair_retrieval_v1(
    legal_rows: Sequence[Any] | Mapping[str, Any],
    legal_cell_embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    top_k: int,
    block_size: int = 512,
    u_projection: np.ndarray | Sequence[Sequence[float]] | None = None,
    v_projection: np.ndarray | Sequence[Sequence[float]] | None = None,
    feature_weights: CheapPairFeatureWeightsV1 | None = None,
) -> DirectPairRetrievalResultV1:
    """Return exact top-k direct-retrieval pair candidates.

    The implementation is blockwise over first-row ranges, but it scores every
    valid unordered pair exactly.  Ties at a block boundary are retained and
    resolved only in the final deterministic sort.
    """

    rows = parse_legal_rows_v1(legal_rows)
    embeddings = np.asarray(legal_cell_embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("legal_cell_embeddings must be a 2D array")
    n = len(rows)
    if embeddings.shape[0] != n:
        raise ValueError(
            f"legal_cell_embeddings row count {embeddings.shape[0]} does not match legal rows {n}"
        )
    k = int(top_k)
    block = max(1, int(block_size))
    scored_pair_count = n * max(0, n - 1) // 2
    if k <= 0 or n < 2:
        return DirectPairRetrievalResultV1(candidates=(), scored_pair_count=scored_pair_count, block_size=block)

    u = _project_embeddings(embeddings, u_projection, projection_name="u_projection")
    v = _project_embeddings(embeddings, v_projection, projection_name="v_projection")
    if u.shape != v.shape:
        raise ValueError(f"direct retrieval projection shape mismatch: u={u.shape}, v={v.shape}")
    weights = feature_weights or CheapPairFeatureWeightsV1()
    coords = np.asarray([row.cell for row in rows], dtype=np.int32)
    legal_by_id = {row.row_id: row for row in rows}
    keep_per_block = min(k, scored_pair_count)
    collected: list[DirectPairRetrievalCandidateV1] = []

    for start in range(0, n, block):
        stop = min(start + block, n)
        scores = 0.5 * (u[start:stop] @ v.T + v[start:stop] @ u.T)
        scores = scores.astype(np.float64, copy=False)
        _add_cheap_pair_features(scores, coords[start:stop], coords, weights)
        local_i = np.arange(start, stop, dtype=np.int64)[:, None]
        global_j = np.arange(n, dtype=np.int64)[None, :]
        scores[global_j <= local_i] = -np.inf
        flat = scores.ravel()
        finite = np.isfinite(flat)
        valid_count = int(finite.sum())
        if valid_count <= 0:
            continue
        if valid_count <= keep_per_block:
            candidate_indices = np.flatnonzero(finite)
        else:
            finite_scores = flat[finite]
            threshold = float(np.partition(finite_scores, -keep_per_block)[-keep_per_block])
            candidate_indices = np.flatnonzero(finite & (flat >= threshold))
        for flat_idx in candidate_indices.tolist():
            score = float(flat[flat_idx])
            if not np.isfinite(score):
                continue
            block_i, second_idx = divmod(int(flat_idx), n)
            first_idx = start + block_i
            identity = pair_identity_from_legal_row_ids(
                rows[first_idx].row_id,
                rows[second_idx].row_id,
                legal_by_id,
            )
            collected.append(
                DirectPairRetrievalCandidateV1(identity=identity, score=score, source_rank=0)
            )

    dedup: dict[tuple[int, int], DirectPairRetrievalCandidateV1] = {}
    for candidate in collected:
        key = candidate.identity.row_id_pair
        current = dedup.get(key)
        if current is None or _direct_candidate_sort_key(candidate) < _direct_candidate_sort_key(current):
            dedup[key] = candidate
    ranked = sorted(dedup.values(), key=_direct_candidate_sort_key)[:keep_per_block]
    ranked = [
        DirectPairRetrievalCandidateV1(
            identity=candidate.identity,
            score=candidate.score,
            source_rank=rank,
        )
        for rank, candidate in enumerate(ranked)
    ]
    return DirectPairRetrievalResultV1(
        candidates=tuple(ranked),
        scored_pair_count=scored_pair_count,
        block_size=block,
    )


def _parse_one_legal_row(row: Any) -> LegalRowIdentityV1:
    if isinstance(row, LegalRowIdentityV1):
        return row
    if isinstance(row, Mapping):
        row_id = row.get("row_id", row.get("id"))
        if row_id is None:
            raise ValueError("V1 legal row mapping is missing row_id")
        if "cell" in row:
            q, r = _coord(row["cell"])
        else:
            q, r = int(row["q"]), int(row["r"])
        return LegalRowIdentityV1(row_id=int(row_id), q=q, r=r)
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        if len(row) < 3:
            raise ValueError("V1 legal row tuple must contain row_id, q, r")
        return LegalRowIdentityV1(row_id=int(row[0]), q=int(row[1]), r=int(row[2]))
    return LegalRowIdentityV1(
        row_id=int(getattr(row, "row_id")),
        q=int(getattr(row, "q")),
        r=int(getattr(row, "r")),
    )


def _parse_one_pair_row(row: Any) -> PairRowIdentityV1:
    if isinstance(row, PairRowIdentityV1):
        return row
    if isinstance(row, Mapping):
        row_id = row.get("row_id")
        first_id = row.get("first_legal_row_id")
        second_id = row.get("second_legal_row_id")
        if first_id is None or second_id is None:
            raise ValueError("V1 pair row mapping is missing legal-row IDs")
        if "first" in row:
            first = _coord(row["first"], field_name="first")
        else:
            first = (int(row["first_q"]), int(row["first_r"]))
        if "second" in row:
            second = _coord(row["second"], field_name="second")
        else:
            second = (int(row["second_q"]), int(row["second_r"]))
        raw_pair_key = row.get("pair_row_key", row.get("pair_key_u64"))
        if raw_pair_key is None and isinstance(row.get("pair_key"), int):
            raw_pair_key = row.get("pair_key")
        return PairRowIdentityV1(
            row_id=None if row_id is None else int(row_id),
            first_legal_row_id=int(first_id),
            second_legal_row_id=int(second_id),
            first=first,
            second=second,
            pair_row_key=None if raw_pair_key is None else int(raw_pair_key),
        )
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
        if len(row) < 8:
            raise ValueError("V1 pair row tuple must contain 8 PyO3 row fields")
        return PairRowIdentityV1(
            row_id=int(row[0]),
            first_legal_row_id=int(row[1]),
            second_legal_row_id=int(row[2]),
            first=(int(row[3]), int(row[4])),
            second=(int(row[5]), int(row[6])),
            pair_row_key=int(row[7]),
        )
    return PairRowIdentityV1(
        row_id=getattr(row, "row_id", None),
        first_legal_row_id=int(getattr(row, "first_legal_row_id")),
        second_legal_row_id=int(getattr(row, "second_legal_row_id")),
        first=_coord(getattr(row, "first"), field_name="first"),
        second=_coord(getattr(row, "second"), field_name="second"),
        pair_row_key=getattr(row, "pair_row_key", getattr(row, "pair_key", None)),
    )


def _project_embeddings(
    embeddings: np.ndarray,
    projection: np.ndarray | Sequence[Sequence[float]] | None,
    *,
    projection_name: str,
) -> np.ndarray:
    if projection is None:
        return embeddings.astype(np.float32, copy=False)
    matrix = np.asarray(projection, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{projection_name} must be a 2D array")
    if embeddings.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"{projection_name} input width {matrix.shape[0]} does not match embeddings width {embeddings.shape[1]}"
        )
    return embeddings @ matrix


def _add_cheap_pair_features(
    scores: np.ndarray,
    first_coords: np.ndarray,
    all_coords: np.ndarray,
    weights: CheapPairFeatureWeightsV1,
) -> None:
    if (
        float(weights.axial_distance) == 0.0
        and float(weights.same_line) == 0.0
        and float(weights.same_axis) == 0.0
    ):
        return
    q1 = first_coords[:, 0:1].astype(np.int64)
    r1 = first_coords[:, 1:2].astype(np.int64)
    q2 = all_coords[None, :, 0].astype(np.int64)
    r2 = all_coords[None, :, 1].astype(np.int64)
    dq = q1 - q2
    dr = r1 - r2
    dist = np.maximum.reduce((np.abs(dq), np.abs(dr), np.abs(dq + dr))).astype(np.float64)
    same_line = ((q1 == q2) | (r1 == r2) | ((q1 + r1) == (q2 + r2))).astype(np.float64)
    same_axis = (
        ((q1 == 0) & (q2 == 0))
        | ((r1 == 0) & (r2 == 0))
        | (((q1 + r1) == 0) & ((q2 + r2) == 0))
    ).astype(np.float64)
    scores += float(weights.axial_distance) * dist
    scores += float(weights.same_line) * same_line
    scores += float(weights.same_axis) * same_axis


def _direct_candidate_sort_key(candidate: DirectPairRetrievalCandidateV1) -> tuple[float, int, int, PairKey]:
    return (
        -float(candidate.score),
        candidate.first_legal_row_id,
        candidate.second_legal_row_id,
        candidate.pair_key,
    )


__all__ = [
    "PAIR_SCORER_VERSION_V1",
    "CheapPairFeatureWeightsV1",
    "DirectPairRetrievalCandidateV1",
    "DirectPairRetrievalResultV1",
    "LegalRowIdentityV1",
    "PairCoord",
    "PairKey",
    "PairRowIdentityV1",
    "canonical_pair_key",
    "direct_pair_retrieval_v1",
    "hex_distance",
    "pair_identity_from_legal_row_ids",
    "parse_legal_rows_v1",
    "parse_pair_rows_v1",
    "same_hex_line",
    "same_origin_axis",
]
