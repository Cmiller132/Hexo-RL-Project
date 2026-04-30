"""Canonical graph semantic contract builder for Hexo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import itertools
from typing import Iterable, Sequence

import numpy as np

from hexorl.engine.tactical import scan_tactical_oracle_from_history
from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.history import MoveHistory, turn_state_after
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.contracts.symmetry import (
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
    transform_qr,
)
from hexorl.engine.history import game_from_history
from hexorl.engine.legal import legal_rows_from_history


GRAPH_SCHEMA_VERSION = 2
GRAPH_FEATURE_DIM = 48
RELATION_SCHEMA_VERSION = 1
PAIR_CHUNK_LIMIT = 4096
GRAPH_IPC_TOKEN_CAPACITY = 768
GRAPH_IPC_ACTION_CAPACITY = 768
GRAPH_IPC_PAIR_CAPACITY = PAIR_CHUNK_LIMIT
GRAPH_CAPACITY_STRATEGY = "preserve_legal_stone_tactical_rows_fail_or_chunk_context"
HEX_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
NEIGHBOR_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)
WIN_LENGTH = 6


class GraphTokenType(IntEnum):
    STATE = 0
    TURN = 1
    PLAYER = 2
    STONE = 3
    LEGAL = 4
    HOT_CELL = 5
    WINDOW6 = 6
    LINE = 7
    COVER_SET = 8
    COMPONENT = 9
    PAIR_ACTION = 10


class RelationType(IntEnum):
    NONE = 0
    DISTANCE_BUCKET = 1
    DIRECTION_BUCKET = 2
    SAME_AXIS = 3
    SAME_LINE = 4
    SAME_WINDOW6 = 5
    STONE_IN_WINDOW6 = 6
    LEGAL_IN_WINDOW6 = 7
    LEGAL_IN_COVER_SET = 8
    WINDOW6_TO_COVER_SET = 9
    LINE_TO_WINDOW6 = 10
    LEGAL_TO_PAIR_ACTION = 11
    PAIR_COVERS_THREAT_SET = 12
    SAME_COMPONENT = 13
    AGE_ORDER_BUCKET = 14
    RECENT_MOVE_RELATION = 15
    FIRST_SECOND_PAIR_RELATION = 16
    D6_ORBIT_RELATION = 17


@dataclass(frozen=True)
class GraphSemanticContract:
    """Single-position graph/action contract.

    Padding is added by :func:`collate_graph_batches`; individual batches store
    only real tokens/actions so tests can assert exact legal-action coverage.
    """

    token_features: np.ndarray
    token_type: np.ndarray
    token_qr: np.ndarray
    token_mask: np.ndarray
    legal_token_indices: np.ndarray
    legal_qr: np.ndarray
    legal_mask: np.ndarray
    pair_token_indices: np.ndarray
    pair_first_indices: np.ndarray
    pair_second_indices: np.ndarray
    relation_bias: np.ndarray
    relation_type: np.ndarray
    policy_target: np.ndarray
    opp_legal_qr: np.ndarray
    opp_legal_mask: np.ndarray
    opp_policy_target: np.ndarray
    pair_first_policy_target: np.ndarray
    pair_policy_target: np.ndarray
    tactical_target: np.ndarray
    placements_remaining: int
    current_player: int
    schema_version: int = GRAPH_SCHEMA_VERSION
    relation_schema_version: int = RELATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        array_fields = {
            "token_features": (np.float32, None),
            "token_type": (np.int64, None),
            "token_qr": (np.int32, None),
            "token_mask": (np.bool_, None),
            "legal_token_indices": (np.int64, None),
            "legal_qr": (np.int32, None),
            "legal_mask": (np.bool_, None),
            "pair_token_indices": (np.int64, None),
            "pair_first_indices": (np.int64, None),
            "pair_second_indices": (np.int64, None),
            "relation_bias": (np.float32, None),
            "relation_type": (np.int64, None),
            "policy_target": (np.float32, None),
            "opp_legal_qr": (np.int32, None),
            "opp_legal_mask": (np.bool_, None),
            "opp_policy_target": (np.float32, None),
            "pair_first_policy_target": (np.float32, None),
            "pair_policy_target": (np.float32, None),
            "tactical_target": (np.float32, None),
        }
        for name, (dtype, _shape) in array_fields.items():
            arr = np.array(getattr(self, name), dtype=dtype, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "placements_remaining", int(self.placements_remaining))
        object.__setattr__(self, "current_player", int(self.current_player))
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "relation_schema_version", int(self.relation_schema_version))


@dataclass(frozen=True)
class GraphCapacityReport:
    token_count: int
    legal_count: int
    opp_legal_count: int
    pair_count: int
    max_tokens: int = GRAPH_IPC_TOKEN_CAPACITY
    max_actions: int = GRAPH_IPC_ACTION_CAPACITY
    max_pairs: int = GRAPH_IPC_PAIR_CAPACITY
    strategy: str = GRAPH_CAPACITY_STRATEGY

    @property
    def fits_ipc(self) -> bool:
        return (
            self.token_count <= self.max_tokens
            and self.legal_count <= self.max_actions
            and self.opp_legal_count <= self.max_actions
            and self.pair_count <= self.max_pairs
        )

    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.token_count > self.max_tokens:
            failures.append("graph_token_capacity")
        if self.legal_count > self.max_actions:
            failures.append("graph_legal_action_capacity")
        if self.opp_legal_count > self.max_actions:
            failures.append("graph_opp_legal_action_capacity")
        if self.pair_count > self.max_pairs:
            failures.append("graph_pair_chunk_capacity")
        return tuple(failures)


def graph_capacity_report(graph_batch: GraphSemanticContract) -> GraphCapacityReport:
    return GraphCapacityReport(
        token_count=int(np.asarray(graph_batch.token_features).shape[0]),
        legal_count=int(np.asarray(graph_batch.legal_qr).shape[0]),
        opp_legal_count=int(np.asarray(graph_batch.opp_legal_qr).shape[0]),
        pair_count=int(np.asarray(graph_batch.pair_token_indices).shape[0]),
    )


def validate_graph_ipc_capacity(graph_batch: GraphSemanticContract) -> GraphCapacityReport:
    """Validate the graph IPC capacity policy without dropping semantic rows."""
    report = graph_capacity_report(graph_batch)
    if not report.fits_ipc:
        raise ValueError(
            "global graph IPC capacity exceeded; "
            f"tokens={report.token_count}/{report.max_tokens}, "
            f"legal={report.legal_count}/{report.max_actions}, "
            f"opp_legal={report.opp_legal_count}/{report.max_actions}, "
            f"pairs={report.pair_count}/{report.max_pairs}. "
            f"strategy={report.strategy}; lower max-game length, bucket/microbatch this position, "
            "or score pair rows through chunks. Semantic legal, stone, and tactical rows were not dropped."
        )
    return report


def hex_distance(a: tuple[int, int], b: tuple[int, int] = (0, 0)) -> int:
    dq = int(a[0]) - int(b[0])
    dr = int(a[1]) - int(b[1])
    return max(abs(dq), abs(dr), abs(dq + dr))


def parse_history(history: bytes) -> list[tuple[int, int, int]]:
    return list(MoveHistory.decode(history, source="rust").rows)


def _unique_qr(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in items:
        qr = (int(q), int(r))
        if qr in seen:
            continue
        seen.add(qr)
        out.append(qr)
    return out


def current_turn_state(moves: Sequence[tuple[int, int, int]]) -> tuple[int, int]:
    return turn_state_after(moves)


def legal_moves_for_stones(stones: dict[tuple[int, int], int], radius: int = 8) -> list[tuple[int, int]]:
    if int(radius) != 8:
        raise ValueError("global graph legal rows must use the Rust placement radius 8")
    rows = legal_rows_from_history(
        MoveHistory.from_rows(
            [(player, q, r) for (q, r), player in stones.items()],
            source="fixture",
            allow_fixture=True,
        ),
        near_radius=radius,
        constrain_threats=False,
    )
    return [(int(q), int(r)) for q, r in rows.tolist()]


def _line_id(qr: tuple[int, int], axis: int) -> int:
    q, r = qr
    if axis == 0:
        return r
    if axis == 1:
        return q
    return q + r


def _window_cells(start: tuple[int, int], axis: int) -> list[tuple[int, int]]:
    dq, dr = HEX_DIRECTIONS[axis]
    q, r = start
    return [(q + dq * i, r + dr * i) for i in range(WIN_LENGTH)]


def _active_windows(stones: dict[tuple[int, int], int], legal: Sequence[tuple[int, int]]) -> list[tuple[int, tuple[int, int], int, int, list[tuple[int, int]]]]:
    interesting = set(stones) | set(legal)
    windows: dict[tuple[int, tuple[int, int]], tuple[int, int, list[tuple[int, int]]]] = {}
    for q, r in interesting:
        for axis, (dq, dr) in enumerate(HEX_DIRECTIONS):
            for back in range(WIN_LENGTH):
                start = (q - dq * back, r - dr * back)
                cells = _window_cells(start, axis)
                own = sum(1 for c in cells if stones.get(c) == 0)
                opp = sum(1 for c in cells if stones.get(c) == 1)
                if own + opp == 0:
                    continue
                if own > 0 and opp > 0:
                    continue
                windows[(axis, start)] = (own, opp, cells)
    out = []
    for (axis, start), (own, opp, cells) in sorted(windows.items(), key=lambda item: (item[0][0], item[0][1])):
        out.append((axis, start, own, opp, [c for c in cells if c not in stones]))
    return out


def _component_ids(stones: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    remaining = set(stones)
    comp: dict[tuple[int, int], int] = {}
    cid = 0
    while remaining:
        root = remaining.pop()
        stack = [root]
        comp[root] = cid
        while stack:
            q, r = stack.pop()
            for dq, dr in NEIGHBOR_DIRECTIONS:
                nxt = (q + dq, r + dr)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp[nxt] = cid
                    stack.append(nxt)
        cid += 1
    return comp


def _nearest_distance(
    qr: tuple[int, int],
    cells: Iterable[tuple[int, int]],
    *,
    default: int = 64,
) -> int:
    best = default
    for cell in cells:
        best = min(best, hex_distance(qr, cell))
    return best


def _line_stats(
    qr: tuple[int, int],
    axis: int,
    stones: dict[tuple[int, int], int],
    player: int,
) -> tuple[int, int, int, int]:
    """Return own/opponent counts and longest contiguous runs on one line."""
    line = _line_id(qr, axis)
    own = opp = 0
    own_run = opp_run = 0
    for cell, owner in stones.items():
        if _line_id(cell, axis) != line:
            continue
        if owner == player:
            own += 1
        else:
            opp += 1
    for owner in (player, 1 - player):
        best = 0
        for cell, cell_owner in stones.items():
            if cell_owner != owner or _line_id(cell, axis) != line:
                continue
            dq, dr = HEX_DIRECTIONS[axis]
            prev = (cell[0] - dq, cell[1] - dr)
            if stones.get(prev) == owner:
                continue
            run = 1
            nxt = (cell[0] + dq, cell[1] + dr)
            while stones.get(nxt) == owner:
                run += 1
                nxt = (nxt[0] + dq, nxt[1] + dr)
            best = max(best, run)
        if owner == player:
            own_run = best
        else:
            opp_run = best
    return own, opp, own_run, opp_run


def _features(
    token_type: GraphTokenType,
    qr: tuple[int, int],
    *,
    current_player: int,
    placements_remaining: int,
    owner: int | None = None,
    age: int = 0,
    axis: int = -1,
    count_a: int = 0,
    count_b: int = 0,
    legal_count: int = 0,
    stone_count: int = 0,
    own_stone_count: int = 0,
    opp_stone_count: int = 0,
    nearest_own: int = 64,
    nearest_opp: int = 64,
    nearest_any: int = 64,
    window_empty_count: int = 0,
    window_legal_count: int = 0,
    own_line_count: int = 0,
    opp_line_count: int = 0,
    own_line_run: int = 0,
    opp_line_run: int = 0,
    cover_size: int = 0,
    cover_memberships: int = 0,
    component_size: int = 0,
    pair_distance: int = 0,
    pair_reaches_win: bool = False,
    pair_blocks_threat: bool = False,
    is_win_now: bool = False,
    is_forced_block: bool = False,
    is_open_four: bool = False,
    is_open_five: bool = False,
    is_cover_cell: bool = False,
    hot_window_count: int = 0,
) -> np.ndarray:
    q, r = qr
    dist = hex_distance(qr)
    out = np.zeros(GRAPH_FEATURE_DIM, dtype=np.float32)
    out[0] = float(token_type) / max(float(max(GraphTokenType)), 1.0)
    out[1] = q / 64.0
    out[2] = r / 64.0
    out[3] = (q + r) / 64.0
    out[4] = min(dist, 64) / 64.0
    out[5] = float(current_player)
    out[6] = float(placements_remaining) / 2.0
    out[7] = -1.0 if owner is None else (1.0 if owner == current_player else 0.0)
    out[8] = min(age, 64) / 64.0
    out[9] = float(axis + 1) / 4.0
    out[10] = count_a / 6.0
    out[11] = count_b / 6.0
    out[12] = min(legal_count, 2048) / 2048.0
    out[13] = min(stone_count, 512) / 512.0
    out[14] = min(own_stone_count, 256) / 256.0
    out[15] = min(opp_stone_count, 256) / 256.0
    out[16] = min(nearest_own, 64) / 64.0
    out[17] = min(nearest_opp, 64) / 64.0
    out[18] = min(nearest_any, 64) / 64.0
    out[19] = min(window_empty_count, 6) / 6.0
    out[20] = min(window_legal_count, 6) / 6.0
    out[21] = min(own_line_count, 64) / 64.0
    out[22] = min(opp_line_count, 64) / 64.0
    out[23] = min(own_line_run, 6) / 6.0
    out[24] = min(opp_line_run, 6) / 6.0
    out[25] = min(cover_size, 16) / 16.0
    out[26] = min(cover_memberships, 16) / 16.0
    out[27] = min(component_size, 128) / 128.0
    out[28] = min(pair_distance, 64) / 64.0
    out[29] = 1.0 if pair_reaches_win else 0.0
    out[30] = 1.0 if pair_blocks_threat else 0.0
    out[31] = 1.0 if is_win_now else 0.0
    out[32] = 1.0 if is_forced_block else 0.0
    out[33] = 1.0 if is_open_four else 0.0
    out[34] = 1.0 if is_open_five else 0.0
    out[35] = 1.0 if is_cover_cell else 0.0
    out[36] = min(hot_window_count, 16) / 16.0
    return out


class GraphSemanticBuilder:
    """Canonical owner for graph token, relation, legal-link, and pair-link semantics."""

    def build(
        self,
        history: bytes,
        *,
        policy_target: Sequence[tuple[int, int, float]] = (),
        opp_legal_moves: Sequence[tuple[int, int]] | None = None,
        opp_policy_target: Sequence[tuple[int, int, float]] = (),
        pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]] = (),
        radius: int = 8,
        max_pair_rows: int = PAIR_CHUNK_LIMIT,
        allow_pair_truncation: bool = False,
        include_pair_rows: bool = False,
    ) -> GraphSemanticContract:
        return build_graph_semantic_from_history(
            history,
            policy_target=policy_target,
            opp_legal_moves=opp_legal_moves,
            opp_policy_target=opp_policy_target,
            pair_policy_target=pair_policy_target,
            radius=radius,
            max_pair_rows=max_pair_rows,
            allow_pair_truncation=allow_pair_truncation,
            include_pair_rows=include_pair_rows,
        )


def build_graph_semantic_from_history(
    history: bytes,
    *,
    policy_target: Sequence[tuple[int, int, float]] = (),
    opp_legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_policy_target: Sequence[tuple[int, int, float]] = (),
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]] = (),
    radius: int = 8,
    max_pair_rows: int = PAIR_CHUNK_LIMIT,
    allow_pair_truncation: bool = False,
    include_pair_rows: bool = False,
) -> GraphSemanticContract:
    if int(radius) != 8:
        raise ValueError("global graph legal rows must preserve all Rust-legal moves; radius must be 8")
    moves = parse_history(history)
    stones = {(q, r): player for player, q, r in moves}
    current_player, placements_remaining = current_turn_state(moves)
    engine_state = _engine_state_from_history(history)
    if engine_state is not None:
        legal, current_player, placements_remaining = engine_state
    else:
        legal = legal_moves_for_stones(stones, radius=radius)
    legal_index = {qr: i for i, qr in enumerate(legal)}
    windows = _active_windows(stones, legal)
    comp = _component_ids(stones)
    oracle = scan_tactical_oracle_from_history(
        history,
        legal,
        near_radius=8,
    )

    token_features: list[np.ndarray] = []
    token_type: list[int] = []
    token_qr: list[tuple[int, int]] = []
    token_axis: list[int] = []
    token_age: list[int] = []
    memberships: dict[int, set[tuple[int, int]]] = {}
    own_stones = [qr for qr, owner in stones.items() if owner == current_player]
    opp_stones = [qr for qr, owner in stones.items() if owner != current_player]
    win_now_cells = {(int(q), int(r)) for q, r in getattr(oracle, "win_now_cells", ())}
    forced_cells = {(int(q), int(r)) for q, r in getattr(oracle, "forced_block_cells", ())}
    open_four_cells = {(int(q), int(r)) for q, r in getattr(oracle, "open_four_cells", ())}
    open_five_cells = {(int(q), int(r)) for q, r in getattr(oracle, "open_five_cells", ())}
    cover_cells = {(int(q), int(r)) for q, r in getattr(oracle, "cover_cells", ())}
    hot_window_count_by_cell: dict[tuple[int, int], int] = {}
    for cell_group in (
        win_now_cells,
        forced_cells,
        open_four_cells,
        open_five_cells,
        cover_cells,
    ):
        for cell in cell_group:
            hot_window_count_by_cell[cell] = hot_window_count_by_cell.get(cell, 0) + 1

    def add(tt: GraphTokenType, qr: tuple[int, int], **kwargs) -> int:
        idx = len(token_type)
        token_type.append(int(tt))
        token_qr.append(qr)
        token_axis.append(int(kwargs.get("axis", -1)))
        token_age.append(int(kwargs.get("age", -1)) if tt == GraphTokenType.STONE else -1)
        token_features.append(
            _features(
                tt,
                qr,
                current_player=current_player,
                placements_remaining=placements_remaining,
                legal_count=len(legal),
                stone_count=len(stones),
                own_stone_count=len(own_stones),
                opp_stone_count=len(opp_stones),
                nearest_own=_nearest_distance(qr, own_stones),
                nearest_opp=_nearest_distance(qr, opp_stones),
                nearest_any=_nearest_distance(qr, stones.keys()),
                is_win_now=qr in win_now_cells,
                is_forced_block=qr in forced_cells,
                is_open_four=qr in open_four_cells,
                is_open_five=qr in open_five_cells,
                is_cover_cell=qr in cover_cells,
                hot_window_count=hot_window_count_by_cell.get(qr, 0),
                **kwargs,
            )
        )
        return idx

    add(GraphTokenType.STATE, (0, 0))
    add(GraphTokenType.TURN, (0, 0))
    add(GraphTokenType.PLAYER, (0, 0), owner=current_player)
    add(GraphTokenType.PLAYER, (0, 0), owner=1 - current_player)

    stone_token: dict[tuple[int, int], int] = {}
    for age, (player, q, r) in enumerate(moves):
        stone_token[(q, r)] = add(GraphTokenType.STONE, (q, r), owner=player, age=age)

    legal_token_indices: list[int] = []
    for qr in legal:
        legal_token_indices.append(add(GraphTokenType.LEGAL, qr))

    hot_cells: set[tuple[int, int]] = {
        cell
        for cell in (
            win_now_cells
            | forced_cells
            | open_four_cells
            | open_five_cells
            | cover_cells
        )
        if cell in legal_index
    }
    for qr in sorted(hot_cells):
        add(GraphTokenType.HOT_CELL, qr)

    window_token_by_key: dict[tuple[int, tuple[int, int]], int] = {}
    for axis, start, own, opp, empties in windows:
        center = _window_cells(start, axis)[WIN_LENGTH // 2]
        idx = add(
            GraphTokenType.WINDOW6,
            center,
            axis=axis,
            count_a=own,
            count_b=opp,
            window_empty_count=len(empties),
            window_legal_count=sum(1 for cell in empties if cell in legal_index),
        )
        window_token_by_key[(axis, start)] = idx
        memberships[idx] = set(_window_cells(start, axis))

    line_keys = sorted(
        {
            (axis, _line_id(qr, axis))
            for qr in itertools.chain(stones.keys(), legal)
            for axis in range(3)
        }
    )
    for axis, line in line_keys:
        qr = (0, line) if axis == 0 else (line, 0)
        own_line, opp_line, own_run, opp_run = _line_stats(qr, axis, stones, current_player)
        add(
            GraphTokenType.LINE,
            qr,
            axis=axis,
            own_line_count=own_line,
            opp_line_count=opp_line,
            own_line_run=own_run,
            opp_line_run=opp_run,
        )

    engine_cover_sets: list[set[tuple[int, int]]] = []
    for a, b in getattr(oracle, "cover_pairs", ()):
        cells = {(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))}
        cells = {cell for cell in cells if cell in legal_index}
        if cells:
            engine_cover_sets.append(cells)
    cover_cells = {
        (int(q), int(r))
        for q, r in getattr(oracle, "cover_cells", ())
        if (int(q), int(r)) in legal_index
    }
    forced_cells = {
        (int(q), int(r))
        for q, r in getattr(oracle, "forced_block_cells", ())
        if (int(q), int(r)) in legal_index
    }
    winning_cells = {
        (int(q), int(r))
        for q, r in getattr(oracle, "win_now_cells", ())
        if (int(q), int(r)) in legal_index
    }
    open_cells = {
        (int(q), int(r))
        for rows in (
            getattr(oracle, "open_four_cells", ()),
            getattr(oracle, "open_five_cells", ()),
        )
        for q, r in rows
        if (int(q), int(r)) in legal_index
    }
    residual_cover = cover_cells | forced_cells | winning_cells | open_cells
    if residual_cover:
        engine_cover_sets.append(residual_cover)

    for cells in engine_cover_sets:
        if cells:
            q = round(sum(c[0] for c in cells) / len(cells))
            r = round(sum(c[1] for c in cells) / len(cells))
            idx = add(
                GraphTokenType.COVER_SET,
                (q, r),
                cover_size=len(cells),
                cover_memberships=sum(1 for cell in cells if cell in cover_cells or cell in forced_cells),
            )
            memberships[idx] = set(cells)

    for cid in sorted(set(comp.values())):
        cells = [qr for qr, c in comp.items() if c == cid]
        q = round(sum(c[0] for c in cells) / len(cells))
        r = round(sum(c[1] for c in cells) / len(cells))
        idx = add(GraphTokenType.COMPONENT, (q, r), component_size=len(cells))
        memberships[idx] = set(cells)

    pair_token_indices: list[int] = []
    pair_first_indices: list[int] = []
    pair_second_indices: list[int] = []
    pair_context_first: tuple[int, int] | None = None
    if include_pair_rows:
        candidate_rows = list(legal)
        known_for_pairs: tuple[int, int] | None = None
        if placements_remaining == 1 and moves:
            _last_player, first_q, first_r = moves[-1]
            pair_context_first = (int(first_q), int(first_r))
            known_for_pairs = pair_context_first
            candidate_rows = [pair_context_first] + list(legal)
        possible_pair_rows = (
            len(legal)
            if known_for_pairs is not None
            else len(legal) * max(0, len(legal) - 1) // 2
        )
        if possible_pair_rows == 0 and any(float(row[2]) > 0.0 for row in pair_policy_target):
            raise ValueError("pair_policy_target provided when the pair-action table is empty")
        pair_cap = possible_pair_rows if max_pair_rows is None else int(max_pair_rows)
        if possible_pair_rows > pair_cap and not allow_pair_truncation:
            raise ValueError(
                "global graph pair rows would be truncated: "
                f"{possible_pair_rows} legal pairs exceed max_pair_rows={pair_cap}. "
                "Use a smaller radius/game-length bucket, raise capacity, or enable an "
                "explicit chunking path; silent pair loss is not allowed."
            )
        pair_mode = "capped_fill" if allow_pair_truncation else "full_capped"
        pair_strategy = PairStrategy(
            mode=pair_mode,
            max_pairs=max(1, pair_cap),
            allow_full=not allow_pair_truncation,
        )
        candidate_table = CandidateContractBuilder().build(
            candidate_rows,
            [],
            offset_q=0,
            offset_r=0,
            budget=max(1, len(candidate_rows)),
            storage_width=max(1, len(candidate_rows)),
            critical_actions=candidate_rows,
            source="rust:synthetic",
        )
        pair_table = PairActionTableBuilder().build(
            candidate_table,
            pair_policy_target,
            strategy=pair_strategy,
            legal_moves=legal,
            known_first=known_for_pairs,
            source="rust:synthetic",
        )
        active_pair_rows = np.flatnonzero(pair_table.mask)
        for pair_row in active_pair_rows:
            first = (int(pair_table.rows[pair_row, 0]), int(pair_table.rows[pair_row, 1]))
            second = (int(pair_table.rows[pair_row, 2]), int(pair_table.rows[pair_row, 3]))
            qr = (round((first[0] + second[0]) / 2), round((first[1] + second[1]) / 2))
            pair_cells = {first, second}
            tok = add(
                GraphTokenType.PAIR_ACTION,
                qr,
                pair_distance=hex_distance(first, second),
                pair_reaches_win=bool(pair_cells & win_now_cells),
                pair_blocks_threat=bool(pair_cells & (forced_cells | cover_cells)),
                cover_size=len(pair_cells & cover_cells),
            )
            memberships[tok] = pair_cells
            pair_token_indices.append(tok)
            if known_for_pairs is not None:
                first_token = stone_token[pair_context_first]
                second_ref = int(pair_table.second_candidate_rows[pair_row]) - 1
                pair_first_indices.append(first_token)
                pair_second_indices.append(legal_token_indices[second_ref])
            else:
                pair_first_indices.append(legal_token_indices[int(pair_table.first_candidate_rows[pair_row])])
                pair_second_indices.append(legal_token_indices[int(pair_table.second_candidate_rows[pair_row])])

    relation_type, relation_bias = _build_relations(
        token_type,
        token_qr,
        token_axis,
        token_age,
        memberships,
        pair_token_indices,
        pair_first_indices,
        pair_second_indices,
    )

    policy = _target_for_legal(legal, policy_target, label="policy_target")
    if opp_policy_target and opp_legal_moves is None:
        raise ValueError(
            "opp_policy_target requires an independently keyed opp_legal_moves table; "
            "training it on the source legal rows is not allowed."
        )
    if opp_legal_moves is None:
        opp_legal = _opponent_legal_after_passive_turn(stones, legal, current_player, placements_remaining, radius)
    else:
        occupied = set(stones)
        occupied_rows = [
            (int(qr[0]), int(qr[1]))
            for qr in opp_legal_moves
            if (int(qr[0]), int(qr[1])) in occupied
        ]
        if occupied_rows:
            raise ValueError(f"opp_legal_moves contains occupied cells: {occupied_rows[:8]}")
        opp_legal = _unique_qr(opp_legal_moves)
    opp_policy = _target_for_legal(opp_legal, opp_policy_target, label="opp_policy_target")
    pair_target = _target_for_pairs(
        pair_first_indices,
        pair_second_indices,
        legal_token_indices,
        legal,
        pair_policy_target,
        placements_remaining=placements_remaining,
        pair_context_first=pair_context_first,
    )
    pair_first_target = _pair_first_target_for_legal(
        legal,
        pair_policy_target,
        placements_remaining=placements_remaining,
    )
    tactical_target = _tactical_target_from_oracle(oracle)

    return GraphSemanticContract(
        token_features=np.asarray(token_features, dtype=np.float32),
        token_type=np.asarray(token_type, dtype=np.int64),
        token_qr=np.asarray(token_qr, dtype=np.int32),
        token_mask=np.ones(len(token_type), dtype=np.bool_),
        legal_token_indices=np.asarray(legal_token_indices, dtype=np.int64),
        legal_qr=np.asarray(legal, dtype=np.int32),
        legal_mask=np.ones(len(legal), dtype=np.bool_),
        pair_token_indices=np.asarray(pair_token_indices, dtype=np.int64),
        pair_first_indices=np.asarray(pair_first_indices, dtype=np.int64),
        pair_second_indices=np.asarray(pair_second_indices, dtype=np.int64),
        relation_bias=relation_bias,
        relation_type=relation_type,
        policy_target=policy,
        opp_legal_qr=np.asarray(opp_legal, dtype=np.int32),
        opp_legal_mask=np.ones(len(opp_legal), dtype=np.bool_),
        opp_policy_target=opp_policy,
        pair_first_policy_target=pair_first_target,
        pair_policy_target=pair_target,
        tactical_target=tactical_target,
        placements_remaining=placements_remaining,
        current_player=current_player,
    )


def _target_for_legal(
    legal: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int, float]],
    *,
    label: str,
) -> np.ndarray:
    legal_set = set(legal)
    target_map: dict[tuple[int, int], float] = {}
    missing: list[tuple[int, int]] = []
    for q, r, prob in target:
        if float(prob) <= 0.0:
            continue
        qr = (int(q), int(r))
        if qr not in legal_set:
            missing.append(qr)
            continue
        target_map[qr] = target_map.get(qr, 0.0) + float(prob)
    if missing:
        raise ValueError(f"{label} contains actions outside its legal table: {missing[:8]}")
    out = np.asarray([target_map.get(qr, 0.0) for qr in legal], dtype=np.float32)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _target_for_pairs(
    pair_first_indices: Sequence[int],
    pair_second_indices: Sequence[int],
    legal_token_indices: Sequence[int],
    legal: Sequence[tuple[int, int]],
    target: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    placements_remaining: int,
    pair_context_first: tuple[int, int] | None,
) -> np.ndarray:
    token_to_legal = {tok: i for i, tok in enumerate(legal_token_indices)}
    legal_set = set(legal)
    positive_target = [row for row in target if float(row[2]) > 0.0]
    if positive_target and len(pair_first_indices) == 0:
        raise ValueError("pair_policy_target provided when the pair-action table is empty")
    unordered_target_map: dict[frozenset[tuple[int, int]], float] = {}
    ordered_second_target_map: dict[tuple[int, int], float] = {}
    for first, second, prob in target:
        if prob <= 0.0:
            continue
        a = (int(first[0]), int(first[1]))
        b = (int(second[0]), int(second[1]))
        if a == b:
            raise ValueError(f"duplicate coordinates are illegal for pair policy: {a}")
        if placements_remaining == 1:
            if pair_context_first is None:
                raise ValueError("second-placement pair target requires a known first placement")
            if a != pair_context_first:
                raise ValueError(
                    f"pair policy target first action {a} does not match current turn first placement {pair_context_first}"
                )
            if b not in legal_set:
                raise ValueError(f"pair policy target contains illegal second action: {b}")
            ordered_second_target_map[b] = ordered_second_target_map.get(b, 0.0) + float(prob)
        else:
            if a not in legal_set or b not in legal_set:
                raise ValueError(f"pair policy target contains illegal action pair: {(a, b)}")
            key = frozenset({a, b})
            unordered_target_map[key] = unordered_target_map.get(key, 0.0) + float(prob)
    out = []
    for first_tok, second_tok in zip(pair_first_indices, pair_second_indices):
        b = legal[token_to_legal[int(second_tok)]]
        if placements_remaining == 1:
            out.append(ordered_second_target_map.get(b, 0.0))
        else:
            a = legal[token_to_legal[int(first_tok)]]
            out.append(unordered_target_map.get(frozenset({a, b}), 0.0))
    arr = np.asarray(out, dtype=np.float32)
    total = float(arr.sum())
    if total > 0.0:
        arr /= total
    return arr


def _pair_first_target_for_legal(
    legal: Sequence[tuple[int, int]],
    target: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    placements_remaining: int,
) -> np.ndarray:
    """Project joint pair targets onto legal first-placement rows."""
    out = np.zeros(len(legal), dtype=np.float32)
    if placements_remaining < 2:
        return out
    legal_index = {qr: i for i, qr in enumerate(legal)}
    for first, second, prob in target:
        if float(prob) <= 0.0:
            continue
        a = (int(first[0]), int(first[1]))
        b = (int(second[0]), int(second[1]))
        if a == b:
            continue
        if a in legal_index:
            out[legal_index[a]] += float(prob)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _opponent_legal_after_passive_turn(
    stones: dict[tuple[int, int], int],
    legal: Sequence[tuple[int, int]],
    current_player: int,
    placements_remaining: int,
    radius: int,
) -> list[tuple[int, int]]:
    if placements_remaining > 1 or not legal:
        return list(legal)
    # For the opponent-policy table we need an independently keyed future legal
    # table.  Use a deterministic passive placeholder when no actual future
    # policy table is provided by replay.
    future = dict(stones)
    future[legal[0]] = current_player
    return legal_moves_for_stones(future, radius=radius)


def _engine_state_from_history(history: bytes) -> tuple[list[tuple[int, int]], int, int] | None:
    game = game_from_history(history)
    legal = getattr(game, "legal_moves", lambda: [])()
    current_player = getattr(game, "current_player", 0)
    placements_remaining = getattr(game, "placements_remaining", 1)
    if callable(current_player):
        current_player = current_player()
    if callable(placements_remaining):
        placements_remaining = placements_remaining()
    return (
        _unique_qr((int(q), int(r)) for q, r in legal),
        int(current_player),
        int(placements_remaining),
    )


def _tactical_target_from_oracle(oracle) -> np.ndarray:
    """State-level tactical labels: win, must-block, cover-pair, quiet."""
    out = np.zeros(4, dtype=np.float32)
    status = str(getattr(oracle, "status", "quiet"))
    if getattr(oracle, "win_now_cells", ()):
        out[0] = 1.0
    if getattr(oracle, "forced_block_cells", ()):
        out[1] = 1.0
    if getattr(oracle, "cover_pairs", ()):
        out[2] = 1.0
    if status == "quiet" and out[:3].sum() == 0.0:
        out[3] = 1.0
    return out


def _build_relations(
    token_type: Sequence[int],
    token_qr: Sequence[tuple[int, int]],
    token_axis: Sequence[int],
    token_age: Sequence[int],
    memberships: dict[int, set[tuple[int, int]]],
    pair_token_indices: Sequence[int],
    pair_first_indices: Sequence[int],
    pair_second_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    n = len(token_type)
    rel = np.zeros((n, n), dtype=np.int64)
    bias = np.zeros((1, n, n), dtype=np.float32)
    pair_edges = set(zip(pair_first_indices, pair_second_indices))
    pair_edges |= {(b, a) for a, b in pair_edges}
    pair_tokens = set(int(x) for x in pair_token_indices)
    pair_to_legal = {
        int(tok): {int(first), int(second)}
        for tok, first, second in zip(pair_token_indices, pair_first_indices, pair_second_indices)
    }
    window_tokens = {i for i, tt in enumerate(token_type) if int(tt) == int(GraphTokenType.WINDOW6)}
    cover_tokens = {i for i, tt in enumerate(token_type) if int(tt) == int(GraphTokenType.COVER_SET)}
    line_tokens = {i for i, tt in enumerate(token_type) if int(tt) == int(GraphTokenType.LINE)}
    component_tokens = {i for i, tt in enumerate(token_type) if int(tt) == int(GraphTokenType.COMPONENT)}
    legal_like_tokens = {
        i
        for i, tt in enumerate(token_type)
        if int(tt) in {int(GraphTokenType.LEGAL), int(GraphTokenType.HOT_CELL)}
    }
    stone_tokens = {i for i, tt in enumerate(token_type) if int(tt) == int(GraphTokenType.STONE)}
    max_stone_age = max((age for age in token_age if int(age) >= 0), default=-1)

    def in_membership(member_idx: int, container_idx: int) -> bool:
        return token_qr[member_idx] in memberships.get(container_idx, set())

    def share_membership(i: int, j: int, containers: set[int]) -> bool:
        a = token_qr[i]
        b = token_qr[j]
        return any(
            a in memberships.get(container, set()) and b in memberships.get(container, set())
            for container in containers
        )

    for i in range(n):
        for j in range(n):
            if i == j:
                rel[i, j] = int(RelationType.D6_ORBIT_RELATION)
                bias[0, i, j] = 1.0
                continue
            a = token_qr[i]
            b = token_qr[j]
            dist = hex_distance(a, b)
            bias[0, i, j] = 1.0 / (1.0 + dist)
            age_i = int(token_age[i])
            age_j = int(token_age[j])
            axis_i = int(token_axis[i])
            axis_j = int(token_axis[j])
            if (i, j) in pair_edges:
                rel[i, j] = int(RelationType.FIRST_SECOND_PAIR_RELATION)
            elif i in pair_tokens and j in pair_to_legal.get(i, set()):
                rel[i, j] = int(RelationType.LEGAL_TO_PAIR_ACTION)
            elif j in pair_tokens and i in pair_to_legal.get(j, set()):
                rel[i, j] = int(RelationType.LEGAL_TO_PAIR_ACTION)
            elif i in pair_tokens and j in cover_tokens and memberships.get(j, set()) <= memberships.get(i, set()):
                rel[i, j] = int(RelationType.PAIR_COVERS_THREAT_SET)
            elif j in pair_tokens and i in cover_tokens and memberships.get(i, set()) <= memberships.get(j, set()):
                rel[i, j] = int(RelationType.PAIR_COVERS_THREAT_SET)
            elif i in window_tokens and j in cover_tokens and memberships.get(i, set()) & memberships.get(j, set()):
                rel[i, j] = int(RelationType.WINDOW6_TO_COVER_SET)
            elif j in window_tokens and i in cover_tokens and memberships.get(i, set()) & memberships.get(j, set()):
                rel[i, j] = int(RelationType.WINDOW6_TO_COVER_SET)
            elif i in cover_tokens and j in legal_like_tokens and in_membership(j, i):
                rel[i, j] = int(RelationType.LEGAL_IN_COVER_SET)
            elif j in cover_tokens and i in legal_like_tokens and in_membership(i, j):
                rel[i, j] = int(RelationType.LEGAL_IN_COVER_SET)
            elif (
                i in line_tokens
                and j in window_tokens
                and token_axis[i] == token_axis[j]
                and token_axis[i] >= 0
                and _line_id(a, token_axis[i]) == _line_id(b, token_axis[j])
            ):
                rel[i, j] = int(RelationType.LINE_TO_WINDOW6)
            elif (
                j in line_tokens
                and i in window_tokens
                and token_axis[i] == token_axis[j]
                and token_axis[i] >= 0
                and _line_id(a, token_axis[i]) == _line_id(b, token_axis[j])
            ):
                rel[i, j] = int(RelationType.LINE_TO_WINDOW6)
            elif i in window_tokens and j in stone_tokens and in_membership(j, i):
                rel[i, j] = int(RelationType.STONE_IN_WINDOW6)
            elif j in window_tokens and i in stone_tokens and in_membership(i, j):
                rel[i, j] = int(RelationType.STONE_IN_WINDOW6)
            elif i in window_tokens and j in legal_like_tokens and in_membership(j, i):
                rel[i, j] = int(RelationType.LEGAL_IN_WINDOW6)
            elif j in window_tokens and i in legal_like_tokens and in_membership(i, j):
                rel[i, j] = int(RelationType.LEGAL_IN_WINDOW6)
            elif max_stone_age >= 0 and (age_i == max_stone_age or age_j == max_stone_age) and dist <= 2:
                rel[i, j] = int(RelationType.RECENT_MOVE_RELATION)
            elif age_i >= 0 and age_j >= 0 and age_i != age_j:
                rel[i, j] = int(RelationType.AGE_ORDER_BUCKET)
            elif share_membership(i, j, component_tokens):
                rel[i, j] = int(RelationType.SAME_COMPONENT)
            elif share_membership(i, j, window_tokens):
                rel[i, j] = int(RelationType.SAME_WINDOW6)
            elif axis_i >= 0 and axis_i == axis_j:
                rel[i, j] = int(RelationType.SAME_AXIS)
            elif any(_line_id(a, axis) == _line_id(b, axis) for axis in range(3)):
                rel[i, j] = int(RelationType.SAME_LINE)
            elif dist <= 2:
                rel[i, j] = int(RelationType.DISTANCE_BUCKET)
            elif dist > 0:
                rel[i, j] = int(RelationType.DIRECTION_BUCKET)
    return rel, bias


def collate_graph_batches(batches: Sequence[GraphSemanticContract]) -> GraphSemanticContract:
    if not batches:
        raise ValueError("cannot collate an empty graph batch list")
    max_t = max(b.token_features.shape[0] for b in batches)
    max_a = max(b.legal_qr.shape[0] for b in batches)
    max_o = max(b.opp_legal_qr.shape[0] for b in batches)
    max_p = max(b.pair_token_indices.shape[0] for b in batches)
    bsz = len(batches)

    def pad(shape, dtype, fill=0):
        arr = np.full(shape, fill, dtype=dtype)
        return arr

    token_features = pad((bsz, max_t, GRAPH_FEATURE_DIM), np.float32)
    token_type = pad((bsz, max_t), np.int64)
    token_qr = pad((bsz, max_t, 2), np.int32)
    token_mask = pad((bsz, max_t), np.bool_)
    relation_type = pad((bsz, max_t, max_t), np.int64)
    relation_bias = pad((bsz, 1, max_t, max_t), np.float32)
    legal_token_indices = pad((bsz, max_a), np.int64, -1)
    legal_qr = pad((bsz, max_a, 2), np.int32)
    legal_mask = pad((bsz, max_a), np.bool_)
    policy_target = pad((bsz, max_a), np.float32)
    opp_legal_qr = pad((bsz, max_o, 2), np.int32)
    opp_legal_mask = pad((bsz, max_o), np.bool_)
    opp_policy_target = pad((bsz, max_o), np.float32)
    pair_first_policy_target = pad((bsz, max_a), np.float32)
    pair_token_indices = pad((bsz, max_p), np.int64, -1)
    pair_first_indices = pad((bsz, max_p), np.int64, -1)
    pair_second_indices = pad((bsz, max_p), np.int64, -1)
    pair_policy_target = pad((bsz, max_p), np.float32)
    tactical_target = pad((bsz, 4), np.float32)

    for row, batch in enumerate(batches):
        t = batch.token_features.shape[0]
        a = batch.legal_qr.shape[0]
        o = batch.opp_legal_qr.shape[0]
        p = batch.pair_token_indices.shape[0]
        token_features[row, :t] = batch.token_features
        token_type[row, :t] = batch.token_type
        token_qr[row, :t] = batch.token_qr
        token_mask[row, :t] = True
        relation_type[row, :t, :t] = batch.relation_type
        relation_bias[row, :, :t, :t] = batch.relation_bias
        legal_token_indices[row, :a] = batch.legal_token_indices
        legal_qr[row, :a] = batch.legal_qr
        legal_mask[row, :a] = True
        policy_target[row, :a] = batch.policy_target
        opp_legal_qr[row, :o] = batch.opp_legal_qr
        opp_legal_mask[row, :o] = True
        opp_policy_target[row, :o] = batch.opp_policy_target
        pair_first_policy_target[row, :a] = batch.pair_first_policy_target
        pair_token_indices[row, :p] = batch.pair_token_indices
        pair_first_indices[row, :p] = batch.pair_first_indices
        pair_second_indices[row, :p] = batch.pair_second_indices
        pair_policy_target[row, :p] = batch.pair_policy_target
        tactical_target[row] = batch.tactical_target

    return GraphSemanticContract(
        token_features=token_features,
        token_type=token_type,
        token_qr=token_qr,
        token_mask=token_mask,
        legal_token_indices=legal_token_indices,
        legal_qr=legal_qr,
        legal_mask=legal_mask,
        pair_token_indices=pair_token_indices,
        pair_first_indices=pair_first_indices,
        pair_second_indices=pair_second_indices,
        relation_bias=relation_bias,
        relation_type=relation_type,
        policy_target=policy_target,
        opp_legal_qr=opp_legal_qr,
        opp_legal_mask=opp_legal_mask,
        opp_policy_target=opp_policy_target,
        pair_first_policy_target=pair_first_policy_target,
        pair_policy_target=pair_policy_target,
        tactical_target=tactical_target,
        placements_remaining=-1,
        current_player=-1,
    )
