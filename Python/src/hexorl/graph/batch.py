"""Deterministic all-legal global graph batches for Hexo.

This module is intentionally independent from the crop encoder.  The true
global graph path uses compact move history as the source of truth, preserves
every legal action row, and rebuilds graph data after any D6 transform.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import struct
import time
from typing import Iterable, Sequence

import numpy as np

from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history


GRAPH_SCHEMA_VERSION = 3
GRAPH_FEATURE_DIM = 12
GRAPH_FEATURE_PLACEMENTS_REMAINING = 0
GRAPH_FEATURE_CURRENT_PLAYER = 1
GRAPH_FEATURE_OWNER_RELATIVE = 2
GRAPH_FEATURE_MOVE_AGE = 3
GRAPH_FEATURE_NEAREST_OWN = 4
GRAPH_FEATURE_NEAREST_OPPONENT = 5
GRAPH_FEATURE_WINDOW_OWNER_RELATIVE = 6
GRAPH_FEATURE_WINDOW_STONE_COUNT = 7
GRAPH_FEATURE_WINDOW_EMPTY_COUNT = 8
GRAPH_FEATURE_WINDOW_AXIS = 9
GRAPH_FEATURE_LEGAL_WINDOW_COUNT = 10
RELATION_SCHEMA_VERSION = 2
PAIR_CHUNK_LIMIT = 4096
GRAPH_IPC_TOKEN_CAPACITY = 4096
GRAPH_IPC_ACTION_CAPACITY = 4096
GRAPH_IPC_PAIR_CAPACITY = PAIR_CHUNK_LIMIT
GRAPH_CAPACITY_STRATEGY = "preserve_legal_stone_tactical_rows_fail_or_chunk_context"
HEX_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
WIN_LENGTH = 6


class GraphTokenType(IntEnum):
    STATE = 0
    TURN = 1
    STONE = 2
    LEGAL = 3
    WINDOW6 = 4


class RelationType(IntEnum):
    NONE = 0
    DISTANCE_BUCKET = 1
    DIRECTION_BUCKET = 2
    SAME_AXIS = 3
    SAME_LINE = 4
    SAME_WINDOW6 = 5
    STONE_IN_WINDOW6 = 6
    LEGAL_IN_WINDOW6 = 7
    AGE_ORDER_BUCKET = 14
    RECENT_MOVE_RELATION = 15
    FIRST_SECOND_PAIR_RELATION = 16
    D6_ORBIT_RELATION = 17


@dataclass(frozen=True)
class GraphBatch:
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
    pair_second_policy_target: np.ndarray
    tactical_target: np.ndarray
    placements_remaining: int
    current_player: int
    schema_version: int = GRAPH_SCHEMA_VERSION
    relation_schema_version: int = RELATION_SCHEMA_VERSION
    placements_remaining_by_sample: np.ndarray | None = None


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


def graph_capacity_report(graph_batch: GraphBatch) -> GraphCapacityReport:
    return GraphCapacityReport(
        token_count=int(np.asarray(graph_batch.token_features).shape[0]),
        legal_count=int(np.asarray(graph_batch.legal_qr).shape[0]),
        opp_legal_count=int(np.asarray(graph_batch.opp_legal_qr).shape[0]),
        pair_count=int(np.asarray(graph_batch.pair_token_indices).shape[0]),
    )


def validate_graph_ipc_capacity(graph_batch: GraphBatch) -> GraphCapacityReport:
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


def transform_qr(qr: tuple[int, int], sym_idx: int) -> tuple[int, int]:
    """Apply one of the 12 D6 symmetries to an axial coordinate."""
    q, r = int(qr[0]), int(qr[1])
    sym = int(sym_idx) % 12
    if sym == 0:
        return (q, r)
    if sym == 1:
        return (-r, q + r)
    if sym == 2:
        return (-q - r, q)
    if sym == 3:
        return (-q, -r)
    if sym == 4:
        return (r, -q - r)
    if sym == 5:
        return (q + r, -q)
    if sym == 6:
        return (r, q)
    if sym == 7:
        return (-q, q + r)
    if sym == 8:
        return (-q - r, r)
    if sym == 9:
        return (-r, -q)
    if sym == 10:
        return (q, -q - r)
    return (q + r, -r)


def transform_history(history: bytes, sym_idx: int) -> bytes:
    """Transform compact move-history coordinates while preserving players."""
    if len(history) % 12 != 0:
        raise ValueError("compact move history length must be a multiple of 12")
    if int(sym_idx) % 12 == 0:
        return history
    out = bytearray(len(history))
    for off in range(0, len(history), 12):
        player, q, r = struct.unpack_from("<iii", history, off)
        tq, tr = transform_qr((q, r), sym_idx)
        struct.pack_into("<iii", out, off, int(player), int(tq), int(tr))
    return bytes(out)


def transform_policy_target(
    target: Sequence[tuple[int, int, float]],
    sym_idx: int,
) -> list[tuple[int, int, float]]:
    """Transform global action-keyed policy entries under D6."""
    merged: dict[tuple[int, int], float] = {}
    for q, r, prob in target:
        if float(prob) <= 0.0:
            continue
        qr = transform_qr((int(q), int(r)), sym_idx)
        merged[qr] = merged.get(qr, 0.0) + float(prob)
    return [(q, r, prob) for (q, r), prob in merged.items()]


def transform_pair_policy_target(
    target: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
    sym_idx: int,
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    """Transform pair-action policy entries under D6."""
    transformed = []
    for first, second, prob in target:
        if float(prob) <= 0.0:
            continue
        transformed.append((
            transform_qr(first, sym_idx),
            transform_qr(second, sym_idx),
            float(prob),
        ))
    return transformed


def parse_history(history: bytes) -> list[tuple[int, int, int]]:
    if len(history) % 12 != 0:
        raise ValueError("compact move history length must be a multiple of 12")
    moves: list[tuple[int, int, int]] = []
    for off in range(0, len(history), 12):
        moves.append(struct.unpack_from("<iii", history, off))
    return moves


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
    if not moves:
        return 0, 1
    last_player = moves[-1][0]
    run_len = 0
    for player, _q, _r in reversed(moves):
        if player != last_player:
            break
        run_len += 1
    if len(moves) == 1:
        return 1, 2
    if run_len == 1:
        return last_player, 1
    return 1 - last_player, 2


def legal_moves_for_stones(stones: dict[tuple[int, int], int], radius: int = 8) -> list[tuple[int, int]]:
    if int(radius) != 8:
        raise ValueError("global graph legal rows must use the Rust placement radius 8")
    if not stones:
        return [(0, 0)]
    legal: set[tuple[int, int]] = set()
    radius = 8
    for q, r in stones:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    cell = (q + dq, r + dr)
                    if cell not in stones:
                        legal.add(cell)
    return sorted(legal, key=lambda qr: (hex_distance(qr), qr[0], qr[1]))


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


def _features(
    token_type: GraphTokenType,
    *,
    current_player: int,
    placements_remaining: int,
    owner: int | None = None,
    age: int = 0,
    axis: int = -1,
    count_a: int = 0,
    count_b: int = 0,
    nearest_own: int = 64,
    nearest_opp: int = 64,
    window_empty_count: int = 0,
    legal_window_count: int = 0,
) -> np.ndarray:
    out = np.zeros(GRAPH_FEATURE_DIM, dtype=np.float32)
    out[GRAPH_FEATURE_PLACEMENTS_REMAINING] = float(placements_remaining) / 2.0
    out[GRAPH_FEATURE_CURRENT_PLAYER] = float(current_player)
    out[GRAPH_FEATURE_OWNER_RELATIVE] = 0.0 if owner is None else (1.0 if owner == current_player else -1.0)
    out[GRAPH_FEATURE_MOVE_AGE] = min(max(age, 0), 64) / 64.0
    out[GRAPH_FEATURE_NEAREST_OWN] = min(nearest_own, 64) / 64.0
    out[GRAPH_FEATURE_NEAREST_OPPONENT] = min(nearest_opp, 64) / 64.0
    if token_type == GraphTokenType.WINDOW6:
        current_count = count_a if current_player == 0 else count_b
        opponent_count = count_b if current_player == 0 else count_a
        if current_count > 0:
            out[GRAPH_FEATURE_WINDOW_OWNER_RELATIVE] = 1.0
            out[GRAPH_FEATURE_WINDOW_STONE_COUNT] = min(current_count, WIN_LENGTH) / float(WIN_LENGTH)
        elif opponent_count > 0:
            out[GRAPH_FEATURE_WINDOW_OWNER_RELATIVE] = -1.0
            out[GRAPH_FEATURE_WINDOW_STONE_COUNT] = min(opponent_count, WIN_LENGTH) / float(WIN_LENGTH)
        out[GRAPH_FEATURE_WINDOW_EMPTY_COUNT] = min(window_empty_count, WIN_LENGTH) / float(WIN_LENGTH)
        out[GRAPH_FEATURE_WINDOW_AXIS] = float(axis + 1) / 4.0
    if token_type == GraphTokenType.LEGAL:
        out[GRAPH_FEATURE_LEGAL_WINDOW_COUNT] = min(legal_window_count, 16) / 16.0
    return out


def build_graph_batch_from_history(
    history: bytes,
    *,
    policy_target: Sequence[tuple[int, int, float]] = (),
    legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_policy_target: Sequence[tuple[int, int, float]] = (),
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]] = (),
    radius: int = 8,
    constrain_threats: bool = False,
    max_pair_rows: int = PAIR_CHUNK_LIMIT,
    allow_pair_truncation: bool = False,
    include_pair_rows: bool = True,
    materialize_pair_context_tokens: bool = False,
    max_legal_rows: int | None = None,
    max_context_tokens: int | None = None,
    required_legal_rows: Sequence[tuple[int, int]] = (),
) -> GraphBatch:
    if int(radius) != 8:
        raise ValueError("global graph legal rows must preserve all Rust-legal moves; radius must be 8")
    if materialize_pair_context_tokens:
        raise ValueError("PAIR_ACTION context tokens were removed from the minimal global graph schema")
    moves = parse_history(history)
    stones = {(q, r): player for player, q, r in moves}
    current_player, placements_remaining = current_turn_state(moves)
    engine_state = _engine_state_from_history(
        history,
        radius=radius,
        constrain_threats=bool(constrain_threats) and legal_moves is None,
    )
    if engine_state is not None:
        legal, current_player, placements_remaining = engine_state
    else:
        legal = legal_moves_for_stones(stones, radius=radius)
        if constrain_threats and legal_moves is None:
            legal = _threat_constrained_fallback(history, legal)
    if legal_moves is not None:
        occupied_rows = [
            (int(qr[0]), int(qr[1]))
            for qr in legal_moves
            if (int(qr[0]), int(qr[1])) in stones
        ]
        if occupied_rows:
            raise ValueError(f"legal_moves contains occupied cells: {occupied_rows[:8]}")
        legal = _unique_qr(legal_moves)
    oracle = scan_tactical_oracle_from_history(
        history,
        legal,
        near_radius=8,
    )
    win_now_cells = {(int(q), int(r)) for q, r in getattr(oracle, "win_now_cells", ())}
    forced_cells = {(int(q), int(r)) for q, r in getattr(oracle, "forced_block_cells", ())}
    open_four_cells = {(int(q), int(r)) for q, r in getattr(oracle, "open_four_cells", ())}
    open_five_cells = {(int(q), int(r)) for q, r in getattr(oracle, "open_five_cells", ())}
    cover_cells = {(int(q), int(r)) for q, r in getattr(oracle, "cover_cells", ())}
    context_budget = int(max_context_tokens) if max_context_tokens is not None and int(max_context_tokens) > 0 else None
    if max_legal_rows is not None and int(max_legal_rows) > 0 and len(legal) > int(max_legal_rows):
        legal_set = set(legal)
        required_legal = {
            (int(q), int(r))
            for q, r in required_legal_rows
            if (int(q), int(r)) in legal_set
        }
        for q, r, prob in policy_target:
            if float(prob) > 0.0 and (int(q), int(r)) in legal_set:
                required_legal.add((int(q), int(r)))
        for first, second, prob in pair_policy_target:
            if float(prob) <= 0.0:
                continue
            a = (int(first[0]), int(first[1]))
            b = (int(second[0]), int(second[1]))
            if a in legal_set:
                required_legal.add(a)
            if b in legal_set:
                required_legal.add(b)
        any_stones = tuple(stones.keys())
        nearest_by_legal: dict[tuple[int, int], int] = {}
        if any_stones:
            legal_arr_for_rank = np.asarray(legal, dtype=np.int64).reshape(-1, 2)
            stone_arr_for_rank = np.asarray(any_stones, dtype=np.int64).reshape(-1, 2)
            nearest = np.full(legal_arr_for_rank.shape[0], 64, dtype=np.int64)
            for start in range(0, legal_arr_for_rank.shape[0], 1024):
                chunk = legal_arr_for_rank[start : start + 1024]
                delta = chunk[:, None, :] - stone_arr_for_rank[None, :, :]
                dq_arr = delta[..., 0]
                dr_arr = delta[..., 1]
                dist = np.maximum.reduce((np.abs(dq_arr), np.abs(dr_arr), np.abs(dq_arr + dr_arr)))
                nearest[start : start + chunk.shape[0]] = np.minimum(64, dist.min(axis=1))
            nearest_by_legal = {
                (int(q), int(r)): int(nearest[idx])
                for idx, (q, r) in enumerate(legal_arr_for_rank.tolist())
            }

        def legal_rank(qr: tuple[int, int]) -> tuple[int, int, int, int, int]:
            tactical_rank = 0 if qr in (win_now_cells | forced_cells) else 1
            hot_rank = 0 if qr in (open_four_cells | open_five_cells | cover_cells) else 1
            nearest = nearest_by_legal.get(qr, 0)
            return (tactical_rank, hot_rank, nearest, hex_distance(qr), qr[0] * 4096 + qr[1])

        budget = max(int(max_legal_rows), len(required_legal))
        selected = [qr for qr in legal if qr in required_legal]
        selected.extend(
            qr
            for qr in sorted((qr for qr in legal if qr not in required_legal), key=legal_rank)
            if len(selected) < budget
        )
        selected_set = set(selected)
        legal = [qr for qr in legal if qr in selected_set]
    legal_index = {qr: i for i, qr in enumerate(legal)}
    windows = _active_windows(stones, legal)

    token_features: list[np.ndarray] = []
    token_type: list[int] = []
    token_qr: list[tuple[int, int]] = []
    token_axis: list[int] = []
    token_age: list[int] = []
    memberships: dict[int, set[tuple[int, int]]] = {}
    own_stones = [qr for qr, owner in stones.items() if owner == current_player]
    opp_stones = [qr for qr, owner in stones.items() if owner != current_player]
    own_stone_arr = np.asarray(own_stones, dtype=np.int64).reshape(-1, 2)
    opp_stone_arr = np.asarray(opp_stones, dtype=np.int64).reshape(-1, 2)
    any_stone_arr = np.asarray(list(stones.keys()), dtype=np.int64).reshape(-1, 2)
    nearest_cache: dict[tuple[int, int], tuple[int, int, int]] = {}

    def nearest_from_array(qr: tuple[int, int], cells: np.ndarray, default: int = 64) -> int:
        if cells.size == 0:
            return default
        point = np.asarray(qr, dtype=np.int64)
        delta = cells - point[None, :]
        dq_arr = delta[:, 0]
        dr_arr = delta[:, 1]
        distances = np.maximum.reduce((np.abs(dq_arr), np.abs(dr_arr), np.abs(dq_arr + dr_arr)))
        return int(min(default, int(distances.min())))

    def nearest_distances(qr: tuple[int, int]) -> tuple[int, int, int]:
        key = (int(qr[0]), int(qr[1]))
        cached = nearest_cache.get(key)
        if cached is not None:
            return cached
        cached = (
            nearest_from_array(key, own_stone_arr),
            nearest_from_array(key, opp_stone_arr),
            nearest_from_array(key, any_stone_arr),
        )
        nearest_cache[key] = cached
        return cached

    legal_window_count_by_cell: dict[tuple[int, int], int] = {}
    for axis, start, _own, _opp, _empties in windows:
        for cell in _window_cells(start, axis):
            if cell in legal_index:
                legal_window_count_by_cell[cell] = legal_window_count_by_cell.get(cell, 0) + 1

    def add(tt: GraphTokenType, qr: tuple[int, int], **kwargs) -> int:
        idx = len(token_type)
        nearest_own, nearest_opp, _nearest_any = nearest_distances(qr)
        token_type.append(int(tt))
        token_qr.append(qr)
        token_axis.append(int(kwargs.get("axis", -1)))
        token_age.append(int(kwargs.get("age", -1)) if tt == GraphTokenType.STONE else -1)
        token_features.append(
            _features(
                tt,
                current_player=current_player,
                placements_remaining=placements_remaining,
                nearest_own=nearest_own,
                nearest_opp=nearest_opp,
                legal_window_count=legal_window_count_by_cell.get(qr, 0),
                **kwargs,
            )
        )
        return idx

    add(GraphTokenType.STATE, (0, 0))
    add(GraphTokenType.TURN, (0, 0))

    if context_budget is None:
        stone_start = 0
        window_limit = None
    else:
        stone_limit = min(len(moves), max(16, context_budget // 2))
        stone_start = max(0, len(moves) - stone_limit)
        window_limit = max(16, context_budget // 4)

    stone_token: dict[tuple[int, int], int] = {}
    for age, (player, q, r) in enumerate(moves):
        if age < stone_start:
            continue
        stone_token[(q, r)] = add(GraphTokenType.STONE, (q, r), owner=player, age=age)

    legal_token_indices: list[int] = []
    for qr in legal:
        legal_token_indices.append(add(GraphTokenType.LEGAL, qr))

    if window_limit is not None and len(windows) > window_limit:
        important_cells = win_now_cells | forced_cells | open_four_cells | open_five_cells | cover_cells

        def window_rank(row: tuple[int, tuple[int, int], int, int, tuple[tuple[int, int], ...]]) -> tuple[int, int, int, int, int, int]:
            axis, start, own, opp, _empties = row
            cells = _window_cells(start, axis)
            center = cells[WIN_LENGTH // 2]
            touches_important = any(cell in important_cells for cell in cells)
            return (
                0 if touches_important else 1,
                -max(int(own), int(opp)),
                -(int(own) + int(opp)),
                hex_distance(center),
                int(start[0]),
                int(start[1]),
            )

        windows = sorted(windows, key=window_rank)[:window_limit]
    for axis, start, own, opp, empties in windows:
        center = _window_cells(start, axis)[WIN_LENGTH // 2]
        idx = add(
            GraphTokenType.WINDOW6,
            center,
            axis=axis,
            count_a=own,
            count_b=opp,
            window_empty_count=len(empties),
        )
        memberships[idx] = set(_window_cells(start, axis))

    pair_token_indices: list[int] = []
    pair_first_indices: list[int] = []
    pair_second_indices: list[int] = []
    pair_context_first: tuple[int, int] | None = None
    if not include_pair_rows:
        pass
    elif placements_remaining >= 2:
        total_pair_rows = len(legal) * (len(legal) - 1) // 2
        if (
            max_pair_rows is not None
            and total_pair_rows > int(max_pair_rows)
            and not allow_pair_truncation
        ):
            raise ValueError(
                "global graph pair rows would be truncated: "
                f"{total_pair_rows} legal pairs exceed max_pair_rows={int(max_pair_rows)}. "
                "Use a smaller radius/game-length bucket, raise capacity, or enable an "
                "explicit chunking path; silent pair loss is not allowed."
            )
        pair_limit = total_pair_rows if max_pair_rows is None else min(total_pair_rows, int(max_pair_rows))
        for a_idx in range(len(legal)):
            for b_idx in range(a_idx + 1, len(legal)):
                if len(pair_token_indices) >= pair_limit:
                    break
                pair_token_indices.append(-1)
                pair_first_indices.append(legal_token_indices[a_idx])
                pair_second_indices.append(legal_token_indices[b_idx])
            if len(pair_token_indices) >= pair_limit:
                break
    elif placements_remaining == 1 and moves:
        _last_player, first_q, first_r = moves[-1]
        pair_context_first = (int(first_q), int(first_r))
        first_token = stone_token[pair_context_first]
        total_pair_rows = len(legal)
        if (
            max_pair_rows is not None
            and total_pair_rows > int(max_pair_rows)
            and not allow_pair_truncation
        ):
            raise ValueError(
                "global graph second-placement pair rows would be truncated: "
                f"{total_pair_rows} legal seconds exceed max_pair_rows={int(max_pair_rows)}. "
                "Use a smaller radius/game-length bucket, raise capacity, or enable an "
                "explicit chunking path; silent pair loss is not allowed."
            )
        pair_limit = total_pair_rows if max_pair_rows is None else min(total_pair_rows, int(max_pair_rows))
        for b_idx, _second in enumerate(legal[:pair_limit]):
            pair_token_indices.append(-1)
            pair_first_indices.append(first_token)
            pair_second_indices.append(legal_token_indices[b_idx])

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
    pair_second_target = (
        pair_target.copy()
        if placements_remaining == 1
        else np.zeros_like(pair_target, dtype=np.float32)
    )
    tactical_target = _tactical_target_from_oracle(oracle)

    return GraphBatch(
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
        pair_second_policy_target=pair_second_target,
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


def graph_batch_with_policy_targets(
    graph_batch: GraphBatch,
    *,
    policy_target: Sequence[tuple[int, int, float]] = (),
    opp_legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_policy_target: Sequence[tuple[int, int, float]] = (),
) -> GraphBatch:
    """Reuse cached graph structure while replacing replay-dependent targets."""

    legal = [(int(q), int(r)) for q, r in np.asarray(graph_batch.legal_qr, dtype=np.int32).tolist()]
    policy = _target_for_legal(legal, policy_target, label="policy_target")
    if opp_policy_target and opp_legal_moves is None:
        raise ValueError(
            "opp_policy_target requires an independently keyed opp_legal_moves table; "
            "training it on the source legal rows is not allowed."
        )
    if opp_legal_moves is None:
        opp_mask = np.asarray(graph_batch.opp_legal_mask, dtype=np.bool_)
        opp_legal = [
            (int(q), int(r))
            for q, r in np.asarray(graph_batch.opp_legal_qr, dtype=np.int32)[opp_mask].tolist()
        ]
    else:
        stone_rows = np.flatnonzero(np.asarray(graph_batch.token_type) == int(GraphTokenType.STONE))
        occupied = {
            tuple(int(x) for x in np.asarray(graph_batch.token_qr[int(row)], dtype=np.int32).tolist())
            for row in stone_rows
        }
        occupied_rows = [
            (int(qr[0]), int(qr[1]))
            for qr in opp_legal_moves
            if (int(qr[0]), int(qr[1])) in occupied
        ]
        if occupied_rows:
            raise ValueError(f"opp_legal_moves contains occupied cells: {occupied_rows[:8]}")
        opp_legal = _unique_qr(opp_legal_moves)
    opp_policy = _target_for_legal(opp_legal, opp_policy_target, label="opp_policy_target")
    return replace(
        graph_batch,
        policy_target=policy,
        opp_legal_qr=np.asarray(opp_legal, dtype=np.int32),
        opp_legal_mask=np.ones(len(opp_legal), dtype=np.bool_),
        opp_policy_target=opp_policy,
    )


def graph_batch_with_reference_pair_rows(
    graph_batch: GraphBatch,
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
    *,
    max_pair_rows: int | None = None,
) -> GraphBatch:
    """Attach legal pair reference rows without materializing pair tokens.

    The transformer context keeps all legal action tokens but pair scoring can
    be O(A^2).  For replay training, represent pair rows by references to the
    relevant LEGAL/STONE token indices so the pair heads can train without
    adding pair tokens to the attention context.  When max_pair_rows is set,
    all positive search-observed target rows are preserved and deterministic
    legal negatives fill the remaining budget.
    """

    legal = [(int(q), int(r)) for q, r in np.asarray(graph_batch.legal_qr, dtype=np.int32).tolist()]
    legal_tokens = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
    row_budget = None if max_pair_rows is None else max(0, int(max_pair_rows))
    if graph_batch.placements_remaining >= 2:
        selected_pairs: list[tuple[int, int]] = []
        selected_set: set[tuple[int, int]] = set()
        if row_budget is not None:
            legal_index = {qr: i for i, qr in enumerate(legal)}
            for first, second, prob in pair_policy_target:
                if float(prob) <= 0.0:
                    continue
                a = (int(first[0]), int(first[1]))
                b = (int(second[0]), int(second[1]))
                if a == b:
                    raise ValueError(f"duplicate coordinates are illegal for pair policy: {a}")
                if a not in legal_index or b not in legal_index:
                    raise ValueError(f"pair policy target contains illegal action pair: {(a, b)}")
                pair = tuple(sorted((legal_index[a], legal_index[b])))
                if pair not in selected_set:
                    selected_set.add(pair)
                    selected_pairs.append(pair)
        budget = None if row_budget is None else max(row_budget, len(selected_pairs))
        for a_idx in range(len(legal)):
            for b_idx in range(a_idx + 1, len(legal)):
                pair = (a_idx, b_idx)
                if pair in selected_set:
                    continue
                if budget is not None and len(selected_pairs) >= budget:
                    break
                selected_set.add(pair)
                selected_pairs.append(pair)
            if budget is not None and len(selected_pairs) >= budget:
                break
        pair_first = np.asarray([int(legal_tokens[a_idx]) for a_idx, _ in selected_pairs], dtype=np.int64)
        pair_second = np.asarray([int(legal_tokens[b_idx]) for _, b_idx in selected_pairs], dtype=np.int64)
        pair_context_first = None
    elif graph_batch.placements_remaining == 1:
        stone_tokens = np.flatnonzero(graph_batch.token_type == int(GraphTokenType.STONE))
        if stone_tokens.size == 0:
            pair_first = np.zeros(0, dtype=np.int64)
            pair_second = np.zeros(0, dtype=np.int64)
            pair_context_first = None
        else:
            first_token = int(stone_tokens[-1])
            first_qr = tuple(int(x) for x in graph_batch.token_qr[first_token].tolist())
            selected_seconds: list[int] = []
            selected_set: set[int] = set()
            if row_budget is not None:
                legal_index = {qr: i for i, qr in enumerate(legal)}
                for first, second, prob in pair_policy_target:
                    if float(prob) <= 0.0:
                        continue
                    a = (int(first[0]), int(first[1]))
                    b = (int(second[0]), int(second[1]))
                    if a != first_qr:
                        raise ValueError(
                            f"pair policy target first action {a} does not match current turn first placement {first_qr}"
                        )
                    if b not in legal_index:
                        raise ValueError(f"pair policy target contains illegal second action: {b}")
                    b_idx = legal_index[b]
                    if b_idx not in selected_set:
                        selected_set.add(b_idx)
                        selected_seconds.append(b_idx)
            budget = None if row_budget is None else max(row_budget, len(selected_seconds))
            for b_idx in range(len(legal)):
                if b_idx in selected_set:
                    continue
                if budget is not None and len(selected_seconds) >= budget:
                    break
                selected_set.add(b_idx)
                selected_seconds.append(b_idx)
            pair_first = np.full(len(selected_seconds), first_token, dtype=np.int64)
            pair_second = np.asarray([int(legal_tokens[b_idx]) for b_idx in selected_seconds], dtype=np.int64)
            pair_context_first = first_qr
    else:
        pair_first = np.zeros(0, dtype=np.int64)
        pair_second = np.zeros(0, dtype=np.int64)
        pair_context_first = None

    pair_target = _target_for_pairs(
        pair_first,
        pair_second,
        legal_tokens,
        legal,
        pair_policy_target,
        placements_remaining=int(graph_batch.placements_remaining),
        pair_context_first=pair_context_first,
    )
    pair_first_target = _pair_first_target_for_legal(
        legal,
        pair_policy_target,
        placements_remaining=int(graph_batch.placements_remaining),
    )
    pair_second_target = (
        pair_target.copy()
        if int(graph_batch.placements_remaining) == 1
        else np.zeros_like(pair_target, dtype=np.float32)
    )
    pair_count = int(pair_first.shape[0])
    return GraphBatch(
        token_features=graph_batch.token_features,
        token_type=graph_batch.token_type,
        token_qr=graph_batch.token_qr,
        token_mask=graph_batch.token_mask,
        legal_token_indices=graph_batch.legal_token_indices,
        legal_qr=graph_batch.legal_qr,
        legal_mask=graph_batch.legal_mask,
        pair_token_indices=np.full(pair_count, -1, dtype=np.int64),
        pair_first_indices=pair_first,
        pair_second_indices=pair_second,
        relation_bias=graph_batch.relation_bias,
        relation_type=graph_batch.relation_type,
        policy_target=graph_batch.policy_target,
        opp_legal_qr=graph_batch.opp_legal_qr,
        opp_legal_mask=graph_batch.opp_legal_mask,
        opp_policy_target=graph_batch.opp_policy_target,
        pair_first_policy_target=pair_first_target,
        pair_policy_target=pair_target,
        pair_second_policy_target=pair_second_target,
        tactical_target=graph_batch.tactical_target,
        placements_remaining=graph_batch.placements_remaining,
        current_player=graph_batch.current_player,
        schema_version=graph_batch.schema_version,
        relation_schema_version=graph_batch.relation_schema_version,
    )


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


def _engine_state_from_history(
    history: bytes,
    *,
    radius: int = 8,
    constrain_threats: bool = False,
) -> tuple[list[tuple[int, int]], int, int] | None:
    try:
        import _engine  # type: ignore
    except Exception:
        return None
    engine_cls = getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame", None)
    if engine_cls is None:
        return None
    game = engine_cls()
    for player, q, r in parse_history(history):
        current = getattr(game, "current_player", player)
        current = current() if callable(current) else current
        if int(player) != int(current):
            raise ValueError(f"invalid graph history: player {player} does not match engine current player {current}")
        game.place(int(q), int(r))
    if constrain_threats and hasattr(game, "encode_board_and_legal"):
        _tensor, _offset_q, _offset_r, legal_bytes = game.encode_board_and_legal(
            int(radius),
            True,
        )
        legal_arr = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        legal = [(int(q), int(r)) for q, r in legal_arr.tolist()]
    elif constrain_threats and hasattr(game, "threat_constrained_moves"):
        legal = game.threat_constrained_moves(int(radius))
    else:
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


def _threat_constrained_fallback(
    history: bytes,
    legal: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    oracle = scan_tactical_oracle_from_history(
        history,
        legal,
        near_radius=8,
        allow_python_fallback=True,
    )
    win_now = {(int(q), int(r)) for q, r in getattr(oracle, "win_now_cells", ())}
    forced = {(int(q), int(r)) for q, r in getattr(oracle, "forced_block_cells", ())}
    if win_now:
        allowed = win_now
    elif forced:
        allowed = forced
    else:
        return list(legal)
    return [qr for qr in legal if qr in allowed]


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
    token_type_arr = np.asarray(token_type, dtype=np.int64)
    token_qr_arr = np.asarray(token_qr, dtype=np.int64)
    token_axis_arr = np.asarray(token_axis, dtype=np.int64)
    token_age_arr = np.asarray(token_age, dtype=np.int64)

    q = token_qr_arr[:, 0]
    r = token_qr_arr[:, 1]
    dq = q[:, None] - q[None, :]
    dr = r[:, None] - r[None, :]
    dist = np.maximum.reduce((np.abs(dq), np.abs(dr), np.abs(dq + dr)))
    rel = np.zeros((n, n), dtype=np.int16)
    bias = (1.0 / (1.0 + dist.astype(np.float32)))[None, :, :]
    np.fill_diagonal(rel, int(RelationType.D6_ORBIT_RELATION))

    def assign(mask: np.ndarray, relation: RelationType) -> None:
        rel[(rel == int(RelationType.NONE)) & mask] = int(relation)

    def assign_edges(
        sources: np.ndarray,
        targets: np.ndarray,
        relation: RelationType,
        *,
        symmetric: bool = False,
    ) -> None:
        src = np.asarray(sources, dtype=np.int64).reshape(-1)
        dst = np.asarray(targets, dtype=np.int64).reshape(-1)
        if src.size == 0 or dst.size == 0:
            return
        if src.size == 1 and dst.size > 1:
            src = np.full(dst.shape, int(src[0]), dtype=np.int64)
        elif dst.size == 1 and src.size > 1:
            dst = np.full(src.shape, int(dst[0]), dtype=np.int64)
        elif src.shape != dst.shape:
            raise ValueError("relation edge sources and targets must align")
        valid = (src >= 0) & (src < n) & (dst >= 0) & (dst < n)
        if not np.any(valid):
            return
        src = src[valid]
        dst = dst[valid]
        free = rel[src, dst] == int(RelationType.NONE)
        if np.any(free):
            rel[src[free], dst[free]] = int(relation)
        if symmetric:
            free_rev = rel[dst, src] == int(RelationType.NONE)
            if np.any(free_rev):
                rel[dst[free_rev], src[free_rev]] = int(relation)

    coord_bias = np.int64(1 << 20)
    token_cell_keys = ((q + coord_bias) << np.int64(32)) | (
        (r + coord_bias) & np.int64(0xFFFFFFFF)
    )
    token_rows_by_cell_key: dict[int, list[int]] = {}
    for row, key in enumerate(token_cell_keys.tolist()):
        token_rows_by_cell_key.setdefault(int(key), []).append(int(row))

    def cell_key(cell: tuple[int, int]) -> np.int64:
        cq = np.int64(int(cell[0])) + coord_bias
        cr = np.int64(int(cell[1])) + coord_bias
        return (cq << np.int64(32)) | (cr & np.int64(0xFFFFFFFF))

    empty_i64 = np.asarray((), dtype=np.int64)
    membership_keys: dict[int, np.ndarray] = {}
    for container, cells in memberships.items():
        if not cells:
            membership_keys[int(container)] = empty_i64
            continue
        keys = np.fromiter((cell_key(cell) for cell in cells), dtype=np.int64)
        membership_keys[int(container)] = np.unique(keys)

    rows_for_membership_cache: dict[int, np.ndarray] = {}

    def tokens_for_container(container: int) -> np.ndarray:
        container = int(container)
        cached = rows_for_membership_cache.get(container)
        if cached is not None:
            return cached
        keys = membership_keys.get(container, empty_i64)
        if keys.size == 0:
            rows = empty_i64
        else:
            chunks = [token_rows_by_cell_key.get(int(key)) for key in keys.tolist()]
            present = [chunk for chunk in chunks if chunk]
            if not present:
                rows = empty_i64
            else:
                rows = np.asarray(
                    [row for chunk in present for row in chunk],
                    dtype=np.int64,
                )
                if rows.size > 1:
                    rows = np.unique(rows)
        rows_for_membership_cache[container] = rows
        return rows

    def membership_token_matrix(containers: np.ndarray) -> np.ndarray:
        containers = np.asarray(containers, dtype=np.int64).reshape(-1)
        matrix = np.zeros((containers.size, n), dtype=np.bool_)
        for row, container in enumerate(containers):
            rows = tokens_for_container(int(container))
            if rows.size:
                matrix[row, rows] = True
        return matrix

    def assign_same_membership(containers: np.ndarray, relation: RelationType) -> None:
        containers = np.asarray(containers, dtype=np.int64).reshape(-1)
        for container in containers:
            rows = tokens_for_container(int(container))
            if rows.size == 0:
                continue
            src = np.repeat(rows, rows.size)
            dst = np.tile(rows, rows.size)
            assign_edges(src, dst, relation, symmetric=False)

    window_tokens = np.flatnonzero(token_type_arr == int(GraphTokenType.WINDOW6))
    legal_tokens = np.flatnonzero(token_type_arr == int(GraphTokenType.LEGAL))
    stone_mask = np.zeros(n, dtype=np.bool_)
    legal_token_mask = np.zeros(n, dtype=np.bool_)
    stone_mask[np.flatnonzero(token_type_arr == int(GraphTokenType.STONE))] = True
    legal_token_mask[legal_tokens] = True

    pair_first_arr = np.asarray(pair_first_indices, dtype=np.int64)
    pair_second_arr = np.asarray(pair_second_indices, dtype=np.int64)
    valid_pair_refs = (
        (pair_first_arr >= 0)
        & (pair_first_arr < n)
        & (pair_second_arr >= 0)
        & (pair_second_arr < n)
    )
    if np.any(valid_pair_refs):
        pair_edge_mask = np.zeros((n, n), dtype=np.bool_)
        first = pair_first_arr[valid_pair_refs]
        second = pair_second_arr[valid_pair_refs]
        pair_edge_mask[first, second] = True
        pair_edge_mask[second, first] = True
        assign(pair_edge_mask, RelationType.FIRST_SECOND_PAIR_RELATION)

    if window_tokens.size:
        window_member_rows = membership_token_matrix(window_tokens)
        stone_edges = window_member_rows & stone_mask[None, :]
        window_rows, stone_cols = np.nonzero(stone_edges)
        if window_rows.size:
            assign_edges(
                window_tokens[window_rows],
                stone_cols.astype(np.int64),
                RelationType.STONE_IN_WINDOW6,
                symmetric=True,
            )
        legal_edges = window_member_rows & legal_token_mask[None, :]
        window_rows, legal_cols = np.nonzero(legal_edges)
        if window_rows.size:
            assign_edges(
                window_tokens[window_rows],
                legal_cols.astype(np.int64),
                RelationType.LEGAL_IN_WINDOW6,
                symmetric=True,
            )

    max_stone_age = int(token_age_arr[token_age_arr >= 0].max()) if np.any(token_age_arr >= 0) else -1
    if max_stone_age >= 0:
        assign(
            ((token_age_arr[:, None] == max_stone_age) | (token_age_arr[None, :] == max_stone_age))
            & (dist <= 2),
            RelationType.RECENT_MOVE_RELATION,
        )
    assign(
        (token_age_arr[:, None] >= 0)
        & (token_age_arr[None, :] >= 0)
        & (token_age_arr[:, None] != token_age_arr[None, :]),
        RelationType.AGE_ORDER_BUCKET,
    )

    if window_tokens.size:
        assign_same_membership(window_tokens, RelationType.SAME_WINDOW6)

    assign(
        (token_axis_arr[:, None] >= 0)
        & (token_axis_arr[:, None] == token_axis_arr[None, :]),
        RelationType.SAME_AXIS,
    )
    same_line = (
        (r[:, None] == r[None, :])
        | (q[:, None] == q[None, :])
        | ((q + r)[:, None] == (q + r)[None, :])
    )
    assign(same_line, RelationType.SAME_LINE)
    assign(dist <= 2, RelationType.DISTANCE_BUCKET)
    assign(dist > 0, RelationType.DIRECTION_BUCKET)
    return rel, bias


def collate_graph_batches(
    batches: Sequence[GraphBatch],
    *,
    timings: dict[str, float] | None = None,
) -> GraphBatch:
    started = time.perf_counter()
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
    relation_type = pad((bsz, max_t, max_t), np.int16)
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
    pair_second_policy_target = pad((bsz, max_p), np.float32)
    tactical_target = pad((bsz, 4), np.float32)
    placements_remaining_by_sample = np.zeros(bsz, dtype=np.int64)

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
        pair_second_policy_target[row, :p] = batch.pair_second_policy_target
        tactical_target[row] = batch.tactical_target
        placements_remaining_by_sample[row] = int(batch.placements_remaining)

    collated = GraphBatch(
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
        pair_second_policy_target=pair_second_policy_target,
        tactical_target=tactical_target,
        placements_remaining=-1,
        current_player=-1,
        placements_remaining_by_sample=placements_remaining_by_sample,
    )
    if timings is not None:
        timings["graph_collate_s"] = timings.get("graph_collate_s", 0.0) + (
            time.perf_counter() - started
        )
    return collated
