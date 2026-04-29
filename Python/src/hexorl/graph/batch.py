"""Deterministic all-legal global graph batches for Hexo.

This module is intentionally independent from the crop encoder.  The true
global graph path uses compact move history as the source of truth, preserves
every legal action row, and rebuilds graph data after any D6 transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import itertools
import struct
from typing import Iterable, Sequence

import numpy as np

from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history


GRAPH_SCHEMA_VERSION = 1
GRAPH_FEATURE_DIM = 24
RELATION_SCHEMA_VERSION = 1
PAIR_CHUNK_LIMIT = 4096
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
    tactical_target: np.ndarray
    placements_remaining: int
    current_player: int
    schema_version: int = GRAPH_SCHEMA_VERSION
    relation_schema_version: int = RELATION_SCHEMA_VERSION


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
    return out


def build_graph_batch_from_history(
    history: bytes,
    *,
    policy_target: Sequence[tuple[int, int, float]] = (),
    opp_legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_policy_target: Sequence[tuple[int, int, float]] = (),
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]] = (),
    radius: int = 8,
    max_pair_rows: int = PAIR_CHUNK_LIMIT,
    allow_pair_truncation: bool = False,
    include_pair_rows: bool = True,
) -> GraphBatch:
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

    hot_cells: set[tuple[int, int]] = set()
    for _axis, _start, own, opp, empties in windows:
        if (own >= 4 and opp == 0) or (opp >= 4 and own == 0):
            hot_cells.update(c for c in empties if c in legal_index)
    for qr in sorted(hot_cells):
        add(GraphTokenType.HOT_CELL, qr)

    window_token_by_key: dict[tuple[int, tuple[int, int]], int] = {}
    for axis, start, own, opp, empties in windows:
        center = _window_cells(start, axis)[WIN_LENGTH // 2]
        idx = add(GraphTokenType.WINDOW6, center, axis=axis, count_a=own, count_b=opp)
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
        add(GraphTokenType.LINE, qr, axis=axis)

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
            idx = add(GraphTokenType.COVER_SET, (q, r))
            memberships[idx] = set(cells)

    for cid in sorted(set(comp.values())):
        cells = [qr for qr, c in comp.items() if c == cid]
        q = round(sum(c[0] for c in cells) / len(cells))
        r = round(sum(c[1] for c in cells) / len(cells))
        idx = add(GraphTokenType.COMPONENT, (q, r))
        memberships[idx] = set(cells)

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
                a = legal[a_idx]
                b = legal[b_idx]
                qr = (round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2))
                tok = add(GraphTokenType.PAIR_ACTION, qr)
                memberships[tok] = {a, b}
                pair_token_indices.append(tok)
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
        for b_idx, second in enumerate(legal[:pair_limit]):
            qr = (round((pair_context_first[0] + second[0]) / 2), round((pair_context_first[1] + second[1]) / 2))
            tok = add(GraphTokenType.PAIR_ACTION, qr)
            memberships[tok] = {pair_context_first, second}
            pair_token_indices.append(tok)
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


def graph_batch_with_reference_pair_rows(
    graph_batch: GraphBatch,
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]],
) -> GraphBatch:
    """Attach full legal pair rows without materializing pair tokens.

    The transformer context keeps all legal action tokens but pair scoring can
    be O(A^2).  For replay training, represent pair rows by references to the
    relevant LEGAL/STONE token indices so the pair heads can train over the
    complete table without adding tens of thousands of PAIR_ACTION tokens.
    """

    legal = [(int(q), int(r)) for q, r in np.asarray(graph_batch.legal_qr, dtype=np.int32).tolist()]
    legal_tokens = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
    if graph_batch.placements_remaining >= 2:
        first_rows: list[int] = []
        second_rows: list[int] = []
        for a_idx in range(len(legal)):
            for b_idx in range(a_idx + 1, len(legal)):
                first_rows.append(int(legal_tokens[a_idx]))
                second_rows.append(int(legal_tokens[b_idx]))
        pair_first = np.asarray(first_rows, dtype=np.int64)
        pair_second = np.asarray(second_rows, dtype=np.int64)
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
            pair_first = np.full(len(legal), first_token, dtype=np.int64)
            pair_second = legal_tokens.astype(np.int64, copy=True)
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
    pair_count = int(pair_first.shape[0])
    return GraphBatch(
        token_features=graph_batch.token_features,
        token_type=graph_batch.token_type,
        token_qr=graph_batch.token_qr,
        token_mask=graph_batch.token_mask,
        legal_token_indices=graph_batch.legal_token_indices,
        legal_qr=graph_batch.legal_qr,
        legal_mask=graph_batch.legal_mask,
        pair_token_indices=np.zeros(pair_count, dtype=np.int64),
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


def _engine_state_from_history(history: bytes) -> tuple[list[tuple[int, int]], int, int] | None:
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


def collate_graph_batches(batches: Sequence[GraphBatch]) -> GraphBatch:
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

    return GraphBatch(
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
