"""Tactical oracle adapters for Hexo candidate construction.

Production callers use the Rust engine hot-window oracle.  The Python scanner
remains available only as an explicit fixture/diagnostic helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from hexorl.selfplay.records import action_to_board_index


AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
WIN_LENGTH = 6
PLACEMENT_RADIUS = 8
TACTICAL_SCAN_RADIUS = PLACEMENT_RADIUS


@dataclass(frozen=True)
class TacticalOracleResult:
    win_now_cells: tuple[tuple[int, int], ...]
    forced_block_cells: tuple[tuple[int, int], ...]
    open_four_cells: tuple[tuple[int, int], ...]
    open_five_cells: tuple[tuple[int, int], ...]
    cover_cells: tuple[tuple[int, int], ...]
    cover_pairs: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    outside_crop_cells: tuple[tuple[int, int], ...]
    status: str = "quiet"
    current_player: int = 0
    placements_remaining: int = 1

    @property
    def critical_actions(self) -> tuple[tuple[int, int], ...]:
        return _unique_qr(
            self.win_now_cells
            + self.forced_block_cells
            + self.open_four_cells
            + self.open_five_cells
            + self.cover_cells
        )


def parse_history_state(history_bytes: bytes) -> tuple[dict[tuple[int, int], int], int, int]:
    """Return stones, current player, and placements remaining for compact history."""
    if len(history_bytes) % 12 != 0:
        raise ValueError(f"history_bytes length {len(history_bytes)} is not a multiple of 12")
    stones: dict[tuple[int, int], int] = {}
    current_player = 0
    placements_remaining = 1
    for offset in range(0, len(history_bytes), 12):
        player = int.from_bytes(history_bytes[offset : offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4 : offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8 : offset + 12], "little", signed=True)
        if player != current_player:
            raise ValueError(
                f"Invalid compact history: move {offset // 12} stores player {player}, "
                f"expected {current_player}"
            )
        if (q, r) in stones:
            raise ValueError(f"Invalid compact history: duplicate cell ({q}, {r})")
        stones[(q, r)] = player
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2
    return stones, current_player, placements_remaining


def engine_game_from_history(history_bytes: bytes):
    """Replay compact history through the Rust engine when available."""
    engine_cls = _engine_game_class()
    if engine_cls is None:
        return None
    if len(history_bytes) % 12 != 0:
        raise ValueError(f"history_bytes length {len(history_bytes)} is not a multiple of 12")
    game = engine_cls()
    for offset in range(0, len(history_bytes), 12):
        player = int.from_bytes(history_bytes[offset : offset + 4], "little", signed=True)
        q = int.from_bytes(history_bytes[offset + 4 : offset + 8], "little", signed=True)
        r = int.from_bytes(history_bytes[offset + 8 : offset + 12], "little", signed=True)
        current_player = int(_attr_value(game, "current_player", player))
        if player != current_player:
            raise ValueError(
                f"Invalid compact history: move {offset // 12} stores player {player}, "
                f"expected {current_player}"
            )
        game.place(q, r)
    return game


def legal_moves_from_stones(
    stones: Mapping[tuple[int, int], int],
    near_radius: int = TACTICAL_SCAN_RADIUS,
) -> list[tuple[int, int]]:
    if not stones:
        return [(0, 0)]
    radius = max(0, int(near_radius))
    occupied = set(stones)
    legal: set[tuple[int, int]] = set()
    for q, r in occupied:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    candidate = (q + dq, r + dr)
                    if candidate not in occupied:
                        legal.add(candidate)
    return sorted(legal, key=_cell_sort_key)


def scan_tactical_oracle_from_history(
    history_bytes: bytes,
    legal_moves: Sequence[tuple[int, int]] | None = None,
    *,
    offset_q: int = -16,
    offset_r: int = -16,
    near_radius: int = TACTICAL_SCAN_RADIUS,
    allow_python_fallback: bool = False,
) -> TacticalOracleResult:
    game = engine_game_from_history(history_bytes)
    if game is not None and hasattr(game, "tactical_oracle"):
        return scan_tactical_oracle_from_game(
            game,
            legal_moves,
            offset_q=offset_q,
            offset_r=offset_r,
            near_radius=near_radius,
        )
    if not allow_python_fallback:
        if game is None:
            raise RuntimeError("engine tactical oracle is required but the Rust engine is unavailable")
        raise RuntimeError("engine tactical_oracle method is required for production tactical labels")
    stones, current_player, _placements_remaining = parse_history_state(history_bytes)
    legal = legal_moves if legal_moves is not None else legal_moves_from_stones(stones, near_radius)
    return scan_tactical_oracle(
        stones,
        current_player,
        legal,
        offset_q=offset_q,
        offset_r=offset_r,
    )


def scan_tactical_oracle_from_game(
    game: object,
    legal_moves: Sequence[tuple[int, int]] | None = None,
    *,
    offset_q: int = -16,
    offset_r: int = -16,
    near_radius: int = TACTICAL_SCAN_RADIUS,
    allow_python_fallback: bool = False,
) -> TacticalOracleResult:
    """Return the engine-backed tactical oracle for an already-restored game."""
    pieces = getattr(game, "board_pieces", lambda: [])()
    stones = {(int(q), int(r)): int(player) for q, r, player in pieces}
    current_player = int(_attr_value(game, "current_player", 0))
    placements_remaining = int(_attr_value(game, "placements_remaining", 1))

    if hasattr(game, "tactical_oracle"):
        payload = game.tactical_oracle(int(near_radius))
        result = _result_from_engine_payload(payload, offset_q=offset_q, offset_r=offset_r)
        if legal_moves is None:
            return result
        return _filter_result_to_legal(result, legal_moves)

    if not allow_python_fallback:
        raise RuntimeError("engine tactical_oracle method is required for production tactical labels")

    if hasattr(game, "legal_moves_near"):
        legal_moves = game.legal_moves_near(int(near_radius))
    elif hasattr(game, "legal_moves"):
        legal_moves = game.legal_moves()
    else:
        legal_moves = legal_moves_from_stones(stones, near_radius)
    return scan_tactical_oracle(
        stones,
        current_player,
        legal_moves,
        offset_q=offset_q,
        offset_r=offset_r,
        placements_remaining=placements_remaining,
    )


def _filter_result_to_legal(
    result: TacticalOracleResult,
    legal_moves: Sequence[tuple[int, int]],
) -> TacticalOracleResult:
    """Keep engine oracle labels keyed to an explicit legal table."""
    legal = {(int(q), int(r)) for q, r in legal_moves}

    def cells(rows: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        return _unique_qr((int(q), int(r)) for q, r in rows if (int(q), int(r)) in legal)

    cover_pairs = tuple(
        ((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))
        for a, b in result.cover_pairs
        if (int(a[0]), int(a[1])) in legal and (int(b[0]), int(b[1])) in legal
    )
    return TacticalOracleResult(
        win_now_cells=cells(result.win_now_cells),
        forced_block_cells=cells(result.forced_block_cells),
        open_four_cells=cells(result.open_four_cells),
        open_five_cells=cells(result.open_five_cells),
        cover_cells=cells(result.cover_cells),
        cover_pairs=cover_pairs,
        outside_crop_cells=cells(result.outside_crop_cells),
        status=result.status,
        current_player=result.current_player,
        placements_remaining=result.placements_remaining,
    )


def scan_tactical_oracle(
    stones: Mapping[tuple[int, int], int],
    current_player: int,
    legal_moves: Sequence[tuple[int, int]],
    *,
    offset_q: int = -16,
    offset_r: int = -16,
    placements_remaining: int = 1,
) -> TacticalOracleResult:
    legal = _unique_qr((int(q), int(r)) for q, r in legal_moves if (int(q), int(r)) not in stones)
    legal_set = set(legal)
    player = int(current_player)
    opponent = 1 - player

    win_now = [cell for cell in legal if _would_win(stones, player, cell)]
    own_hot = _hot_windows(stones, player, legal_set)
    opp_hot = _hot_windows(stones, opponent, legal_set)

    open_four: list[tuple[int, int]] = []
    open_five: list[tuple[int, int]] = []
    for window in own_hot:
        if len(window) == 2:
            open_four.extend(window)
        elif len(window) == 1:
            open_five.extend(window)

    forced: list[tuple[int, int]] = []
    for window in opp_hot:
        forced.extend(window)

    cover_pairs = _cover_pairs(opp_hot)
    cover_cells = _unique_qr(cell for pair in cover_pairs for cell in pair)
    forced_unique = _unique_qr(forced)
    tactical = _unique_qr(
        tuple(win_now)
        + tuple(forced_unique)
        + tuple(open_four)
        + tuple(open_five)
        + tuple(cover_cells)
    )
    outside = tuple(cell for cell in tactical if action_to_board_index(cell[0], cell[1], offset_q, offset_r) < 0)

    return TacticalOracleResult(
        status=_fallback_status(bool(win_now), bool(forced_unique), bool(cover_pairs), int(placements_remaining)),
        current_player=player,
        placements_remaining=int(placements_remaining),
        win_now_cells=tuple(_unique_qr(win_now)),
        forced_block_cells=tuple(forced_unique),
        open_four_cells=tuple(_unique_qr(open_four)),
        open_five_cells=tuple(_unique_qr(open_five)),
        cover_cells=tuple(_unique_qr(tuple(forced_unique) + tuple(cover_cells))),
        cover_pairs=tuple(cover_pairs),
        outside_crop_cells=outside,
    )


def _would_win(
    stones: Mapping[tuple[int, int], int],
    player: int,
    cell: tuple[int, int],
) -> bool:
    q, r = cell
    for dq, dr in AXES:
        run = 1
        run += _count_axis(stones, player, q, r, dq, dr)
        run += _count_axis(stones, player, q, r, -dq, -dr)
        if run >= WIN_LENGTH:
            return True
    return False


def _count_axis(
    stones: Mapping[tuple[int, int], int],
    player: int,
    q: int,
    r: int,
    dq: int,
    dr: int,
) -> int:
    count = 0
    cq, cr = q + dq, r + dr
    while stones.get((cq, cr)) == player:
        count += 1
        cq += dq
        cr += dr
    return count


def _hot_windows(
    stones: Mapping[tuple[int, int], int],
    player: int,
    legal_set: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    opponent = 1 - int(player)
    windows: set[tuple[tuple[int, int], ...]] = set()
    anchors = {cell for cell, owner in stones.items() if owner == player}
    for cell in anchors:
        for dq, dr in AXES:
            for start_idx in range(WIN_LENGTH):
                sq = cell[0] - start_idx * dq
                sr = cell[1] - start_idx * dr
                cells = tuple((sq + i * dq, sr + i * dr) for i in range(WIN_LENGTH))
                own_count = 0
                blocked = False
                empties: list[tuple[int, int]] = []
                for pos in cells:
                    owner = stones.get(pos)
                    if owner == player:
                        own_count += 1
                    elif owner == opponent:
                        blocked = True
                        break
                    elif pos in legal_set:
                        empties.append(pos)
                    else:
                        blocked = True
                        break
                if not blocked and own_count >= 4 and own_count + len(empties) == WIN_LENGTH:
                    windows.add(tuple(sorted(empties, key=_cell_sort_key)))
    return sorted(windows, key=lambda window: (len(window), tuple(_cell_sort_key(c) for c in window)))


def _cover_pairs(
    hot_windows: Sequence[tuple[tuple[int, int], ...]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if not hot_windows:
        return []
    cells = _unique_qr(cell for window in hot_windows for cell in window)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in combinations(cells, 2):
        if all(a in window or b in window for window in hot_windows):
            pairs.append((a, b) if a <= b else (b, a))
    return sorted(set(pairs), key=lambda pair: (_cell_sort_key(pair[0]), _cell_sort_key(pair[1])))


def _cell_sort_key(cell: tuple[int, int]) -> tuple[int, int, int]:
    q, r = int(cell[0]), int(cell[1])
    return (max(abs(q), abs(r), abs(q + r)), q, r)


def _unique_qr(items: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for q, r in items:
        cell = (int(q), int(r))
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return tuple(sorted(out, key=_cell_sort_key))


def _fallback_status(has_win: bool, has_forced: bool, has_pairs: bool, placements_remaining: int) -> str:
    if has_win:
        return "winning_turn"
    if not has_forced:
        return "quiet"
    if placements_remaining <= 1 or has_pairs:
        return "must_block"
    return "unblockable"


def _result_from_engine_payload(
    payload: Mapping[str, object],
    *,
    offset_q: int,
    offset_r: int,
) -> TacticalOracleResult:
    win_now = _unique_qr(_coerce_cells(payload.get("win_now_cells", ())))
    forced = _unique_qr(_coerce_cells(payload.get("forced_block_cells", ())))
    cover = _unique_qr(_coerce_cells(payload.get("cover_cells", ())))
    open_four = _unique_qr(_coerce_cells(payload.get("open_four_cells", ())))
    open_five = _unique_qr(_coerce_cells(payload.get("open_five_cells", ())))
    pairs = _coerce_pairs(payload.get("cover_pairs", ()))
    tactical = _unique_qr(win_now + forced + cover + open_four + open_five)
    outside = tuple(cell for cell in tactical if action_to_board_index(cell[0], cell[1], offset_q, offset_r) < 0)
    return TacticalOracleResult(
        status=str(payload.get("status", "quiet")),
        current_player=int(payload.get("current_player", 0)),
        placements_remaining=int(payload.get("placements_remaining", 1)),
        win_now_cells=win_now,
        forced_block_cells=forced,
        open_four_cells=open_four,
        open_five_cells=open_five,
        cover_cells=cover,
        cover_pairs=pairs,
        outside_crop_cells=outside,
    )


def _coerce_cells(raw: object) -> tuple[tuple[int, int], ...]:
    return tuple((int(cell[0]), int(cell[1])) for cell in raw or ())


def _coerce_pairs(raw: object) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    pairs = []
    for first, second in raw or ():
        a = (int(first[0]), int(first[1]))
        b = (int(second[0]), int(second[1]))
        pairs.append((a, b) if a <= b else (b, a))
    return tuple(sorted(set(pairs), key=lambda pair: (_cell_sort_key(pair[0]), _cell_sort_key(pair[1]))))


def _engine_game_class():
    try:
        import _engine  # type: ignore
    except Exception:
        return None
    return getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame", None)


def _attr_value(obj: object, name: str, default: int | bool | None = None):
    value = getattr(obj, name, default)
    return value() if callable(value) else value
