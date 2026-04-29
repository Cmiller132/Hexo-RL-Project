"""Pure Python full-board tactical scan for Hexo candidate construction."""

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
) -> TacticalOracleResult:
    stones, current_player, _placements_remaining = parse_history_state(history_bytes)
    legal = legal_moves if legal_moves is not None else legal_moves_from_stones(stones, near_radius)
    return scan_tactical_oracle(
        stones,
        current_player,
        legal,
        offset_q=offset_q,
        offset_r=offset_r,
    )


def scan_tactical_oracle(
    stones: Mapping[tuple[int, int], int],
    current_player: int,
    legal_moves: Sequence[tuple[int, int]],
    *,
    offset_q: int = -16,
    offset_r: int = -16,
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
    anchors = set(stones) | set(legal_set)
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
