"""Replayable Phase 3 tactical and outside-window evaluation suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from hexorl.action_contract.tactical_oracle import (
    legal_moves_from_stones,
    scan_tactical_oracle,
)


Move = tuple[int, int, int]
Cell = tuple[int, int]
PlayerFn = Callable[[list[Move], int, int], tuple[int | None, int | None]]


@dataclass(frozen=True)
class SuitePosition:
    suite: str
    move_history: tuple[Move, ...]
    current_player: int
    expected_action_set: tuple[Cell, ...]
    expected_pair_set: tuple[tuple[Cell, Cell], ...]
    legal_count: int
    board_span: int
    source_label: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TacticalSuiteResult:
    score: float
    passed: int
    total: int
    positions: tuple[dict[str, object], ...]


def phase3_tactical_suite_positions() -> tuple[SuitePosition, ...]:
    """Return the required Phase 3 replay fixtures.

    The fixtures are authored histories, then the expected action sets are
    derived from the full-board tactical oracle so outside-crop actions remain
    part of the ground truth.
    """

    return (
        _oracle_position("win-now", _p0_line_history(5, start_q=0), "phase3:win_now"),
        _oracle_position("forced-block", _p1_line_history(5, start_q=10), "phase3:forced_block"),
        _oracle_position("open-four", _p0_line_history(4, start_q=0), "phase3:open_four"),
        _oracle_position("open-five", _p0_line_history(5, start_q=0), "phase3:open_five"),
        _oracle_position("two-placement cover", _p1_two_hot_windows_history(), "phase3:two_placement_cover"),
        _oracle_position("outside-window win", _p0_line_history(5, start_q=30), "phase3:outside_window_win"),
        _oracle_position("outside-window block", _p1_line_history(5, start_q=30), "phase3:outside_window_block"),
        _oracle_position("separated-cluster long-span", _separated_cluster_history(), "phase3:long_span"),
        _late_game_high_legal_position(),
    )


def evaluate_tactical_suite(
    player_fn: PlayerFn,
    *,
    positions: Iterable[SuitePosition] | None = None,
    time_ms: int = 100,
) -> TacticalSuiteResult:
    rows: list[dict[str, object]] = []
    passed = 0
    selected_positions = tuple(positions or phase3_tactical_suite_positions())
    for position in selected_positions:
        q, r = player_fn(list(position.move_history), time_ms, position.current_player)
        move = None if q is None or r is None else (int(q), int(r))
        selected_pair = None
        if move is not None and position.expected_pair_set:
            q2, r2 = player_fn(
                [*position.move_history, (position.current_player, move[0], move[1])],
                time_ms,
                position.current_player,
            )
            second = None if q2 is None or r2 is None else (int(q2), int(r2))
            if second is not None:
                selected_pair = _normalize_pair(move, second)
            ok = selected_pair in {_normalize_pair(*pair) for pair in position.expected_pair_set}
        else:
            ok = move in set(position.expected_action_set)
        passed += int(ok)
        rows.append(
            {
                **position.to_dict(),
                "selected_action": move,
                "selected_pair": selected_pair,
                "passed": ok,
            }
        )
    total = len(selected_positions)
    return TacticalSuiteResult(
        score=passed / max(total, 1),
        passed=passed,
        total=total,
        positions=tuple(rows),
    )


def replay_position(position: SuitePosition) -> tuple[dict[Cell, int], int]:
    stones: dict[Cell, int] = {}
    current_player = 0
    placements_remaining = 1
    for idx, (player, q, r) in enumerate(position.move_history):
        if int(player) != current_player:
            raise ValueError(f"{position.suite} move {idx} has player {player}, expected {current_player}")
        cell = (int(q), int(r))
        if cell in stones:
            raise ValueError(f"{position.suite} duplicate cell {cell}")
        stones[cell] = int(player)
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2
    if current_player != position.current_player:
        raise ValueError(
            f"{position.suite} current_player {position.current_player}, replay produced {current_player}"
        )
    return stones, current_player


def _oracle_position(suite: str, history: tuple[Move, ...], source_label: str) -> SuitePosition:
    stones, current_player = _history_state(history)
    legal = legal_moves_from_stones(stones, near_radius=8)
    oracle = scan_tactical_oracle(stones, current_player, legal, offset_q=-16, offset_r=-16)
    if suite == "win-now":
        expected = oracle.win_now_cells
    elif suite == "outside-window win":
        expected = oracle.win_now_cells
    elif suite == "forced-block":
        expected = oracle.forced_block_cells
    elif suite == "outside-window block":
        expected = oracle.forced_block_cells
    elif suite == "open-four":
        expected = oracle.open_four_cells
    elif suite == "open-five":
        expected = oracle.open_five_cells
    elif suite == "two-placement cover":
        expected = oracle.cover_cells
    else:
        expected = oracle.critical_actions
    if not expected:
        expected = tuple(legal[:1])
    pair_set = tuple(oracle.cover_pairs)
    if suite == "two-placement cover" and not pair_set:
        left = tuple(cell for cell in expected if cell[0] < 20)
        right = tuple(cell for cell in expected if cell[0] >= 20)
        pair_set = tuple(_normalize_pair(a, b) for a in left for b in right)
    return SuitePosition(
        suite=suite,
        move_history=history,
        current_player=current_player,
        expected_action_set=tuple(expected),
        expected_pair_set=pair_set,
        legal_count=len(legal),
        board_span=_board_span(stones),
        source_label=source_label,
    )


def _late_game_high_legal_position() -> SuitePosition:
    history: list[Move] = [(0, 0, 0)]
    rings = [
        ((1, 0), (0, 1)),
        ((-1, 1), (-1, 0)),
        ((0, -1), (1, -1)),
        ((2, 0), (0, 2)),
        ((-2, 2), (-2, 0)),
        ((0, -2), (2, -2)),
    ]
    player = 1
    for a, b in rings:
        history.append((player, a[0], a[1]))
        history.append((player, b[0], b[1]))
        player = 1 - player
    stones, current_player = _history_state(tuple(history))
    legal = legal_moves_from_stones(stones, near_radius=8)
    return SuitePosition(
        suite="late-game high-legal-count",
        move_history=tuple(history),
        current_player=current_player,
        expected_action_set=tuple(legal[: min(8, len(legal))]),
        expected_pair_set=(),
        legal_count=len(legal),
        board_span=_board_span(stones),
        source_label="phase3:late_game_high_legal_count",
    )


def _p0_line_history(length: int, *, start_q: int) -> tuple[Move, ...]:
    p0_cells = [(start_q + idx, 0) for idx in range(length)]
    if (0, 0) not in p0_cells:
        p0_cells = [(0, 0), *p0_cells]
    return _paired_history(p0_cells, _distractors(player=1, start_q=-20, count=12))


def _p1_line_history(length: int, *, start_q: int) -> tuple[Move, ...]:
    p1_cells = [(start_q + idx, 0) for idx in range(length)]
    return _paired_history(_distractors(player=0, start_q=-20, count=12), p1_cells)


def _p1_two_hot_windows_history() -> tuple[Move, ...]:
    p1_cells = [(10 + idx, 0) for idx in range(5)] + [(30 + idx, 0) for idx in range(5)]
    return _paired_history(_distractors(player=0, start_q=-30, count=16), p1_cells)


def _separated_cluster_history() -> tuple[Move, ...]:
    p0_cells = [(0, 0), (1, 0), (2, 0), (40, -5), (41, -5), (42, -5)]
    return _paired_history(p0_cells, _distractors(player=1, start_q=-20, count=12))


def _paired_history(p0_cells: list[Cell], p1_cells: list[Cell]) -> tuple[Move, ...]:
    p0 = [cell for cell in p0_cells if cell != (0, 0)]
    history: list[Move] = [(0, 0, 0)]
    turn = 1
    p0_idx = 0
    p1_idx = 0
    filler_idx = 0
    while p0_idx < len(p0) or p1_idx < len(p1_cells):
        if turn == 1:
            pair = p1_cells[p1_idx : p1_idx + 2]
            p1_idx += len(pair)
            while len(pair) < 2:
                filler_idx += 1
                pair.append((-80 - filler_idx, -80 - len(pair)))
            history.extend((1, q, r) for q, r in pair)
            turn = 0
        else:
            pair = p0[p0_idx : p0_idx + 2]
            p0_idx += len(pair)
            while len(pair) < 2:
                filler_idx += 1
                pair.append((80 + filler_idx, 80 + len(pair)))
            history.extend((0, q, r) for q, r in pair)
            turn = 1
    if turn == 0:
        history.extend((0, q, r) for q, r in [(90 + filler_idx, 0), (91 + filler_idx, 0)])
        turn = 1
    if turn == 1:
        history.extend((1, q, r) for q, r in [(-90 - filler_idx, 0), (-91 - filler_idx, 0)])
    return tuple(history)


def _distractors(*, player: int, start_q: int, count: int) -> list[Cell]:
    r = 10 if player == 0 else -10
    return [(start_q - 3 * idx, r + 2 * idx) for idx in range(count)]


def _history_state(history: tuple[Move, ...]) -> tuple[dict[Cell, int], int]:
    stones: dict[Cell, int] = {}
    current_player = 0
    placements_remaining = 1
    for player, q, r in history:
        stones[(int(q), int(r))] = int(player)
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2
    return stones, current_player


def _board_span(stones: dict[Cell, int]) -> int:
    if not stones:
        return 0
    qs = [q for q, _r in stones]
    rs = [r for _q, r in stones]
    return max(max(qs) - min(qs), max(rs) - min(rs))


def _normalize_pair(left: Cell, right: Cell) -> tuple[Cell, Cell]:
    return (left, right) if left <= right else (right, left)
