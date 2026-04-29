"""Engine, reference-oracle, D6, and tactical-rule invariants."""

from __future__ import annotations

import random
import struct
from collections.abc import Iterable, Mapping, Sequence

import pytest

from hexorl.action_contract.tactical_oracle import (
    TACTICAL_SCAN_RADIUS,
    legal_moves_from_stones,
    scan_tactical_oracle_from_history,
)

_engine = pytest.importorskip("_engine")

AXES: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))
WIN_LENGTH = 6
PLACEMENT_RADIUS = 8

AXIS_Q_WIN_HISTORY: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, -1),
    (1, 1, -1),
    (0, 1, 0),
    (0, 2, 0),
    (1, -1, 1),
    (1, 0, -2),
    (0, 3, 0),
    (0, 4, 0),
    (1, -1, 0),
    (1, -2, 1),
    (0, 5, 0),
)


def test_engine_opening_and_two_placement_turn_order():
    game = _engine.PyHexGame()

    assert game.current_player == 0
    assert game.placements_remaining == 1
    assert set(game.legal_moves()) == {(0, 0)}
    with pytest.raises(ValueError, match="origin"):
        game.place(1, 0)

    assert game.place(0, 0) is True
    assert game.current_player == 1
    assert game.placements_remaining == 2

    assert game.place(1, 0) is False
    assert game.current_player == 1
    assert game.placements_remaining == 1

    assert game.place(0, 1) is True
    assert game.current_player == 0
    assert game.placements_remaining == 2

    assert game.place(-1, 1) is False
    assert game.current_player == 0
    assert game.placements_remaining == 1

    assert game.place(-1, 0) is True
    assert game.current_player == 1
    assert game.placements_remaining == 2


def test_engine_legal_moves_match_radius_rule():
    for moves in _sample_histories():
        game = _game_from_history(moves)
        stones = _stones_from_moves(moves)

        expected = _reference_legal_moves(stones, winner=_reference_winner(stones))
        actual = set(game.legal_moves())

        assert actual == expected, f"history={moves!r}"
        assert not (actual & set(stones))
        if stones and actual:
            for cell in actual:
                assert min(_hex_distance(cell, stone) for stone in stones) <= PLACEMENT_RADIUS


def test_engine_winner_matches_reference_axis_scan():
    axis_wins = [
        [(i, 0, 0) for i in range(WIN_LENGTH)],
        [(0, i, 0) for i in range(WIN_LENGTH)],
        [(i, -i, 0) for i in range(WIN_LENGTH)],
    ]
    for pieces in axis_wins:
        game = _engine.PyHexGame()
        game.set_position(pieces, 0, 2)
        stones = {(q, r): player for q, r, player in pieces}
        assert game.winner == _reference_winner(stones)

    for moves in _sample_histories() + [list(AXIS_Q_WIN_HISTORY)]:
        game = _game_from_history(moves)
        stones = _stones_from_moves(moves)
        assert game.winner == _reference_winner(stones), f"history={moves!r}"


def test_engine_rejects_post_terminal_moves():
    game = _game_from_history(AXIS_Q_WIN_HISTORY)

    assert game.is_over
    assert game.winner == 0
    assert game.placements_remaining == 0
    assert game.legal_moves() == []
    with pytest.raises(ValueError, match="over"):
        game.place(0, 1)

    loaded = _engine.PyHexGame()
    loaded.set_position([(0, i, 0) for i in range(WIN_LENGTH)], 0, 2)
    assert loaded.is_over
    assert loaded.placements_remaining == 0
    assert loaded.legal_moves() == []
    with pytest.raises(ValueError, match="over"):
        loaded.place(0, 1)


def test_compact_replay_reconstructs_terminal_state():
    original = _game_from_history(AXIS_Q_WIN_HISTORY)
    history = original.move_history_bytes()
    moves = _unpack_history(history)
    replayed = _game_from_history(moves)

    assert moves == list(AXIS_Q_WIN_HISTORY)
    assert set(replayed.board_pieces()) == set(original.board_pieces())
    assert replayed.move_history() == original.move_history()
    assert replayed.is_over == original.is_over
    assert replayed.winner == original.winner
    assert replayed.current_player == original.current_player
    assert replayed.placements_remaining == original.placements_remaining


def test_rust_and_python_legal_moves_agree_on_random_histories():
    for moves in _sample_histories(seed=91, count=10, max_len=28):
        game = _game_from_history(moves)
        stones = _stones_from_moves(moves)
        expected = _reference_legal_moves(stones, winner=_reference_winner(stones))
        assert set(game.legal_moves()) == expected


def test_rust_and_python_winner_scans_agree_on_random_histories():
    for moves in _sample_histories(seed=123, count=10, max_len=36):
        game = _game_from_history(moves)
        stones = _stones_from_moves(moves)
        assert game.winner == _reference_winner(stones), f"history={moves!r}"


def test_rust_and_python_window_oracles_agree_on_tactical_fixtures():
    fixtures = [
        ([(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)], 0, 2),
        ([(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2),
        ([(0, 0, 0), (1, -1, 0), (2, -2, 0), (3, -3, 0), (4, -4, 0)], 1, 2),
    ]

    for pieces, current_player, remaining in fixtures:
        game = _engine.PyHexGame()
        game.set_position(pieces, current_player, remaining)
        stones = {(q, r): player for q, r, player in pieces}
        for player in (0, 1):
            rust_windows = {
                tuple((q, r) for q, r, _occupied in window)
                for window in game.get_threat_windows(player)
            }
            assert rust_windows == _reference_hot_windows(stones, player)


def test_d6_all_12_preserve_engine_state():
    for moves in _sample_histories(seed=7, count=8, max_len=24) + [list(AXIS_Q_WIN_HISTORY)]:
        original = _game_from_history(moves)
        original_pieces = {(q, r, player) for q, r, player in original.board_pieces()}
        original_winner = original.winner

        for sym in range(12):
            transformed_moves = _transform_history(moves, sym)
            transformed = _game_from_history(transformed_moves)
            expected_pieces = {(*_d6(q, r, sym), player) for q, r, player in original_pieces}

            assert {(q, r, player) for q, r, player in transformed.board_pieces()} == expected_pieces
            assert transformed.current_player == original.current_player
            assert transformed.placements_remaining == original.placements_remaining
            assert transformed.winner == original_winner


def test_d6_legal_mask_bijection_random_histories():
    for moves in _sample_histories(seed=19, count=8, max_len=22):
        game = _game_from_history(moves)
        if game.is_over:
            continue
        legal = set(game.legal_moves())

        for sym in range(12):
            transformed = _game_from_history(_transform_history(moves, sym))
            expected = {_d6(q, r, sym) for q, r in legal}
            assert set(transformed.legal_moves()) == expected


def test_tactical_oracle_default_legal_generation_is_exact_placement_radius():
    legal = set(legal_moves_from_stones({(0, 0): 0}, near_radius=TACTICAL_SCAN_RADIUS))

    assert TACTICAL_SCAN_RADIUS == 3
    assert (TACTICAL_SCAN_RADIUS, 0) in legal
    assert (0, TACTICAL_SCAN_RADIUS) in legal
    assert (TACTICAL_SCAN_RADIUS, -TACTICAL_SCAN_RADIUS) in legal
    assert (TACTICAL_SCAN_RADIUS + 1, 0) not in legal
    assert (0, 0) not in legal
    assert len(legal) == 3 * TACTICAL_SCAN_RADIUS * (TACTICAL_SCAN_RADIUS + 1)


def test_tactical_oracle_default_radius_finds_forced_cover_pair_from_history():
    history = _pack_history(
        [
            (0, 0, 0),
            (1, 0, -1),
            (1, 0, -2),
            (0, 1, 0),
            (0, 2, 0),
            (1, 1, -1),
            (1, 1, -2),
            (0, 3, 0),
            (0, 2, -2),
        ]
    )

    result = scan_tactical_oracle_from_history(history, near_radius=TACTICAL_SCAN_RADIUS)

    assert {(-1, 0), (4, 0)} <= set(result.forced_block_cells)
    assert ((-1, 0), (4, 0)) in result.cover_pairs
    assert set(result.critical_actions) <= set(result.cover_cells)
    stones = _stones_from_moves(_unpack_history(history))
    for cell in result.critical_actions:
        assert min(_hex_distance(cell, stone) for stone in stones) <= TACTICAL_SCAN_RADIUS


def test_tactical_oracle_outputs_are_d6_equivariant():
    moves = [
        (0, 0, 0),
        (1, 0, -1),
        (1, 0, -2),
        (0, 1, 0),
        (0, 2, 0),
        (1, 1, -1),
        (1, 1, -2),
        (0, 3, 0),
        (0, 2, -2),
    ]
    base = scan_tactical_oracle_from_history(_pack_history(moves), near_radius=TACTICAL_SCAN_RADIUS)

    for sym in range(12):
        transformed = scan_tactical_oracle_from_history(
            _pack_history(_transform_history(moves, sym)),
            near_radius=TACTICAL_SCAN_RADIUS,
        )

        assert transformed.status == base.status
        assert set(transformed.win_now_cells) == {_d6(q, r, sym) for q, r in base.win_now_cells}
        assert set(transformed.forced_block_cells) == {
            _d6(q, r, sym) for q, r in base.forced_block_cells
        }
        assert set(transformed.cover_cells) == {_d6(q, r, sym) for q, r in base.cover_cells}
        expected_pairs = {
            tuple(sorted((_d6(a[0], a[1], sym), _d6(b[0], b[1], sym))))
            for a, b in base.cover_pairs
        }
        actual_pairs = {tuple(sorted(pair)) for pair in transformed.cover_pairs}
        assert actual_pairs == expected_pairs


def _game_from_history(moves: Sequence[tuple[int, int, int]]):
    game = _engine.PyHexGame()
    for player, q, r in moves:
        assert player == game.current_player
        game.place(q, r)
    return game


def _sample_histories(
    *,
    seed: int = 17,
    count: int = 8,
    max_len: int = 24,
) -> list[list[tuple[int, int, int]]]:
    rng = random.Random(seed)
    histories: list[list[tuple[int, int, int]]] = [[], [(0, 0, 0)]]
    for _case in range(count):
        game = _engine.PyHexGame()
        moves: list[tuple[int, int, int]] = []
        target_len = rng.randint(1, max_len)
        for _step in range(target_len):
            legal = sorted(game.legal_moves())
            if not legal:
                break
            q, r = rng.choice(legal)
            player = game.current_player
            game.place(q, r)
            moves.append((player, q, r))
            histories.append(list(moves))
            if game.is_over:
                break
    return histories


def _stones_from_moves(moves: Iterable[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    return {(q, r): player for player, q, r in moves}


def _reference_legal_moves(
    stones: Mapping[tuple[int, int], int],
    *,
    winner: int | None = None,
) -> set[tuple[int, int]]:
    if winner is not None:
        return set()
    if not stones:
        return {(0, 0)}
    legal: set[tuple[int, int]] = set()
    for q, r in stones:
        for dq in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
            for dr in range(-PLACEMENT_RADIUS, PLACEMENT_RADIUS + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= PLACEMENT_RADIUS:
                    cell = (q + dq, r + dr)
                    if cell not in stones:
                        legal.add(cell)
    return legal


def _reference_winner(stones: Mapping[tuple[int, int], int]) -> int | None:
    for player in (0, 1):
        player_stones = {cell for cell, owner in stones.items() if owner == player}
        for q, r in sorted(player_stones):
            for dq, dr in AXES:
                if (q - dq, r - dr) in player_stones:
                    continue
                run = 0
                cq, cr = q, r
                while (cq, cr) in player_stones:
                    run += 1
                    cq += dq
                    cr += dr
                if run >= WIN_LENGTH:
                    return player
    return None


def _reference_hot_windows(
    stones: Mapping[tuple[int, int], int],
    player: int,
) -> set[tuple[tuple[int, int], ...]]:
    player_cells = {cell for cell, owner in stones.items() if owner == player}
    opponent = 1 - player
    windows: set[tuple[tuple[int, int], ...]] = set()
    for q, r in player_cells:
        for dq, dr in AXES:
            for back in range(WIN_LENGTH):
                start = (q - back * dq, r - back * dr)
                cells = tuple((start[0] + i * dq, start[1] + i * dr) for i in range(WIN_LENGTH))
                own = sum(1 for cell in cells if stones.get(cell) == player)
                opp = sum(1 for cell in cells if stones.get(cell) == opponent)
                if own >= 4 and opp == 0 and own < WIN_LENGTH:
                    windows.add(cells)
    return windows


def _pack_history(moves: Sequence[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", player, q, r))
    return bytes(out)


def _unpack_history(history: bytes) -> list[tuple[int, int, int]]:
    return [struct.unpack_from("<iii", history, offset) for offset in range(0, len(history), 12)]


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def _d6(q: int, r: int, sym: int) -> tuple[int, int]:
    sym %= 12
    if sym == 0:
        return q, r
    if sym == 1:
        return -r, q + r
    if sym == 2:
        return -q - r, q
    if sym == 3:
        return -q, -r
    if sym == 4:
        return r, -q - r
    if sym == 5:
        return q + r, -q
    if sym == 6:
        return r, q
    if sym == 7:
        return -q, q + r
    if sym == 8:
        return -q - r, r
    if sym == 9:
        return -r, -q
    if sym == 10:
        return q, -q - r
    return q + r, -r


def _transform_history(moves: Sequence[tuple[int, int, int]], sym: int) -> list[tuple[int, int, int]]:
    return [(player, *_d6(q, r, sym)) for player, q, r in moves]
