import pytest

from hexorl.action_contract.candidates import build_candidate_batch
from hexorl.action_contract.tactical_oracle import (
    TACTICAL_SCAN_RADIUS,
    scan_tactical_oracle,
    scan_tactical_oracle_from_game,
    scan_tactical_oracle_from_history,
)


def test_full_board_oracle_finds_win_outside_crop():
    stones = {(30 + i, 0): 0 for i in range(5)}
    legal = [(29, 0), (35, 0), (0, 0)]

    result = scan_tactical_oracle(stones, 0, legal, offset_q=-16, offset_r=-16)

    assert (35, 0) in result.win_now_cells
    assert (35, 0) in result.open_five_cells
    assert (35, 0) in result.outside_crop_cells


def test_full_board_oracle_finds_forced_block_outside_crop():
    stones = {(30 + i, 0): 1 for i in range(5)}
    legal = [(29, 0), (35, 0), (0, 0)]

    result = scan_tactical_oracle(stones, 0, legal, offset_q=-16, offset_r=-16)

    assert {(29, 0), (35, 0)} <= set(result.forced_block_cells)
    assert {(29, 0), (35, 0)} <= set(result.cover_cells)
    assert {(29, 0), (35, 0)} <= set(result.outside_crop_cells)


def test_full_board_oracle_reports_open_four_and_cover_pairs():
    stones = {(30 + i, 0): 0 for i in range(4)}
    legal = [(29, 0), (34, 0), (35, 0)]

    result = scan_tactical_oracle(stones, 0, legal, offset_q=-16, offset_r=-16)

    assert {(29, 0), (34, 0), (35, 0)} & set(result.open_four_cells)


def test_candidate_builder_includes_oracle_critical_cells_outside_crop():
    stones = {(30 + i, 0): 1 for i in range(5)}
    legal = [(29, 0), (35, 0), (0, 0)]
    oracle = scan_tactical_oracle(stones, 0, legal, offset_q=-16, offset_r=-16)

    cand = build_candidate_batch(
        legal,
        [],
        offset_q=-16,
        offset_r=-16,
        budget=1,
        forced_block_moves=oracle.forced_block_cells,
        cover_cells=oracle.cover_cells,
    )

    represented = {tuple(qr) for qr in cand.qr[cand.mask]}
    assert {(29, 0), (35, 0)} <= represented
    assert cand.recall_forced_block == pytest.approx(1.0)
    assert cand.discovery_forced_block == pytest.approx(1.0)


def test_engine_backed_oracle_uses_hot_window_cells_with_radius_three():
    _engine = pytest.importorskip("_engine")
    game_cls = getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame")
    game = game_cls()
    game.set_position([(0, 0, 0)] + [(1 + i, 0, 1) for i in range(5)], 0, 2)

    payload = game.tactical_oracle(TACTICAL_SCAN_RADIUS)

    assert payload["status"] == "must_block"
    assert payload["current_player"] == 0
    assert (6, 0) in payload["forced_block_cells"]
    assert (6, 0) in payload["cover_cells"]


def test_engine_backed_oracle_stays_engine_backed_with_explicit_legal_rows():
    _engine = pytest.importorskip("_engine")
    game_cls = getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame")
    game = game_cls()
    game.set_position([(0, 0, 0)] + [(1 + i, 0, 1) for i in range(5)], 0, 2)

    result = scan_tactical_oracle_from_game(
        game,
        legal_moves=[(6, 0), (30, 30)],
        near_radius=TACTICAL_SCAN_RADIUS,
    )

    assert result.status == "must_block"
    assert result.forced_block_cells == ((6, 0),)
    assert result.cover_cells == ((6, 0),)


def test_history_oracle_rejects_illegal_origin_via_engine_replay():
    pytest.importorskip("_engine")
    bad_opening = (0).to_bytes(4, "little", signed=True)
    bad_opening += (1).to_bytes(4, "little", signed=True)
    bad_opening += (0).to_bytes(4, "little", signed=True)

    with pytest.raises(ValueError, match="origin"):
        scan_tactical_oracle_from_history(bad_opening)


def test_history_oracle_requires_engine_oracle_for_production(monkeypatch):
    import hexorl.action_contract.tactical_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "_engine_game_class", lambda: None)
    with pytest.raises(RuntimeError, match="engine tactical oracle is required"):
        scan_tactical_oracle_from_history(b"")

    result = scan_tactical_oracle_from_history(b"", allow_fixture_scan=True)
    assert result.win_now_cells == ()
