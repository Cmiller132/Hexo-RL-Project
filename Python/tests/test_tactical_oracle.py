import pytest

from hexorl.action_contract.candidates import build_candidate_batch
from hexorl.action_contract.tactical_oracle import scan_tactical_oracle


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
