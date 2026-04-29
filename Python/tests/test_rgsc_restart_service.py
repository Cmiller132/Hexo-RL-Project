import struct

import pytest

from hexorl.selfplay.records import GameRecord, PositionRecord
from hexorl.selfplay.rgsc import RGSCRestartService, encode_move_history, restore_game_from_history


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


class _FakeHexGame:
    def __init__(self):
        self.current_player = 0
        self.placements_remaining = 1
        self.is_over = False
        self._stones = set()

    def place(self, q: int, r: int):
        if self.is_over:
            raise ValueError("game over")
        if (q, r) in self._stones:
            raise ValueError("occupied")
        if not self._stones and (q, r) != (0, 0):
            raise ValueError("opening must be origin")
        self._stones.add((q, r))
        self.placements_remaining -= 1
        if self.placements_remaining == 0:
            self.current_player = 1 - self.current_player
            self.placements_remaining = 2


def test_rgsc_restart_restores_current_player_and_turn_phase():
    history = _move(0, 0, 0) + _move(1, 1, 0)

    restored = restore_game_from_history(history, _FakeHexGame, max_game_moves=20)

    assert restored.ok
    assert restored.move_count == 2
    assert restored.current_player == 1
    assert restored.placements_remaining == 1


def test_rgsc_restart_rejects_illegal_or_stale_history():
    bad_player = _move(0, 0, 0) + _move(0, 1, 0)
    restored = restore_game_from_history(bad_player, _FakeHexGame, max_game_moves=20)
    assert not restored.ok
    assert "player_mismatch" in restored.reason

    malformed = restore_game_from_history(b"bad", _FakeHexGame, max_game_moves=20)
    assert not malformed.ok
    assert "multiple of 12" in malformed.reason


def test_rgsc_restart_samples_from_prb_when_beta_one():
    history = _move(0, 0, 0) + _move(1, 1, 0)
    service = RGSCRestartService(beta=1.0, capacity=4, seed=7, enabled=True)
    assert service.prb.add(history, regret=3.0, rank_score=3.0, game_id=1)

    decision = service.maybe_restart(_FakeHexGame, max_game_moves=20)

    assert decision.attempted
    assert decision.used
    assert decision.move_history == history
    assert decision.move_count == 2
    assert service.metrics["rgsc_restart_successes"] == pytest.approx(1.0)


def test_prb_ema_update_after_restart_game():
    history = _move(0, 0, 0) + _move(1, 1, 0)
    service = RGSCRestartService(
        beta=1.0,
        capacity=4,
        ema_alpha=0.5,
        seed=11,
        enabled=True,
    )
    service.prb.add(history, regret=1.0, rank_score=1.0, game_id=1)
    record = GameRecord(
        positions=[
            PositionRecord(
                history,
                {1: 1.0},
                0.0,
                player=1,
                selected_action_value=0.0,
                outcome=1.0,
                game_id=2,
                regret_rank=4.0,
                regret_value=4.0,
                regret_weight=1.0,
            )
        ],
        outcome=1.0,
        game_id=2,
    )

    service.observe_game(record, restart_entry_index=0)

    entries = service.prb.get_entries()
    assert entries[0].refresh_count == 1
    assert entries[0].observed_regret == pytest.approx(4.0)
    assert service.metrics["rgsc_prb_refreshes"] == pytest.approx(1.0)


def test_rgsc_tree_node_states_can_enter_prb():
    service = RGSCRestartService(beta=1.0, capacity=4, seed=13, enabled=True)
    history = encode_move_history([(0, 0, 0), (1, 1, 0), (1, 2, 0)])

    inserted = service.observe_tree_node_candidates(
        [(history, 2.5, 3.5)],
        game_id=7,
    )

    assert inserted == 1
    entries = service.prb.get_entries()
    assert entries[0].move_history == history
    assert entries[0].rank_score == pytest.approx(2.5)
    assert entries[0].observed_regret == pytest.approx(3.5)
    assert entries[0].source == "mcts_tree_node_scored_candidate"
    assert service.metrics["rgsc_tree_node_insertions"] == pytest.approx(1.0)


def test_rgsc_tree_node_source_is_persisted_honestly():
    service = RGSCRestartService(beta=1.0, capacity=4, seed=17, enabled=True)
    history = encode_move_history([(0, 0, 0), (1, 1, 0)])

    service.observe_tree_node_candidates(
        [(history, 1.0, 1.0)],
        game_id=3,
        score_source="mcts_tree_node_depth_heuristic",
    )

    assert service.prb.get_entries()[0].source == "mcts_tree_node_depth_heuristic"
