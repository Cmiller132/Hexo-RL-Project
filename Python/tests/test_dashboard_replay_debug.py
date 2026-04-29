import struct

import pytest

from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.replay import replay_game


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def test_replay_endpoint_returns_policy_weights_regret_and_candidate_debug(tmp_path):
    pytest.importorskip("_engine")
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    history = _move(0, 0, 0)
    game_id = store.insert_game_with_positions(
        run_id="debug-run",
        game_id="debug-game",
        source="unit",
        final_move_history=history,
        outcome=0.0,
        positions=[
            {
                "turn_index": 1,
                "player": 1,
                "move_history": history,
                "root_value": 0.25,
                "policy_target": {},
                "debug": {
                    "is_full_search": False,
                    "outcome": 0.0,
                    "selected_action_value": -0.5,
                    "value_weight": 0.0,
                    "policy_weight": 0.25,
                    "opp_policy_weight": 0.75,
                    "regret_weight": 0.0,
                    "regret_rank": 0.4,
                    "regret_value": 0.4,
                    "policy_target_v2": [(1, 0, 0.7), (0, 1, 0.3)],
                    "opp_policy_target_v2": [(-1, 0, 1.0)],
                    "pair_policy_target_v2": [((1, 0), (0, 1), 1.0)],
                    "sparse_prior_stage": 2,
                    "sparse_prior_root_candidate_count": 12,
                    "sparse_prior_leaf_candidate_count": 5.5,
                    "sparse_prior_root_hit_frac": 0.8,
                    "sparse_prior_leaf_hit_frac": 0.6,
                    "fallback_prior_use": 0.2,
                    "fallback_prior_use_on_mcts_top1": 0.0,
                    "fallback_prior_use_on_mcts_top4": 0.25,
                    "fallback_prior_use_on_mcts_top8": 0.125,
                    "pair_prior_candidate_count": 4,
                    "pair_prior_hit_frac": 0.0,
                    "pair_fallback_prior_use": 1.0,
                },
            }
        ],
    )

    payload = replay_game(store, game_id)
    position = payload["positions"][0]

    assert position["root_value"] == pytest.approx(0.25)
    assert position["selected_action_value"] == pytest.approx(-0.5)
    assert position["final_outcome"] == pytest.approx(0.0)
    assert position["per_step_error"] == pytest.approx(0.25)
    assert position["regret_rank"] == pytest.approx(0.4)
    assert position["regret_value"] == pytest.approx(0.4)
    assert position["value_weight"] == pytest.approx(0.0)
    assert position["policy_weight"] == pytest.approx(0.25)
    assert position["opp_policy_weight"] == pytest.approx(0.75)
    assert position["regret_weight"] == pytest.approx(0.0)
    assert position["policy_target_v2"] == [[1, 0, 0.7], [0, 1, 0.3]]
    assert position["opp_policy_target_v2"] == [[-1, 0, 1.0]]
    assert position["pair_policy_target_v2"] == [[[1, 0], [0, 1], 1.0]]
    assert position["prior_sources"]["sparse_prior_stage"] == 2
    assert position["prior_sources"]["fallback_prior_use"] == pytest.approx(0.2)
    assert position["prior_sources"]["pair_fallback_prior_use"] == pytest.approx(1.0)

    candidates = position["debug"]["candidate_rows"]
    assert candidates["available"] is True
    assert candidates["candidate_count"] > 0
    represented = {(row["q"], row["r"]) for row in candidates["rows"]}
    assert {(1, 0), (0, 1)} <= represented
    assert candidates["missing_mass"] == pytest.approx(0.0)
