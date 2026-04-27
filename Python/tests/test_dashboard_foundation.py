import struct

import numpy as np
import pytest
import torch

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import evaluate_all, get_prototype
from hexorl.dashboard.checkpoints import index_checkpoint
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.fixtures import ClassicalFixtureConfig, generate_classical_fixtures
from hexorl.dashboard.play import apply_move, create_session, session_payload, undo_move
from hexorl.dashboard.recorder import RunRecorder
from hexorl.eval.players import NoisyModelPlayer, NoisyPolicyConfig
from hexorl.eval.arena import ArenaStats, MatchResult
from hexorl.model.network import HexNet
from hexorl.selfplay.records import GameRecord, PositionRecord, action_to_board_index


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def test_dashboard_store_records_game_and_json_payloads(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    recorder = RunRecorder(store, "test-run", tmp_path / "events.jsonl")
    history = _move(0, 0, 0) + _move(1, 1, 0)
    game = GameRecord(
        positions=[
            PositionRecord(b"", {action_to_board_index(0, 0): 1.0}, 0.2, player=0),
            PositionRecord(_move(0, 0, 0), {action_to_board_index(1, 0): 1.0}, -0.1, player=1, turn_index=1),
        ],
        outcome=1.0,
        game_id=7,
        game_length=2,
        final_move_history=history,
    )

    game_row_id = recorder.game(game, source="unit")
    rows = store.rows("SELECT * FROM games WHERE game_id=?", (game_row_id,))
    assert rows[0]["final_history_b64"] == history
    assert rows[0]["payload_json"]["positions"] == 2
    positions = store.rows("SELECT * FROM positions WHERE game_id=? ORDER BY turn_index", (game_row_id,))
    assert len(positions) == 2
    assert positions[0]["policy_json"]
    assert (tmp_path / "events.jsonl").exists()


def test_checkpoint_indexing_extracts_current_hexorl_metadata(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    model = HexNet(channels=4, blocks=1, heads=["policy", "value", "axis", "axis_delta_norm"])
    path = tmp_path / "epoch_0003.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 3,
            "global_step": 12,
        },
        path,
    )

    result = index_checkpoint(path, store, run_id="ckpt-run")

    assert result.is_loadable
    assert result.epoch == 3
    assert result.global_step == 12
    assert result.model_heads == ["axis", "axis_delta_norm", "policy", "value"]
    rows = store.rows("SELECT * FROM checkpoints WHERE checkpoint_id=?", (result.checkpoint_id,))
    assert rows[0]["sha256"] == result.sha256


def test_replay_and_play_session_roundtrip(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    session = create_session(store)
    payload = session_payload(store, session.session_id)
    assert payload["position"]["current_player"] == 0
    legal = payload["position"]["legal_moves"]
    move = legal[0]

    apply_move(store, session.session_id, move["q"], move["r"])
    after = session_payload(store, session.session_id)
    assert after["position"]["turn_index"] == 1
    assert after["position"]["stones"][0] == {"q": move["q"], "r": move["r"], "player": 0}

    apply_move(store, session.session_id, 1, -1)
    second = session_payload(store, session.session_id)
    assert second["position"]["turn_index"] == 2
    assert second["position"]["stones"][1] == {"q": 1, "r": -1, "player": 1}
    assert second["position"]["moves"][-1] == {"player": 1, "q": 1, "r": -1}

    undo_move(store, session.session_id)
    undone = session_payload(store, session.session_id)
    assert undone["position"]["turn_index"] == 1


def test_classical_axis_fixture_generation(tmp_path):
    pytest.importorskip("_engine")
    store = DashboardStore(tmp_path / "dashboard.sqlite3")

    fixtures = generate_classical_fixtures(
        store,
        ClassicalFixtureConfig(
            count=2,
            examples_per_move_count=2,
            move_counts=(2, 4),
            time_ms=1,
            max_depth=1,
            near_radius=2,
            noise_level=0.05,
            seed=11,
        ),
    )

    assert len(fixtures) == 4
    assert fixtures[0]["session_id"]
    assert fixtures[0]["move_count"] == 2
    assert [fixture["payload"]["target_moves"] for fixture in fixtures] == [2, 2, 4, 4]
    assert fixtures[0]["payload"]["source"] == "rust_classical_selfplay"
    loaded = session_payload(store, fixtures[0]["session_id"])
    assert loaded["position"]["moves"]
    assert loaded["payload"]["mode"] == "axis_fixture"


def test_axis_policy_prototypes_are_python_tunable():
    position = AxisPolicyInput(
        stones=[{"player": 0, "q": 0, "r": 0}, {"player": 1, "q": 1, "r": 0}],
        legal_moves=[{"q": -1, "r": 0}, {"q": 0, "r": 1}, {"q": 2, "r": 0}],
        current_player=0,
        metadata={"placements_remaining": 2},
    )
    results = evaluate_all(position, {"dual_axis_strength": {"w4": 1.0, "w5": 1.0}})
    assert {r["prototype_id"] for r in results} == {
        "dual_axis_strength",
        "dual_axis_strength_tail",
        "dual_axis_strength_legacy_weights",
        "dual_axis_strength_hot",
        "legacy_axis_influence",
        "exp_delta_fork",
        "exp_delta_norm",
        "exp_delta_soft_strong",
        "exp_delta_balance",
        "exp_cross_axis_pivot",
    }
    assert any(r["cells"] for r in results)
    tail = next(r for r in results if r["prototype_id"] == "dual_axis_strength_tail")
    assert tail["debug_terms"]["aggregation"] == "best_tail"
    delta = next(r for r in results if r["prototype_id"] == "exp_delta_fork")
    assert len(delta["axis_summaries"]) == 6
    assert delta["debug_terms"]["target_kind"] == "diagnostic_legal_delta_not_training_target"
    delta_norm = next(r for r in results if r["prototype_id"] == "exp_delta_norm")
    assert delta_norm["debug_terms"]["variant"] == "norm"
    first_cell = next(r["cells"][0] for r in results if r["cells"])
    assert {"q", "r", "score", "axes", "own_axes", "opp_axes", "net_axes", "owner"} <= set(first_cell)


def test_axis_strength_requires_feasible_six_cell_window():
    position = AxisPolicyInput(
        stones=[{"player": 0, "q": -16, "r": -16}],
        legal_moves=[{"q": -16, "r": -16}],
        current_player=0,
        offset_q=-16,
        offset_r=-16,
    )

    result = get_prototype("dual_axis_strength").compute(position)

    assert result.axis_maps[0, 0, 0] > 0.0
    assert result.axis_maps[1, 0, 0] > 0.0
    assert result.axis_maps[2, 0, 0] == 0.0


def test_delta_norm_fast_window_scan_matches_cell_reference():
    from hexorl.axis_policy.experiments import (
        _placement_delta_axes,
        _placement_delta_base_maps,
        _strength_array,
    )

    position = AxisPolicyInput(
        stones=[
            {"player": 0, "q": 0, "r": 0},
            {"player": 0, "q": 1, "r": 0},
            {"player": 1, "q": 0, "r": 1},
        ],
        legal_moves=[
            {"q": 2, "r": 0},
            {"q": 1, "r": -1},
            {"q": -1, "r": 1},
        ],
        current_player=0,
        offset_q=-16,
        offset_r=-16,
    )
    proto = get_prototype("exp_delta_norm")
    params = {spec.name: spec.default for spec in proto.parameters}
    strength = _strength_array(params)
    maps = _placement_delta_base_maps(position, strength, params)

    for move in position.legal_moves:
        q, r = int(move["q"]), int(move["r"])
        ij = (q - position.offset_q, r - position.offset_r)
        own, opp = _placement_delta_axes(
            q,
            r,
            position.own_stones,
            position.opp_stones,
            strength,
            params,
            position.offset_q,
            position.offset_r,
        )
        assert np.allclose(maps[:3, ij[0], ij[1]], own)
        assert np.allclose(maps[3:, ij[0], ij[1]], opp)


def test_fastapi_dashboard_smoke(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    app = create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing")
    app.state.store.insert_game(
        run_id="api-run",
        game_id="game-with-bytes",
        source="unit",
        final_move_history=_move(0, 0, 0),
        payload={"fixture": True},
    )
    client = TestClient(app)

    assert client.get("/api/health").json()["ok"]
    games = client.get("/api/games").json()
    assert games[0]["move_count"] == 1
    assert games[0]["payload"] == {"fixture": True}
    assert "final_history_b64" not in games[0]
    assert client.get("/api/axis/prototypes").json()
    created = client.post("/api/session/create", json={}).json()
    assert created["session_id"]
    axis = client.post("/api/axis/evaluate", json={"session_id": created["session_id"]}).json()
    assert axis["results"]
    assert client.get("/api/axis/fixtures").json() == []

    spread = client.post(
        "/api/axis/evaluate",
        json={
            "prototype_id": "exp_delta_fork",
            "position": {
                "current_player": 0,
                "offset_q": -16,
                "offset_r": -16,
                "stones": [
                    {"player": 0, "q": 40, "r": -40},
                    {"player": 1, "q": -40, "r": 40},
                ],
                "legal_moves": [
                    {"q": 41, "r": -40},
                    {"q": 39, "r": -40},
                    {"q": -41, "r": 40},
                    {"q": -39, "r": 40},
                ],
            },
        },
    ).json()
    assert spread["cells"]
    assert spread["debug_terms"]["legal_cells_scored"] > 0


def test_noisy_model_player_reproducible_without_engine():
    model = HexNet(channels=4, blocks=1, heads=["policy", "value"])
    a = NoisyModelPlayer(model, config=NoisyPolicyConfig(seed=123))
    b = NoisyModelPlayer(model, config=NoisyPolicyConfig(seed=123))
    assert a([], 0, 0) == b([], 0, 0)


def test_noisy_model_player_chooses_legal_origin_when_engine_available():
    pytest.importorskip("_engine")
    model = HexNet(channels=4, blocks=1, heads=["policy", "value"])
    player = NoisyModelPlayer(model, config=NoisyPolicyConfig(seed=1))
    assert player([], 0, 0) == (0, 0)


def test_eval_players_use_model_dtype(monkeypatch):
    from hexorl.eval import arena as arena_mod
    from hexorl.eval import players as players_mod

    model = _DtypeCheckingPolicy(torch.float16)
    monkeypatch.setattr(players_mod, "HAS_ENGINE", False)
    assert players_mod.NoisyModelPlayer(model)([], 0, 0) is not None

    monkeypatch.setattr(arena_mod, "HAS_ENGINE", False)
    assert arena_mod.model_move_fn(model, temperature=0.0)([], 0, 0) is not None


def test_arena_stats_reports_reason_counts():
    stats = ArenaStats(
        total_games=3,
        results=[
            MatchResult(0, 1.0, 0.0, 3, 0.0, True, "terminal"),
            MatchResult(1, 0.0, 1.0, 0, 0.0, False, "crash:dtype"),
            MatchResult(1, 0.0, 1.0, 1, 0.0, True, "illegal:occupied"),
        ],
    )
    assert stats.reason_counts == {
        "terminal": 1,
        "crash:dtype": 1,
        "illegal:occupied": 1,
    }


class _DtypeCheckingPolicy(torch.nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=dtype), requires_grad=False)

    def forward(self, x):
        assert x.dtype == self.weight.dtype
        return {"policy": torch.zeros(x.shape[0], 33 * 33, device=x.device)}
