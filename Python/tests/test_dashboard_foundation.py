import struct

import pytest
import torch

from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import evaluate_all
from hexorl.dashboard.checkpoints import index_checkpoint
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.play import apply_move, create_session, session_payload, undo_move
from hexorl.dashboard.recorder import RunRecorder
from hexorl.eval.players import NoisyModelPlayer, NoisyPolicyConfig
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
    model = HexNet(channels=4, blocks=1, heads=["policy", "value", "axis"])
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
    assert result.model_heads == ["axis", "policy", "value"]
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


def test_axis_policy_prototypes_are_python_tunable():
    position = AxisPolicyInput(
        stones=[{"player": 0, "q": 0, "r": 0}, {"player": 1, "q": 1, "r": 0}],
        legal_moves=[{"q": -1, "r": 0}, {"q": 0, "r": 1}, {"q": 2, "r": 0}],
        current_player=0,
        metadata={"placements_remaining": 2},
    )
    results = evaluate_all(position, {"threat_window_strength": {"own_weight": 2.0}})
    assert {r["prototype_id"] for r in results} == {
        "legacy_axis_influence",
        "threat_window_strength",
        "axis_development",
        "multi_line_threats",
    }
    assert any(r["top"] for r in results)
    first_top = next(r["top"][0] for r in results if r["top"])
    assert {"q", "r", "axes", "owner"} <= set(first_top)


def test_fastapi_dashboard_smoke(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    app = create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing")
    client = TestClient(app)

    assert client.get("/api/health").json()["ok"]
    assert client.get("/api/axis/prototypes").json()
    created = client.post("/api/session/create", json={}).json()
    assert created["session_id"]
    axis = client.post("/api/axis/evaluate", json={"session_id": created["session_id"]}).json()
    assert axis["results"]


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
