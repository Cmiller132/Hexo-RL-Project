import base64
import json
import struct

import numpy as np
import pytest
import torch

from hexorl.contracts.candidates import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_VERSION
from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import evaluate_all, get_prototype
from hexorl.dashboard.checkpoints import index_checkpoint
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.fixtures import ClassicalFixtureConfig, generate_classical_fixtures
from hexorl.dashboard.play import apply_move, create_session, session_payload, undo_move
from hexorl.dashboard.recorder import RunRecorder
from hexorl.dashboard.render import MatchSnapshotOptions, render_match_snapshot_png
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


def test_dashboard_store_repairs_partial_v1_schema(tmp_path):
    db_path = tmp_path / "dashboard.sqlite3"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (1, 1.0)")

    store = DashboardStore(db_path)
    store.upsert_run("repaired-run")

    rows = store.rows("SELECT run_id FROM runs")
    assert rows == [{"run_id": "repaired-run"}]


def test_match_snapshot_renderer_outputs_png_bytes():
    history = (
        _move(0, 0, 0)
        + _move(1, 1, 0)
        + _move(1, 1, -1)
        + _move(0, -1, 0)
        + _move(0, -1, 1)
    )

    png = render_match_snapshot_png(
        history,
        options=MatchSnapshotOptions(width=420, height=320, title="unit snapshot"),
        metadata={"run_id": "unit-run", "game_id": 3, "source": "unit"},
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png[:32]
    assert len(png) > 1000


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
    snapshot = client.get(f"/api/games/{games[0]['game_id']}/snapshot.png?width=360&height=280")
    assert snapshot.status_code == 200
    assert snapshot.headers["content-type"] == "image/png"
    assert snapshot.content.startswith(b"\x89PNG\r\n\x1a\n")
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


def test_suite_dashboard_scores_and_stage_fallback(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    run_root = tmp_path / "suite"
    trial_dir = run_root / "trials" / "trial-a"
    trial_dir.mkdir(parents=True)
    store = DashboardStore(trial_dir / "dashboard.sqlite3")
    ckpt = trial_dir / "epoch_0002.pt"
    ckpt.write_bytes(b"fake")
    store.upsert_checkpoint(
        path=ckpt,
        sha256="abc",
        run_id="trial-a",
        epoch=2,
        global_step=20,
        is_loadable=True,
    )
    (trial_dir / "trial.json").write_text(
        json.dumps(
            {
                "trial_id": "trial-a",
                "family": {"name": "best_current_33", "architecture": "cnn"},
                "static": {"full_sims": 800, "pcr_low_sims": 192},
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "LATEST.json").write_text(
        json.dumps(
            {
                "stage": "3B_static_asha",
                "epoch": 2,
                "train": {"loss_total": 1.5},
                "selfplay": {"positions_per_min": 600.0},
                "checkpoint_path": str(ckpt),
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "scores.jsonl").write_text(
        json.dumps({"scheduler_score": 0.4242, "stage": "3B_static_asha"}) + "\n",
        encoding="utf-8",
    )
    (run_root / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "stage_start", "stage": "3A_calibration", "time": 1.0}),
                json.dumps({"event": "trial_epoch_complete", "stage": "3B_static_asha", "trial_id": "trial-a", "time": 2.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "manifest.json").write_text(json.dumps({"args": {"max_game_moves": 384}}), encoding="utf-8")

    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=run_root))

    best = client.get("/api/suite/best-checkpoints").json()
    assert best[0]["score"] == pytest.approx(0.4242)
    status = client.get("/api/suite/status").json()
    assert status["latest_stage"] == "3B_static_asha"
    assert status["best_score"] == pytest.approx(0.4242)


def test_dashboard_debug_contract_and_graph_endpoints(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing"))
    contracts = client.get("/api/debug/contracts").json()
    assert contracts["candidate"]["feature_version"] == CANDIDATE_FEATURE_VERSION
    assert contracts["candidate"]["feature_names"] == list(CANDIDATE_FEATURE_NAMES)
    assert contracts["graph"]["schema_version"] >= 1
    assert "LEGAL" in contracts["graph"]["token_types"]

    history = _move(0, 0, 0) + _move(1, 1, 0)
    payload = {"history_b64": base64.b64encode(history).decode("ascii")}
    graph = client.post("/api/debug/graph", json=payload).json()
    assert graph["legal_count"] > 0
    assert graph["token_counts"]["LEGAL"] == graph["legal_count"]
    assert graph["relation_schema_version"] >= 1

    d6 = client.post("/api/debug/d6", json=payload).json()
    assert d6["symmetry_count"] == 12
    assert len(d6["transforms"]) == 12
    assert all(item["graph"]["legal_count"] == item["legal_count"] for item in d6["transforms"])
    first = d6["transforms"][0]["contracts"]
    assert first["dense_legal_mask"]["legal_count"] == d6["transforms"][0]["legal_count"]
    assert first["sparse_candidates"]["feature_version"] == CANDIDATE_FEATURE_VERSION
    assert first["axis"]["prototype_count"] > 0
    assert "graph_targets" in first


def test_dashboard_pair_policy_inference_returns_pair_logits(tmp_path):
    pytest.importorskip("_engine")
    from hexorl.dashboard.model_cache import CachedModel, ModelCache

    model = HexNet(
        channels=4,
        blocks=1,
        heads=["policy", "value", "pair_policy"],
    )
    cache = ModelCache()
    cache._models["pair-unit"] = CachedModel(
        "pair-unit",
        tmp_path / "in-memory.pt",
        model,
        torch.device("cpu"),
    )
    cache._order.append("pair-unit")

    result = cache.infer_history("pair-unit", _move(0, 0, 0))

    assert result["heads"]["pair_policy"]
    top = result["heads"]["pair_policy"][0]
    assert {"first", "second", "logit"} <= set(top)
    assert top["first"] != top["second"]


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
