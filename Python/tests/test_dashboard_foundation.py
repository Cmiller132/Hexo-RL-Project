import base64
import json
import struct

import numpy as np
import pytest
import torch

from hexorl.action_contract.candidates import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_VERSION
from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import evaluate_all, get_prototype
from hexorl.dashboard.checkpoints import index_checkpoint
from hexorl.dashboard.app import _suite_games, _suite_runs, _suite_store_for_run
from hexorl.dashboard.db import DashboardStore
from hexorl.dashboard.fixtures import ClassicalFixtureConfig, generate_classical_fixtures
from hexorl.dashboard.play import apply_move, create_session, session_payload, undo_move
from hexorl.dashboard.recorder import RunRecorder
from hexorl.dashboard.render import MatchSnapshotOptions, render_match_snapshot_png
from hexorl.eval.players import NoisyModelPlayer, NoisyPolicyConfig
from hexorl.eval.arena import ArenaStats, MatchResult
from hexorl.models.families.network import HexNet
from hexorl.models.families.global_graph import GlobalHexGraphNet
from hexorl.selfplay.records import GameRecord, PositionRecord, action_to_board_index


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def _must_block_moves() -> list[tuple[int, int, int]]:
    return [
        (0, 0, 0),
        (1, 0, 5),
        (1, 0, 6),
        (0, 1, 0),
        (0, 2, 0),
        (1, 1, 5),
        (1, 1, 6),
        (0, 3, 0),
        (0, 4, 0),
    ]


class _GraphPrefersLastLegal(torch.nn.Module):
    hexorl_architecture = "global_xattn_0"
    graph_context_tokens = 64
    graph_legal_rows = 64

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, legal_mask, **kwargs):
        width = int(legal_mask.shape[1])
        logits = torch.arange(width, device=legal_mask.device, dtype=torch.float32)
        logits = logits.unsqueeze(0) + self.weight
        return {
            "policy_place": logits,
            "value": torch.zeros(1, 65, device=legal_mask.device),
        }


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


def test_suite_games_include_direct_phase3_trial_dashboards(tmp_path):
    suite_root = tmp_path / "run" / "phase_normal_dashboard_suite"
    suite_trial_dir = suite_root / "trials" / "global_xattn_0__none__v1"
    phase3_trial_dir = tmp_path / "run" / "phase3_trials" / "global_graph768_champion__none__v1__phase3_t0000"
    suite_trial_dir.mkdir(parents=True)
    phase3_trial_dir.mkdir(parents=True)

    def record_game(trial_dir, run_id, game_id):
        store = DashboardStore(trial_dir / "dashboard.sqlite3")
        recorder = RunRecorder(store, run_id, trial_dir / "events.jsonl")
        history = _move(0, 0, 0) + _move(1, 1, 0)
        game = GameRecord(
            positions=[],
            outcome=0.0,
            game_id=game_id,
            game_length=2,
            final_move_history=history,
        )
        recorder.game(game, source="selfplay", epoch=14, payload={"terminal_reason": "unit"})

    record_game(
        suite_trial_dir,
        "optuna_mcts500_graphfix_20260507_01_global_xattn_0__none__v1",
        1,
    )
    record_game(
        phase3_trial_dir,
        "phase3_global_graph768_champion__none__v1__phase3_t0000",
        2,
    )

    games = _suite_games(suite_root, limit=10)
    assert [row["trial_id"] for row in games] == [
        "global_graph768_champion__none__v1__phase3_t0000",
        "global_xattn_0__none__v1",
    ]
    assert _suite_store_for_run(
        suite_root, "phase3_global_graph768_champion__none__v1__phase3_t0000"
    ) is not None
    runs = _suite_runs(suite_root)
    assert {row["suite_trial_id"] for row in runs} == {
        "global_graph768_champion__none__v1__phase3_t0000",
        "global_xattn_0__none__v1",
    }


def test_run_recorder_dashboard_write_failure_does_not_crash_training(tmp_path):
    class FailingMetricStore:
        path = tmp_path / "readonly.sqlite3"

        def upsert_run(self, _run_id):
            return None

        def record_metric(self, *_args, **_kwargs):
            raise RuntimeError("readonly database")

    store = FailingMetricStore()
    recorder = RunRecorder(store, "resilient-run", tmp_path / "events.jsonl")

    assert recorder.metric({"loss": 1.0}, phase="train", epoch=1, global_step=12) == -1
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert rows[-1]["event_type"] == "dashboard_write_failed"
    assert rows[-1]["payload"]["operation"] == "record_metric"
    assert rows[-1]["payload"]["error"] == "readonly database"


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


def test_replay_routes_default_to_compact_payloads(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    app = create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing")
    game_id = app.state.store.insert_game_with_positions(
        run_id="api-run",
        game_id="long-game",
        source="unit",
        final_move_history=_move(0, 0, 0) + _move(1, 1, 0) + _move(0, 0, 1),
        positions=[
            {
                "turn_index": 1,
                "player": 1,
                "move_history": _move(0, 0, 0),
                "root_value": 0.0,
                "policy_target": {},
                "debug": {},
            }
        ],
    )
    client = TestClient(app)

    summary = client.get(f"/api/games/{game_id}/replay").json()
    assert len(summary["moves"]) == 3
    assert summary["positions"] == []

    full = client.get(f"/api/games/{game_id}/replay?include_positions=true").json()
    assert len(full["positions"]) == 1

    compact_position = client.get(f"/api/games/{game_id}/position/3?compact=true").json()
    assert compact_position["turn_index"] == 3
    assert "moves" not in compact_position
    assert compact_position["stones"]


def test_suite_game_lookup_accepts_dashboard_run_id_that_differs_from_trial_dir(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    run_root = tmp_path / "suite"
    trial_dir = run_root / "trials" / "candidate-a"
    trial_dir.mkdir(parents=True)
    store = DashboardStore(trial_dir / "dashboard.sqlite3")
    run_id = "production_run_candidate_a"
    game_id = store.insert_game(
        run_id=run_id,
        game_id="suite-game",
        source="selfplay",
        final_move_history=_move(0, 0, 0),
    )
    store.upsert_checkpoint(
        path=trial_dir / "epoch_0001.pt",
        sha256="abc",
        run_id=run_id,
        epoch=1,
        global_step=12,
        is_loadable=True,
    )
    client = TestClient(
        create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=run_root)
    )

    games = client.get(f"/api/games?run_id={run_id}").json()
    assert games[0]["game_id"] == game_id
    assert games[0]["run_id"] == run_id

    checkpoints = client.get(f"/api/checkpoints?run_id={run_id}").json()
    assert checkpoints[0]["run_id"] == run_id
    assert checkpoints[0]["trial_id"] == "candidate-a"


def test_suite_dashboard_derives_phase1_trial_metrics_without_latest_json(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    run_root = tmp_path / "suite"
    trial_dir = run_root / "trials" / "global_xattn_0__none__v1"
    trial_dir.mkdir(parents=True)
    run_id = "production_global_xattn"
    store = DashboardStore(trial_dir / "dashboard.sqlite3")
    store.upsert_run(run_id)
    ckpt = trial_dir / "epoch_0008.pt"
    ckpt.write_bytes(b"fake")
    store.upsert_checkpoint(
        path=ckpt,
        sha256="abc",
        run_id=run_id,
        epoch=8,
        global_step=96,
        is_loadable=True,
    )
    store.record_metric(
        run_id,
        phase="selfplay",
        metrics={
            "positions_per_min": 2400.0,
            "truncation_rate": 0.125,
            "recorder_failures": 0,
            "terminal_reason_win": 10,
        },
    )
    store.record_metric(
        run_id,
        phase="train",
        epoch=8,
        global_step=96,
        metrics={
            "checkpoint_path": str(ckpt),
            "train": {
                "epoch": 8,
                "loss_total": 5.5,
                "loss_value": 0.7,
                "value_weight_mean": 1.0,
            },
            "buffer": {
                "avg_missing_target_policy_mass": 0.01,
                "avg_candidate_recall_mcts_top1": 0.9,
            },
        },
    )
    (trial_dir / "full_config.json").write_text(
        json.dumps(
            {
                "model": {"architecture": "global_xattn_0", "channels": 128, "blocks": 16},
                "selfplay": {"mcts_simulations": 512, "max_game_moves": 500, "states_per_epoch": 3000},
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "optuna_trial.json").write_text(
        json.dumps({"trial_number": 0, "params": {"candidate_id": "global_xattn_0__none__v1"}}),
        encoding="utf-8",
    )
    (run_root / "manifest.json").write_text(json.dumps({"run_id": "unit"}), encoding="utf-8")

    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=run_root))

    trials = client.get("/api/suite/trials").json()
    assert trials[0]["trial_id"] == "global_xattn_0__none__v1"
    assert trials[0]["run_id"] == run_id
    assert trials[0]["architecture"] == "global_xattn_0"
    assert trials[0]["epoch"] == 8
    assert trials[0]["loss_total"] == pytest.approx(5.5)
    assert trials[0]["positions_per_sec"] == pytest.approx(40.0)
    assert trials[0]["truncation_rate"] == pytest.approx(0.125)
    assert trials[0]["mcts_simulations"] == 512

    status = client.get("/api/suite/status").json()
    assert status["leading_trial_id"] == "global_xattn_0__none__v1"
    assert status["current_trial_id"] == "global_xattn_0__none__v1"
    assert status["current_model"] == "xattn 0"
    assert status["total_checkpoints"] == 1

    detail = client.get("/api/suite/trials/global_xattn_0__none__v1").json()
    assert detail["run_id"] == run_id
    assert detail["config"]["selfplay"]["max_game_moves"] == 500


def test_suite_events_merge_trial_events_and_suppress_game_spam(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hexorl.dashboard.app import create_app

    run_root = tmp_path / "suite"
    trial_dir = run_root / "trials" / "global_xattn_0__none__v1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "dashboard.sqlite3").write_bytes(b"")
    (trial_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_type": "game_recorded",
                        "phase": "selfplay",
                        "epoch": 4,
                        "payload": {"game_id": 1},
                        "time": 10.0,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "metric",
                        "phase": "selfplay",
                        "payload": {
                            "positions_done": 3000,
                            "positions_per_min": 1800.0,
                            "truncation_rate": 0.25,
                        },
                        "time": 11.0,
                    }
                ),
                json.dumps(
                    {
                        "event": "training_signal_warning",
                        "metric": "truncation_rate",
                        "value": 0.25,
                        "threshold": 0.2,
                        "time": 12.0,
                    }
                ),
                json.dumps(
                    {
                        "event_type": "epoch_complete",
                        "phase": "epoch",
                        "epoch": 4,
                        "payload": {"elapsed_s": 99.0},
                        "time": 13.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=run_root))

    events = client.get("/api/suite/events").json()

    assert [event["event"] for event in events] == ["metric", "training_signal_warning", "epoch_complete"]
    assert events[0]["trial_label"] == "xattn 0"
    assert events[0]["positions_per_sec"] == pytest.approx(30.0)
    assert "3,000 positions" in events[0]["message"]
    assert events[1]["severity"] == "warning"
    assert "truncation_rate" in events[1]["message"]

    status = client.get("/api/suite/status").json()
    assert status["event_count"] == 3
    assert status["warning_count"] == 1
    assert status["last_event_name"] == "epoch_complete"


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


def test_noisy_model_player_supports_global_graph_policy_place_eval():
    engine = pytest.importorskip("_engine")
    model = GlobalHexGraphNet(
        channels=8,
        layers=1,
        heads=2,
        architecture="global_xattn_0",
        output_heads=["policy_place", "value"],
    )
    model.graph_context_tokens = 64
    model.graph_legal_rows = 64
    player = NoisyModelPlayer(
        model,
        config=NoisyPolicyConfig(seed=7, temperature=1e-4, top_p=1.0),
    )

    q, r = player([], 0, 0)

    game = engine.HexGame()
    game.place(int(q), int(r))


def test_noisy_model_player_global_graph_eval_uses_threat_constrained_legal_rows():
    engine = pytest.importorskip("_engine")
    model = _GraphPrefersLastLegal()
    player = NoisyModelPlayer(
        model,
        config=NoisyPolicyConfig(seed=7, temperature=1e-4, top_p=1.0),
    )

    q, r = player(_must_block_moves(), 0, 1)

    game = engine.HexGame()
    for _player, mq, mr in _must_block_moves():
        game.place(int(mq), int(mr))
    _tensor, _offset_q, _offset_r, legal_bytes = game.encode_board_and_legal(8, True)
    threat_legal = {
        tuple(row)
        for row in np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2).tolist()
    }
    assert threat_legal == {(-1, 0), (5, 0)}
    assert (q, r) in threat_legal


def test_arena_model_move_fn_routes_global_graph_models_to_policy_place():
    engine = pytest.importorskip("_engine")
    from hexorl.eval import arena as arena_mod

    model = GlobalHexGraphNet(
        channels=8,
        layers=1,
        heads=2,
        architecture="global_graph_full_0",
        output_heads=["policy_place", "value"],
    )
    model.graph_context_tokens = 64
    model.graph_legal_rows = 64

    q, r = arena_mod.model_move_fn(model, temperature=0.0)([], 0, 0)

    game = engine.HexGame()
    game.place(int(q), int(r))


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
