from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from hexorl.dashboard.app import create_app
from hexorl.dashboard.db import DashboardStore


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _suite_root(tmp_path: Path) -> Path:
    root = tmp_path / "suite"
    trial = root / "trials" / "trial-a"
    trial.mkdir(parents=True)
    _write_json(root / "manifest.json", {"host": {"cpu_count": 8}, "scheduler": {"stages": [{"stage": "seed"}]}, "budget": {"positions": 1000}})
    _write_json(root / "state.json", {"stage": "seed", "trials": [{"trial_id": "trial-a", "last_score": 1.25, "family": {"name": "cnn", "architecture": "cnn"}, "runtime_sweep": {"selected": {"workers": 2}, "probes": [{"workers": 1, "batch": 16, "positions_per_sec": 40.0, "stable": True}]}}]})
    _write_json(trial / "trial.json", {"family": {"name": "cnn", "architecture": "cnn", "channels": 32, "blocks": 2}, "static": {"candidate_budget": 128}})
    _write_json(trial / "LATEST.json", {"stage": "seed", "epoch": 3, "selfplay": {"positions_per_min": 600, "workers_alive": 2, "workers_total": 2}, "train": {"loss_total": 0.5}})
    _write_jsonl(root / "events.jsonl", [{"event": "stage_start", "stage": "seed", "trial_id": "trial-a", "time": 1.0}])
    _write_jsonl(trial / "events.jsonl", [{"event": "trial_epoch_complete", "stage": "seed", "trial_id": "trial-a", "time": 2.0}])
    _write_jsonl(trial / "scores.jsonl", [{"epoch": 1, "score": 1.0, "scheduler_score": 1.25}])
    store = DashboardStore(trial / "dashboard.sqlite3")
    store.upsert_run("trial-a", name="trial-a")
    store.record_metric("trial-a", epoch=1, global_step=10, phase="train", metrics={"loss_total": 0.5, "loss_policy": 0.2})
    store.upsert_checkpoint(path=trial / "ckpt.pt", sha256="abc", run_id="trial-a", epoch=1, is_loadable=False)
    return root


def test_suite_refactor_endpoints_read_synthetic_run_root(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=_suite_root(tmp_path)))
    status = client.get("/api/suite/status").json()
    assert status["enabled"] is True
    assert status["best_trial_id"] == "trial-a"
    detail = client.get("/api/suite/trials/trial-a").json()
    assert detail["trial_id"] == "trial-a"
    assert "scores" not in detail
    assert client.get("/api/suite/trials/trial-a/scores").json()[0]["scheduler_score"] == 1.25
    assert client.get("/api/suite/trials/trial-a/events").json()[0]["event"] == "trial_epoch_complete"
    assert client.get("/api/suite/trials/trial-a/loss-curve").json()[0]["loss_total"] == 0.5
    assert client.get("/api/suite/trials/trial-a/runtime-sweep").json()["selected"]["workers"] == 2
    assert client.get("/api/suite/manifest").json()["manifest"]["budget"]["positions"] == 1000
    assert client.get("/api/suite/scheduler").json()["current_stage"] == "seed"
    assert client.get("/api/suite/runtime-sweep").json()["probes"][0]["trial_id"] == "trial-a"
    assert "families" in client.get("/api/suite/family-space").json()


def test_dashboard_openapi_snapshot_matches_artifact(tmp_path: Path) -> None:
    app = create_app(tmp_path / "dashboard.sqlite3", frontend_dist=tmp_path / "missing", run_root=_suite_root(tmp_path))
    snapshot = Path("Docs/refactor/artifacts/dashboard_openapi_snapshot.json")
    assert app.openapi() == json.loads(snapshot.read_text(encoding="utf-8"))
