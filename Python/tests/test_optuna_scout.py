import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from hexorl.autotune import CandidateRecipe, ModelRecipe, PairStrategySpec
from hexorl.config import Config
from hexorl.tuning.optuna_scout import (
    EpochScoutEpochRunner,
    PHASE1_STUDY_NAME,
    PHASE1_TARGET_SCALAR,
    Phase1OptunaScoutController,
    ScoutHardFailure,
    ScoutQuantumResult,
)
from hexorl.tuning.runtime_probe import RuntimeKnobs


def _candidate(architecture_id: str, pair_mode: str = "none") -> CandidateRecipe:
    kwargs = {}
    if pair_mode != "none":
        kwargs["output_heads"] = [
            "policy_place",
            "value",
            "policy_pair_first",
            "policy_pair_joint",
            "policy_pair_second",
        ]
    return CandidateRecipe(
        model=ModelRecipe(architecture_id=architecture_id, **kwargs),
        pair_strategy=(
            PairStrategySpec(mode="none")
            if pair_mode == "none"
            else PairStrategySpec(mode=pair_mode, pair_row_budget=256)
        ),
    )


class RecordingRunner:
    def __init__(self, fail_candidates=None):
        self.epochs = []
        self.fail_candidates = set(fail_candidates or ())

    def run_quantum(self, request):
        if request.candidate.candidate_id in self.fail_candidates:
            raise ScoutHardFailure("simulated_hard_failure", details={"epoch": request.start_epoch})
        scorecards = []
        latest_checkpoint = None
        for epoch in range(request.start_epoch, request.end_epoch + 1):
            self.epochs.append((request.candidate.candidate_id, epoch))
            checkpoint = request.paths.checkpoints_dir / f"epoch_{epoch:04d}.ckpt"
            checkpoint.write_text(f"{request.candidate.candidate_id}:{epoch}\n", encoding="utf-8")
            latest_checkpoint = str(checkpoint)
            scorecards.append(
                {
                    "candidate_id": request.candidate.candidate_id,
                    "epoch": epoch,
                    PHASE1_TARGET_SCALAR: 100.0 + epoch,
                    "checkpoint_path": latest_checkpoint,
                }
            )
        return ScoutQuantumResult(
            completed_epochs=request.end_epoch,
            scorecards=tuple(scorecards),
            latest_checkpoint_path=latest_checkpoint,
            latest_scorecard_path=str(request.paths.scorecards_jsonl),
        )


class ConfigRecordingRunner(RecordingRunner):
    def __init__(self):
        super().__init__()
        self.runtime_rows = []

    def run_quantum(self, request):
        self.runtime_rows.append(
            (
                request.config.selfplay.num_workers,
                request.config.selfplay.batch_size_per_worker,
                request.config.inference.max_batch_size,
                request.config.inference.max_wait_us,
                request.config.selfplay.games_per_epoch,
                request.config.selfplay.states_per_epoch,
                request.config.train.batches_per_epoch,
            )
        )
        return super().run_quantum(request)


def test_phase1_scout_uses_one_sqlite_study_nop_pruner_and_enqueues_once(tmp_path):
    candidates = (
        _candidate("global_xattn_0"),
        _candidate("global_line_window_0"),
    )
    runner = RecordingRunner()
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="scout",
        candidates=candidates,
        runner=runner,
        min_epochs=2,
        quantum_epochs=2,
    )

    study = controller.create_or_resume_study()
    assert study.study_name == PHASE1_STUDY_NAME
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert Path(controller.storage.removeprefix("sqlite:///")).exists()
    assert len(study.get_trials(deepcopy=False)) == 2

    resumed = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="scout",
        candidates=candidates,
        runner=runner,
        min_epochs=2,
        quantum_epochs=2,
    ).create_or_resume_study()

    trials = resumed.get_trials(deepcopy=False)
    assert len(trials) == 2
    assert {trial.user_attrs["candidate_id"] for trial in trials} == {c.candidate_id for c in candidates}
    assert resumed.user_attrs["phase1_pruner"] == "NopPruner"


def test_round_robin_two_epoch_quanta_and_lifecycle_attrs_reach_epoch_floor(tmp_path):
    candidates = (
        _candidate("global_xattn_0"),
        _candidate("global_line_window_0"),
    )
    runner = RecordingRunner()
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="rr",
        candidates=candidates,
        runner=runner,
        min_epochs=4,
        quantum_epochs=2,
    )

    summary = controller.run()

    a = candidates[0].candidate_id
    b = candidates[1].candidate_id
    assert runner.epochs == [(a, 1), (a, 2), (b, 1), (b, 2), (a, 3), (a, 4), (b, 3), (b, 4)]
    assert summary.completed is True
    assert all(row["hexo_status"] == "completed" for row in summary.candidate_statuses.values())
    assert all(row["completed_epochs"] == 4 for row in summary.candidate_statuses.values())

    study = controller.create_or_resume_study()
    for trial in study.get_trials(deepcopy=False):
        attrs = trial.user_attrs
        assert trial.state == optuna.trial.TrialState.COMPLETE
        assert attrs["candidate_id"] in {a, b}
        assert attrs["architecture_id"] in {"global_xattn_0", "global_line_window_0"}
        assert attrs["pair_strategy"] == "none"
        assert attrs["recipe_schema_version"] == 1
        assert len(attrs["config_hash"]) == 64
        assert Path(attrs["run_dir"]).name == attrs["candidate_id"]
        assert attrs["completed_epochs"] == 4
        assert attrs["hexo_status"] == "completed"
        assert attrs["quarantine_reason"] is None
        assert Path(attrs["latest_checkpoint_path"]).exists()
        assert Path(attrs["latest_scorecard_path"]).exists()
        assert trial.intermediate_values[4] == pytest.approx(104.0)


def test_resume_continues_existing_stable_candidate_trials(tmp_path):
    candidates = (
        _candidate("global_xattn_0"),
        _candidate("global_line_window_0"),
    )
    first_runner = RecordingRunner()
    first = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="resume",
        candidates=candidates,
        runner=first_runner,
        min_epochs=4,
        quantum_epochs=2,
    )
    first_summary = first.run(max_quanta=1)
    first_trials = {
        trial.user_attrs["candidate_id"]: trial.number
        for trial in first.create_or_resume_study().get_trials(deepcopy=False)
    }

    second_runner = RecordingRunner()
    second = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="resume",
        candidates=candidates,
        runner=second_runner,
        min_epochs=4,
        quantum_epochs=2,
    )
    final_summary = second.run()
    second_trials = {
        trial.user_attrs["candidate_id"]: trial.number
        for trial in second.create_or_resume_study().get_trials(deepcopy=False)
    }

    assert first_summary.completed is False
    assert first_trials == second_trials
    assert second_runner.epochs == [
        (candidates[1].candidate_id, 1),
        (candidates[1].candidate_id, 2),
        (candidates[0].candidate_id, 3),
        (candidates[0].candidate_id, 4),
        (candidates[1].candidate_id, 3),
        (candidates[1].candidate_id, 4),
    ]
    assert final_summary.completed is True


def test_hard_failure_quarantines_candidate_and_scout_continues(tmp_path):
    candidates = (
        _candidate("global_xattn_0"),
        _candidate("global_line_window_0"),
    )
    failing_id = candidates[0].candidate_id
    runner = RecordingRunner(fail_candidates={failing_id})
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="quarantine",
        candidates=candidates,
        runner=runner,
        min_epochs=2,
        quantum_epochs=2,
    )

    summary = controller.run()

    assert summary.completed is True
    assert summary.candidate_statuses[failing_id]["hexo_status"] == "quarantined"
    assert summary.candidate_statuses[failing_id]["quarantine_reason"] == "simulated_hard_failure"
    assert summary.candidate_statuses[candidates[1].candidate_id]["hexo_status"] == "completed"

    study = controller.create_or_resume_study()
    trials = {trial.user_attrs["candidate_id"]: trial for trial in study.get_trials(deepcopy=False)}
    failed_trial = trials[failing_id]
    survivor_trial = trials[candidates[1].candidate_id]
    assert failed_trial.state == optuna.trial.TrialState.FAIL
    assert failed_trial.user_attrs["hexo_status"] == "quarantined"
    assert failed_trial.user_attrs["quarantine_reason"] == "simulated_hard_failure"
    assert failed_trial.user_attrs.get("debug_bundle_path") is None
    assert survivor_trial.state == optuna.trial.TrialState.COMPLETE
    quarantine_json = Path(failed_trial.user_attrs["run_dir"]) / "quarantine.json"
    payload = json.loads(quarantine_json.read_text(encoding="utf-8"))
    assert payload["strength_pruning"] is False


def test_candidate_first_artifacts_and_pair_attrs_are_written(tmp_path):
    candidates = (
        _candidate("global_pair_twostage_0", "root_pair_mcts"),
    )
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="artifacts",
        candidates=candidates,
        runner=RecordingRunner(),
        min_epochs=2,
        quantum_epochs=2,
    )

    controller.run()

    candidate_dir = tmp_path / "runs" / "artifacts" / "candidates" / candidates[0].candidate_id
    assert (tmp_path / "runs" / "artifacts" / "study_manifest.json").exists()
    assert (candidate_dir / "candidate_manifest.json").exists()
    assert (candidate_dir / "recipe.json").exists()
    assert (candidate_dir / "full_config.toml").exists()
    assert (candidate_dir / "optuna_trial.json").exists()
    assert (candidate_dir / "scorecards.jsonl").read_text(encoding="utf-8").count("\n") == 2
    manifest = json.loads((candidate_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == candidates[0].candidate_id
    assert manifest["pair_strategy_mode"] == "root_pair_mcts"


def test_resume_stops_when_checkpoint_lineage_is_ambiguous(tmp_path):
    candidates = (
        _candidate("global_xattn_0"),
    )
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="lineage",
        candidates=candidates,
        runner=RecordingRunner(),
        min_epochs=4,
        quantum_epochs=2,
    )
    controller.run(max_quanta=1)
    study = controller.create_or_resume_study()
    trial = study.get_trials(deepcopy=False)[0]
    Path(trial.user_attrs["latest_checkpoint_path"]).unlink()

    with pytest.raises(RuntimeError, match="checkpoint lineage is missing"):
        Phase1OptunaScoutController(
            runs_root=tmp_path / "runs",
            run_id="lineage",
            candidates=candidates,
            runner=RecordingRunner(),
            min_epochs=4,
            quantum_epochs=2,
        ).create_or_resume_study()


def test_epoch_scout_runner_advances_real_epoch_api_with_candidate_artifacts(tmp_path, monkeypatch):
    candidate = _candidate("global_xattn_0")
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="epoch_runner",
        candidates=(candidate,),
        runner=RecordingRunner(),
        min_epochs=2,
        quantum_epochs=2,
    )
    paths = controller.writer.write_candidate(candidate, controller._configs[candidate.candidate_id])
    calls = []

    def fake_run_hexo_epoch(cfg, *, trainer, output_dir, recorder_run_dir, recorder_run_id, **kwargs):
        epoch = len(calls) + 1
        checkpoint = output_dir / f"epoch_{epoch:04d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"epoch:{epoch}\n", encoding="utf-8")
        calls.append(
            {
                "trainer": trainer,
                "output_dir": output_dir,
                "recorder_run_dir": recorder_run_dir,
                "recorder_run_id": recorder_run_id,
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(
            trainer=f"trainer-{epoch}",
            train_stats={
                "epoch": epoch,
                "loss_total": 10.0 / epoch,
                "loss_policy_place": 6.0 / epoch,
                "loss_value": 3.0 / epoch,
                "pair_policy_weight_mean": 0.25 * epoch,
                "batches_per_sec": 2.0 * epoch,
                "graph_peak_cuda_allocated_mb": 100.0 * epoch,
                "graph_microbatch_oom_retries": 0.0,
                "graph_microbatch_nonfinite_retries": 0.0,
            },
            buffer_stats={
                "size": 100 * epoch,
                "avg_missing_target_policy_mass": 0.0,
                "avg_candidate_recall_mcts_top1": 1.0,
                "pair_prior_hit_frac": 0.0,
                "pair_fallback_prior_use": 1.0,
                "pair_fallback_prior_use_on_mcts_top1": 1.0,
                "pair_fallback_prior_use_on_mcts_top4": 1.0,
                "pair_fallback_prior_use_on_mcts_top8": 1.0,
            },
            checkpoint_path=checkpoint,
            elapsed_s=2.5,
        )

    monkeypatch.setattr("hexorl.tuning.optuna_scout._run_hexo_epoch", fake_run_hexo_epoch)
    runner = EpochScoutEpochRunner(bootstrap_games=3, use_selfplay=False)
    result = runner.run_quantum(
        SimpleNamespace(
            candidate=candidate,
            config=controller._configs[candidate.candidate_id],
            paths=paths,
            trial_number=0,
            start_epoch=1,
            end_epoch=2,
            run_id="epoch_runner",
            study_name=PHASE1_STUDY_NAME,
        )
    )

    assert result.completed_epochs == 2
    assert result.latest_checkpoint_path == str(paths.checkpoints_dir / "epoch_0002.pt")
    assert [row["epoch"] for row in result.scorecards] == [1, 2]
    assert PHASE1_TARGET_SCALAR in result.scorecards[-1]
    assert result.scorecards[-1]["scalar_score"] == pytest.approx(0.0)
    assert result.scorecards[-1]["component_metrics"][PHASE1_TARGET_SCALAR] == pytest.approx(0.0)
    assert result.scorecards[-1]["component_metrics"]["train_loss"] == pytest.approx(5.0)
    assert result.scorecards[-1]["component_metrics"]["loss_total"] == pytest.approx(5.0)
    assert result.scorecards[-1]["component_metrics"]["loss_policy_place"] == pytest.approx(3.0)
    assert result.scorecards[-1]["component_metrics"]["loss_value"] == pytest.approx(1.5)
    assert result.scorecards[-1]["component_metrics"]["pair_policy_weight_mean"] == pytest.approx(0.5)
    assert result.scorecards[-1]["component_metrics"]["batches_per_sec"] == pytest.approx(4.0)
    assert result.scorecards[-1]["component_metrics"]["graph_peak_cuda_allocated_mb"] == pytest.approx(200.0)
    assert result.scorecards[-1]["component_metrics"]["avg_candidate_recall_mcts_top1"] == pytest.approx(1.0)
    assert result.scorecards[-1]["component_metrics"]["pair_prior_hit_frac"] == pytest.approx(0.0)
    assert result.scorecards[-1]["component_metrics"]["pair_fallback_prior_use"] == pytest.approx(0.0)
    assert result.scorecards[-1]["component_metrics"]["pair_fallback_prior_use_on_mcts_top8"] == pytest.approx(0.0)
    assert result.scorecards[-1]["checkpoint_lineage"]["checkpoint_path"].endswith("epoch_0002.pt")
    assert result.scorecards[-1]["hard_gates"] == {"hard_pass": True, "failures": []}
    assert str(paths.events_jsonl) in result.scorecards[-1]["evidence_paths"]
    assert calls[0]["trainer"] is None
    assert calls[1]["trainer"] == "trainer-1"
    assert calls[0]["kwargs"]["bootstrap_games"] == 3
    assert calls[0]["kwargs"]["use_selfplay"] is False


def test_runtime_probe_selection_applies_knobs_before_training(tmp_path):
    candidate = _candidate("global_xattn_0")
    runner = ConfigRecordingRunner()
    selected = RuntimeKnobs(2, 5, 64, 300)
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="runtime_selected",
        candidates=(candidate,),
        runner=runner,
        min_epochs=2,
        quantum_epochs=2,
        runtime_probe_candidates=(RuntimeKnobs(1, 4, 32, 200), selected),
        runtime_probe_runner=lambda knobs, index: {
            "candidate": knobs.to_legacy_candidate(),
            "ok": True,
            "positions": 40 if knobs == selected else 10,
            "elapsed_s": 10.0,
            "positions_per_second": 4.0 if knobs == selected else 1.0,
            "memory": {"unsafe": False},
        },
        runtime_probe_host_profile={"host": "test"},
        runtime_probe_code_hash="code",
    )

    controller.run(max_quanta=1)

    assert runner.runtime_rows == [(2, 5, 64, 300, 0, 3000, 12)]
    study = controller.create_or_resume_study()
    trial = study.get_trials(deepcopy=False)[0]
    assert trial.user_attrs["runtime_probe_status"] == "selected"
    assert trial.user_attrs["runtime_probe_selected_knobs"]["workers"] == 2
    full_config = json.loads(
        (tmp_path / "runs" / "runtime_selected" / "candidates" / candidate.candidate_id / "full_config.json")
        .read_text(encoding="utf-8")
    )
    assert full_config["selfplay"]["num_workers"] == 2
    assert full_config["selfplay"]["games_per_epoch"] == 0
    assert full_config["selfplay"]["states_per_epoch"] == 3000
    assert full_config["train"]["batches_per_epoch"] == 12


def test_runtime_probe_speed_quarantine_stops_candidate_but_not_scout(tmp_path):
    candidates = (_candidate("global_xattn_0"), _candidate("global_line_window_0"))
    runner = RecordingRunner()
    controller = Phase1OptunaScoutController(
        runs_root=tmp_path / "runs",
        run_id="runtime_quarantine",
        candidates=candidates,
        runner=runner,
        min_epochs=2,
        quantum_epochs=2,
        runtime_probe_candidates=(RuntimeKnobs(1, 4, 32, 200),),
        runtime_probe_runner=lambda knobs, index: {
            "candidate": knobs.to_legacy_candidate(),
            "ok": True,
            "positions": 10,
            "elapsed_s": 10.0,
            "positions_per_second": 1.0,
            "memory": {"unsafe": False},
        },
    )

    summary = controller.run(max_quanta=2)

    assert summary.candidate_statuses[candidates[0].candidate_id]["hexo_status"] == "quarantined"
    assert summary.candidate_statuses[candidates[1].candidate_id]["hexo_status"] == "quarantined"
    assert runner.epochs == []
    for candidate in candidates:
        quarantine = (
            tmp_path / "runs" / "runtime_quarantine" / "candidates" / candidate.candidate_id / "quarantine.json"
        )
        assert quarantine.exists()
