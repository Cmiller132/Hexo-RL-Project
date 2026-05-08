import json
from dataclasses import asdict

import pytest

from hexorl.autotune import CandidateRecipe, ModelRecipe
from hexorl.config import Config
from hexorl.eval.arena import MatchResult
from hexorl.eval.scorecard import ScorecardRecord, append_scorecard
from hexorl.tuning.fixed_classical_eval import FixedClassicalEvalSettings
from hexorl.tuning.optuna_tuning import phase3_study_spec
from hexorl.tuning.phase3_runner import (
    DryRunPhase3TrialRunner,
    Phase3OptunaTpeRunner,
    Phase3StudySpec,
)
from hexorl.tuning.phase3_supervisor import (
    ActiveProcess,
    Phase3AutonomousSupervisor,
    Phase3StudyTerminalCounts,
    next_round_target,
    phase3_terminal_counts,
)


def test_phase3_supervisor_next_target_uses_min_terminal_count():
    empty = (
        Phase3StudyTerminalCounts("a", "sqlite:///a", "a", 0, 0, 0, 0, 0),
        Phase3StudyTerminalCounts("b", "sqlite:///b", "b", 0, 0, 0, 0, 0),
    )
    one_each = (
        Phase3StudyTerminalCounts("a", "sqlite:///a", "a", 1, 1, 0, 0, 1),
        Phase3StudyTerminalCounts("b", "sqlite:///b", "b", 1, 1, 0, 0, 1),
    )
    uneven = (
        Phase3StudyTerminalCounts("a", "sqlite:///a", "a", 3, 3, 0, 0, 3),
        Phase3StudyTerminalCounts("b", "sqlite:///b", "b", 1, 1, 0, 0, 1),
    )
    capped = (
        Phase3StudyTerminalCounts("a", "sqlite:///a", "a", 256, 256, 0, 0, 256),
        Phase3StudyTerminalCounts("b", "sqlite:///b", "b", 256, 256, 0, 0, 256),
    )

    assert next_round_target(empty, max_trials_per_study=256) == 1
    assert next_round_target(one_each, max_trials_per_study=256) == 2
    assert next_round_target(uneven, max_trials_per_study=256) == 2
    assert next_round_target(capped, max_trials_per_study=256) is None


def test_phase3_supervisor_stop_marker_prevents_round_launch(tmp_path):
    run_dir, _spec_path = _phase3_supervisor_fixture(tmp_path)
    (run_dir / "phase3_autosupervisor.stop").write_text("paused\n", encoding="utf-8")
    launched = []

    summary = Phase3AutonomousSupervisor(
        run_dir=run_dir,
        max_rounds=1,
        mirror_dashboards=False,
        process_finder=lambda: [],
        round_runner=lambda **_kwargs: launched.append(True) or 0,
    ).run()

    assert summary.status == "paused"
    assert summary.stopped_reason == "stop_marker_present"
    assert launched == []


def test_phase3_supervisor_active_lock_prevents_duplicate(tmp_path):
    run_dir, _spec_path = _phase3_supervisor_fixture(tmp_path)
    (run_dir / "phase3_autosupervisor.lock.json").write_text(
        json.dumps({"pid": 12345}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="phase3_(runner|supervisor)_already_active"):
        Phase3AutonomousSupervisor(
            run_dir=run_dir,
            max_rounds=1,
            mirror_dashboards=False,
            process_finder=lambda: [ActiveProcess(12345, "python scripts/run_phase3_autonomous_supervisor.py")],
            round_runner=lambda **_kwargs: 0,
        ).run()


def test_phase3_supervisor_dry_run_advances_two_studies_to_round_three(tmp_path):
    run_dir, spec_path = _phase3_supervisor_fixture(tmp_path)
    outcomes = [
        MatchResult(winner=0, side_a_score=1.0, side_b_score=0.0, moves=60, time_ms=1.0, opening_is_black=True),
        MatchResult(winner=-1, side_a_score=0.0, side_b_score=0.0, moves=100, time_ms=1.0, opening_is_black=False, reason="max_moves"),
    ]

    def run_round(*, target_trials, summary_path, stdout_path, stderr_path):
        stdout_path.write_text(f"target={target_trials}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        Phase3OptunaTpeRunner(
            run_dir=run_dir,
            spec_path=spec_path,
            trial_runner=DryRunPhase3TrialRunner(),
            n_trials_per_study=target_trials,
            trial_epochs=2,
            fixed_eval_settings=FixedClassicalEvalSettings(games_per_candidate=2, max_moves=100),
            fixed_eval_game_runner=lambda game_index, _seed: outcomes[game_index],
            summary_path=summary_path,
        ).run()
        return 0

    summary = Phase3AutonomousSupervisor(
        run_dir=run_dir,
        start_target="auto",
        max_trials_per_study=256,
        max_rounds=3,
        mirror_dashboards=False,
        process_finder=lambda: [],
        round_runner=run_round,
        summary_path=run_dir / "phase3_autosupervisor_summary.json",
    ).run()

    assert summary.status == "paused"
    assert summary.stopped_reason == "max_rounds_reached"
    assert [item.target_trials_per_study for item in summary.rounds] == [1, 2, 3]
    counts = phase3_terminal_counts(tuple(_load_specs(spec_path)))
    assert {item.promoted_candidate_id: item.trials_terminal for item in counts} == {
        "global_graph768_champion__none__v1": 3,
        "global_xattn_0__none__v1": 3,
    }
    assert (run_dir / "phase3_autosupervisor_events.jsonl").exists()
    assert (run_dir / "phase3_autosupervisor_summary.json").exists()
    assert (run_dir / "phase3_review" / "phase3_trial_ranking_report.json").exists()
    assert (run_dir / "champion_selection_report_phase3.json").exists()


def test_phase3_supervisor_stops_when_round_adds_failed_trials(tmp_path):
    optuna = pytest.importorskip("optuna")
    run_dir, spec_path = _phase3_supervisor_fixture(tmp_path)
    specs = _load_specs(spec_path)

    def run_round(*, target_trials, summary_path, stdout_path, stderr_path):
        stdout_path.write_text(f"target={target_trials}\n", encoding="utf-8")
        stderr_path.write_text("forced failure\n", encoding="utf-8")
        for spec in specs:
            study = optuna.create_study(
                study_name=spec.study_name,
                storage=spec.storage,
                direction=spec.direction,
                load_if_exists=True,
            )
            trial = study.ask()
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        summary_path.write_text(json.dumps({"forced_failure": True}), encoding="utf-8")
        return 0

    summary = Phase3AutonomousSupervisor(
        run_dir=run_dir,
        max_rounds=3,
        mirror_dashboards=False,
        process_finder=lambda: [],
        round_runner=run_round,
        summary_path=run_dir / "phase3_autosupervisor_summary.json",
    ).run()

    assert summary.status == "failed"
    assert summary.rounds[0].status == "failed"
    assert summary.rounds[0].reason.startswith("phase3_trial_failures:")
    assert len(summary.rounds) == 1


def _phase3_supervisor_fixture(tmp_path):
    run_dir = tmp_path / "run"
    spec_path = run_dir / "phase2_review" / "phase3_study_specs.json"
    spec_path.parent.mkdir(parents=True)
    (run_dir / "phase3_studies").mkdir(parents=True)
    specs = []
    for rank, (promoted_id, architecture_id) in enumerate(
        (
            ("global_xattn_0__none__v1", "global_xattn_0"),
            ("global_graph768_champion__none__v1", "global_graph768_champion"),
        ),
        start=1,
    ):
        source_dir = run_dir / "candidates" / promoted_id
        source_dir.mkdir(parents=True)
        cfg = CandidateRecipe(model=ModelRecipe(architecture_id=architecture_id)).materialize_config(Config())
        (source_dir / "full_config.json").write_text(
            json.dumps(cfg.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        checkpoint = source_dir / "checkpoints" / "epoch_0012.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"checkpoint")
        source_scorecard = source_dir / "scorecards.jsonl"
        append_scorecard(
            source_scorecard,
            ScorecardRecord(
                candidate_id=promoted_id,
                scalar_score=0.42 - rank * 0.01,
                component_metrics={
                    "classical_survival_lcb": 0.42 - rank * 0.01,
                    "classical_survival_games": 4.0,
                    "illegal_or_crash_rate": 0.0,
                },
                hard_gates={"hard_pass": True, "failures": []},
                checkpoint_lineage={"checkpoint_path": str(checkpoint)},
                evidence_paths=(str(source_dir / "fixed_classical_epoch_0012_games.jsonl"),),
                epoch=12,
                completed_epochs=12,
                metadata={"classical_survival_lcb": {"games": 4, "score": 0.42 - rank * 0.01}},
            ),
        )
        base_spec = phase3_study_spec(
            architecture_id=architecture_id,
            pair_mode="none",
            storage=f"sqlite:///{(run_dir / 'phase3_studies' / (promoted_id + '.sqlite3')).as_posix()}",
            seed=11,
        )
        specs.append(
            Phase3StudySpec(
                **{
                    **asdict(base_spec),
                    "metadata": {
                        **base_spec.metadata,
                        "promoted_candidate_id": promoted_id,
                        "phase2_rank": rank,
                        "phase2_classical_survival_lcb": 0.42 - rank * 0.01,
                        "phase2_checkpoint_lineage": {"checkpoint_path": str(checkpoint)},
                        "phase2_evidence_paths": [str(source_dir / "fixed_classical_epoch_0012_games.jsonl")],
                        "phase2_scorecard_path": str(source_scorecard),
                    },
                }
            )
        )
    spec_path.write_text(json.dumps([asdict(spec) for spec in specs], indent=2, sort_keys=True), encoding="utf-8")
    return run_dir, spec_path


def _load_specs(spec_path):
    return [Phase3StudySpec(**item) for item in json.loads(spec_path.read_text(encoding="utf-8"))]
