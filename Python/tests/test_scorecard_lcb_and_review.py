import json
import sqlite3
from types import SimpleNamespace

import pytest

from hexorl.eval.arena import MatchResult
from hexorl.eval.scorecard import (
    CLASSICAL_CONFIDENCE_Z,
    ClassicalGameEvidence,
    ScorecardRecord,
    append_scorecard,
    build_classical_scorecard_record,
    classical_survival_lcb,
    read_scorecards,
)
from hexorl.tuning.fixed_classical_eval import (
    FixedClassicalEvalSettings,
    evaluate_candidate_fixed_classical,
)
from hexorl.tuning.optuna_tuning import (
    HexoScorecardFloorPruner,
    create_phase3_floor_pruner,
    mark_trial_hexo_artifacts,
    phase3_study_specs_from_phase2_report,
    phase3_study_name,
    phase3_study_spec,
)
from hexorl.tuning.review import rank_phase2_survivors
from hexorl.tuning.scorecard_repair import repair_phase1_scorecards_from_dashboard


def test_classical_survival_lcb_recomputes_from_fixed_classical_evidence():
    rows = [
        ClassicalGameEvidence(
            outcome="loss",
            moves=50,
            max_moves=100,
            illegal_or_crash_penalty=0.0,
            confidence_method="normal_95",
            opponent_id="fixed_strong",
            seed=1,
        ),
        ClassicalGameEvidence(
            outcome="draw",
            moves=100,
            max_moves=100,
            illegal_or_crash_penalty=0.0,
            confidence_method="normal_95",
            opponent_id="fixed_strong",
            seed=2,
        ),
        ClassicalGameEvidence(
            outcome="win",
            moves=61,
            max_moves=100,
            illegal_or_crash_penalty=0.0,
            confidence_method="normal_95",
            opponent_id="fixed_strong",
            seed=3,
        ),
        ClassicalGameEvidence(
            outcome="loss",
            moves=10,
            max_moves=100,
            illegal_or_crash_penalty=1.0,
            confidence_method="normal_95",
            opponent_id="fixed_strong",
            seed=4,
        ),
    ]

    result = classical_survival_lcb([row.to_dict() for row in rows])
    scores = [0.5, 1.0, 1.15, -0.9]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    expected = mean - CLASSICAL_CONFIDENCE_Z["normal_95"] * (variance**0.5) / (len(scores) ** 0.5)

    assert result.score == pytest.approx(expected)
    assert result.mean == pytest.approx(mean)
    assert result.games == 4
    assert result.opponent_ids == ("fixed_strong",)
    assert result.seeds == (1, 2, 3, 4)
    assert result.illegal_or_crash_penalty_total == pytest.approx(1.0)


def test_scorecards_append_and_read_traceable_classical_records(tmp_path):
    evidence_path = tmp_path / "fixed_classical_games.jsonl"
    rows = [
        {
            "outcome": "loss",
            "moves": 80,
            "max_moves": 100,
            "illegal_or_crash_penalty": 0.0,
            "confidence_method": "normal_95",
            "opponent_id": "fixed_strong",
            "seed": 10,
        },
        {
            "outcome": "draw",
            "moves": 100,
            "max_moves": 100,
            "illegal_or_crash_penalty": 0.0,
            "confidence_method": "normal_95",
            "opponent_id": "fixed_strong",
            "seed": 11,
        },
    ]
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    scorecard_path = tmp_path / "scorecards.jsonl"

    first = build_classical_scorecard_record(
        candidate_id="global_xattn_0__none__v1",
        evidence_path=evidence_path,
        component_metrics={"generated_positions_per_second": 2.5},
        hard_gates={"hard_pass": True, "failures": []},
        study_id="study_architecture_scout_v1",
        trial_id=7,
        config_hash="abc",
        checkpoint_lineage={"checkpoint": "epoch_0012.pt", "parent": "epoch_0010.pt"},
        epoch=12,
    )
    append_scorecard(scorecard_path, first)
    append_scorecard(
        scorecard_path,
        ScorecardRecord.from_mapping(
            {
                **first.to_dict(),
                "scalar_score": first.scalar_score + 0.1,
                "epoch": 14,
                "completed_epochs": 14,
            }
        ),
    )

    loaded = read_scorecards(scorecard_path)
    assert len(scorecard_path.read_text(encoding="utf-8").splitlines()) == 2
    assert [row.epoch for row in loaded] == [12, 14]
    assert loaded[0].candidate_id == "global_xattn_0__none__v1"
    assert loaded[0].study_id == "study_architecture_scout_v1"
    assert loaded[0].trial_id == 7
    assert loaded[0].config_hash == "abc"
    assert loaded[0].checkpoint_lineage["parent"] == "epoch_0010.pt"
    assert loaded[0].evidence_paths == (str(evidence_path),)
    assert "classical_survival_lcb" in loaded[0].component_metrics

    append_scorecard(
        scorecard_path,
        {
            "candidate_id": "global_xattn_0__none__v1",
            "classical_survival_lcb": 0.25,
            "component_metrics": {"classical_survival_lcb": 0.25},
            "hard_gates": {"hard_pass": True, "failures": []},
            "checkpoint_path": "runs/candidate/checkpoints/epoch_0016.pt",
            "epoch": 16,
            "completed_epochs": 16,
        },
    )
    loaded = read_scorecards(scorecard_path)
    assert loaded[-1].scalar_score == pytest.approx(0.25)
    assert loaded[-1].metadata["extra_fields"]["checkpoint_path"].endswith("epoch_0016.pt")


def test_phase2_review_ranks_epoch12_survivors_and_excludes_failed_candidates():
    report = rank_phase2_survivors(
        [
            _record("a", 12, 0.40),
            _record("b", 12, 0.55),
            _record("quarantined", 12, 9.0, status="quarantined"),
            _record("below_floor", 11, 8.0),
            _record("hard_fail", 12, 7.0, failures=["illegal_or_crash_rate"]),
        ]
    )

    assert [row.candidate_id for row in report.ranked] == ["b", "a"]
    assert [row.rank for row in report.ranked] == [1, 2]
    excluded = {row.candidate_id: row.reason for row in report.excluded}
    assert excluded == {
        "below_floor": "below_epoch_floor",
        "hard_fail": "hard_failed",
        "quarantined": "runtime_quarantine",
    }


def test_phase2_review_rejects_placeholder_lcb_without_fixed_classical_games():
    report = rank_phase2_survivors(
        [
            _record("real", 12, 0.40),
            ScorecardRecord(
                candidate_id="placeholder",
                scalar_score=0.0,
                component_metrics={"classical_survival_lcb": 0.0},
                hard_gates={"hard_pass": True, "failures": []},
                checkpoint_lineage={"checkpoint_path": "runs/placeholder/checkpoints/epoch_0012.pt"},
                evidence_paths=("runs/placeholder/dashboard.sqlite3",),
                epoch=12,
                completed_epochs=12,
            ),
        ]
    )

    assert [row.candidate_id for row in report.ranked] == ["real"]
    excluded = {row.candidate_id: row.reason for row in report.excluded}
    assert excluded["placeholder"] == "missing_classical_evidence"


def test_phase2_review_missing_created_at_rows_do_not_shadow_fixed_classical_evidence():
    placeholder = {
        "candidate_id": "candidate",
        "scalar_score": 0.0,
        "component_metrics": {"classical_survival_lcb": 0.0},
        "hard_gates": {"hard_pass": True, "failures": []},
        "evidence_paths": ["runs/candidate/dashboard.sqlite3"],
        "epoch": 12,
        "completed_epochs": 12,
    }
    fixed = ScorecardRecord.from_mapping(
        {
            **_record("candidate", 12, 0.40).to_dict(),
            "created_at": "2026-05-07T08:34:23+00:00",
        }
    )

    assert ScorecardRecord.from_mapping(placeholder).created_at == ""
    report = rank_phase2_survivors([placeholder, fixed])

    assert [row.candidate_id for row in report.ranked] == ["candidate"]
    assert report.ranked[0].classical_survival_lcb == pytest.approx(0.40)
    assert report.excluded == ()


def test_phase1_scorecard_repair_appends_dashboard_train_components(tmp_path):
    candidate_id = "global_xattn_0__none__v1"
    candidate_dir = tmp_path / "run" / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "candidate_manifest.json").write_text(
        json.dumps({"candidate_id": candidate_id, "pair_strategy_mode": "none"}),
        encoding="utf-8",
    )
    scorecard_path = candidate_dir / "scorecards.jsonl"
    append_scorecard(
        scorecard_path,
        ScorecardRecord(
            candidate_id=candidate_id,
            scalar_score=0.0,
            component_metrics={"classical_survival_lcb": 0.0, "train_loss": 0.0},
            hard_gates={"hard_pass": True, "failures": []},
            evidence_paths=(str(candidate_dir / "events.jsonl"),),
            epoch=4,
            completed_epochs=4,
        ),
    )
    dashboard_path = candidate_dir / "dashboard.sqlite3"
    con = sqlite3.connect(dashboard_path)
    try:
        con.execute(
            "create table metrics (metric_id integer primary key, phase text, epoch integer, "
            "global_step integer, metrics_json text, created_at text)"
        )
        con.execute(
            "insert into metrics (phase, epoch, global_step, metrics_json, created_at) values (?, ?, ?, ?, ?)",
            (
                "train",
                4,
                48,
                json.dumps(
                    {
                        "train": {
                            "loss_total": 6.5,
                            "loss_policy_place": 5.25,
                            "loss_value": 1.2,
                            "pair_policy_weight_mean": 0.8,
                            "batches_per_sec": 0.3,
                            "graph_microbatch_oom_retries": 0,
                            "graph_microbatch_nonfinite_retries": 0,
                        },
                        "buffer": {
                            "avg_missing_target_policy_mass": 0.0,
                            "avg_candidate_recall_mcts_top1": 1.0,
                            "critical_overflow_count": 0,
                            "pair_prior_hit_frac": 0.0,
                            "pair_fallback_prior_use": 1.0,
                            "pair_fallback_prior_use_on_mcts_top1": 1.0,
                            "pair_fallback_prior_use_on_mcts_top8": 1.0,
                        },
                    }
                ),
                "2026-05-07T01:00:00Z",
            ),
        )
        con.commit()
    finally:
        con.close()

    summary = repair_phase1_scorecards_from_dashboard(tmp_path / "run")

    assert summary.appended_rows == 1
    loaded = read_scorecards(scorecard_path)
    assert len(loaded) == 2
    repaired = loaded[-1]
    assert repaired.component_metrics["train_loss"] == pytest.approx(6.5)
    assert repaired.component_metrics["loss_total"] == pytest.approx(6.5)
    assert repaired.component_metrics["loss_policy_place"] == pytest.approx(5.25)
    assert repaired.component_metrics["avg_candidate_recall_mcts_top1"] == pytest.approx(1.0)
    assert repaired.component_metrics["pair_prior_hit_frac"] == pytest.approx(0.0)
    assert repaired.component_metrics["pair_fallback_prior_use"] == pytest.approx(0.0)
    assert repaired.component_metrics["pair_fallback_prior_use_on_mcts_top1"] == pytest.approx(0.0)
    assert repaired.component_metrics["pair_fallback_prior_use_on_mcts_top8"] == pytest.approx(0.0)
    assert repaired.metadata["scorecard_repair"]["source"] == "dashboard_train_metric"
    assert str(dashboard_path) in repaired.evidence_paths

    second = repair_phase1_scorecards_from_dashboard(tmp_path / "run")
    assert second.appended_rows == 0
    assert len(read_scorecards(scorecard_path)) == 2


def test_fixed_classical_eval_appends_evidence_scorecard_and_is_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    candidate_dir = run_dir / "candidates" / "global_xattn_0__none__v1"
    candidate_dir.mkdir(parents=True)
    checkpoint = candidate_dir / "checkpoints" / "epoch_0012.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    scorecard_path = candidate_dir / "scorecards.jsonl"
    append_scorecard(
        scorecard_path,
        ScorecardRecord(
            candidate_id=candidate_dir.name,
            scalar_score=0.0,
            component_metrics={"classical_survival_lcb": 0.0, "train_loss": 3.5},
            hard_gates={"hard_pass": True, "failures": []},
            checkpoint_lineage={"checkpoint_path": str(checkpoint)},
            evidence_paths=(str(candidate_dir / "dashboard.sqlite3"),),
            epoch=12,
            completed_epochs=12,
        ),
    )
    outcomes = [
        MatchResult(winner=1, side_a_score=0.0, side_b_score=1.0, moves=80, time_ms=1.0, opening_is_black=True, reason="terminal"),
        MatchResult(winner=0, side_a_score=1.0, side_b_score=0.0, moves=61, time_ms=1.0, opening_is_black=False, reason="terminal"),
        MatchResult(winner=-1, side_a_score=0.0, side_b_score=0.0, moves=100, time_ms=1.0, opening_is_black=True, reason="max_moves"),
    ]

    result = evaluate_candidate_fixed_classical(
        candidate_dir,
        run_dir=run_dir,
        settings=FixedClassicalEvalSettings(games_per_candidate=3, max_moves=100),
        game_runner=lambda game_index, _seed: outcomes[game_index],
    )

    assert result.status == "ready"
    assert result.appended_games == 3
    assert result.appended_scorecards == 1
    evidence_path = candidate_dir / "fixed_classical_epoch_0012_games.jsonl"
    assert evidence_path.exists()
    evidence_rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    assert [row["outcome"] for row in evidence_rows] == ["classical_win", "model_win", "survived"]
    loaded = read_scorecards(scorecard_path)
    assert len(loaded) == 2
    fixed = loaded[-1]
    assert fixed.component_metrics["classical_survival_games"] == pytest.approx(3.0)
    assert fixed.component_metrics["classical_win_rate"] == pytest.approx(1.0 / 3.0)
    assert fixed.component_metrics["classical_draw_rate"] == pytest.approx(1.0 / 3.0)
    assert fixed.metadata["fixed_classical_eval"]["games_requested"] == 3
    assert str(evidence_path) in fixed.evidence_paths

    second = evaluate_candidate_fixed_classical(
        candidate_dir,
        run_dir=run_dir,
        settings=FixedClassicalEvalSettings(games_per_candidate=3, max_moves=100),
        game_runner=lambda game_index, _seed: outcomes[game_index],
    )
    assert second.appended_games == 0
    assert second.appended_scorecards == 0
    assert len(read_scorecards(scorecard_path)) == 2


def test_phase3_study_spec_names_per_family_pair_mode_and_tpe_floor_pruner():
    spec = phase3_study_spec(
        architecture_id="global_pair_twostage_0",
        pair_mode="root_pair_mcts",
        storage="sqlite:///runs/test/optuna.sqlite3",
        seed=123,
    )

    assert spec.study_name == "study_global_pair_twostage_0__root_pair_mcts__schedule_v1"
    assert phase3_study_name("global_xattn_0", "none") == "study_global_xattn_0__none__schedule_v1"
    assert spec.sampler == {
        "type": "TPESampler",
        "multivariate": True,
        "group": True,
        "n_startup_trials": 8,
        "seed": 123,
    }
    assert spec.pruner["signal_floor_epoch"] == 12
    assert spec.pruner["requires_scorecard_written"] is True
    assert spec.pruner["delegate"]["type"] == "SuccessiveHalvingPruner"
    assert "pair_prior_mix" in spec.search_scope["knobs"]
    assert spec.metadata["hexo_hard_gates_before_optuna"] is True


def test_phase3_specs_from_phase2_report_preserve_promotion_evidence():
    report = rank_phase2_survivors(
        [
            _record("global_pair_twostage_0__root_pair_mcts__v1", 12, 0.55),
            _record("global_xattn_0__none__v1", 12, 0.40),
        ]
    )

    specs = phase3_study_specs_from_phase2_report(
        report,
        storage_template="sqlite:///runs/phase3/<candidate_id>.sqlite3",
        max_promoted=1,
        seed=99,
    )

    assert len(specs) == 1
    assert specs[0].architecture_id == "global_pair_twostage_0"
    assert specs[0].pair_mode == "root_pair_mcts"
    assert specs[0].storage.endswith("global_pair_twostage_0__root_pair_mcts__v1.sqlite3")
    assert specs[0].metadata["promoted_candidate_id"] == "global_pair_twostage_0__root_pair_mcts__v1"
    assert specs[0].metadata["phase2_rank"] == 1
    assert specs[0].metadata["phase2_evidence_paths"]


def test_mark_trial_hexo_artifacts_records_traceable_attrs():
    trial = _TrialAttrs()

    mark_trial_hexo_artifacts(
        trial,
        scorecard_path="runs/candidate/scorecards.jsonl",
        checkpoint_path="runs/candidate/checkpoints/epoch_0012.pt",
        evidence_paths=("runs/candidate/fixed_classical_games.jsonl",),
        hard_gates={"hard_pass": True, "failures": []},
        debug_bundle_paths=("runs/candidate/debug_bundles/probe",),
        extra_attrs={"hexo_candidate_id": "candidate"},
    )

    assert trial.user_attrs["scorecards_written"] is True
    assert trial.user_attrs["final_scorecard_path"] == "runs/candidate/scorecards.jsonl"
    assert trial.user_attrs["hexo_checkpoint_path"].endswith("epoch_0012.pt")
    assert trial.user_attrs["hexo_evidence_paths"] == ["runs/candidate/fixed_classical_games.jsonl"]
    assert trial.user_attrs["hexo_hard_gate_passed"] is True
    assert trial.user_attrs["hexo_artifacts_recorded"] is True
    assert trial.user_attrs["hexo_candidate_id"] == "candidate"


def test_floor_pruner_refuses_before_signal_floor_and_before_scorecards(tmp_path):
    delegate = _DelegatePruner(result=True)
    pruner = HexoScorecardFloorPruner(signal_floor_epoch=12, delegate=delegate)

    assert pruner.prune(None, SimpleNamespace(last_step=11, user_attrs={"scorecards_written": True})) is False
    assert delegate.calls == 0
    assert pruner.prune(None, SimpleNamespace(last_step=12, user_attrs={})) is False
    assert delegate.calls == 0

    assert pruner.prune(None, SimpleNamespace(last_step=12, user_attrs={"scorecards_written": True})) is True
    assert delegate.calls == 1

    scorecard_path = tmp_path / "scorecards.jsonl"
    scorecard_path.write_text("{}", encoding="utf-8")
    assert pruner.prune(
        None,
        SimpleNamespace(last_step=None, intermediate_values={12: 0.1}, user_attrs={"final_scorecard_path": str(scorecard_path)}),
    ) is True
    assert delegate.calls == 2


def test_create_phase3_floor_pruner_uses_optuna_successive_halving_when_available():
    optuna = pytest.importorskip("optuna")

    pruner = create_phase3_floor_pruner(signal_floor_epoch=12, reduction_factor=2)

    assert isinstance(pruner, HexoScorecardFloorPruner)
    assert isinstance(pruner.delegate, optuna.pruners.SuccessiveHalvingPruner)


class _DelegatePruner:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def prune(self, study, trial):
        self.calls += 1
        return self.result


class _TrialAttrs:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def _record(candidate_id, epoch, score, status="healthy", failures=None):
    return ScorecardRecord(
        candidate_id=candidate_id,
        scalar_score=score,
        component_metrics={
            "classical_survival_lcb": score,
            "classical_survival_games": 2.0,
            "illegal_or_crash_rate": 0.0,
        },
        hard_gates={"hard_pass": not failures, "failures": failures or []},
        checkpoint_lineage={"checkpoint_path": f"runs/{candidate_id}/checkpoints/epoch_{epoch:04d}.pt"},
        evidence_paths=(f"runs/{candidate_id}/fixed_classical_games.jsonl",),
        epoch=epoch,
        completed_epochs=epoch,
        status=status,
        metadata={
            "classical_survival_lcb": {
                "games": 2,
                "score": score,
            }
        },
    )
