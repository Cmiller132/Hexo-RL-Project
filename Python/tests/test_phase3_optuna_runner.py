import json
import sqlite3
from dataclasses import asdict

import pytest

from hexorl.autotune import CandidateRecipe, ModelRecipe
from hexorl.config import Config
from hexorl.eval.arena import MatchResult
from hexorl.eval.scorecard import ScorecardRecord, append_scorecard, read_scorecards
from hexorl.tuning.fixed_classical_eval import FixedClassicalEvalSettings
from hexorl.tuning.optuna_tuning import phase3_study_spec
from hexorl.tuning.phase3_runner import (
    DryRunPhase3TrialRunner,
    Phase3OptunaTpeRunner,
    Phase3StudySpec,
    apply_phase3_params,
)


def test_apply_phase3_params_preserves_architecture_pair_mode_and_heads():
    cfg = CandidateRecipe(model=ModelRecipe(architecture_id="global_xattn_0")).materialize_config(Config())
    tuned = apply_phase3_params(
        cfg,
        {
            "lr_multiplier": 0.75,
            "weight_decay": 2e-4,
            "c_puct": 1.8,
            "scaled_alpha_total": 0.4,
            "value_loss_weight": 1.25,
        },
        pair_mode="none",
    )

    assert tuned.model.architecture == cfg.model.architecture
    assert tuned.model.pair_strategy == cfg.model.pair_strategy
    assert list(tuned.model.heads) == list(cfg.model.heads)
    assert tuned.train.peak_lr == pytest.approx(cfg.train.peak_lr * 0.75)
    assert tuned.train.weight_decay == pytest.approx(2e-4)
    assert tuned.selfplay.c_puct == pytest.approx(1.8)
    assert tuned.selfplay.dirichlet_alpha == pytest.approx(0.4)
    assert tuned.train.loss_weights["value"] == pytest.approx(1.25)


def test_phase3_runner_consumes_specs_writes_scorecards_and_is_resume_safe(tmp_path):
    run_dir = tmp_path / "run"
    promoted_id = "global_xattn_0__none__v1"
    source_dir = run_dir / "candidates" / promoted_id
    source_dir.mkdir(parents=True)
    cfg = CandidateRecipe(model=ModelRecipe(architecture_id="global_xattn_0")).materialize_config(Config())
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
            scalar_score=0.42,
            component_metrics={
                "classical_survival_lcb": 0.42,
                "classical_survival_games": 4.0,
                "illegal_or_crash_rate": 0.0,
            },
            hard_gates={"hard_pass": True, "failures": []},
            checkpoint_lineage={"checkpoint_path": str(checkpoint)},
            evidence_paths=(str(source_dir / "fixed_classical_epoch_0012_games.jsonl"),),
            epoch=12,
            completed_epochs=12,
            metadata={"classical_survival_lcb": {"games": 4, "score": 0.42}},
        ),
    )
    base_spec = phase3_study_spec(
        architecture_id="global_xattn_0",
        pair_mode="none",
        storage=f"sqlite:///{(tmp_path / 'phase3.sqlite3').as_posix()}",
        seed=11,
    )
    spec = Phase3StudySpec(
        **{
            **asdict(base_spec),
            "metadata": {
                **base_spec.metadata,
                "promoted_candidate_id": promoted_id,
                "phase2_rank": 1,
                "phase2_classical_survival_lcb": 0.42,
                "phase2_checkpoint_lineage": {"checkpoint_path": str(checkpoint)},
                "phase2_evidence_paths": [str(source_dir / "fixed_classical_epoch_0012_games.jsonl")],
                "phase2_scorecard_path": str(source_scorecard),
            },
        }
    )
    spec_path = run_dir / "phase2_review" / "phase3_study_specs.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps([asdict(spec)], indent=2, sort_keys=True), encoding="utf-8")
    outcomes = [
        MatchResult(winner=0, side_a_score=1.0, side_b_score=0.0, moves=60, time_ms=1.0, opening_is_black=True),
        MatchResult(winner=-1, side_a_score=0.0, side_b_score=0.0, moves=100, time_ms=1.0, opening_is_black=False, reason="max_moves"),
    ]

    summary = Phase3OptunaTpeRunner(
        run_dir=run_dir,
        spec_path=spec_path,
        trial_runner=DryRunPhase3TrialRunner(),
        n_trials_per_study=1,
        trial_epochs=2,
        fixed_eval_settings=FixedClassicalEvalSettings(games_per_candidate=2, max_moves=100),
        fixed_eval_game_runner=lambda game_index, _seed: outcomes[game_index],
        summary_path=run_dir / "phase3_runner_summary.json",
    ).run()

    assert summary.studies[0].trials_started == 1
    assert summary.studies[0].trials_completed == 1
    assert summary.studies[0].trials_pruned == 0
    assert summary.studies[0].trials_terminal == 1
    phase3_scorecards = list((run_dir / "phase3_trials").glob("*/scorecards.jsonl"))
    assert len(phase3_scorecards) == 1
    scorecards = read_scorecards(phase3_scorecards[0])
    assert scorecards[-1].component_metrics["classical_survival_games"] == pytest.approx(2.0)
    assert scorecards[-1].evidence_paths
    assert scorecards[-1].hard_pass is True
    con = sqlite3.connect(tmp_path / "phase3.sqlite3")
    try:
        attrs = dict(
            con.execute(
                "select key, value_json from trial_user_attributes where key in "
                "('scorecards_written', 'hexo_artifacts_recorded', 'hexo_hard_gate_passed')"
            ).fetchall()
        )
        assert attrs["scorecards_written"] == "true"
        assert attrs["hexo_artifacts_recorded"] == "true"
        assert attrs["hexo_hard_gate_passed"] == "true"
    finally:
        con.close()

    resumed = Phase3OptunaTpeRunner(
        run_dir=run_dir,
        spec_path=spec_path,
        trial_runner=DryRunPhase3TrialRunner(),
        n_trials_per_study=1,
        trial_epochs=2,
        fixed_eval_settings=FixedClassicalEvalSettings(games_per_candidate=2, max_moves=100),
        fixed_eval_game_runner=lambda game_index, _seed: outcomes[game_index],
    ).run()

    assert resumed.studies[0].trials_existing == 1
    assert resumed.studies[0].trials_started == 0
