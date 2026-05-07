import json
from pathlib import Path

import hexorl.tuning as tuning
from hexorl.eval.scorecard import ScorecardRecord, append_scorecard
from hexorl.tuning.champion import (
    build_champion_selection_report_from_scorecard_files,
    select_champion_from_scorecards,
    write_champion_selection_report,
)


def test_champion_selection_uses_hexo_scorecard_not_optuna_value(tmp_path):
    winner = _scorecard(
        "global_graph768_champion__none",
        scalar_score=0.72,
        optuna_value=0.10,
        checkpoint_path="runs/scout/candidates/winner/checkpoints/epoch_12.pt",
        scorecard_path=tmp_path / "winner_scorecards.jsonl",
    )
    runner_up = _scorecard(
        "global_pair_twostage__root_pair_mcts",
        scalar_score=0.61,
        optuna_value=999.0,
        checkpoint_path="runs/scout/candidates/runner/checkpoints/epoch_12.pt",
        scorecard_path=tmp_path / "runner_scorecards.jsonl",
    )

    report = select_champion_from_scorecards(
        [runner_up, winner],
        reproduction_command="python scripts/run_phase1_optuna_scout.py --runs-root runs --run-id scout --production",
    )
    assert report.selected is not None
    assert report.selected.candidate_id == winner.candidate_id
    assert report.runner_up is not None
    assert report.runner_up.candidate_id == runner_up.candidate_id
    assert report.metadata["optuna_value_role"] == "trace_only_not_ranking_authority"
    assert report.runner_up_comparison["scalar_delta"] == winner.scalar_score - runner_up.scalar_score

    report_path = tmp_path / "champion_report.json"
    write_champion_selection_report(report_path, report)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["selected"]["checkpoint_lineage"]["checkpoint_path"].endswith("epoch_12.pt")
    assert payload["selected"]["gates"]["hard_pass"] is True
    assert payload["reproduction_command"].endswith("--production")


def test_champion_selection_rejects_failed_floor_and_missing_evidence(tmp_path):
    eligible = _scorecard(
        "eligible",
        scalar_score=0.40,
        scorecard_path=tmp_path / "eligible_scorecards.jsonl",
    )
    failed = _scorecard(
        "failed",
        scalar_score=0.90,
        hard_pass=False,
        gate_failures=("illegal_or_crash_rate",),
        scorecard_path=tmp_path / "failed_scorecards.jsonl",
    )
    below_floor = _scorecard(
        "below_floor",
        scalar_score=0.80,
        completed_epochs=10,
        scorecard_path=tmp_path / "below_floor_scorecards.jsonl",
    )
    missing_evidence = _scorecard(
        "missing_evidence",
        scalar_score=0.70,
        evidence_paths=(),
        scorecard_path=tmp_path / "missing_scorecards.jsonl",
    )
    missing_lineage = ScorecardRecord.from_mapping(
        {
            **_scorecard(
                "missing_lineage",
                scalar_score=0.69,
                scorecard_path=tmp_path / "missing_lineage_scorecards.jsonl",
            ).to_dict(),
            "checkpoint_lineage": {},
        }
    )
    placeholder_lcb = ScorecardRecord.from_mapping(
        {
            **_scorecard(
                "placeholder_lcb",
                scalar_score=0.0,
                scorecard_path=tmp_path / "placeholder_scorecards.jsonl",
            ).to_dict(),
            "component_metrics": {"classical_survival_lcb": 0.0, "illegal_or_crash_rate": 0.0},
            "evidence_paths": ("runs/placeholder/dashboard.sqlite3",),
            "metadata": {"score_source": "epoch_result_classical_survival_lcb_or_zero"},
        }
    )

    report = select_champion_from_scorecards(
        [failed, below_floor, missing_evidence, missing_lineage, placeholder_lcb, eligible],
        reproduction_command="python scripts/run_phase1_optuna_scout.py --runs-root runs --run-id scout --production",
    )

    assert report.selected is not None
    assert report.selected.candidate_id == "eligible"
    rejected = {item.candidate_id: item.reason for item in report.rejected}
    assert rejected == {
        "below_floor": "below_epoch_floor",
        "failed": "hard_failed",
        "missing_evidence": "missing_evidence",
        "missing_lineage": "missing_checkpoint_lineage",
        "placeholder_lcb": "missing_classical_evidence",
    }


def test_champion_report_loads_scorecard_files_and_preserves_traceability(tmp_path):
    scorecard_path = tmp_path / "scorecards.jsonl"
    append_scorecard(
        scorecard_path,
        _scorecard(
            "candidate_a",
            scalar_score=0.55,
            scorecard_path=scorecard_path,
        ),
    )
    append_scorecard(
        scorecard_path,
        _scorecard(
            "candidate_b",
            scalar_score=0.59,
            scorecard_path=scorecard_path,
        ),
    )

    report = build_champion_selection_report_from_scorecard_files(
        [scorecard_path],
        reproduction_command="python scripts/run_phase1_optuna_scout.py --runs-root runs --run-id scout --production",
    )

    assert report.selected is not None
    assert report.selected.candidate_id == "candidate_b"
    assert report.selected.scorecard_paths == (str(scorecard_path),)
    assert report.selected.evidence_paths == ("runs/candidate_b/fixed_classical_games.jsonl",)


def test_active_tuning_package_no_longer_exports_legacy_schedulers():
    assert not hasattr(tuning, "ASHARungTable")
    assert not hasattr(tuning, "BOHBSampler")
    assert not hasattr(tuning, "PB2Scheduler")
    assert hasattr(tuning, "Phase1OptunaScoutController")
    assert hasattr(tuning, "select_champion_from_scorecards")


def test_production_import_audit_retired_legacy_scheduler_surface():
    root = Path(__file__).resolve().parents[2]
    scan_files = list((root / "Python" / "src" / "hexorl").rglob("*.py"))
    scan_files.append(root / "scripts" / "run_phase1_optuna_scout.py")
    allowed = {
        "Python/src/hexorl/tuning/asha.py",
        "Python/src/hexorl/tuning/bohb.py",
        "Python/src/hexorl/tuning/pb2.py",
    }
    forbidden = ("ASHARungTable", "BOHBSampler", "PB2Scheduler", "run_phase3_48h_autotune.py")
    violations: list[str] = []
    for path in scan_files:
        rel = path.relative_to(root).as_posix()
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{rel}: {token}")

    legacy_script = (root / "scripts" / "run_phase3_48h_autotune.py").read_text(encoding="utf-8")
    assert violations == []
    assert "LEGACY_NON_PRODUCTION_SUPERVISOR = True" in legacy_script
    assert "--allow-legacy-scheduler" in legacy_script


def test_pair_mode_import_audit_has_no_diagnostic_full_pair_in_active_surfaces():
    root = Path(__file__).resolve().parents[2]
    scan_files = [
        *list((root / "Python" / "src" / "hexorl").rglob("*.py")),
        root / "scripts" / "run_phase1_optuna_scout.py",
        root / "scripts" / "phase00_finalize_artifacts.py",
        root / "Docs" / "GLOBAL_GRAPH_MODEL_EXPLAINER.md",
        root / "Docs" / "OPTUNA_SEQUENTIAL_AUTOTUNING_PLAN.md",
    ]

    violations = [
        path.relative_to(root).as_posix()
        for path in scan_files
        if "diagnostic_full_pair" in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def _scorecard(
    candidate_id: str,
    *,
    scalar_score: float,
    optuna_value: float | None = None,
    completed_epochs: int = 12,
    hard_pass: bool = True,
    gate_failures: tuple[str, ...] = (),
    checkpoint_path: str | None = None,
    evidence_paths: tuple[str, ...] | None = None,
    scorecard_path: Path | None = None,
) -> ScorecardRecord:
    return ScorecardRecord(
        candidate_id=candidate_id,
        scalar_name="classical_survival_lcb",
        scalar_score=scalar_score,
        component_metrics={
            "classical_survival_lcb": scalar_score,
            "classical_survival_games": 2.0,
            "illegal_or_crash_rate": 0.0 if hard_pass else 1.0,
        },
        hard_gates={"hard_pass": hard_pass, "failures": list(gate_failures)},
        study_id="study_architecture_scout_v1",
        trial_id=f"trial-{candidate_id}",
        config_hash=f"hash-{candidate_id}",
        checkpoint_lineage={
            "checkpoint_path": checkpoint_path or f"runs/{candidate_id}/checkpoints/epoch_12.pt",
            "parent_checkpoint": "",
        },
        evidence_paths=evidence_paths
        if evidence_paths is not None
        else (f"runs/{candidate_id}/fixed_classical_games.jsonl",),
        epoch=completed_epochs,
        completed_epochs=completed_epochs,
        metadata={
            "classical_survival_lcb": {
                "games": 2,
                "score": scalar_score,
            },
            "optuna_value": optuna_value,
            "scorecard_path": str(scorecard_path or ""),
        },
    )
