"""Production Phase 3 Optuna TPE trial runner.

The runner consumes Phase 3 study specs produced from Phase 2 review artifacts,
starts schedule/search child trials from promoted checkpoints, and records Hexo
scorecard evidence before allowing Optuna pruning or ranking to matter.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from hexorl.config import Config
from hexorl.eval.scorecard import ScorecardRecord, append_scorecard, read_scorecards
from hexorl.models.assembly import build_model_from_config
from hexorl.models.loading import restore_model_weights
from hexorl.tuning.fixed_classical_eval import (
    FixedClassicalEvalSettings,
    evaluate_candidate_fixed_classical,
)
from hexorl.tuning.optuna_tuning import (
    Phase3StudySpec,
    create_phase3_study,
    mark_trial_hexo_artifacts,
)
from hexorl.tuning.review import build_phase2_promotion_report_from_scorecard_files


PHASE3_TARGET_SCALAR = "classical_survival_lcb"
PHASE3_TRIALS_DIRNAME = "phase3_trials"
_PAIR_HEADS = ("policy_pair_first", "policy_pair_joint", "policy_pair_second")


@dataclass(frozen=True)
class Phase3TrialRequest:
    spec: Phase3StudySpec
    promoted_candidate_id: str
    phase3_candidate_id: str
    trial_number: int
    trial_dir: Path
    source_candidate_dir: Path
    source_checkpoint_path: Path
    source_scorecard_path: Path
    config: Config
    params: dict[str, float]
    source_epoch: int
    target_epoch: int


@dataclass(frozen=True)
class Phase3TrialRunResult:
    completed_epochs: int
    latest_checkpoint_path: str
    train_scorecard_path: str
    event_paths: tuple[str, ...] = ()
    debug_bundle_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Phase3StudyRunSummary:
    study_name: str
    storage: str
    promoted_candidate_id: str
    trials_requested: int
    trials_existing: int
    trials_started: int
    trials_completed: int
    trials_pruned: int
    trials_failed: int
    trials_terminal: int
    best_value: float | None = None
    best_trial_number: int | None = None


@dataclass(frozen=True)
class Phase3RunnerSummary:
    run_dir: str
    spec_path: str
    target_trial_epochs: int
    fixed_classical_games: int
    studies: tuple[Phase3StudyRunSummary, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "run_dir": self.run_dir,
            "spec_path": self.spec_path,
            "target_trial_epochs": self.target_trial_epochs,
            "fixed_classical_games": self.fixed_classical_games,
            "studies": [asdict(item) for item in self.studies],
        }


class Phase3HardFailure(RuntimeError):
    """A hard trial failure that should fail one Optuna trial and continue."""


class Phase3TrialRunner(Protocol):
    def run_trial(self, request: Phase3TrialRequest) -> Phase3TrialRunResult:
        """Run one configured Phase 3 child trial."""


@dataclass
class DryRunPhase3TrialRunner:
    """Fast deterministic runner for smoke/resume tests."""

    score_base: float = 0.25

    def run_trial(self, request: Phase3TrialRequest) -> Phase3TrialRunResult:
        checkpoints_dir = request.trial_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint = checkpoints_dir / f"epoch_{request.target_epoch:04d}.pt"
        latest_checkpoint.write_text(
            json.dumps(
                {
                    "candidate_id": request.phase3_candidate_id,
                    "dry_run": True,
                    "params": request.params,
                    "target_epoch": request.target_epoch,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        scorecard_path = request.trial_dir / "scorecards.jsonl"
        append_scorecard(
            scorecard_path,
            _train_scorecard(
                request,
                epoch=request.target_epoch,
                checkpoint_path=latest_checkpoint,
                parent_checkpoint=request.source_checkpoint_path,
                elapsed_s=0.001,
                train_stats={
                    "loss_total": max(0.01, self.score_base),
                    "loss_policy_place": max(0.01, self.score_base),
                    "loss_value": 0.0,
                    "batches_per_sec": 1.0,
                },
                buffer_stats={"size": 32},
            ),
        )
        _append_jsonl(
            request.trial_dir / "events.jsonl",
            {
                "event": "phase3_dry_run_completed",
                "candidate_id": request.phase3_candidate_id,
                "target_epoch": request.target_epoch,
            },
        )
        return Phase3TrialRunResult(
            completed_epochs=request.target_epoch,
            latest_checkpoint_path=str(latest_checkpoint),
            train_scorecard_path=str(scorecard_path),
            event_paths=(str(request.trial_dir / "events.jsonl"),),
        )


@dataclass
class EpochPhase3TrialRunner:
    """Real self-play/training runner for Phase 3 child trials."""

    bootstrap_games: int = 0
    use_selfplay: bool = True
    train: bool = True
    device: Any | None = None

    def run_trial(self, request: Phase3TrialRequest) -> Phase3TrialRunResult:
        import torch

        from hexorl.dashboard.recorder import RunRecorder
        from hexorl.epoch import run_epoch
        from hexorl.train.trainer import Trainer

        if self.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(self.device, torch.device):
            device = self.device
        else:
            device = torch.device(str(self.device))

        model = build_model_from_config(request.config, device=device, inference=False)
        checkpoint = torch.load(request.source_checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        restore_model_weights(model, state, allow_partial=False)
        trainer = Trainer(model, request.config, [], device=device)
        trainer.epoch = int(request.source_epoch)
        trainer.global_step = int(request.source_epoch) * int(request.config.train.batches_per_epoch)

        checkpoints_dir = request.trial_dir / "checkpoints"
        recorder = RunRecorder.for_run_dir(
            request.trial_dir,
            run_id=f"phase3_{request.phase3_candidate_id}",
        )
        latest_checkpoint: Path | None = None
        parent_checkpoint: Path | str = request.source_checkpoint_path
        for expected_epoch in range(request.source_epoch + 1, request.target_epoch + 1):
            result = run_epoch(
                request.config,
                trainer=trainer,
                output_dir=checkpoints_dir,
                bootstrap_games=self.bootstrap_games,
                use_selfplay=self.use_selfplay,
                train=self.train,
                device=device,
                recorder=recorder,
            )
            trainer = result.trainer or trainer
            latest_epoch = int((result.train_stats or {}).get("epoch", expected_epoch))
            latest_checkpoint = result.checkpoint_path or latest_checkpoint
            if latest_checkpoint is None:
                raise Phase3HardFailure("phase3_trial_missing_checkpoint")
            append_scorecard(
                request.trial_dir / "scorecards.jsonl",
                _train_scorecard(
                    request,
                    epoch=latest_epoch,
                    checkpoint_path=latest_checkpoint,
                    parent_checkpoint=parent_checkpoint,
                    elapsed_s=float(result.elapsed_s),
                    train_stats=result.train_stats,
                    buffer_stats=result.buffer_stats,
                ),
            )
            _append_jsonl(
                request.trial_dir / "events.jsonl",
                {
                    "event": "phase3_epoch_completed",
                    "candidate_id": request.phase3_candidate_id,
                    "epoch": latest_epoch,
                    "checkpoint_path": str(latest_checkpoint),
                    "elapsed_s": float(result.elapsed_s),
                },
            )
            parent_checkpoint = latest_checkpoint

        if latest_checkpoint is None:
            raise Phase3HardFailure("phase3_trial_no_epochs_completed")
        return Phase3TrialRunResult(
            completed_epochs=int(request.target_epoch),
            latest_checkpoint_path=str(latest_checkpoint),
            train_scorecard_path=str(request.trial_dir / "scorecards.jsonl"),
            event_paths=(str(request.trial_dir / "events.jsonl"),),
        )


class Phase3OptunaTpeRunner:
    """Run promoted Phase 3 TPE studies from saved Hexo evidence."""

    def __init__(
        self,
        *,
        run_dir: Path | str,
        spec_path: Path | str,
        trial_runner: Phase3TrialRunner | None = None,
        n_trials_per_study: int = 1,
        trial_epochs: int = 2,
        fixed_eval_settings: FixedClassicalEvalSettings | None = None,
        fixed_eval_game_runner: Callable[[int, int], Any] | None = None,
        max_studies: int | None = None,
        summary_path: Path | str | None = None,
    ) -> None:
        if int(n_trials_per_study) <= 0:
            raise ValueError("n_trials_per_study must be positive")
        if int(trial_epochs) <= 0:
            raise ValueError("trial_epochs must be positive")
        self.run_dir = Path(run_dir)
        self.spec_path = Path(spec_path)
        self.trial_runner = trial_runner or EpochPhase3TrialRunner()
        self.n_trials_per_study = int(n_trials_per_study)
        self.trial_epochs = int(trial_epochs)
        self.fixed_eval_settings = fixed_eval_settings or FixedClassicalEvalSettings(games_per_candidate=20)
        self.fixed_eval_game_runner = fixed_eval_game_runner
        self.max_studies = max_studies
        self.summary_path = Path(summary_path) if summary_path is not None else None

    def run(self) -> Phase3RunnerSummary:
        specs = load_phase3_study_specs(self.spec_path)
        if self.max_studies is not None:
            specs = specs[: int(self.max_studies)]
        summaries: list[Phase3StudyRunSummary] = []
        for spec in specs:
            summaries.append(self._run_study(spec))
        summary = Phase3RunnerSummary(
            run_dir=str(self.run_dir),
            spec_path=str(self.spec_path),
            target_trial_epochs=self.trial_epochs,
            fixed_classical_games=int(self.fixed_eval_settings.games_per_candidate),
            studies=tuple(summaries),
        )
        if self.summary_path is not None:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(
                json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return summary

    def _run_study(self, spec: Phase3StudySpec) -> Phase3StudyRunSummary:
        optuna = _import_optuna()
        study = create_phase3_study(
            architecture_id=spec.architecture_id,
            pair_mode=spec.pair_mode,
            storage=spec.storage,
            seed=_seed_from_spec(spec),
            load_if_exists=True,
        )
        existing_trials = len(study.get_trials(deepcopy=False))
        remaining = max(0, self.n_trials_per_study - existing_trials)
        started = 0
        if remaining:
            study.optimize(
                lambda trial: self._objective(spec, study, trial),
                n_trials=remaining,
                catch=(Phase3HardFailure,),
                gc_after_trial=True,
            )
            started = remaining
        trials = study.get_trials(deepcopy=False)
        completed = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE)
        pruned = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.PRUNED)
        failed = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.FAIL)
        best_value: float | None = None
        best_trial_number: int | None = None
        if completed:
            best = study.best_trial
            best_value = float(best.value) if best.value is not None else None
            best_trial_number = int(best.number)
        return Phase3StudyRunSummary(
            study_name=spec.study_name,
            storage=spec.storage,
            promoted_candidate_id=str(spec.metadata.get("promoted_candidate_id", "")),
            trials_requested=self.n_trials_per_study,
            trials_existing=existing_trials,
            trials_started=started,
            trials_completed=completed,
            trials_pruned=pruned,
            trials_failed=failed,
            trials_terminal=completed + pruned + failed,
            best_value=best_value,
            best_trial_number=best_trial_number,
        )

    def _objective(self, spec: Phase3StudySpec, study: Any, trial: Any) -> float:
        promoted_candidate_id = str(spec.metadata.get("promoted_candidate_id", ""))
        if not promoted_candidate_id:
            raise Phase3HardFailure("phase3_spec_missing_promoted_candidate_id")
        source_candidate_dir = self.run_dir / "candidates" / promoted_candidate_id
        source_scorecard_path = Path(str(spec.metadata.get("phase2_scorecard_path", "")))
        if not source_scorecard_path.exists():
            source_scorecard_path = source_candidate_dir / "scorecards.jsonl"
        source_record = _latest_fixed_classical_scorecard(source_scorecard_path)
        source_epoch = max(source_record.completed_epochs, source_record.epoch)
        source_checkpoint_path = _checkpoint_path_for_record(
            source_record,
            source_candidate_dir=source_candidate_dir,
            run_dir=self.run_dir,
        )
        params = suggest_phase3_params(trial, spec)
        base_config = Config.model_validate(
            json.loads((source_candidate_dir / "full_config.json").read_text(encoding="utf-8"))
        )
        trial_config = apply_phase3_params(base_config, params, pair_mode=spec.pair_mode)
        _validate_child_identity(base_config, trial_config, spec)

        phase3_candidate_id = f"{promoted_candidate_id}__phase3_t{int(trial.number):04d}"
        trial_dir = self.run_dir / PHASE3_TRIALS_DIRNAME / phase3_candidate_id
        target_epoch = source_epoch + self.trial_epochs
        request = Phase3TrialRequest(
            spec=spec,
            promoted_candidate_id=promoted_candidate_id,
            phase3_candidate_id=phase3_candidate_id,
            trial_number=int(trial.number),
            trial_dir=trial_dir,
            source_candidate_dir=source_candidate_dir,
            source_checkpoint_path=source_checkpoint_path,
            source_scorecard_path=source_scorecard_path,
            config=trial_config,
            params=params,
            source_epoch=source_epoch,
            target_epoch=target_epoch,
        )
        _write_trial_manifest(request, trial)
        trial.set_user_attr("hexo_status", "running")
        trial.set_user_attr("hexo_phase", "phase3_per_family_tuning")
        trial.set_user_attr("hexo_promoted_candidate_id", promoted_candidate_id)
        trial.set_user_attr("hexo_phase3_candidate_id", phase3_candidate_id)
        trial.set_user_attr("hexo_trial_dir", str(trial_dir))
        trial.set_user_attr("hexo_source_checkpoint_path", str(source_checkpoint_path))
        trial.set_user_attr("hexo_target_epoch", target_epoch)

        try:
            run_result = self.trial_runner.run_trial(request)
            eval_result = evaluate_candidate_fixed_classical(
                trial_dir,
                run_dir=self.run_dir,
                settings=self.fixed_eval_settings,
                game_runner=self.fixed_eval_game_runner,
            )
        except Exception as exc:
            _write_debug_failure_bundle(trial_dir, exc)
            trial.set_user_attr("hexo_status", "failed")
            trial.set_user_attr("hexo_failure_reason", f"{type(exc).__name__}:{exc}")
            raise Phase3HardFailure(f"phase3_trial_failed:{type(exc).__name__}:{exc}") from exc

        if eval_result.scalar_score is None:
            trial.set_user_attr("hexo_status", "failed")
            trial.set_user_attr("hexo_failure_reason", eval_result.reason or "fixed_classical_missing_score")
            raise Phase3HardFailure(eval_result.reason or "fixed_classical_missing_score")
        latest_scorecard = _latest_fixed_classical_scorecard(Path(eval_result.scorecard_path))
        mark_trial_hexo_artifacts(
            trial,
            scorecard_path=eval_result.scorecard_path,
            checkpoint_path=eval_result.checkpoint_path or run_result.latest_checkpoint_path,
            evidence_paths=latest_scorecard.evidence_paths,
            hard_gates=latest_scorecard.hard_gates,
            debug_bundle_paths=run_result.debug_bundle_paths,
            extra_attrs={
                "hexo_status": "completed" if latest_scorecard.hard_pass else "hard_failed",
                "hexo_phase3_candidate_id": phase3_candidate_id,
                "hexo_promoted_candidate_id": promoted_candidate_id,
                "hexo_trial_dir": str(trial_dir),
                "hexo_completed_epochs": int(run_result.completed_epochs),
                "hexo_fixed_classical_games": int(
                    latest_scorecard.component_metrics.get("classical_survival_games", 0.0)
                ),
                "hexo_phase3_params": params,
                "hexo_phase2_classical_survival_lcb": float(
                    spec.metadata.get("phase2_classical_survival_lcb", 0.0) or 0.0
                ),
            },
        )
        score = float(latest_scorecard.scalar_score)
        trial.report(score, step=int(latest_scorecard.completed_epochs))
        if not latest_scorecard.hard_pass:
            raise Phase3HardFailure("hexo_hard_gate_failed")
        if trial.should_prune():
            raise _import_optuna().TrialPruned("phase3_pruned_after_scorecard_floor")
        _append_jsonl(
            trial_dir / "events.jsonl",
            {
                "event": "phase3_trial_completed",
                "candidate_id": phase3_candidate_id,
                "promoted_candidate_id": promoted_candidate_id,
                "score": score,
                "scorecard_path": eval_result.scorecard_path,
                "evidence_path": eval_result.evidence_path,
            },
        )
        return score


def load_phase3_study_specs(path: Path | str) -> tuple[Phase3StudySpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("phase3 study specs must be a JSON list")
    return tuple(Phase3StudySpec(**item) for item in payload)


def suggest_phase3_params(trial: Any, spec: Phase3StudySpec) -> dict[str, float]:
    knobs = set(spec.search_scope.get("knobs", ()))
    params: dict[str, float] = {}
    if "lr_multiplier" in knobs:
        params["lr_multiplier"] = float(trial.suggest_float("lr_multiplier", 0.5, 1.5, log=True))
    if "weight_decay" in knobs:
        params["weight_decay"] = float(trial.suggest_float("weight_decay", 3e-5, 3e-4, log=True))
    if "c_puct" in knobs:
        params["c_puct"] = float(trial.suggest_float("c_puct", 1.0, 2.25))
    if "c_puct_init" in knobs:
        params["c_puct_init"] = float(trial.suggest_float("c_puct_init", 8_000.0, 30_000.0, log=True))
    if "dirichlet_fraction" in knobs:
        params["dirichlet_fraction"] = float(trial.suggest_float("dirichlet_fraction", 0.10, 0.35))
    if "scaled_alpha_total" in knobs:
        params["scaled_alpha_total"] = float(trial.suggest_float("scaled_alpha_total", 0.12, 0.80, log=True))
    if "pcr_low_sim_prob" in knobs:
        params["pcr_low_sim_prob"] = float(trial.suggest_float("pcr_low_sim_prob", 0.50, 0.90))
    if "recency_decay" in knobs:
        params["recency_decay"] = float(trial.suggest_float("recency_decay", 0.96, 0.995))
    if "value_loss_weight" in knobs:
        params["value_loss_weight"] = float(trial.suggest_float("value_loss_weight", 0.50, 1.50))
    if "auxiliary_loss_weight" in knobs:
        params["auxiliary_loss_weight"] = float(trial.suggest_float("auxiliary_loss_weight", 0.01, 0.15, log=True))
    if "pair_loss_weight" in knobs:
        params["pair_loss_weight"] = float(trial.suggest_float("pair_loss_weight", 0.02, 0.20, log=True))
    if "pair_prior_mix" in knobs:
        params["pair_prior_mix"] = float(trial.suggest_float("pair_prior_mix", 0.15, 0.55))
    return params


def apply_phase3_params(config: Config, params: Mapping[str, float], *, pair_mode: str) -> Config:
    data = config.model_copy(deep=True).model_dump(mode="json")
    train = data.setdefault("train", {})
    selfplay = data.setdefault("selfplay", {})
    buffer = data.setdefault("buffer", {})
    model = data.setdefault("model", {})

    if "lr_multiplier" in params:
        train["peak_lr"] = float(train.get("peak_lr", 0.0)) * float(params["lr_multiplier"])
    if "weight_decay" in params:
        train["weight_decay"] = float(params["weight_decay"])
    if "c_puct" in params:
        selfplay["c_puct"] = float(params["c_puct"])
    if "c_puct_init" in params:
        selfplay["c_puct_init"] = float(params["c_puct_init"])
    if "dirichlet_fraction" in params:
        selfplay["dirichlet_fraction"] = float(params["dirichlet_fraction"])
    if "scaled_alpha_total" in params:
        selfplay["dirichlet_alpha"] = float(params["scaled_alpha_total"])
    if "pcr_low_sim_prob" in params:
        selfplay["pcr_low_sim_prob"] = float(params["pcr_low_sim_prob"])
    if "recency_decay" in params:
        buffer["recency_decay"] = float(params["recency_decay"])

    loss_weights = dict(train.get("loss_weights", {}))
    if "value_loss_weight" in params:
        loss_weights["value"] = float(params["value_loss_weight"])
    if "auxiliary_loss_weight" in params:
        auxiliary = float(params["auxiliary_loss_weight"])
        for key in ("tactical", "legal_token_quality"):
            if key in loss_weights:
                loss_weights[key] = auxiliary
    if pair_mode != "none" and "pair_loss_weight" in params:
        pair_weight = float(params["pair_loss_weight"])
        for head in _PAIR_HEADS:
            if head in loss_weights:
                loss_weights[head] = pair_weight
    if pair_mode != "none" and "pair_prior_mix" in params:
        model["pair_prior_mix"] = float(params["pair_prior_mix"])
    train["loss_weights"] = loss_weights
    return Config.model_validate(data)


def phase3_scorecard_paths_for_run(run_dir: Path | str) -> tuple[Path, ...]:
    root = Path(run_dir) / PHASE3_TRIALS_DIRNAME
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.glob("*/scorecards.jsonl") if path.exists() and path.stat().st_size > 0))


def rerank_phase3_trials(run_dir: Path | str, *, min_epoch_floor: int = 12) -> Any:
    paths = phase3_scorecard_paths_for_run(run_dir)
    return build_phase2_promotion_report_from_scorecard_files(paths, min_epoch_floor=min_epoch_floor)


def _train_scorecard(
    request: Phase3TrialRequest,
    *,
    epoch: int,
    checkpoint_path: Path,
    parent_checkpoint: Path | str,
    elapsed_s: float,
    train_stats: Mapping[str, Any],
    buffer_stats: Mapping[str, Any],
) -> ScorecardRecord:
    metrics = {
        PHASE3_TARGET_SCALAR: 0.0,
        "epoch_seconds": float(elapsed_s),
        "buffer_size": _finite_float(buffer_stats.get("size", 0.0)),
        "selfplay_games_per_epoch": float(request.config.selfplay.games_per_epoch),
        "selfplay_states_per_epoch": float(request.config.selfplay.states_per_epoch),
        "train_batches_per_epoch": float(request.config.train.batches_per_epoch),
        "train_loss": _train_stat_float(train_stats, "loss_total", "loss", "total"),
        "loss_total": _train_stat_float(train_stats, "loss_total", "loss", "total"),
        "loss_policy_place": _train_stat_float(train_stats, "loss_policy_place", "policy_place", "policy"),
        "loss_value": _train_stat_float(train_stats, "loss_value", "value"),
        "pair_policy_weight_mean": _train_stat_float(train_stats, "pair_policy_weight_mean"),
        "batches_per_sec": _train_stat_float(train_stats, "batches_per_sec"),
        "graph_peak_cuda_allocated_mb": _train_stat_float(train_stats, "graph_peak_cuda_allocated_mb"),
        "graph_microbatch_oom_retries": _train_stat_float(train_stats, "graph_microbatch_oom_retries"),
        "graph_microbatch_nonfinite_retries": _train_stat_float(train_stats, "graph_microbatch_nonfinite_retries"),
    }
    return ScorecardRecord(
        candidate_id=request.phase3_candidate_id,
        scalar_name=PHASE3_TARGET_SCALAR,
        scalar_score=0.0,
        component_metrics=metrics,
        hard_gates={"hard_pass": True, "failures": []},
        config_hash="",
        checkpoint_lineage={
            "checkpoint_path": str(checkpoint_path),
            "parent_checkpoint": str(parent_checkpoint),
            "source_checkpoint_path": str(request.source_checkpoint_path),
        },
        evidence_paths=(
            str(request.trial_dir / "events.jsonl"),
            str(checkpoint_path),
            str(request.trial_dir / "dashboard.sqlite3"),
        ),
        epoch=int(epoch),
        completed_epochs=int(epoch),
        status="healthy",
        metadata={
            "run_id": request.trial_dir.parent.parent.name,
            "phase": "phase3_per_family_tuning",
            "study_name": request.spec.study_name,
            "trial_number": int(request.trial_number),
            "promoted_candidate_id": request.promoted_candidate_id,
            "phase3_params": dict(request.params),
            "score_source": "phase3_train_row_pending_fixed_classical",
        },
    )


def _latest_fixed_classical_scorecard(scorecard_path: Path | str) -> ScorecardRecord:
    records = [
        record
        for record in read_scorecards(scorecard_path)
        if record.hard_pass
        and record.scalar_name == PHASE3_TARGET_SCALAR
        and float(record.component_metrics.get("classical_survival_games", 0.0) or 0.0) > 0.0
        and record.evidence_paths
    ]
    if not records:
        raise Phase3HardFailure(f"no fixed-classical scorecard evidence in {scorecard_path}")
    return max(records, key=lambda record: (max(record.completed_epochs, record.epoch), record.created_at))


def _checkpoint_path_for_record(record: ScorecardRecord, *, source_candidate_dir: Path, run_dir: Path) -> Path:
    value = record.checkpoint_lineage.get("checkpoint_path")
    if not value:
        value = record.metadata.get("extra_fields", {}).get("checkpoint_path")
    if not value:
        epoch = max(record.completed_epochs, record.epoch)
        value = source_candidate_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
    path = Path(str(value))
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, run_dir.parent.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise Phase3HardFailure(f"source checkpoint path is missing: {value}")


def _write_trial_manifest(request: Phase3TrialRequest, trial: Any) -> None:
    request.trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = request.trial_dir / "full_config.json"
    config_path.write_text(
        json.dumps(request.config.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_config = request.source_candidate_dir / "full_config.json"
    if source_config.exists():
        shutil.copyfile(source_config, request.trial_dir / "source_full_config.json")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": request.phase3_candidate_id,
        "promoted_candidate_id": request.promoted_candidate_id,
        "architecture_id": request.spec.architecture_id,
        "pair_mode": request.spec.pair_mode,
        "study_name": request.spec.study_name,
        "trial_number": int(request.trial_number),
        "optuna_params": dict(request.params),
        "source_candidate_dir": str(request.source_candidate_dir),
        "source_checkpoint_path": str(request.source_checkpoint_path),
        "source_scorecard_path": str(request.source_scorecard_path),
        "source_epoch": int(request.source_epoch),
        "target_epoch": int(request.target_epoch),
        "optuna_distributions": {
            key: str(value) for key, value in getattr(trial, "distributions", {}).items()
        },
    }
    (request.trial_dir / "trial_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_debug_failure_bundle(trial_dir: Path, exc: Exception) -> None:
    bundle = trial_dir / "debug_bundles" / f"phase3_failure_{int(time.time())}"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "failure.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "type": type(exc).__name__,
                "message": str(exc),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_child_identity(base: Config, child: Config, spec: Phase3StudySpec) -> None:
    if base.model.architecture != child.model.architecture:
        raise Phase3HardFailure("phase3_child_changed_architecture")
    if base.model.pair_strategy != child.model.pair_strategy:
        raise Phase3HardFailure("phase3_child_changed_pair_strategy")
    if child.model.architecture != spec.architecture_id:
        raise Phase3HardFailure("phase3_spec_architecture_mismatch")
    if child.model.pair_strategy != spec.pair_mode:
        raise Phase3HardFailure("phase3_spec_pair_mode_mismatch")
    if list(base.model.heads) != list(child.model.heads):
        raise Phase3HardFailure("phase3_child_changed_model_heads")


def _seed_from_spec(spec: Phase3StudySpec) -> int | None:
    seed = spec.sampler.get("seed")
    return int(seed) if seed is not None else None


def _train_stat_float(values: Mapping[str, Any] | None, *keys: str) -> float:
    if not values:
        return 0.0
    for key in keys:
        if key not in values:
            continue
        result = _finite_float(values.get(key))
        if math.isfinite(result):
            return result
    return 0.0


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _import_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("Phase 3 Optuna runner requires optuna to be installed") from exc
    return optuna
