"""Queued/resumable Optuna controller for the Phase 0/1 architecture scout."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

try:
    import optuna as _OPTUNA
    from optuna.trial import TrialState
except ModuleNotFoundError:
    _OPTUNA = None
    TrialState = None  # type: ignore[assignment]

from hexorl.autotune import CandidateArtifactPaths, CandidateArtifactWriter, CandidateRecipe, config_hash
from hexorl.autotune.recipes import candidate_recipes_from_config
from hexorl.config import Config
from hexorl.tuning.runtime_probe import (
    RuntimeCalibrationCache,
    RuntimeKnobs,
    RuntimeProbe,
    apply_runtime_knobs,
    identity_from_config,
)


PHASE1_STUDY_NAME = "study_architecture_scout_v1"
PHASE1_TARGET_SCALAR = "classical_survival_lcb"
TERMINAL_STATUSES = {"completed", "quarantined", "failed", "metric_pruned"}


@dataclass(frozen=True)
class ScoutQuantumRequest:
    candidate: CandidateRecipe
    config: Config
    paths: CandidateArtifactPaths
    trial_number: int
    start_epoch: int
    end_epoch: int
    run_id: str
    study_name: str


@dataclass(frozen=True)
class ScoutQuantumResult:
    completed_epochs: int
    scorecards: tuple[dict[str, Any], ...] = ()
    latest_checkpoint_path: str | None = None
    latest_scorecard_path: str | None = None
    events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ScoutRunSummary:
    study_name: str
    storage: str
    run_dir: str
    completed: bool
    candidate_statuses: dict[str, dict[str, Any]]
    quanta_executed: int


class ScoutHardFailure(RuntimeError):
    """A Hexo hard sentinel failure that quarantines one candidate."""

    def __init__(
        self,
        reason: str,
        *,
        debug_bundle_path: str | Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.debug_bundle_path = str(debug_bundle_path) if debug_bundle_path is not None else None
        self.details = dict(details or {})


class ScoutEpochRunner(Protocol):
    def run_quantum(self, request: ScoutQuantumRequest) -> ScoutQuantumResult:
        """Run a candidate for ``request.start_epoch..request.end_epoch``."""


@dataclass
class DryRunScoutEpochRunner:
    """Deterministic smoke runner for controller tests and CPU-only dry runs."""

    fail_candidates: set[str] = field(default_factory=set)
    score_base: float = 10.0

    def run_quantum(self, request: ScoutQuantumRequest) -> ScoutQuantumResult:
        if request.candidate.candidate_id in self.fail_candidates:
            bundle = request.paths.debug_bundles_dir / f"epoch_{request.start_epoch}_hard_failure"
            bundle.mkdir(parents=True, exist_ok=True)
            (bundle / "runtime_telemetry.json").write_text(
                json.dumps(
                    {
                        "candidate_id": request.candidate.candidate_id,
                        "failed_epoch": request.start_epoch,
                        "reason": "simulated_hard_failure",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise ScoutHardFailure(
                "simulated_hard_failure",
                debug_bundle_path=bundle,
                details={"failed_epoch": request.start_epoch},
            )

        scorecards: list[dict[str, Any]] = []
        checkpoint_path: Path | None = None
        for epoch in range(request.start_epoch, request.end_epoch + 1):
            checkpoint_path = request.paths.checkpoints_dir / f"epoch_{epoch:04d}.ckpt"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "candidate_id": request.candidate.candidate_id,
                        "epoch": epoch,
                        "dry_run": True,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            scorecards.append(
                {
                    "candidate_id": request.candidate.candidate_id,
                    "epoch": epoch,
                    PHASE1_TARGET_SCALAR: self.score_base + float(epoch),
                    "checkpoint_path": str(checkpoint_path),
                    "dry_run": True,
                    "hard_failures": 0,
                }
            )
        latest_checkpoint = str(checkpoint_path) if checkpoint_path is not None else None
        return ScoutQuantumResult(
            completed_epochs=request.end_epoch,
            scorecards=tuple(scorecards),
            latest_checkpoint_path=latest_checkpoint,
            latest_scorecard_path=str(request.paths.scorecards_jsonl),
            events=(
                {
                    "event": "dry_run_quantum_completed",
                    "start_epoch": request.start_epoch,
                    "end_epoch": request.end_epoch,
                },
            ),
        )


@dataclass
class EpochScoutEpochRunner:
    """Production scout runner that advances candidates through real epochs."""

    bootstrap_games: int = 0
    use_selfplay: bool = True
    train: bool = True
    device: Any | None = None
    _trainers: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def run_quantum(self, request: ScoutQuantumRequest) -> ScoutQuantumResult:
        trainer = self._trainers.get(request.candidate.candidate_id)
        if trainer is None and request.start_epoch > 1:
            checkpoint = _latest_checkpoint_before(request.paths.checkpoints_dir, request.start_epoch)
            if checkpoint is None:
                raise RuntimeError(
                    f"cannot resume {request.candidate.candidate_id} at epoch {request.start_epoch}: "
                    "previous checkpoint is missing"
                )
            trainer = _load_trainer_from_checkpoint(request.config, checkpoint, self.device)

        scorecards: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        latest_checkpoint_path: str | None = None
        latest_epoch = request.start_epoch - 1
        for expected_epoch in range(request.start_epoch, request.end_epoch + 1):
            result = _run_hexo_epoch(
                request.config,
                trainer=trainer,
                output_dir=request.paths.checkpoints_dir,
                recorder_run_dir=request.paths.candidate_dir,
                recorder_run_id=f"{request.run_id}_{request.candidate.candidate_id}",
                bootstrap_games=self.bootstrap_games,
                use_selfplay=self.use_selfplay,
                train=self.train,
                device=self.device,
            )
            trainer = result.trainer
            if trainer is not None:
                self._trainers[request.candidate.candidate_id] = trainer
            latest_epoch = int(result.train_stats.get("epoch", expected_epoch)) if result.train_stats else expected_epoch
            latest_checkpoint_path = str(result.checkpoint_path) if result.checkpoint_path else latest_checkpoint_path
            scalar_score = _epoch_result_phase1_scalar(result)
            scorecards.append(
                {
                    "candidate_id": request.candidate.candidate_id,
                    "epoch": latest_epoch,
                    "completed_epochs": latest_epoch,
                    "scalar_name": PHASE1_TARGET_SCALAR,
                    "scalar_score": scalar_score,
                    PHASE1_TARGET_SCALAR: scalar_score,
                    "component_metrics": {
                        PHASE1_TARGET_SCALAR: scalar_score,
                        "epoch_seconds": float(result.elapsed_s),
                        "buffer_size": float(result.buffer_stats.get("size", 0.0) or 0.0),
                        "selfplay_games_per_epoch": float(request.config.selfplay.games_per_epoch),
                        "selfplay_states_per_epoch": float(request.config.selfplay.states_per_epoch),
                        "selfplay_max_game_moves": float(request.config.selfplay.max_game_moves),
                        "selfplay_mcts_simulations": float(request.config.selfplay.mcts_simulations),
                        "selfplay_pcr_low_sims": float(request.config.selfplay.pcr_low_sims),
                        "train_batches_per_epoch": float(request.config.train.batches_per_epoch),
                        "train_loss": _train_stat_float(result.train_stats, "loss_total", "loss"),
                        "loss_total": _train_stat_float(result.train_stats, "loss_total", "loss"),
                        "loss_policy_place": _train_stat_float(result.train_stats, "loss_policy_place"),
                        "loss_value": _train_stat_float(result.train_stats, "loss_value"),
                        "value_weight_mean": _train_stat_float(result.train_stats, "value_weight_mean"),
                        "value_weight_zero_frac": _train_stat_float(result.train_stats, "value_weight_zero_frac"),
                        "value_effective_samples": _value_effective_samples(
                            request.config, result.train_stats
                        ),
                        "pair_policy_weight_mean": _train_stat_float(result.train_stats, "pair_policy_weight_mean"),
                        "batches_per_sec": _train_stat_float(result.train_stats, "batches_per_sec"),
                        "graph_peak_cuda_allocated_mb": _train_stat_float(
                            result.train_stats, "graph_peak_cuda_allocated_mb"
                        ),
                        "graph_microbatch_oom_retries": _train_stat_float(
                            result.train_stats, "graph_microbatch_oom_retries"
                        ),
                        "graph_microbatch_nonfinite_retries": _train_stat_float(
                            result.train_stats, "graph_microbatch_nonfinite_retries"
                        ),
                        "avg_missing_target_policy_mass": _train_stat_float(
                            result.buffer_stats, "avg_missing_target_policy_mass"
                        ),
                        "avg_target_policy_mass_outside_window": _train_stat_float(
                            result.buffer_stats, "avg_target_policy_mass_outside_window"
                        ),
                        "avg_candidate_recall_mcts_top1": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_mcts_top1"
                        ),
                        "avg_candidate_recall_mcts_top4": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_mcts_top4"
                        ),
                        "avg_candidate_recall_mcts_top8": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_mcts_top8"
                        ),
                        "avg_candidate_recall_winning_move": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_winning_move"
                        ),
                        "avg_candidate_recall_forced_block": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_forced_block"
                        ),
                        "avg_candidate_recall_two_placement_cover": _train_stat_float(
                            result.buffer_stats, "avg_candidate_recall_two_placement_cover"
                        ),
                        "critical_overflow_count": _train_stat_float(result.buffer_stats, "critical_overflow_count"),
                        "selfplay_games_done": _train_stat_float(result.buffer_stats, "games_done"),
                        "selfplay_positions_done": _train_stat_float(result.buffer_stats, "positions_done"),
                        "truncated_games": _train_stat_float(result.buffer_stats, "truncated_games"),
                        "truncation_rate": _train_stat_float(result.buffer_stats, "truncation_rate"),
                        "terminal_reason_max_game_moves": _train_stat_float(
                            result.buffer_stats, "terminal_reason_max_game_moves"
                        ),
                        "pair_prior_hit_frac": _pair_buffer_stat_float(
                            request.config, result.buffer_stats, "pair_prior_hit_frac"
                        ),
                        "pair_fallback_prior_use": _pair_buffer_stat_float(
                            request.config, result.buffer_stats, "pair_fallback_prior_use"
                        ),
                        "pair_fallback_prior_use_on_mcts_top1": _pair_buffer_stat_float(
                            request.config, result.buffer_stats, "pair_fallback_prior_use_on_mcts_top1"
                        ),
                        "pair_fallback_prior_use_on_mcts_top4": _pair_buffer_stat_float(
                            request.config, result.buffer_stats, "pair_fallback_prior_use_on_mcts_top4"
                        ),
                        "pair_fallback_prior_use_on_mcts_top8": _pair_buffer_stat_float(
                            request.config, result.buffer_stats, "pair_fallback_prior_use_on_mcts_top8"
                        ),
                    },
                    "hard_gates": {"hard_pass": True, "failures": []},
                    "checkpoint_path": latest_checkpoint_path,
                    "checkpoint_lineage": {
                        "checkpoint_path": latest_checkpoint_path,
                        "parent_checkpoint": str(_latest_checkpoint_before(request.paths.checkpoints_dir, expected_epoch))
                        if expected_epoch > 1
                        else "",
                    },
                    "evidence_paths": [
                        str(request.paths.events_jsonl),
                        *( [latest_checkpoint_path] if latest_checkpoint_path else [] ),
                    ],
                    "status": "healthy",
                    "metadata": {
                        "score_source": "epoch_result_classical_survival_lcb_or_zero",
                        "run_id": request.run_id,
                        "study_name": request.study_name,
                        "trial_number": request.trial_number,
                    },
                }
            )
            events.extend(
                _training_signal_warning_events(
                    request.candidate.candidate_id,
                    latest_epoch,
                    scorecards[-1]["component_metrics"],
                )
            )
            events.append(
                {
                    "event": "epoch_runner_completed",
                    "epoch": latest_epoch,
                    "expected_epoch": expected_epoch,
                    "checkpoint_path": latest_checkpoint_path,
                    "elapsed_s": float(result.elapsed_s),
                }
            )
        return ScoutQuantumResult(
            completed_epochs=latest_epoch,
            scorecards=tuple(scorecards),
            latest_checkpoint_path=latest_checkpoint_path,
            latest_scorecard_path=str(request.paths.scorecards_jsonl),
            events=tuple(events),
        )


class Phase1OptunaScoutController:
    """Durable Phase 0/1 scout ledger and round-robin executor."""

    def __init__(
        self,
        *,
        runs_root: str | Path,
        run_id: str,
        base_config: Config | None = None,
        candidates: tuple[CandidateRecipe, ...] | None = None,
        storage: str | None = None,
        study_name: str = PHASE1_STUDY_NAME,
        runner: ScoutEpochRunner | None = None,
        seed: int | None = None,
        min_epochs: int | None = None,
        quantum_epochs: int | None = None,
        runtime_probe_runner: Callable[[RuntimeKnobs, int], Any] | None = None,
        runtime_probe_candidates: tuple[RuntimeKnobs, ...] | None = None,
        runtime_probe_cache_path: str | Path | None = None,
        runtime_probe_host_profile: dict[str, Any] | None = None,
        runtime_probe_code_hash: str = "unknown",
    ) -> None:
        self.runs_root = Path(runs_root)
        self.run_id = str(run_id)
        self.base_config = base_config or Config()
        self.candidates = candidates or candidate_recipes_from_config(self.base_config)
        if not self.candidates:
            raise ValueError("Phase 1 scout requires at least one candidate")
        self.study_name = study_name
        self.storage = scout_storage_url(self.runs_root, self.run_id, storage)
        self.runner = runner or DryRunScoutEpochRunner()
        self.seed = seed
        self.min_epochs = int(min_epochs or self.base_config.autotune.scout.min_epochs)
        self.quantum_epochs = int(quantum_epochs or self.base_config.autotune.scout.schedule_quantum_epochs)
        self.runtime_probe_runner = runtime_probe_runner
        self.runtime_probe_candidates = runtime_probe_candidates
        self.runtime_probe_cache_path = Path(runtime_probe_cache_path) if runtime_probe_cache_path is not None else None
        self.runtime_probe_host_profile = dict(runtime_probe_host_profile or {})
        self.runtime_probe_code_hash = str(runtime_probe_code_hash)
        if self.min_epochs <= 0:
            raise ValueError("min_epochs must be positive")
        if self.quantum_epochs <= 0:
            raise ValueError("quantum_epochs must be positive")
        self.writer = CandidateArtifactWriter(self.runs_root, self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._configs = {
            candidate.candidate_id: _scout_epoch_budget_config(candidate.materialize_config(self.base_config))
            for candidate in self.candidates
        }
        self._candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(self._candidate_ids)) != len(self._candidate_ids):
            raise ValueError("candidate ids must be unique")

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.run_id

    def create_or_resume_study(self) -> Any:
        optuna = _import_optuna()
        _ensure_sqlite_parent(self.storage)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.NopPruner(),
            load_if_exists=True,
        )
        self._enqueue_candidate_plan_once(study)
        self._ensure_candidate_artifacts_and_attrs(study)
        self._validate_resume_lineage(study)
        self._finalize_completed_running_trials(study)
        self._write_study_manifest(study)
        return study

    def run(self, *, max_quanta: int | None = None) -> ScoutRunSummary:
        study = self.create_or_resume_study()
        quanta_executed = 0
        while True:
            progressed = False
            for index in self._round_robin_indices(study):
                if max_quanta is not None and quanta_executed >= max_quanta:
                    return self._summary(study, quanta_executed)
                candidate = self.candidates[index]
                if not self._candidate_needs_quantum(study, candidate):
                    continue
                self._run_candidate_quantum(study, index, candidate)
                quanta_executed += 1
                progressed = True
            if not progressed:
                break
        return self._summary(study, quanta_executed)

    def _enqueue_candidate_plan_once(self, study: Any) -> None:
        plan_hash = _candidate_plan_hash(self.candidates)
        existing_hash = study.user_attrs.get("candidate_plan_hash")
        if existing_hash is not None:
            if existing_hash != plan_hash:
                existing_plan = [str(candidate_id) for candidate_id in (study.user_attrs.get("candidate_plan") or [])]
                if existing_plan != list(self._candidate_ids):
                    raise RuntimeError("resumed Optuna study has a different Phase 1 candidate plan")
                study.set_user_attr("candidate_plan_hash", plan_hash)
            self._validate_one_trial_per_candidate(study)
            return

        for index, candidate in enumerate(self.candidates):
            cfg = self._configs[candidate.candidate_id]
            paths = self.writer.paths_for(candidate.candidate_id)
            study.enqueue_trial(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_index": index,
                    "architecture_id": candidate.model.architecture_id,
                    "pair_strategy": candidate.pair_strategy.mode,
                },
                user_attrs=self._initial_user_attrs(candidate, cfg, paths),
                skip_if_exists=True,
            )
        study.set_user_attr("hexo_phase", "phase1_architecture_scout")
        study.set_user_attr("candidate_plan_hash", plan_hash)
        study.set_user_attr("candidate_plan", list(self._candidate_ids))
        study.set_user_attr("phase1_pruner", "NopPruner")
        study.set_user_attr("phase1_sampler", "queued_tpe_shell")
        study.set_user_attr("metric_pruning_before_epoch_12", "disabled")
        study.set_user_attr("min_epochs", self.min_epochs)
        study.set_user_attr("schedule_quantum_epochs", self.quantum_epochs)
        study.set_user_attr("next_candidate_index", 0)
        self._validate_one_trial_per_candidate(study)

    def _ensure_candidate_artifacts_and_attrs(self, study: Any) -> None:
        trials = self._candidate_trials(study)
        for candidate in self.candidates:
            cfg = self._configs[candidate.candidate_id]
            trial = trials[candidate.candidate_id]
            selected_payload = trial.user_attrs.get("runtime_probe_selected_knobs")
            if isinstance(selected_payload, dict):
                cfg = cfg.model_copy(deep=True)
                apply_runtime_knobs(cfg, RuntimeKnobs.from_mapping(selected_payload))
                self._configs[candidate.candidate_id] = cfg
            paths = self.writer.paths_for(candidate.candidate_id)
            attrs = self._initial_user_attrs(candidate, cfg, paths)
            existing_hash = trial.user_attrs.get("config_hash")
            next_hash = attrs["config_hash"]
            if existing_hash is not None and existing_hash != next_hash:
                if _completed_epochs(trial) > 0 or trial.user_attrs.get("latest_checkpoint_path"):
                    raise RuntimeError(
                        f"resumed candidate config hash changed for {candidate.candidate_id} "
                        "after checkpoint lineage exists"
                    )
            updated_trial_attrs = dict(trial.user_attrs)
            for key, value in attrs.items():
                should_update = key not in trial.user_attrs or (
                    key in {"config_hash", "full_config"}
                    and existing_hash is not None
                    and existing_hash != next_hash
                    and _completed_epochs(trial) == 0
                    and not trial.user_attrs.get("latest_checkpoint_path")
                )
                if should_update:
                    self._set_trial_user_attr(study, trial, key, value)
                    updated_trial_attrs[key] = value
            paths = self.writer.write_candidate(
                candidate,
                cfg,
                optuna_trial={
                    "number": trial.number,
                    "state": trial.state.name,
                    "params": dict(trial.params),
                    "user_attrs": updated_trial_attrs,
                },
                study_name=self.study_name,
                trial_number=trial.number,
            )

    def _run_candidate_quantum(self, study: Any, candidate_index: int, candidate: CandidateRecipe) -> None:
        trial = self._activate_candidate_trial(study, candidate)
        completed_epochs = _completed_epochs(trial)
        start_epoch = completed_epochs + 1
        end_epoch = min(self.min_epochs, completed_epochs + self.quantum_epochs)
        paths = self.writer.paths_for(candidate.candidate_id)
        cfg = self._configs[candidate.candidate_id]
        self._set_lifecycle_attrs(
            study,
            trial,
            hexo_status="running",
            completed_epochs=completed_epochs,
            quarantine_reason=None,
        )
        self.writer.append_event(
            paths,
            {
                "event": "quantum_started",
                "candidate_id": candidate.candidate_id,
                "trial_number": trial.number,
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
            },
        )
        request = ScoutQuantumRequest(
            candidate=candidate,
            config=cfg,
            paths=paths,
            trial_number=trial.number,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            run_id=self.run_id,
            study_name=self.study_name,
        )
        try:
            self._ensure_runtime_probe(study, trial, candidate, cfg, paths)
            result = self.runner.run_quantum(request)
        except ScoutHardFailure as exc:
            self._quarantine_candidate(study, trial, candidate, paths, completed_epochs, exc)
            self._set_next_candidate_index(study, candidate_index)
            return

        latest_scorecard_path = result.latest_scorecard_path or str(paths.scorecards_jsonl)
        for event in result.events:
            self.writer.append_event(paths, {"candidate_id": candidate.candidate_id, **event})
        for scorecard in result.scorecards:
            self.writer.append_scorecard(paths, scorecard)
            epoch = int(scorecard.get("epoch", result.completed_epochs))
            if PHASE1_TARGET_SCALAR in scorecard:
                self._set_intermediate_value(study, trial, epoch, float(scorecard[PHASE1_TARGET_SCALAR]))

        new_completed = int(result.completed_epochs)
        if new_completed < end_epoch:
            raise RuntimeError(
                f"runner completed epoch {new_completed}, expected at least {end_epoch}; "
                "checkpoint lineage would be ambiguous"
            )
        latest_checkpoint_path = result.latest_checkpoint_path or trial.user_attrs.get("latest_checkpoint_path")
        if latest_checkpoint_path and not Path(str(latest_checkpoint_path)).exists():
            raise RuntimeError(f"latest checkpoint path does not exist: {latest_checkpoint_path}")

        status = "completed" if new_completed >= self.min_epochs else "healthy"
        self._set_lifecycle_attrs(
            study,
            trial,
            hexo_status=status,
            completed_epochs=new_completed,
            quarantine_reason=None,
            latest_checkpoint_path=latest_checkpoint_path,
            latest_scorecard_path=latest_scorecard_path,
        )
        self.writer.append_event(
            paths,
            {
                "event": "quantum_completed",
                "candidate_id": candidate.candidate_id,
                "trial_number": trial.number,
                "completed_epochs": new_completed,
                "hexo_status": status,
            },
        )
        if status == "completed" and trial.state != TrialState.COMPLETE:
            value = _latest_scalar(result.scorecards, trial)
            study.tell(trial.number, value, skip_if_finished=True)
        self._set_next_candidate_index(study, candidate_index)

    def _ensure_runtime_probe(
        self,
        study: Any,
        trial: Any,
        candidate: CandidateRecipe,
        cfg: Config,
        paths: CandidateArtifactPaths,
    ) -> None:
        selected_payload = trial.user_attrs.get("runtime_probe_selected_knobs")
        if isinstance(selected_payload, dict):
            apply_runtime_knobs(cfg, RuntimeKnobs.from_mapping(selected_payload))
            self._set_lifecycle_attrs(
                study,
                trial,
                config_hash=config_hash(cfg),
                full_config=cfg.model_dump(mode="json"),
            )
            return
        if self.runtime_probe_runner is None or not bool(self.base_config.autotune.runtime_probe.enabled):
            return

        cache_path = self.runtime_probe_cache_path or (self.run_dir / "runtime_calibration_cache.json")
        cache = RuntimeCalibrationCache.load(cache_path)
        candidates = self.runtime_probe_candidates or _default_runtime_probe_candidates(cfg)
        identity = identity_from_config(
            candidate_id=candidate.candidate_id,
            config=cfg,
            host_profile=self.runtime_probe_host_profile,
            code_hash=self.runtime_probe_code_hash,
            config_hash=config_hash(cfg),
            architecture_contract_version=candidate.model.architecture_contract_version,
            recipe_schema_version=str(candidate.schema_version),
            optuna_trial_number=trial.number,
            extra={"study_name": self.study_name, "run_id": self.run_id},
        )
        decision = RuntimeProbe(
            identity=identity,
            candidates=candidates,
            runner=self.runtime_probe_runner,
            cache=cache,
            speed_threshold=float(self.base_config.autotune.runtime_probe.speed_quarantine_positions_per_sec),
            debug_bundle_root=paths.debug_bundles_dir,
            repro_command=[
                "python",
                "scripts/run_phase1_optuna_scout.py",
                "--runs-root",
                str(self.runs_root),
                "--run-id",
                self.run_id,
                "--production",
            ],
        ).run()
        if decision.quarantined:
            quarantine = decision.quarantine
            raise ScoutHardFailure(
                quarantine.reason if quarantine is not None else "runtime_probe_speed_quarantine",
                debug_bundle_path=decision.debug_bundle_path,
                details={
                    "runtime_probe_status": "quarantined",
                    "runtime_probe_cache_key": identity.cache_key(),
                    "runtime_probe_results": [result.to_dict() for result in decision.results],
                },
            )
        if decision.selected is None:
            raise ScoutHardFailure(
                "runtime_probe_no_selected_knobs",
                details={"runtime_probe_cache_key": identity.cache_key()},
            )

        apply_runtime_knobs(cfg, decision.selected)
        selected = decision.selected.to_legacy_candidate()
        self._set_lifecycle_attrs(
            study,
            trial,
            config_hash=config_hash(cfg),
            full_config=cfg.model_dump(mode="json"),
            runtime_probe_status="selected",
            runtime_probe_cache_hit=decision.cache_hit,
            runtime_probe_selected_knobs=selected,
            runtime_probe_cache_key=identity.cache_key(),
        )
        trial_attrs = dict(trial.user_attrs)
        trial_attrs.update(
            {
                "config_hash": config_hash(cfg),
                "full_config": cfg.model_dump(mode="json"),
                "runtime_probe_status": "selected",
                "runtime_probe_cache_hit": decision.cache_hit,
                "runtime_probe_selected_knobs": selected,
                "runtime_probe_cache_key": identity.cache_key(),
            }
        )
        self.writer.write_candidate(
            candidate,
            cfg,
            optuna_trial={
                "number": trial.number,
                "state": trial.state.name,
                "params": dict(trial.params),
                "user_attrs": trial_attrs,
            },
            study_name=self.study_name,
            trial_number=trial.number,
        )
        self.writer.append_event(
            paths,
            {
                "event": "runtime_probe_selected",
                "candidate_id": candidate.candidate_id,
                "trial_number": trial.number,
                "cache_hit": decision.cache_hit,
                "selected": selected,
            },
        )

    def _round_robin_indices(self, study: Any) -> list[int]:
        start = int(study.user_attrs.get("next_candidate_index", 0) or 0) % len(self.candidates)
        return list(range(start, len(self.candidates))) + list(range(0, start))

    def _set_next_candidate_index(self, study: Any, candidate_index: int) -> None:
        study.set_user_attr("next_candidate_index", (int(candidate_index) + 1) % len(self.candidates))

    def _quarantine_candidate(
        self,
        study: Any,
        trial: Any,
        candidate: CandidateRecipe,
        paths: CandidateArtifactPaths,
        completed_epochs: int,
        exc: ScoutHardFailure,
    ) -> None:
        quarantine_payload = {
            "candidate_id": candidate.candidate_id,
            "trial_number": trial.number,
            "reason": exc.reason,
            "details": exc.details,
            "debug_bundle_path": exc.debug_bundle_path,
            "completed_epochs": completed_epochs,
            "strength_pruning": False,
        }
        quarantine_path = paths.candidate_dir / "quarantine.json"
        _atomic_write_json(quarantine_path, quarantine_payload)
        self.writer.append_event(paths, {"event": "candidate_quarantined", **quarantine_payload})
        self._set_lifecycle_attrs(
            study,
            trial,
            hexo_status="quarantined",
            completed_epochs=completed_epochs,
            quarantine_reason=exc.reason,
            debug_bundle_path=exc.debug_bundle_path,
            latest_scorecard_path=trial.user_attrs.get("latest_scorecard_path"),
            latest_checkpoint_path=trial.user_attrs.get("latest_checkpoint_path"),
        )
        if trial.state != TrialState.FAIL:
            study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)

    def _activate_candidate_trial(self, study: Any, candidate: CandidateRecipe) -> Any:
        trial = self._candidate_trials(study)[candidate.candidate_id]
        if trial.state == TrialState.WAITING:
            active = study.ask()
            fixed_candidate = active.suggest_categorical("candidate_id", self._candidate_ids)
            active.suggest_int("candidate_index", 0, len(self._candidate_ids) - 1)
            active.suggest_categorical(
                "architecture_id",
                sorted({recipe.model.architecture_id for recipe in self.candidates}),
            )
            active.suggest_categorical(
                "pair_strategy",
                sorted({recipe.pair_strategy.mode for recipe in self.candidates}),
            )
            if fixed_candidate != candidate.candidate_id:
                raise RuntimeError(
                    "Optuna queued trial order no longer matches the Phase 1 round-robin candidate plan"
                )
            trial = self._candidate_trials(study)[candidate.candidate_id]
        if trial.state not in {TrialState.RUNNING, TrialState.WAITING}:
            if trial.state == TrialState.COMPLETE:
                return trial
            raise RuntimeError(f"candidate {candidate.candidate_id} is in non-runnable state {trial.state.name}")
        return trial

    def _candidate_needs_quantum(self, study: Any, candidate: CandidateRecipe) -> bool:
        trial = self._candidate_trials(study)[candidate.candidate_id]
        status = str(trial.user_attrs.get("hexo_status", "pending"))
        if status in TERMINAL_STATUSES or trial.state in {TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED}:
            return False
        return _completed_epochs(trial) < self.min_epochs

    def _validate_one_trial_per_candidate(self, study: Any) -> None:
        trials = self._candidate_trials(study)
        missing = sorted(set(self._candidate_ids) - set(trials))
        extra = sorted(set(trials) - set(self._candidate_ids))
        if missing or extra:
            raise RuntimeError(f"candidate/trial mapping mismatch missing={missing} extra={extra}")

    def _candidate_trials(self, study: Any) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        duplicates: list[str] = []
        for trial in study.get_trials(deepcopy=False):
            candidate_id = trial.user_attrs.get("candidate_id")
            if candidate_id is None:
                continue
            candidate_id = str(candidate_id)
            if candidate_id in mapping:
                duplicates.append(candidate_id)
            mapping[candidate_id] = trial
        if duplicates:
            raise RuntimeError(f"multiple Optuna trials map to the same candidate ids: {sorted(duplicates)}")
        return mapping

    def _validate_resume_lineage(self, study: Any) -> None:
        for trial in self._candidate_trials(study).values():
            completed_epochs = _completed_epochs(trial)
            checkpoint = trial.user_attrs.get("latest_checkpoint_path")
            if completed_epochs > 0 and not checkpoint and trial.user_attrs.get("hexo_status") != "quarantined":
                raise RuntimeError(
                    f"candidate {trial.user_attrs.get('candidate_id')} has completed epochs without checkpoint lineage"
                )
            if checkpoint and not Path(str(checkpoint)).exists():
                raise RuntimeError(
                    f"candidate {trial.user_attrs.get('candidate_id')} checkpoint lineage is missing: {checkpoint}"
                )

    def _finalize_completed_running_trials(self, study: Any) -> None:
        for trial in self._candidate_trials(study).values():
            if trial.state == TrialState.RUNNING and trial.user_attrs.get("hexo_status") == "completed":
                study.tell(trial.number, _latest_scalar((), trial), skip_if_finished=True)

    def _initial_user_attrs(
        self,
        candidate: CandidateRecipe,
        cfg: Config,
        paths: CandidateArtifactPaths,
    ) -> dict[str, Any]:
        cfg_hash = config_hash(cfg)
        return {
            "candidate_id": candidate.candidate_id,
            "architecture_id": candidate.model.architecture_id,
            "pair_strategy": candidate.pair_strategy.mode,
            "recipe_schema_version": candidate.schema_version,
            "config_hash": cfg_hash,
            "full_config": cfg.model_dump(mode="json"),
            "run_dir": str(paths.candidate_dir),
            "completed_epochs": 0,
            "hexo_status": "pending",
            "quarantine_reason": None,
            "latest_checkpoint_path": None,
            "latest_scorecard_path": None,
        }

    def _set_lifecycle_attrs(self, study: Any, trial: Any, **attrs: Any) -> None:
        for key, value in attrs.items():
            self._set_trial_user_attr(study, trial, key, value)

    @staticmethod
    def _set_trial_user_attr(study: Any, trial: Any, key: str, value: Any) -> None:
        study._storage.set_trial_user_attr(trial._trial_id, key, value)

    @staticmethod
    def _set_intermediate_value(study: Any, trial: Any, epoch: int, value: float) -> None:
        study._storage.set_trial_intermediate_value(trial._trial_id, int(epoch), float(value))

    def _write_study_manifest(self, study: Any) -> None:
        manifest = {
            "study_name": self.study_name,
            "storage": self.storage,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "candidate_plan": list(self._candidate_ids),
            "candidate_plan_hash": _candidate_plan_hash(self.candidates),
            "min_epochs": self.min_epochs,
            "schedule_quantum_epochs": self.quantum_epochs,
            "pruner": "NopPruner",
            "target_scalar": PHASE1_TARGET_SCALAR,
            "trial_numbers_by_candidate": {
                candidate_id: trial.number for candidate_id, trial in self._candidate_trials(study).items()
            },
        }
        _atomic_write_json(self.run_dir / "study_manifest.json", manifest)

    def _summary(self, study: Any, quanta_executed: int) -> ScoutRunSummary:
        statuses: dict[str, dict[str, Any]] = {}
        complete = True
        for candidate in self.candidates:
            trial = self._candidate_trials(study)[candidate.candidate_id]
            attrs = trial.user_attrs
            status = str(attrs.get("hexo_status", "pending"))
            completed_epochs = _completed_epochs(trial)
            statuses[candidate.candidate_id] = {
                "trial_number": trial.number,
                "trial_state": trial.state.name,
                "hexo_status": status,
                "completed_epochs": completed_epochs,
                "quarantine_reason": attrs.get("quarantine_reason"),
                "latest_checkpoint_path": attrs.get("latest_checkpoint_path"),
                "latest_scorecard_path": attrs.get("latest_scorecard_path"),
            }
            if status != "quarantined" and completed_epochs < self.min_epochs:
                complete = False
        return ScoutRunSummary(
            study_name=self.study_name,
            storage=self.storage,
            run_dir=str(self.run_dir),
            completed=complete,
            candidate_statuses=statuses,
            quanta_executed=quanta_executed,
        )


def scout_storage_url(runs_root: str | Path, run_id: str, storage: str | None = None) -> str:
    if storage is None or storage == "sqlite:///runs/<run_id>/optuna.sqlite3":
        path = Path(runs_root) / str(run_id) / "optuna.sqlite3"
        return _sqlite_url(path)
    resolved = storage.replace("<run_id>", str(run_id))
    if not resolved.startswith("sqlite:///"):
        return resolved
    raw_path = resolved.removeprefix("sqlite:///")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return _sqlite_url(path)


def _latest_scalar(scorecards: tuple[dict[str, Any], ...], trial: Any) -> float:
    for scorecard in reversed(scorecards):
        if PHASE1_TARGET_SCALAR in scorecard:
            return float(scorecard[PHASE1_TARGET_SCALAR])
    if trial.intermediate_values:
        last_step = max(trial.intermediate_values)
        return float(trial.intermediate_values[last_step])
    return 0.0


def _default_runtime_probe_candidates(cfg: Config) -> tuple[RuntimeKnobs, ...]:
    base = RuntimeKnobs(
        selfplay_workers=max(1, int(cfg.selfplay.num_workers)),
        batch_size_per_worker=max(1, int(cfg.selfplay.batch_size_per_worker)),
        inference_max_batch_size=max(1, int(cfg.inference.max_batch_size)),
        inference_max_wait_us=max(1, int(cfg.inference.max_wait_us)),
    )
    lower_workers = max(1, base.selfplay_workers // 2)
    conservative = RuntimeKnobs(
        selfplay_workers=lower_workers,
        batch_size_per_worker=base.batch_size_per_worker,
        inference_max_batch_size=base.inference_max_batch_size,
        inference_max_wait_us=base.inference_max_wait_us,
    )
    if conservative == base:
        return (base,)
    return (conservative, base)


def _scout_epoch_budget_config(cfg: Config) -> Config:
    """Size production scout epochs from the scout contract, not base training defaults."""

    min_positions = max(1, int(cfg.autotune.scout.min_generated_selfplay_positions_per_epoch))
    batch_size = max(1, int(cfg.train.batch_size))
    tuned = cfg.model_copy(deep=True)
    tuned.selfplay.games_per_epoch = 0
    tuned.selfplay.states_per_epoch = min_positions
    tuned.train.batches_per_epoch = max(1, math.ceil(min_positions / batch_size))
    return Config.model_validate(tuned.model_dump(mode="json"))


def _epoch_result_phase1_scalar(result: Any) -> float:
    train_stats = getattr(result, "train_stats", {}) or {}
    if PHASE1_TARGET_SCALAR in train_stats:
        return float(train_stats[PHASE1_TARGET_SCALAR])
    component_metrics = train_stats.get("component_metrics", {})
    if isinstance(component_metrics, dict) and PHASE1_TARGET_SCALAR in component_metrics:
        return float(component_metrics[PHASE1_TARGET_SCALAR])
    return 0.0


def _train_stat_float(train_stats: dict[str, Any] | None, *keys: str) -> float:
    if not train_stats:
        return 0.0
    for key in keys:
        value = train_stats.get(key)
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return 0.0


def _value_effective_samples(cfg: Config, train_stats: dict[str, Any] | None) -> float:
    zero_frac = min(max(_train_stat_float(train_stats, "value_weight_zero_frac"), 0.0), 1.0)
    return float(cfg.train.batch_size) * max(0.0, 1.0 - zero_frac)


def _training_signal_warning_events(
    candidate_id: str,
    epoch: int,
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    truncation_rate = float(metrics.get("truncation_rate", 0.0) or 0.0)
    value_zero_frac = float(metrics.get("value_weight_zero_frac", 0.0) or 0.0)
    if truncation_rate > 0.25:
        events.append(
            {
                "event": "training_signal_warning",
                "candidate_id": candidate_id,
                "epoch": int(epoch),
                "metric": "truncation_rate",
                "value": truncation_rate,
                "threshold": 0.25,
                "action": "monitor_not_strength_prune",
            }
        )
    if value_zero_frac > 0.50:
        events.append(
            {
                "event": "training_signal_warning",
                "candidate_id": candidate_id,
                "epoch": int(epoch),
                "metric": "value_weight_zero_frac",
                "value": value_zero_frac,
                "threshold": 0.50,
                "action": "monitor_not_strength_prune",
            }
        )
    return events


def _pair_buffer_stat_float(cfg: Config, buffer_stats: dict[str, Any] | None, key: str) -> float:
    if str(getattr(cfg.model, "pair_strategy", "none")) == "none":
        return 0.0
    return _train_stat_float(buffer_stats, key)


def _latest_checkpoint_before(checkpoints_dir: Path, start_epoch: int) -> Path | None:
    target_epoch = int(start_epoch) - 1
    if target_epoch <= 0:
        return None
    exact = checkpoints_dir / f"epoch_{target_epoch:04d}.pt"
    if exact.exists():
        return exact
    candidates = sorted(checkpoints_dir.glob("epoch_*.pt"))
    before = [
        path for path in candidates
        if _checkpoint_epoch(path) is not None and int(_checkpoint_epoch(path) or 0) <= target_epoch
    ]
    return before[-1] if before else None


def _checkpoint_epoch(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def _load_trainer_from_checkpoint(cfg: Config, checkpoint_path: Path, device: Any | None) -> Any:
    from hexorl.models.assembly import build_model_from_config
    from hexorl.train.trainer import Trainer

    model = build_model_from_config(cfg, device=device, inference=False)
    trainer = Trainer(model, cfg, [], device=device)
    trainer.load_checkpoint(checkpoint_path)
    return trainer


def _run_hexo_epoch(
    cfg: Config,
    *,
    trainer: Any | None,
    output_dir: Path,
    recorder_run_dir: Path,
    recorder_run_id: str,
    bootstrap_games: int,
    use_selfplay: bool,
    train: bool,
    device: Any | None,
) -> Any:
    from hexorl.dashboard.recorder import RunRecorder
    from hexorl.epoch import run_epoch

    recorder = RunRecorder.for_run_dir(recorder_run_dir, run_id=recorder_run_id)
    return run_epoch(
        cfg,
        trainer=trainer,
        output_dir=output_dir,
        bootstrap_games=bootstrap_games,
        use_selfplay=use_selfplay,
        train=train,
        device=device,
        recorder=recorder,
    )


def _completed_epochs(trial: Any) -> int:
    return int(trial.user_attrs.get("completed_epochs") or 0)


def _candidate_plan_hash(candidates: tuple[CandidateRecipe, ...]) -> str:
    payload = [
        {
            "candidate_id": candidate.candidate_id,
            "architecture_id": candidate.model.architecture_id,
            "pair_strategy": candidate.pair_strategy.mode,
            "recipe_schema_version": candidate.schema_version,
        }
        for candidate in candidates
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_sqlite_parent(storage: str) -> None:
    if not storage.startswith("sqlite:///"):
        return
    path = Path(storage.removeprefix("sqlite:///"))
    path.parent.mkdir(parents=True, exist_ok=True)


def _sqlite_url(path: str | Path) -> str:
    absolute = Path(path).resolve()
    return f"sqlite:///{absolute.as_posix()}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _import_optuna() -> Any:
    if _OPTUNA is None:
        raise RuntimeError("Phase 1 Optuna scout requires optuna to be installed")
    return _OPTUNA
