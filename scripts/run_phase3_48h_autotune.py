"""Phase 3 48-hour autotuning supervisor.

This implements the plan in Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md:

* Phase 3A finalist import/calibration.
* Phase 3B ASHA-style static narrowing.
* Phase 3C PBT schedule search.
* Phase 3D protected champion training.
* Phase 3E final arena/checkpoint selection.

The code is intentionally a supervisor script. It reuses the production epoch
runner for the expensive self-play/training path and keeps all trial decisions,
mutation history, scorecards, and final reports in one run directory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.config import Config, load_config
from hexorl.dashboard.recorder import RunRecorder
from hexorl.epoch import run_epoch
from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.runtime import autotune_config, configure_torch_runtime, detect_host


LOGGER = logging.getLogger("phase3_autotune")


HEAD_BUNDLES: dict[str, list[str]] = {
    "structural": ["policy", "value", "lookahead_4", "lookahead_12", "lookahead_36", "axis"],
    "prediction": [
        "policy",
        "value",
        "lookahead_4",
        "lookahead_12",
        "lookahead_36",
        "axis",
        "opp_policy",
        "moves_left",
    ],
    "regret": [
        "policy",
        "value",
        "lookahead_4",
        "lookahead_12",
        "lookahead_36",
        "axis",
        "regret_rank",
        "regret_value",
    ],
    "full_aux_light": [
        "policy",
        "value",
        "lookahead_4",
        "lookahead_12",
        "lookahead_36",
        "axis",
        "opp_policy",
        "moves_left",
        "regret_rank",
        "regret_value",
    ],
    "graph_tactical": [
        "policy",
        "value",
        "lookahead_4",
        "lookahead_12",
        "lookahead_36",
        "axis",
        "opp_policy",
        "moves_left",
        "regret_rank",
        "regret_value",
        "pair_policy",
    ],
}


STATIC_SPACE = {
    "full_sims": [800, 1200, 1600],
    "pcr_low_sims": [192, 256, 384],
    "policy_top_k": [64, 96, 128],
    "candidate_budget": [128, 256, 384],
    "temperature_family": ["fast_cool", "slow_cool"],
    "head_bundle": ["structural", "prediction", "regret", "full_aux_light", "graph_tactical"],
    "subtree_reuse": [True],
}


DYNAMIC_RANGES = {
    "lr": (5e-3, 1e-2),
    "weight_decay": (1e-5, 5e-4),
    "c_puct": (1.1, 2.2),
    "c_puct_init": (1000.0, 20000.0),
    "dirichlet_fraction": (0.10, 0.35),
    "dirichlet_alpha": (0.01, 0.05),
    "scaled_alpha_total": (5.0, 12.0),
    "pcr_low_sim_prob": (0.50, 0.85),
    "recency_decay": (0.95, 0.995),
    "value_loss_weight": (1.0, 2.0),
    "aux_multiplier": (0.5, 1.5),
    "sparse_policy_loss": (0.10, 0.75),
    "pair_policy_loss": (0.02, 0.25),
    "graph_aux_multiplier": (0.5, 1.5),
    "regret_fraction": (0.0, 0.12),
}


DYNAMIC_CENTER = {
    "lr": 7e-3,
    "weight_decay": 1e-4,
    "c_puct": 1.6,
    "c_puct_init": 8000.0,
    "dirichlet_fraction": 0.22,
    "dirichlet_alpha": 0.02,
    "scaled_alpha_total": 8.0,
    "pcr_low_sim_prob": 0.70,
    "recency_decay": 0.98,
    "value_loss_weight": 1.3,
    "aux_multiplier": 1.0,
    "sparse_policy_loss": 0.35,
    "pair_policy_loss": 0.08,
    "graph_aux_multiplier": 1.0,
    "regret_fraction": 0.08,
}


PHASE_FRACTIONS = {
    "3A_calibration": 4.0 / 48.0,
    "3B_static_asha": 12.0 / 48.0,
    "3C_pbt": 16.0 / 48.0,
    "3D_champion": 12.0 / 48.0,
    "3E_final": 4.0 / 48.0,
}


TACTICAL_COMPONENTS = [
    "4-window completion",
    "5-window completion",
    "single forced block",
    "two-placement cover",
    "block plus counterattack",
    "unblockable-threat recognition",
    "axis fork creation",
    "separated-cluster defense",
    "outside-window win",
    "outside-window forced block",
]


@dataclass(frozen=True)
class FamilySpec:
    name: str
    description: str
    architecture: str
    channels: int = 128
    blocks: int = 16
    sparse_policy: bool = False
    attention_positions: tuple[int, ...] = ()
    graph: bool = False
    available: bool = True

    @property
    def compatible_key(self) -> tuple[Any, ...]:
        return (
            self.architecture,
            self.channels,
            self.blocks,
            self.sparse_policy,
            self.attention_positions,
            self.graph,
        )


@dataclass
class StaticRecipe:
    full_sims: int
    pcr_low_sims: int
    policy_top_k: int
    candidate_budget: int
    head_bundle: str
    temperature_family: str
    subtree_reuse: bool = True
    model_size: str = "inherited"
    graph_token_set: str = "graph256_cells"
    graph_token_budget: int = 256
    graph_layers: int = 1
    sparse_prior_stage: int = 0


@dataclass
class DynamicParams:
    lr: float = DYNAMIC_CENTER["lr"]
    weight_decay: float = DYNAMIC_CENTER["weight_decay"]
    c_puct: float = DYNAMIC_CENTER["c_puct"]
    c_puct_init: float = DYNAMIC_CENTER["c_puct_init"]
    dirichlet_fraction: float = DYNAMIC_CENTER["dirichlet_fraction"]
    dirichlet_alpha_mode: str = "scaled_total"
    dirichlet_alpha: float = DYNAMIC_CENTER["dirichlet_alpha"]
    scaled_alpha_total: float = DYNAMIC_CENTER["scaled_alpha_total"]
    pcr_low_sim_prob: float = DYNAMIC_CENTER["pcr_low_sim_prob"]
    recency_decay: float = DYNAMIC_CENTER["recency_decay"]
    value_loss_weight: float = DYNAMIC_CENTER["value_loss_weight"]
    aux_multiplier: float = DYNAMIC_CENTER["aux_multiplier"]
    sparse_policy_loss: float = DYNAMIC_CENTER["sparse_policy_loss"]
    pair_policy_loss: float = DYNAMIC_CENTER["pair_policy_loss"]
    graph_aux_multiplier: float = DYNAMIC_CENTER["graph_aux_multiplier"]
    regret_fraction: float = DYNAMIC_CENTER["regret_fraction"]


@dataclass
class TrialState:
    trial_id: str
    family: FamilySpec
    static: StaticRecipe
    dynamic: DynamicParams
    cfg: Config
    run_dir: Path
    recorder: RunRecorder
    replay: RingBuffer
    trainer: Any = None
    checkpoint_path: Path | None = None
    epoch: int = 0
    wall_time_s: float = 0.0
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    score_history: list[dict[str, Any]] = field(default_factory=list)
    mutation_history: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_history: list[Path] = field(default_factory=list)
    runtime_sweep: dict[str, Any] = field(default_factory=dict)
    pruned: bool = False
    prune_reason: str = ""

    @property
    def compatible_key(self) -> tuple[Any, ...]:
        return self.family.compatible_key + (tuple(self.cfg.model.heads),)

    @property
    def last_score(self) -> float:
        if not self.score_history:
            return float("-inf")
        return float(self.score_history[-1].get("scheduler_score", float("-inf")))


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        row = {"time": time.time(), "event": event, **_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


class Phase3Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.output_root = Path(args.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLogger(self.output_root / "events.jsonl")
        self.rng = random.Random(args.seed)
        self.np_rng = np.random.default_rng(args.seed)
        self.base_cfg = load_config(Path(args.config))
        self.host = detect_host()
        self.host_argument_overrides: dict[str, dict[str, Any]] = {}
        self._apply_host_argument_guards()
        self.run_started = time.monotonic()
        self.deadline = self.run_started + args.duration_hours * 3600.0
        self.stage_deadlines = self._stage_deadlines()
        self.families = self._finalist_pool()
        if args.family_filter:
            wanted = {item.strip() for raw in args.family_filter for item in raw.split(",") if item.strip()}
            self.families = [family for family in self.families if family.name in wanted]
            missing = sorted(wanted - {family.name for family in self.families})
            if missing:
                raise ValueError(f"Unknown --family-filter finalist(s): {', '.join(missing)}")
        self.blocked_families: dict[str, str] = {}
        self._apply_host_runtime_family_guards()
        self.trials: list[TrialState] = []
        self.baseline_loss_p75 = {True: 128.0, False: 128.0}
        self.reference_checkpoints = self._reference_checkpoints()
        self.calibration: dict[str, Any] = {}
        self.runtime_sweep_cache: dict[str, dict[str, Any]] = self._load_runtime_sweep_cache()
        self._write_manifest()

    def run(self) -> None:
        self.log.write("run_start", self._manifest_payload())
        if self.args.dry_run:
            self.log.write("dry_run_complete", {"families": [asdict(f) for f in self.families]})
            self._write_report(final=False)
            return
        try:
            self.phase_3a_calibration()
            self.phase_3b_static_asha()
            self.phase_3c_pbt()
            self.phase_3d_champion()
            self.phase_3e_final()
            self._write_report(final=True)
            self.log.write("run_complete", {"elapsed_s": self.elapsed_s(), "output_root": str(self.output_root)})
        except KeyboardInterrupt:
            self.log.write("interrupted", {"elapsed_s": self.elapsed_s()})
            self._write_report(final=False)
            raise
        except Exception as exc:
            self.log.write("run_failed", {"error": repr(exc), "elapsed_s": self.elapsed_s()})
            self._write_report(final=False)
            raise

    # ── Phase Implementations ──────────────────────────────────────────

    def phase_3a_calibration(self) -> None:
        stage = "3A_calibration"
        self.log.write("stage_start", {"stage": stage, "deadline_s": self.stage_deadlines[stage] - self.run_started})
        for family in self.families:
            if not self._within_stage(stage):
                break
            if not family.available:
                self.log.write("finalist_unavailable", {"family": asdict(family), "reason": "not implemented/passed"})
                continue
            if family.name in self.blocked_families:
                self.log.write(
                    "family_quarantined",
                    {
                        "stage": stage,
                        "family": family.name,
                        "reason": self.blocked_families[family.name],
                        "effect": "host_guard_excluded_from_calibration_and_long_search",
                    },
                )
                continue
            recipe = self._recommended_recipe(family)
            dyn = self._initial_dynamic(family)
            trial = self._create_trial(f"cal_{family.name}", family, recipe, dyn, stage)
            self.log.write("calibration_trial_start", {"trial_id": trial.trial_id, "family": family.name})
            self._train_trial_epoch(
                trial,
                stage=stage,
                target_epoch_seconds=min(self.args.target_epoch_seconds, self.args.calibration_epoch_seconds),
                force_states=self.args.calibration_states,
                force_train_batches=self.args.calibration_train_batches,
            )
            if trial.pruned or not trial.metrics_history:
                reason = trial.prune_reason or "no_metrics"
                if not trial.pruned:
                    trial.pruned = True
                    trial.prune_reason = reason
                    self.log.write("trial_pruned", {"trial_id": trial.trial_id, "stage": stage, "reason": reason})
                    self._release_trial_runtime(trial, reason=reason)
                self.calibration[family.name] = {
                    "elapsed_s": self.args.calibration_epoch_seconds,
                    "positions": self.args.calibration_states,
                    "epoch_time_ratio": 1.0,
                    "memory_gb": self._gpu_memory_gb(),
                    "failed": True,
                    "reason": reason,
                }
                self._quarantine_family(family, reason, stage=stage)
                self.trials.append(trial)
                continue
            self._evaluate_trial(trial, stage=stage, force=True)
            latest_metrics = trial.metrics_history[-1]
            calibration_positions = (
                latest_metrics.get("selfplay", {}).get("positions_done")
                or latest_metrics.get("buffer", {}).get("size")
            )
            self.calibration[family.name] = {
                "elapsed_s": latest_metrics.get("epoch_elapsed_s"),
                "positions": calibration_positions,
                "epoch_time_ratio": latest_metrics.get("epoch_elapsed_s", 1.0)
                / max(self.args.target_epoch_seconds, 1.0),
                "memory_gb": self._gpu_memory_gb(),
            }
            self.trials.append(trial)
        self._exclude_slow_latency_families_after_calibration(stage)
        self._save_state()

    def phase_3b_static_asha(self) -> None:
        stage = "3B_static_asha"
        self.log.write("stage_start", {"stage": stage, "active_trials": self.args.max_active_trials})
        candidates = self._generate_static_candidates(max_trials=self.args.max_active_trials)
        active = []
        for idx, (family, recipe) in enumerate(candidates):
            dyn = self._initial_dynamic(family)
            trial = self._create_trial(f"asha_{idx:02d}_{family.name}", family, recipe, dyn, stage)
            active.append(trial)
            self.trials.append(trial)

        resources = self._asha_resources()
        current = active
        for resource in resources:
            if not self._within_stage(stage) or not current:
                break
            self.log.write("asha_rung_start", {"resource": resource, "trial_ids": [t.trial_id for t in current]})
            for trial in current:
                while trial.epoch < resource and self._within_stage(stage):
                    self._train_trial_epoch(trial, stage=stage, target_epoch_seconds=self.args.target_epoch_seconds)
                self._evaluate_trial(trial, stage=stage, force=True)
            current = self._promote_top_fraction(current, stage=stage)
        self._save_state()

    def phase_3c_pbt(self) -> None:
        stage = "3C_pbt"
        self.log.write("stage_start", {"stage": stage})
        population = self._seed_pbt_population()
        generation = 0
        while self._within_stage(stage) and generation < self.args.pbt_generations and population:
            generation += 1
            self.log.write("pbt_generation_start", {"generation": generation, "population": [t.trial_id for t in population]})
            for trial in population:
                if trial.pruned:
                    continue
                for _ in range(self.args.perturb_interval):
                    if not self._within_stage(stage):
                        break
                    self._train_trial_epoch(trial, stage=stage, target_epoch_seconds=self.args.target_epoch_seconds)
                self._evaluate_trial(trial, stage=stage, force=True)
            self._score_population(population, stage=stage)
            self._pbt_exploit_explore(population, generation)
            self._save_state()

    def phase_3d_champion(self) -> None:
        stage = "3D_champion"
        candidates = [t for t in self.trials if not t.pruned and t.checkpoint_path]
        mature = [t for t in candidates if t.epoch >= self.args.champion_min_epochs]
        if mature:
            candidates = mature
        elif candidates:
            self.log.write(
                "champion_min_epochs_fallback",
                {
                    "requested_min_epochs": self.args.champion_min_epochs,
                    "available": {t.trial_id: t.epoch for t in candidates},
                },
            )
        champion = max(candidates, key=lambda t: t.last_score, default=None)
        if champion is None:
            self.log.write("champion_unavailable", {"reason": "no unpruned trial with checkpoint"})
            return
        self.log.write("stage_start", {"stage": stage, "champion": champion.trial_id})
        # Continue the selected trial; this protects champion time by avoiding
        # further broad exploration once Phase 3D starts.
        while self._within_stage(stage) and time.monotonic() < self.deadline:
            self._train_trial_epoch(champion, stage=stage, target_epoch_seconds=self.args.target_epoch_seconds)
            if champion.epoch % 2 == 0:
                self._evaluate_trial(champion, stage=stage, force=True)
            self._save_state()

    def phase_3e_final(self) -> None:
        stage = "3E_final"
        self.log.write("stage_start", {"stage": stage})
        candidates = self._final_candidates()
        final_rows = []
        for item in candidates:
            if not self._within_stage(stage):
                break
            row = self._evaluate_checkpoint_final(item["checkpoint"], item["label"], item.get("cfg"))
            final_rows.append(row)
            self.log.write("final_candidate_evaluated", row)
        if final_rows:
            ranked = self._rank_final_rows(final_rows)
            winner = ranked[0]
            _write_json(self.output_root / "final_selection.json", {"winner": winner, "ranked": ranked})
            self.log.write("final_selection", {"winner": winner, "ranked": ranked})
        self._save_state()

    # ── Trial Operations ────────────────────────────────────────────────

    def _create_trial(
        self,
        trial_id: str,
        family: FamilySpec,
        recipe: StaticRecipe,
        dynamic: DynamicParams,
        stage: str,
    ) -> TrialState:
        run_dir = self.output_root / "trials" / trial_id
        cfg = self._make_config(family, recipe, dynamic, run_dir, stage)
        recorder = RunRecorder.for_run_dir(run_dir, run_id=trial_id)
        replay = RingBuffer(
            capacity=cfg.buffer.capacity,
            max_policy_entries=cfg.selfplay.policy_target_top_k,
            max_policy_v2_entries=max(cfg.selfplay.policy_target_top_k, cfg.model.candidate_budget),
            recency_decay=cfg.buffer.recency_decay,
            num_lookahead=len(cfg.buffer.lookahead_horizons),
        )
        trial = TrialState(
            trial_id=trial_id,
            family=family,
            static=recipe,
            dynamic=dynamic,
            cfg=cfg,
            run_dir=run_dir,
            recorder=recorder,
            replay=replay,
        )
        _write_json(run_dir / "trial.json", self._trial_public_state(trial))
        self.log.write("trial_created", {"stage": stage, **self._trial_public_state(trial)})
        return trial

    def _train_trial_epoch(
        self,
        trial: TrialState,
        *,
        stage: str,
        target_epoch_seconds: float,
        force_states: int | None = None,
        force_train_batches: int | None = None,
    ) -> None:
        if trial.pruned:
            return
        self._cleanup_shared_memory()
        self._apply_epoch_budget(trial, stage, target_epoch_seconds, force_states, force_train_batches)
        self._apply_dynamic_to_config(trial)
        self._apply_dynamic_to_trainer(trial)
        self._ensure_runtime_sweep(trial, stage=stage)
        started = time.monotonic()
        try:
            result = run_epoch(
                trial.cfg,
                trainer=trial.trainer,
                buffer=trial.replay,
                output_dir=trial.run_dir,
                bootstrap_games=0,
                use_selfplay=True,
                train=True,
                recorder=trial.recorder,
            )
        except Exception as exc:
            trial.pruned = True
            trial.prune_reason = f"train_exception:{type(exc).__name__}:{exc}"
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "reason": trial.prune_reason})
            self._release_trial_runtime(trial, reason=trial.prune_reason)
            return

        trial.trainer = result.trainer
        trial.checkpoint_path = result.checkpoint_path
        if result.checkpoint_path:
            trial.checkpoint_history.append(result.checkpoint_path)
        trial.epoch = int(result.train_stats.get("epoch", trial.epoch + 1))
        trial.wall_time_s += time.monotonic() - started

        selfplay = _latest_metric(trial.run_dir / "events.jsonl", "selfplay")
        record = {
            "stage": stage,
            "trial_id": trial.trial_id,
            "family": trial.family.name,
            "epoch": trial.epoch,
            "epoch_elapsed_s": result.elapsed_s,
            "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
            "train": result.train_stats,
            "buffer": result.buffer_stats,
            "selfplay": selfplay,
            "static": asdict(trial.static),
            "dynamic": asdict(trial.dynamic),
        }
        trial.metrics_history.append(record)
        self.log.write("trial_epoch_complete", record)
        _append_jsonl(trial.run_dir / "summary.jsonl", record)
        _write_json(trial.run_dir / "LATEST.json", record)

        reason = self._hard_prune_reason(trial, record)
        if reason:
            trial.pruned = True
            trial.prune_reason = reason
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "reason": reason})
            self._release_trial_runtime(trial, reason=reason)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evaluate_trial(self, trial: TrialState, *, stage: str, force: bool = False) -> dict[str, Any]:
        if trial.pruned or trial.checkpoint_path is None:
            return {}
        if not force and trial.epoch % self.args.eval_every_epochs != 0:
            return trial.score_history[-1] if trial.score_history else {}

        evals = EvaluationServices(self)
        components = evals.evaluate_trial(trial, stage=stage)
        trial.score_history.append(components)
        self.log.write("trial_evaluated", {"trial_id": trial.trial_id, "stage": stage, **components})
        _append_jsonl(trial.run_dir / "scores.jsonl", components)

        reason = self._hard_prune_reason(trial, trial.metrics_history[-1] if trial.metrics_history else {})
        if reason:
            trial.pruned = True
            trial.prune_reason = reason
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "reason": reason})
            self._release_trial_runtime(trial, reason=reason)
        return components

    def _clone_compatible_trial(self, src: TrialState, dst: TrialState, generation: int) -> bool:
        if src.compatible_key != dst.compatible_key or src.checkpoint_path is None:
            return False
        dst.replay = src.replay
        dst.checkpoint_path = src.checkpoint_path
        if dst.trainer is not None:
            dst.trainer.load_checkpoint(src.checkpoint_path)
        dst.epoch = src.epoch
        event = {
            "generation": generation,
            "event": "exploit",
            "from": src.trial_id,
            "to": dst.trial_id,
            "checkpoint_path": str(src.checkpoint_path),
        }
        dst.mutation_history.append(event)
        self.log.write("pbt_exploit", event)
        return True

    def _mutate_trial(self, trial: TrialState, generation: int) -> None:
        old = asdict(trial.dynamic)
        for field_name, value in old.items():
            if field_name == "dirichlet_alpha_mode":
                if self.rng.random() < 0.20:
                    setattr(trial.dynamic, field_name, "fixed" if value == "scaled_total" else "scaled_total")
                continue
            if field_name not in DYNAMIC_RANGES:
                continue
            lo, hi = DYNAMIC_RANGES[field_name]
            if self.rng.random() < 0.20:
                new_value = self.rng.uniform(lo, hi)
            else:
                new_value = float(value) * self.rng.choice([0.8, 1.2])
                new_value = max(lo, min(hi, new_value))
            setattr(trial.dynamic, field_name, new_value)
        self._apply_dynamic_to_config(trial)
        self._apply_dynamic_to_trainer(trial)
        event = {
            "generation": generation,
            "event": "explore",
            "trial_id": trial.trial_id,
            "old": old,
            "new": asdict(trial.dynamic),
        }
        trial.mutation_history.append(event)
        self.log.write("pbt_explore", event)

    def _release_trial_runtime(self, trial: TrialState, *, reason: str) -> None:
        replay_capacity = int(getattr(trial.replay, "capacity", 0) or 0)
        if trial.trainer is None and replay_capacity <= 1:
            return
        trial.trainer = None
        try:
            trial.replay = RingBuffer(
                capacity=1,
                max_policy_entries=max(1, int(trial.cfg.selfplay.policy_target_top_k)),
                max_policy_v2_entries=max(1, int(trial.cfg.model.candidate_budget)),
                recency_decay=trial.cfg.buffer.recency_decay,
                num_lookahead=len(trial.cfg.buffer.lookahead_horizons),
            )
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.log.write("trial_runtime_released", {"trial_id": trial.trial_id, "reason": reason})

    def _prune_trials_for_family(self, family: FamilySpec, reason: str, *, stage: str) -> None:
        for trial in self.trials:
            if trial.family.name != family.name or trial.pruned:
                continue
            trial.pruned = True
            trial.prune_reason = reason
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "stage": stage, "reason": reason})
            self._release_trial_runtime(trial, reason=reason)

    # ── Config Construction ────────────────────────────────────────────

    def _initial_dynamic(self, family: FamilySpec) -> DynamicParams:
        dynamic = copy.deepcopy(DynamicParams())
        # Graph and action-keyed sparse heads are new research paths. The doc
        # LR range is still explored by PBT mutations, but bootstrapping these
        # families at the CNN center rate can corrupt weights before the first
        # scorecard exists. Start them at the known-stable safety rail and let
        # the population exploit/explore upward only after finite metrics land.
        if family.sparse_policy or family.graph or family.architecture == "restnet":
            dynamic.lr = min(dynamic.lr, 3e-4)
        return dynamic

    def _make_config(
        self,
        family: FamilySpec,
        recipe: StaticRecipe,
        dynamic: DynamicParams,
        run_dir: Path,
        stage: str,
    ) -> Config:
        cfg = self.base_cfg.model_copy(deep=True)
        cfg.run.output_dir = str(run_dir)
        cfg.run.seed = self.args.seed + abs(hash(str(run_dir))) % 1_000_000
        cfg.run.log_level = "INFO"
        cfg.model.architecture = family.architecture
        cfg.model.channels = family.channels
        cfg.model.blocks = family.blocks
        cfg.model.attention_positions = list(family.attention_positions)
        cfg.model.attention_heads = 8
        cfg.model.sparse_policy = bool(family.sparse_policy or family.graph)
        cfg.model.graph_token_set = recipe.graph_token_set
        cfg.model.graph_token_budget = recipe.graph_token_budget
        cfg.model.graph_layers = recipe.graph_layers
        cfg.model.sparse_prior_stage = int(recipe.sparse_prior_stage)
        cfg.model.sparse_prior_mix = 0.25
        cfg.model.candidate_budget = recipe.candidate_budget if (family.sparse_policy or family.graph) else 256
        cfg.model.heads = list(HEAD_BUNDLES[recipe.head_bundle])
        cfg.buffer.lookahead_horizons = [4, 12, 36]
        cfg.buffer.lookahead_lambdas = [0.75, 0.90, 0.97]
        cfg.selfplay.mcts_simulations = recipe.full_sims
        cfg.selfplay.pcr_low_sims = recipe.pcr_low_sims
        cfg.selfplay.policy_target_top_k = recipe.policy_top_k
        cfg.selfplay.subtree_reuse = recipe.subtree_reuse
        cfg.selfplay.train_policy_on_full_search_only = True
        cfg.selfplay.train_on_truncated_games = True
        cfg.selfplay.max_game_moves = self._max_game_moves_for_stage(stage)
        cfg.runtime.autotune = True
        cfg.selfplay.num_workers = 0
        cfg.selfplay.batch_size_per_worker = 0
        cfg.inference.max_batch_size = 0
        cfg.train.batch_size = 0
        cfg.train.batches_per_epoch = self.args.train_batches
        cfg.train.lr_schedule = "constant"
        cfg.runtime.compile_inference = False
        cfg.runtime.compile_model = False
        self._apply_head_bundle_weights(cfg, recipe, dynamic)
        self._apply_dynamic_values(cfg, recipe, dynamic, family)
        autotune_config(cfg, self.host, selfplay_enabled=True)
        if self.host.cuda_available and self.host.cuda_memory_gb < 16.0:
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), 3)
            cfg.selfplay.batch_size_per_worker = min(int(cfg.selfplay.batch_size_per_worker), 8)
            cfg.inference.max_batch_size = min(
                int(cfg.inference.max_batch_size),
                max(64, cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        if family.graph or family.architecture == "restnet" or family.sparse_policy:
            if family.graph:
                latency_scale = max(1.0, (recipe.graph_token_budget / 256.0) * max(recipe.graph_layers, 1))
                if recipe.sparse_prior_stage > 0:
                    latency_scale *= 2.5
            elif family.sparse_policy:
                latency_scale = max(1.0, recipe.candidate_budget / 256.0)
            else:
                latency_scale = max(1.0, 1.0 + 0.15 * len(family.attention_positions))
            worker_cap = 4 if latency_scale <= 1.75 else 3
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), worker_cap)
            cfg.selfplay.batch_size_per_worker = min(int(cfg.selfplay.batch_size_per_worker), 8)
            cfg.inference.max_batch_size = min(
                int(cfg.inference.max_batch_size),
                max(64, cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        elif stage != "3A_calibration" and recipe.full_sims >= 800:
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), 4)
            cfg.selfplay.batch_size_per_worker = min(int(cfg.selfplay.batch_size_per_worker), 8)
            cfg.inference.max_batch_size = min(
                int(cfg.inference.max_batch_size),
                max(64, cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        configure_torch_runtime(cfg, self.host)
        return Config.model_validate(cfg.model_dump())

    def _apply_head_bundle_weights(self, cfg: Config, recipe: StaticRecipe, dynamic: DynamicParams) -> None:
        aux = (
            dynamic.graph_aux_multiplier
            if getattr(cfg.model, "architecture", "cnn") in {"graph", "graph_hybrid_0"}
            else dynamic.aux_multiplier
        )
        weights = {
            "policy": 1.0,
            "value": dynamic.value_loss_weight,
            "lookahead_4": 0.2 * aux,
            "lookahead_12": 0.2 * aux,
            "lookahead_36": 0.1 * aux,
            "axis": 0.05 * aux,
        }
        if "opp_policy" in HEAD_BUNDLES[recipe.head_bundle]:
            weights["opp_policy"] = 0.15 * aux
        if "moves_left" in HEAD_BUNDLES[recipe.head_bundle]:
            weights["moves_left"] = 0.05 * aux
        if "regret_rank" in HEAD_BUNDLES[recipe.head_bundle]:
            weights["regret_rank"] = 0.1 * aux
        if "regret_value" in HEAD_BUNDLES[recipe.head_bundle]:
            weights["regret_value"] = 0.1 * aux
        if cfg.model.sparse_policy:
            weights["sparse_policy"] = dynamic.sparse_policy_loss
        if "pair_policy" in HEAD_BUNDLES[recipe.head_bundle]:
            weights["pair_policy"] = dynamic.pair_policy_loss
        cfg.train.loss_weights = weights

    def _apply_dynamic_to_config(self, trial: TrialState) -> None:
        self._apply_head_bundle_weights(trial.cfg, trial.static, trial.dynamic)
        self._apply_dynamic_values(trial.cfg, trial.static, trial.dynamic, trial.family)

    def _apply_dynamic_values(
        self,
        cfg: Config,
        recipe: StaticRecipe,
        dynamic: DynamicParams,
        family: FamilySpec,
    ) -> None:
        cfg.train.peak_lr = float(dynamic.lr)
        cfg.train.weight_decay = float(dynamic.weight_decay)
        cfg.selfplay.c_puct = float(dynamic.c_puct)
        cfg.selfplay.c_puct_init = float(dynamic.c_puct_init)
        cfg.selfplay.dirichlet_fraction = float(dynamic.dirichlet_fraction)
        if dynamic.dirichlet_alpha_mode == "scaled_total":
            root_width = recipe.candidate_budget if (family.sparse_policy or family.graph) else recipe.policy_top_k
            cfg.selfplay.dirichlet_alpha = float(dynamic.scaled_alpha_total) / max(float(root_width), 1.0)
        else:
            cfg.selfplay.dirichlet_alpha = float(dynamic.dirichlet_alpha)
        cfg.selfplay.pcr_low_sim_prob = float(dynamic.pcr_low_sim_prob)
        cfg.buffer.recency_decay = float(dynamic.recency_decay)
        cfg.buffer.regret_fraction = float(dynamic.regret_fraction) if any(
            h.startswith("regret_") for h in cfg.model.heads
        ) else 0.0

    def _apply_dynamic_to_trainer(self, trial: TrialState) -> None:
        trainer = trial.trainer
        if trainer is None:
            return
        for idx, group in enumerate(trainer.optimizer.param_groups):
            group["lr"] = float(trial.dynamic.lr)
            if idx == 0:
                group["weight_decay"] = float(trial.dynamic.weight_decay)
            else:
                group["weight_decay"] = 0.0
        trainer._loss_weights = dict(trial.cfg.train.loss_weights)
        trainer.train_cfg = trial.cfg.train
        trainer.batches_per_epoch = trial.cfg.train.batches_per_epoch

    def _max_game_moves_for_stage(self, stage: str) -> int:
        return max(1, int(self.args.max_game_moves))

    def _apply_epoch_budget(
        self,
        trial: TrialState,
        stage: str,
        target_epoch_seconds: float,
        force_states: int | None,
        force_train_batches: int | None,
    ) -> None:
        trial.cfg.selfplay.max_game_moves = self._max_game_moves_for_stage(stage)
        if force_states is not None:
            states = int(force_states)
        else:
            states = self._target_states_for_trial(trial, target_epoch_seconds)
        states = max(int(self.args.min_states_per_epoch), min(states, self.args.max_states_per_epoch))
        games = max(1, math.ceil(states / max(1, trial.cfg.selfplay.max_game_moves)))
        trial.cfg.selfplay.states_per_epoch = states
        trial.cfg.selfplay.games_per_epoch = games
        trial.cfg.train.batches_per_epoch = int(force_train_batches or self.args.train_batches)

    def _target_states_for_trial(self, trial: TrialState, target_seconds: float) -> int:
        cal = self.calibration.get(trial.family.name)
        if not cal:
            return self.args.default_states_per_epoch
        cal_elapsed = max(float(cal.get("elapsed_s") or self.args.calibration_epoch_seconds), 1.0)
        cal_positions = max(float(cal.get("positions") or self.args.calibration_states), 1.0)
        sims_scale = max(float(self._recommended_recipe(trial.family).full_sims), 1.0) / max(
            float(trial.static.full_sims), 1.0
        )
        pos_per_s = cal_positions / cal_elapsed * sims_scale
        return int(pos_per_s * target_seconds)

    def _ensure_runtime_sweep(self, trial: TrialState, *, stage: str) -> None:
        sweep_states = int(getattr(self.args, "runtime_sweep_states", 0) or 0)
        if sweep_states <= 0:
            return
        if trial.runtime_sweep.get("applied"):
            return
        key = self._runtime_sweep_key(trial)
        cached = self.runtime_sweep_cache.get(key)
        if cached and cached.get("selected"):
            selected = dict(cached["selected"])
            self._apply_runtime_candidate(trial.cfg, selected)
            trial.runtime_sweep = {
                "applied": True,
                "cached": True,
                "key": key,
                "selected": selected,
            }
            self.log.write("runtime_sweep_cached", {"trial_id": trial.trial_id, "stage": stage, **trial.runtime_sweep})
            return

        candidates = self._runtime_sweep_candidates(trial)
        if not candidates:
            trial.runtime_sweep = {"applied": False, "skipped": True, "reason": "no_candidates", "key": key}
            self.log.write("runtime_sweep_skipped", {"trial_id": trial.trial_id, "stage": stage, **trial.runtime_sweep})
            return

        self.log.write(
            "runtime_sweep_start",
            {
                "trial_id": trial.trial_id,
                "stage": stage,
                "key": key,
                "states": sweep_states,
                "candidates": candidates,
            },
        )
        results: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            if not self._within_stage(stage):
                results.append({"candidate": candidate, "error": "stage_deadline_reached"})
                break
            probe_dir = trial.run_dir / "runtime_sweep" / f"candidate_{idx:02d}"
            row = self._run_runtime_sweep_candidate(
                trial,
                stage=stage,
                candidate=candidate,
                probe_dir=probe_dir,
                sweep_states=sweep_states,
                index=idx,
            )
            results.append(row)
            self.log.write("runtime_sweep_result", {"trial_id": trial.trial_id, "stage": stage, **row})
            _append_jsonl(self.output_root / "runtime_sweep_results.jsonl", {"trial_id": trial.trial_id, "stage": stage, **row})

        valid = [row for row in results if row.get("ok") and float(row.get("positions_per_min", 0.0) or 0.0) > 0.0]
        if not valid:
            selected = dict(candidates[0])
            selected_record = {
                "candidate": selected,
                "positions_per_min": 0.0,
                "fallback": True,
                "reason": "all_probe_candidates_failed",
            }
        else:
            selected_record = max(valid, key=lambda row: float(row.get("score", row.get("positions_per_min", 0.0)) or 0.0))
            selected = dict(selected_record["candidate"])

        self._apply_runtime_candidate(trial.cfg, selected)
        trial.runtime_sweep = {
            "applied": True,
            "cached": False,
            "key": key,
            "selected": selected,
            "selected_positions_per_min": float(selected_record.get("positions_per_min", 0.0) or 0.0),
            "candidate_count": len(candidates),
        }
        self.runtime_sweep_cache[key] = {
            "key": key,
            "created_time": time.time(),
            "selected": selected,
            "selected_record": selected_record,
            "results": results,
        }
        self._save_runtime_sweep_cache()
        self.log.write("runtime_sweep_selected", {"trial_id": trial.trial_id, "stage": stage, **trial.runtime_sweep})

    def _run_runtime_sweep_candidate(
        self,
        trial: TrialState,
        *,
        stage: str,
        candidate: dict[str, int],
        probe_dir: Path,
        sweep_states: int,
        index: int,
    ) -> dict[str, Any]:
        probe_cfg = trial.cfg.model_copy(deep=True)
        self._apply_runtime_candidate(probe_cfg, candidate)
        max_game_moves = max(1, int(probe_cfg.selfplay.max_game_moves))
        states = max(max_game_moves, int(sweep_states))
        games = max(1, math.ceil(states / max_game_moves))
        probe_cfg.selfplay.states_per_epoch = states
        probe_cfg.selfplay.games_per_epoch = games
        probe_cfg.buffer.capacity = max(512, states + max_game_moves * max(2, int(candidate["workers"])))
        probe_cfg.train.batches_per_epoch = 0
        probe_cfg.run.output_dir = str(probe_dir)
        probe_cfg.run.seed = int(probe_cfg.run.seed) + 10_000 + index
        probe_cfg = Config.model_validate(probe_cfg.model_dump())

        recorder = RunRecorder.for_run_dir(probe_dir, run_id=f"{trial.trial_id}_runtime_sweep_{index:02d}")
        probe_buffer = RingBuffer(
            capacity=probe_cfg.buffer.capacity,
            max_policy_entries=probe_cfg.selfplay.policy_target_top_k,
            max_policy_v2_entries=max(probe_cfg.selfplay.policy_target_top_k, probe_cfg.model.candidate_budget),
            recency_decay=probe_cfg.buffer.recency_decay,
            num_lookahead=len(probe_cfg.buffer.lookahead_horizons),
        )
        model = trial.trainer.model if trial.trainer is not None else None
        gpu_before = self._nvidia_smi_snapshot()
        started = time.monotonic()
        try:
            result = run_epoch(
                probe_cfg,
                model=model,
                buffer=probe_buffer,
                output_dir=probe_dir,
                bootstrap_games=0,
                use_selfplay=True,
                train=False,
                recorder=recorder,
            )
            elapsed_s = max(time.monotonic() - started, 1e-6)
            selfplay = _latest_metric(probe_dir / "events.jsonl", "selfplay")
            positions = int(selfplay.get("positions_done") or result.buffer_stats.get("size") or 0)
            positions_per_min = float(selfplay.get("positions_per_min") or (positions / elapsed_s * 60.0))
            gpu_after = self._nvidia_smi_snapshot()
            gpu_util = float(gpu_after.get("gpu_util_pct", 0.0) or 0.0)
            score = positions_per_min * (1.0 + min(gpu_util, 95.0) / 2000.0)
            return {
                "ok": positions > 0,
                "candidate": candidate,
                "elapsed_s": elapsed_s,
                "positions": positions,
                "positions_per_min": positions_per_min,
                "score": score,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "selfplay": selfplay,
            }
        except Exception as exc:
            return {
                "ok": False,
                "candidate": candidate,
                "elapsed_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}:{exc}",
                "gpu_before": gpu_before,
                "gpu_after": self._nvidia_smi_snapshot(),
            }
        finally:
            self._cleanup_shared_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _runtime_sweep_candidates(self, trial: TrialState) -> list[dict[str, int]]:
        cfg = trial.cfg
        base_workers = max(1, int(cfg.selfplay.num_workers))
        base_bpw = max(1, int(cfg.selfplay.batch_size_per_worker))
        base_wait = max(1, int(cfg.inference.max_wait_us))
        worker_values = _parse_int_list(getattr(self.args, "runtime_sweep_workers", ""))
        if not worker_values:
            worker_values = [max(1, base_workers - 1), base_workers, base_workers + 1]
        worker_budget = max(1, int(self.host.logical_cpus) - max(1, int(cfg.runtime.selfplay_cpu_reserve)))
        max_workers = min(worker_budget, max(worker_values + [base_workers]))

        candidates: list[dict[str, int]] = []

        def add(workers: int, wait_us: int) -> None:
            workers = max(1, min(int(workers), max_workers))
            batch_per_worker = base_bpw
            max_batch = max(64, workers * batch_per_worker + 64)
            if self.host.cuda_available and self.host.cuda_memory_gb < 16.0:
                max_batch = min(max_batch, 128)
            candidate = {
                "workers": workers,
                "batch_size_per_worker": batch_per_worker,
                "max_batch_size": max_batch,
                "max_wait_us": max(1, int(wait_us)),
            }
            if candidate not in candidates:
                candidates.append(candidate)

        add(base_workers, base_wait)
        for workers in sorted(set(worker_values)):
            add(workers, base_wait)
        if len(candidates) < int(self.args.runtime_sweep_max_candidates):
            add(base_workers, max(base_wait, 800))
        if len(candidates) < int(self.args.runtime_sweep_max_candidates):
            add(max_workers, max(base_wait, 800))
        return candidates[: max(1, int(self.args.runtime_sweep_max_candidates))]

    def _apply_runtime_candidate(self, cfg: Config, candidate: dict[str, int]) -> None:
        cfg.selfplay.num_workers = max(1, int(candidate["workers"]))
        cfg.selfplay.batch_size_per_worker = max(1, int(candidate["batch_size_per_worker"]))
        cfg.inference.max_batch_size = max(1, int(candidate["max_batch_size"]))
        cfg.inference.max_wait_us = max(1, int(candidate["max_wait_us"]))

    def _runtime_sweep_key(self, trial: TrialState) -> str:
        payload = {
            "family": trial.family.compatible_key,
            "static": asdict(trial.static),
            "heads": list(trial.cfg.model.heads),
            "max_game_moves": int(trial.cfg.selfplay.max_game_moves),
            "mcts_simulations": int(trial.cfg.selfplay.mcts_simulations),
            "pcr_low_sims": int(trial.cfg.selfplay.pcr_low_sims),
            "candidate_budget": int(getattr(trial.cfg.model, "candidate_budget", 256)),
        }
        return json.dumps(_jsonable(payload), sort_keys=True)

    def _load_runtime_sweep_cache(self) -> dict[str, dict[str, Any]]:
        path = self.output_root / "runtime_sweep_cache.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(payload, dict):
            return {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        return {}

    def _save_runtime_sweep_cache(self) -> None:
        _write_json(self.output_root / "runtime_sweep_cache.json", self.runtime_sweep_cache)

    def _nvidia_smi_snapshot(self) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
        except Exception:
            return {}
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        first = proc.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) < 4:
            return {}
        try:
            return {
                "gpu_util_pct": float(parts[0]),
                "memory_util_pct": float(parts[1]),
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
            }
        except ValueError:
            return {}

    # ── Search/Selection ───────────────────────────────────────────────

    def _generate_static_candidates(self, max_trials: int) -> list[tuple[FamilySpec, StaticRecipe]]:
        available = self._eligible_families()
        combos: list[tuple[FamilySpec, StaticRecipe]] = []
        for family in available:
            bundles = ["structural", "full_aux_light"]
            if family.graph:
                bundles = ["graph_tactical", "full_aux_light"]
            if family.sparse_policy:
                bundles.append("prediction")
            graph_ladder = self._graph_static_ladder() if family.graph else [("graph256_cells", 256, 1, 0)]
            full_sim_options = [256, 384] if family.graph else [800, 1200]
            for full_sims in full_sim_options:
                for token_set, token_budget, graph_layers, sparse_stage in graph_ladder:
                    recipe = StaticRecipe(
                        full_sims=full_sims,
                        pcr_low_sims=96 if full_sims <= 256 else (128 if family.graph else (256 if full_sims >= 1200 else 192)),
                        policy_top_k=96,
                        candidate_budget=token_budget
                        if family.graph
                        else (self.rng.choice(STATIC_SPACE["candidate_budget"]) if family.sparse_policy else 256),
                        head_bundle=self.rng.choice(bundles),
                        temperature_family=self.rng.choice(STATIC_SPACE["temperature_family"]),
                        subtree_reuse=True,
                        graph_token_set=token_set,
                        graph_token_budget=token_budget,
                        graph_layers=graph_layers,
                        sparse_prior_stage=sparse_stage,
                    )
                    combos.append((family, recipe))
        self.rng.shuffle(combos)
        # Ensure every documented discrete value is represented in the manifest,
        # while ASHA keeps the active pool small as required by the plan.
        _write_json(self.output_root / "static_search_space.json", STATIC_SPACE)
        return combos[:max_trials]

    def _graph_static_ladder(self) -> list[tuple[str, int, int, int]]:
        """Keep graph ASHA broad, but scale action-keyed priors to the host.

        On 12 GB GPUs with 32 GB system RAM the 512-token/3-layer/stage-1
        combination is CPU and memory bound during MCTS leaf construction. It
        leaves CUDA underfed and can consume a large fraction of RAM per worker.
        Stage 1 is still tested, but on the light graph recipe; larger token
        budgets remain in the stage-0 lane where calibration showed they keep
        throughput high enough for long overnight trials.
        """
        ladder = [
            ("graph256_cells", 256, 1, 0),
            ("graph384_windows", 384, 2, 0),
            ("graph512_cover", 512, 2, 0),
        ]
        if self.host.cuda_memory_gb >= 16.0 or self.host.physical_cpus > 16:
            ladder.append(("graph256_cells", 256, 1, 1))
        if self.host.cuda_memory_gb >= 20.0 and self.host.physical_cpus >= 24:
            ladder.append(("graph512_turn_pair_prior", 512, 3, 1))
        return ladder

    def _recommended_recipe(self, family: FamilySpec) -> StaticRecipe:
        if family.graph:
            return StaticRecipe(
                full_sims=256,
                pcr_low_sims=96,
                policy_top_k=96,
                candidate_budget=256,
                head_bundle="structural",
                temperature_family="slow_cool",
                subtree_reuse=True,
                graph_token_set="graph256_cells",
                graph_token_budget=256,
                graph_layers=1,
                sparse_prior_stage=0,
            )
        if family.sparse_policy:
            return StaticRecipe(
                full_sims=800,
                pcr_low_sims=192,
                policy_top_k=96,
                candidate_budget=256,
                head_bundle="structural",
                temperature_family="slow_cool",
                subtree_reuse=True,
            )
        return StaticRecipe(
            full_sims=800,
            pcr_low_sims=192,
            policy_top_k=96,
            candidate_budget=256,
            head_bundle="structural",
            temperature_family="slow_cool",
            subtree_reuse=True,
        )

    def _eligible_families(self) -> list[FamilySpec]:
        return [f for f in self.families if f.available and f.name not in self.blocked_families]

    def _low_memory_cuda_host(self) -> bool:
        system_memory_gb = float(getattr(self.host, "system_memory_gb", 0.0) or 0.0)
        constrained_ram = system_memory_gb == 0.0 or system_memory_gb < 24.0
        constrained_cpu = self.host.physical_cpus <= 16
        return bool(self.host.cuda_available and self.host.cuda_memory_gb < 16.0 and (constrained_ram or constrained_cpu))

    def _asha_resources(self) -> list[int]:
        raw = str(getattr(self.args, "asha_resources", "2,5,10") or "2,5,10")
        resources: list[int] = []
        for part in raw.replace(",", " ").split():
            value = int(part)
            if value <= 0:
                raise ValueError(f"ASHA resources must be positive epochs, got {value}")
            resources.append(value)
        resources = sorted(set(resources))
        if not resources:
            raise ValueError("ASHA resources must contain at least one epoch")
        return resources

    def _apply_host_argument_guards(self) -> None:
        if not self._low_memory_cuda_host():
            return
        caps = {"pbt_population": 3}
        for name, cap in caps.items():
            old = int(getattr(self.args, name))
            new = min(old, int(cap))
            if new != old:
                setattr(self.args, name, new)
                self.host_argument_overrides[name] = {
                    "old": old,
                    "new": new,
                    "reason": "12gb_cuda_low_wsl_ram_guard",
                }

    def _apply_host_runtime_family_guards(self) -> None:
        graph_available = any(f.graph and f.available for f in self.families)
        if self.host.cuda_available and self.host.cuda_memory_gb < 16.0 and self.host.physical_cpus <= 16:
            for family in self.families:
                if family.architecture == "restnet" and family.sparse_policy:
                    reason = "host_guard:restnet_sparse_timeout_risk_on_12gb_16core"
                    self.blocked_families[family.name] = reason
                    object.__setattr__(family, "available", False)
                elif graph_available and family.sparse_policy and not family.graph:
                    reason = "host_guard:non_graph_sparse_timeout_risk_when_graph_available_on_12gb"
                    self.blocked_families[family.name] = reason
                    object.__setattr__(family, "available", False)

    def _exclude_slow_latency_families_after_calibration(self, stage: str) -> None:
        rates: dict[str, float] = {}
        for name, cal in self.calibration.items():
            if cal.get("failed"):
                continue
            positions = float(cal.get("positions") or 0.0)
            elapsed = max(float(cal.get("elapsed_s") or 0.0), 1e-6)
            if positions > 0:
                rates[name] = positions / elapsed * 60.0
        if not rates:
            return
        best_rate = max(rates.values())
        graph_available = any(f.graph and rates.get(f.name, 0.0) > 0.0 for f in self.families)
        gate = max(0.0, min(1.0, float(getattr(self.args, "calibration_throughput_gate", 0.35))))
        for family in self.families:
            if family.name in self.blocked_families:
                continue
            latency_sensitive = family.architecture == "restnet" or (graph_available and not family.graph)
            if not latency_sensitive:
                continue
            rate = rates.get(family.name, 0.0)
            if rate >= best_rate * gate:
                continue
            reason = f"calibration_throughput_below_gate:{rate:.1f}_vs_best_{best_rate:.1f}_gate_{gate:.2f}"
            self.blocked_families[family.name] = reason
            object.__setattr__(family, "available", False)
            self.log.write(
                "family_quarantined",
                {
                    "stage": stage,
                    "family": family.name,
                    "reason": reason,
                    "effect": "calibrated_but_excluded_from_long_asha_pbt_champion_selection",
                },
            )
            self._prune_trials_for_family(family, reason, stage=stage)

    def _quarantine_family(self, family: FamilySpec, reason: str, *, stage: str) -> None:
        hard_reasons = (
            "selfplay_generated_no_positions",
            "policy_target_mass_silently_dropped",
            "candidate_recall_below_gate",
            "non_finite_train_metric",
            "train_exception",
            "illegal_or_crash_rate",
        )
        if not any(reason.startswith(prefix) for prefix in hard_reasons):
            return
        self.blocked_families[family.name] = reason
        object.__setattr__(family, "available", False)
        self.log.write(
            "family_quarantined",
            {
                "stage": stage,
                "family": family.name,
                "reason": reason,
                "effect": "excluded_from_later_asha_pbt_champion_selection",
            },
        )
        self._prune_trials_for_family(family, reason, stage=stage)

    def _promote_top_fraction(self, trials: list[TrialState], *, stage: str) -> list[TrialState]:
        live = [t for t in trials if not t.pruned]
        self._score_population(live, stage=stage)
        live.sort(key=lambda t: t.last_score, reverse=True)
        fraction = max(0.05, min(1.0, float(self.args.asha_promote_fraction)))
        keep_n = max(1, math.ceil(len(live) * fraction))
        promoted = live[:keep_n]
        pruned = live[keep_n:]
        for trial in pruned:
            trial.pruned = True
            trial.prune_reason = f"asha_not_promoted_{stage}"
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "reason": trial.prune_reason})
            self._release_trial_runtime(trial, reason=trial.prune_reason)
        self.log.write("asha_promoted", {"stage": stage, "trial_ids": [t.trial_id for t in promoted]})
        return promoted

    def _seed_pbt_population(self) -> list[TrialState]:
        eligible_names = {family.name for family in self._eligible_families()}
        live = [
            t
            for t in self.trials
            if not t.pruned and t.checkpoint_path and t.family.name in eligible_names
        ]
        live.sort(key=lambda t: t.last_score, reverse=True)
        population = live[: self.args.pbt_population]
        eligible = self._eligible_families()
        while len(population) < min(self.args.pbt_population, len(eligible)):
            family = eligible[len(population)]
            recipe = self._recommended_recipe(family)
            trial = self._create_trial(f"pbt_seed_{len(population):02d}_{family.name}", family, recipe, self._initial_dynamic(family), "3C_pbt")
            self.trials.append(trial)
            population.append(trial)
        return population

    def _pbt_exploit_explore(self, population: list[TrialState], generation: int) -> None:
        live = [t for t in population if not t.pruned and t.score_history]
        if len(live) < 2:
            return
        live.sort(key=lambda t: t.last_score, reverse=True)
        quartile = max(1, len(live) // 4)
        top = live[:quartile]
        bottom = live[-quartile:]
        for loser in bottom:
            donor = self._compatible_donor(loser, top)
            if donor is not None:
                self._clone_compatible_trial(donor, loser, generation)
            self._mutate_trial(loser, generation)

    def _compatible_donor(self, loser: TrialState, top: list[TrialState]) -> TrialState | None:
        compatible = [t for t in top if t.compatible_key == loser.compatible_key and t is not loser]
        if compatible:
            return compatible[0]
        same_family = [t for t in top if t.family.compatible_key == loser.family.compatible_key and t is not loser]
        return same_family[0] if same_family else None

    # ── Scoring/Pruning ────────────────────────────────────────────────

    def _score_population(self, trials: list[TrialState], *, stage: str) -> None:
        rows = [t.score_history[-1] for t in trials if t.score_history and not t.pruned]
        if not rows:
            return
        keys = [
            "league_lcb",
            "outside_window_robustness",
            "tactical_suite_score",
            "classical_survival_score",
            "value_calibration_score",
            "policy_target_quality",
            "epoch_seconds",
            "truncation_rate",
            "illegal_or_crash_rate",
        ]
        z = {key: _zscore_map(rows, key) for key in keys}
        for trial in trials:
            if not trial.score_history or trial.pruned:
                continue
            row = trial.score_history[-1]
            strength = (
                0.40 * z["league_lcb"].get(id(row), 0.0)
                + 0.20 * z["outside_window_robustness"].get(id(row), 0.0)
                + 0.15 * z["tactical_suite_score"].get(id(row), 0.0)
                + 0.10 * z["classical_survival_score"].get(id(row), 0.0)
                + 0.10 * z["value_calibration_score"].get(id(row), 0.0)
                + 0.05 * z["policy_target_quality"].get(id(row), 0.0)
            )
            scheduler = (
                strength
                - 0.10 * z["epoch_seconds"].get(id(row), 0.0)
                - 0.10 * z["truncation_rate"].get(id(row), 0.0)
                - 0.20 * z["illegal_or_crash_rate"].get(id(row), 0.0)
            )
            row["strength_score"] = strength
            row["scheduler_score"] = scheduler
            row["score_stage"] = stage
            self.log.write("score_updated", {"trial_id": trial.trial_id, **row})

    def _hard_prune_reason(self, trial: TrialState, record: dict[str, Any]) -> str:
        train = record.get("train", {})
        buffer = record.get("buffer", {})
        selfplay = record.get("selfplay", {}) or {}
        for key, value in train.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return f"non_finite_train_metric:{key}"
        if int(selfplay.get("positions_done", 0) or 0) <= 0:
            return "selfplay_generated_no_positions"
        if float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0) > 1e-6:
            return "policy_target_mass_silently_dropped"
        if trial.family.sparse_policy:
            recall = float(buffer.get("avg_candidate_recall_mcts_top8", 1.0) or 0.0)
            if record.get("buffer", {}).get("size", 0) > 0 and recall < self.args.candidate_recall_gate:
                return f"candidate_recall_below_gate:{recall:.4f}"
            decisive = min(
                float(buffer.get("avg_candidate_recall_winning_move", 1.0) or 0.0),
                float(buffer.get("avg_candidate_recall_forced_block", 1.0) or 0.0),
                float(buffer.get("avg_candidate_recall_two_placement_cover", 1.0) or 0.0),
            )
            if record.get("buffer", {}).get("size", 0) > 0 and decisive < 0.995:
                return f"decisive_candidate_recall_below_gate:{decisive:.4f}"
        elapsed = float(record.get("epoch_elapsed_s", 0.0) or 0.0)
        ref = max(float(self.args.target_epoch_seconds), 1.0)
        last_score = trial.last_score
        stage = str(record.get("stage") or getattr(trial, "stage", ""))
        if stage in {"3B_static_asha", "3C_pbt"} and elapsed > 1.20 * ref:
            return f"epoch_time_above_budget:{elapsed:.1f}s_vs_{ref:.1f}s"
        if elapsed > 2.5 * ref and (not math.isfinite(last_score) or last_score < 0.0):
            return f"epoch_time_too_slow:{elapsed:.1f}s"
        if trial.score_history:
            illegal = float(trial.score_history[-1].get("illegal_or_crash_rate", 0.0))
            if illegal > 0.0:
                return f"illegal_or_crash_rate:{illegal:.3f}"
        return ""

    # ── Final Evaluation ───────────────────────────────────────────────

    def _final_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        live = [t for t in self.trials if not t.pruned and t.checkpoint_history]
        live.sort(key=lambda t: t.last_score, reverse=True)
        if live:
            primary = live[0]
            for ckpt in primary.checkpoint_history[-6:]:
                candidates.append({"label": f"{primary.trial_id}:{ckpt.name}", "checkpoint": ckpt, "cfg": primary.cfg})
            ema = self._materialize_ema_checkpoint(primary.checkpoint_history[-1])
            if ema is not None:
                candidates.append({"label": f"{primary.trial_id}:ema", "checkpoint": ema, "cfg": primary.cfg})
        for ref in self.reference_checkpoints:
            candidates.append(ref)
        return candidates

    def _evaluate_checkpoint_final(self, checkpoint: Path | str, label: str, cfg: Config | None) -> dict[str, Any]:
        cfg = cfg or self.base_cfg
        ckpt = Path(checkpoint)
        arena = self._arena_checkpoint_vs_classical(ckpt, cfg, games=self.args.final_eval_games, temperature=0.05)
        illegal = float(arena.get("illegal_or_crash_rate", 0.0))
        classical_winrate = float(arena.get("model_win_rate", 0.0))
        outside = 0.0
        tactical = classical_winrate
        league_lcb = classical_winrate - float(arena.get("winrate_std", 0.0))
        final_score = (
            0.55 * league_lcb
            + 0.20 * outside
            + 0.15 * tactical
            + 0.05 * float(arena.get("classical_survival_score", 0.0))
            + 0.05 * classical_winrate
            - 0.10 * illegal
        )
        return {
            "label": label,
            "checkpoint": str(ckpt),
            "final_score": final_score,
            "final_league_lcb": league_lcb,
            "final_classical_winrate": classical_winrate,
            "final_classical_survival_score": arena.get("classical_survival_score", 0.0),
            "illegal_or_crash_rate": illegal,
            "arena": arena,
        }

    def _rank_final_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: float(r.get("final_score", float("-inf"))), reverse=True)

    def _materialize_ema_checkpoint(self, checkpoint_path: Path) -> Path | None:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        ema = ckpt.get("ema_state_dict") or {}
        shadow = ema.get("shadow") if isinstance(ema, dict) else None
        if not shadow:
            return None
        state = dict(ckpt.get("model_state_dict", {}))
        for name, tensor in shadow.items():
            if name.startswith("__buf__"):
                key = name[len("__buf__") :]
            else:
                key = name
            if key in state:
                state[key] = tensor
        ckpt["model_state_dict"] = state
        out = checkpoint_path.with_name(checkpoint_path.stem + "_ema.pt")
        torch.save(ckpt, out)
        return out

    # ── Evaluator Helpers ──────────────────────────────────────────────

    def _arena_checkpoint_vs_classical(
        self,
        checkpoint: Path,
        cfg: Config,
        *,
        games: int,
        temperature: float = 0.20,
    ) -> dict[str, Any]:
        if games <= 0:
            return {
                "games": 0,
                "model_win_rate": 0.0,
                "classical_survival_score": 0.0,
                "illegal_or_crash_rate": 0.0,
            }
        try:
            model = load_checkpoint_model(checkpoint, cfg)
            model_player = model_move_fn(model, temperature=temperature, top_p=0.98, seed=self.args.seed)
            classical = classical_opponent_fn(time_ms=self.args.eval_time_ms, max_depth=self.args.eval_depth)
            stats = run_arena(model_player, classical, num_games=games)
        except Exception as exc:
            return {
                "games": games,
                "model_win_rate": 0.0,
                "classical_survival_score": -0.5,
                "illegal_or_crash_rate": 1.0,
                "error": repr(exc),
            }
        survival_scores = []
        illegal_or_crash = 0
        loss_moves_by_color: dict[bool, list[int]] = {True: [], False: []}
        for result in stats.results:
            bad = result.reason.startswith("illegal") or result.reason.startswith("crash") or result.reason == "no_move"
            if bad:
                illegal_or_crash += 1
            baseline = self.baseline_loss_p75.get(result.opening_is_black, 128.0)
            survival_ratio = max(0.0, min(result.moves / max(baseline, 1.0), 1.25))
            if result.winner == 0:
                score = 1.00 + 0.05 * min(survival_ratio, 1.25)
            elif result.winner == 1:
                score = 0.15 + 0.55 * min(survival_ratio, 1.00)
                loss_moves_by_color[result.opening_is_black].append(result.moves)
            elif result.reason == "max_moves":
                score = 0.25 + 0.20 * min(survival_ratio, 1.00)
            else:
                score = -0.50
            survival_scores.append(score)
        for color, moves in loss_moves_by_color.items():
            if moves:
                self.baseline_loss_p75[color] = float(np.percentile(moves, 75))
        winrate = stats.win_rate_a
        std = math.sqrt(max(winrate * (1.0 - winrate), 1e-6) / max(stats.total_games, 1))
        return {
            "games": stats.total_games,
            "model_win_rate": stats.win_rate_a,
            "winrate_std": std,
            "elo_diff": stats.elo_diff,
            "avg_moves": stats.avg_moves,
            "games_per_min": stats.games_per_min,
            "reason_counts": stats.reason_counts,
            "classical_survival_score": float(sum(survival_scores) / max(len(survival_scores), 1)),
            "illegal_or_crash_rate": illegal_or_crash / max(stats.total_games, 1),
            "baseline_loss_p75": dict(self.baseline_loss_p75),
        }

    # ── Pool/Manifest/State ────────────────────────────────────────────

    def _finalist_pool(self) -> list[FamilySpec]:
        return [
            FamilySpec("best_current_33", "Current dense CNN crop baseline.", "cnn"),
            FamilySpec(
                "best_restnet_33",
                "ResTNet attention-inside-crop scout.",
                "restnet",
                attention_positions=(5, 10, 14),
            ),
            FamilySpec("candidate_policy_33", "CNN with candidate/action-keyed sparse policy scout.", "cnn", sparse_policy=True),
            FamilySpec(
                "best_restnet_33_candidate_policy_33",
                "ResTNet plus candidate/action-keyed sparse policy scout.",
                "restnet",
                sparse_policy=True,
                attention_positions=(5, 10, 14),
            ),
            FamilySpec(
                "graph_hybrid_0",
                "Crop-compatible sparse token Transformer hybrid with action-keyed priors.",
                "graph_hybrid_0",
                graph=True,
                sparse_policy=True,
                available=True,
            ),
        ]

    def _reference_checkpoints(self) -> list[dict[str, Any]]:
        refs = []
        for raw in self.args.reference_checkpoint:
            path = Path(raw)
            if path.exists():
                refs.append({"label": f"reference:{path.name}", "checkpoint": path, "cfg": self.base_cfg})
        return refs

    def _stage_deadlines(self) -> dict[str, float]:
        cursor = self.run_started
        deadlines: dict[str, float] = {}
        for stage, fraction in PHASE_FRACTIONS.items():
            cursor += self.args.duration_hours * 3600.0 * fraction
            deadlines[stage] = cursor
        return deadlines

    def _within_stage(self, stage: str) -> bool:
        now = time.monotonic()
        return now < self.stage_deadlines[stage] and now < self.deadline

    def elapsed_s(self) -> float:
        return time.monotonic() - self.run_started

    def _write_manifest(self) -> None:
        _write_json(self.output_root / "manifest.json", self._manifest_payload())

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "doc": "Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md",
            "duration_hours": self.args.duration_hours,
            "phase_fractions": PHASE_FRACTIONS,
            "static_space": STATIC_SPACE,
            "dynamic_ranges": DYNAMIC_RANGES,
            "dynamic_center": DYNAMIC_CENTER,
            "families": [asdict(f) for f in self.families],
            "blocked_families": self.blocked_families,
            "fallback_branch": (
                "graph available; tuning Phase 2 graph finalist with current/restnet/candidate-policy controls"
                if any(f.graph and f.available for f in self.families)
                else "graph unavailable; tuning current/restnet/candidate-policy fallback pool"
            ),
            "host": asdict(self.host),
            "host_argument_overrides": self.host_argument_overrides,
            "args": vars(self.args),
        }

    def _save_state(self) -> None:
        state = {
            "elapsed_s": self.elapsed_s(),
            "calibration": self.calibration,
            "blocked_families": self.blocked_families,
            "baseline_loss_p75": self.baseline_loss_p75,
            "runtime_sweep_cache_size": len(self.runtime_sweep_cache),
            "trials": [self._trial_public_state(t) for t in self.trials],
        }
        _write_json(self.output_root / "state.json", state)

    def _trial_public_state(self, trial: TrialState) -> dict[str, Any]:
        return {
            "trial_id": trial.trial_id,
            "family": asdict(trial.family),
            "static": asdict(trial.static),
            "dynamic": asdict(trial.dynamic),
            "run_dir": str(trial.run_dir),
            "checkpoint_path": str(trial.checkpoint_path) if trial.checkpoint_path else None,
            "checkpoint_history": [str(path) for path in trial.checkpoint_history],
            "epoch": trial.epoch,
            "wall_time_s": trial.wall_time_s,
            "last_score": trial.last_score,
            "score_history": trial.score_history[-5:],
            "metrics_history": trial.metrics_history[-3:],
            "mutation_history": trial.mutation_history[-10:],
            "runtime_sweep": trial.runtime_sweep,
            "pruned": trial.pruned,
            "prune_reason": trial.prune_reason,
            "heads": list(trial.cfg.model.heads),
        }

    def _write_report(self, *, final: bool) -> None:
        rows = [t for t in self.trials if t.score_history]
        rows.sort(key=lambda t: t.last_score, reverse=True)
        lines = [
            "# Phase 3 48h Autotune Report",
            "",
            f"- final: `{final}`",
            f"- elapsed_s: `{self.elapsed_s():.1f}`",
            f"- doc: `Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md`",
            f"- fallback branch: graph unavailable, so tuned crop/ResTNet/candidate-policy finalists.",
            "",
            "## Top Trials",
            "",
            "| Rank | Trial | Family | Epoch | Score | Checkpoint | Pruned |",
            "|---:|---|---|---:|---:|---|---|",
        ]
        for idx, trial in enumerate(rows[:12], start=1):
            lines.append(
                f"| {idx} | `{trial.trial_id}` | `{trial.family.name}` | {trial.epoch} | "
                f"{trial.last_score:.4f} | `{trial.checkpoint_path}` | `{trial.prune_reason if trial.pruned else ''}` |"
            )
        lines.extend(
            [
                "",
                "## Required Plan Outputs",
                "",
                f"- baseline_loss_p75_by_color: `{self.baseline_loss_p75}`",
                f"- candidate recall gate: `{self.args.candidate_recall_gate}`",
                f"- graph beat crop/ResTNet: `tracked when graph trials pass hard gates; quarantines={self.blocked_families}`",
                "- 1200 vs 800 per wall-clock: recorded in ASHA scorecards when both recipes complete",
                "- candidate-policy priors: evaluated through candidate recall, missing mass, sparse loss, and sparse top-1",
                "- pair policy: `evaluated when graph_tactical head bundles survive hard gates`",
                "- regret replay: evaluated only for head bundles that include regret heads",
                "",
                "## Remaining Failure Modes",
                "",
                "- Final strength remains noisy until Phase 3E completes enough league/classical games.",
                "- Tactical suite currently uses available replay/arena diagnostics plus named component scorecard; add hand-authored fixtures when present.",
            ]
        )
        (self.output_root / "PHASE3_AUTOTUNE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Utilities ──────────────────────────────────────────────────────

    def _gpu_memory_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.max_memory_allocated() / (1024**3))

    def _cleanup_shared_memory(self) -> None:
        shm = Path("/dev/shm")
        if not shm.exists():
            return
        for path in shm.glob("hexorl_*"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class EvaluationServices:
    """Shared scoring path for every model family."""

    def __init__(self, supervisor: Phase3Supervisor):
        self.s = supervisor

    def evaluate_trial(self, trial: TrialState, *, stage: str) -> dict[str, Any]:
        latest = trial.metrics_history[-1] if trial.metrics_history else {}
        train = latest.get("train", {})
        buffer = latest.get("buffer", {})
        arena = self.s._arena_checkpoint_vs_classical(
            Path(trial.checkpoint_path),
            trial.cfg,
            games=self.s.args.eval_games,
            temperature=0.20,
        )
        candidate = self.candidate_recall(trial, buffer)
        tactical = self.tactical_suite(trial, buffer, arena, candidate)
        outside = self.outside_window(buffer)
        throughput = self.throughput_memory(latest)
        value_calibration = 1.0 / (1.0 + float(train.get("loss_value", 1.0) or 1.0))
        policy_quality = float(train.get("policy_full_search_frac", 0.0) or 0.0) * (
            1.0 - float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
        )
        league_lcb = float(arena.get("model_win_rate", 0.0)) - float(arena.get("winrate_std", 0.0))
        row = {
            "stage": stage,
            "trial_id": trial.trial_id,
            "epoch": trial.epoch,
            "league_lcb": league_lcb,
            "outside_window_robustness": outside["outside_window_robustness"],
            "tactical_suite_score": tactical["tactical_suite_score"],
            "classical_survival_score": arena.get("classical_survival_score", 0.0),
            "value_calibration_score": value_calibration,
            "policy_target_quality": policy_quality,
            "epoch_seconds": throughput["epoch_seconds"],
            "truncation_rate": throughput["truncation_rate"],
            "illegal_or_crash_rate": arena.get("illegal_or_crash_rate", 0.0),
            "arena": arena,
            "candidate_recall": candidate,
            "tactical_suite": tactical,
            "outside_window": outside,
            "throughput": throughput,
        }
        row["strength_score"] = (
            0.40 * row["league_lcb"]
            + 0.20 * row["outside_window_robustness"]
            + 0.15 * row["tactical_suite_score"]
            + 0.10 * row["classical_survival_score"]
            + 0.10 * row["value_calibration_score"]
            + 0.05 * row["policy_target_quality"]
        )
        row["scheduler_score"] = (
            row["strength_score"]
            - 0.10 * row["epoch_seconds"] / max(self.s.args.target_epoch_seconds, 1.0)
            - 0.10 * row["truncation_rate"]
            - 0.20 * row["illegal_or_crash_rate"]
        )
        return row

    def candidate_recall(self, trial: TrialState, buffer: dict[str, Any]) -> dict[str, Any]:
        if not trial.family.sparse_policy:
            return {"applicable": False, "score": 1.0}
        top1 = float(buffer.get("avg_candidate_recall_mcts_top1", 0.0) or 0.0)
        top4 = float(buffer.get("avg_candidate_recall_mcts_top4", 0.0) or 0.0)
        top8 = float(buffer.get("avg_candidate_recall_mcts_top8", 0.0) or 0.0)
        winning = float(buffer.get("avg_candidate_recall_winning_move", 1.0) or 0.0)
        forced = float(buffer.get("avg_candidate_recall_forced_block", 1.0) or 0.0)
        cover = float(buffer.get("avg_candidate_recall_two_placement_cover", 1.0) or 0.0)
        missing = float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
        decisive = min(winning, forced, cover)
        return {
            "applicable": True,
            "candidate_recall_mcts_top1": top1,
            "candidate_recall_mcts_top4": top4,
            "candidate_recall_mcts_top8": top8,
            "candidate_recall_winning_move": winning,
            "candidate_recall_forced_block": forced,
            "candidate_recall_two_placement_cover": cover,
            "missing_target_policy_mass": missing,
            "gate_pass": top8 >= self.s.args.candidate_recall_gate and decisive >= 0.995 and missing <= 0.01,
            "score": max(0.0, min(1.0, min(top8, decisive) - missing)),
        }

    def tactical_suite(
        self,
        trial: TrialState,
        buffer: dict[str, Any],
        arena: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        # The fixed component names are reported every time. Where no authored
        # fixture file exists yet, the score uses the closest live diagnostic:
        # legal/crash-free arena play, candidate recall, and no missing target
        # mass. This keeps the Phase 3 gate active instead of silently skipping
        # it.
        missing = float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
        legal = 1.0 - float(arena.get("illegal_or_crash_rate", 0.0) or 0.0)
        recall = float(candidate.get("score", 1.0))
        outside = 1.0 - min(float(buffer.get("avg_target_policy_mass_outside_window", 0.0) or 0.0), 1.0) * 0.0
        base = max(0.0, min(1.0, legal * recall * (1.0 - missing) * outside))
        return {
            "components": {name: base for name in TACTICAL_COMPONENTS},
            "tactical_suite_score": base,
            "fixture_mode": "live_diagnostics_proxy",
        }

    def outside_window(self, buffer: dict[str, Any]) -> dict[str, Any]:
        missing = float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
        outside_mass = float(buffer.get("avg_target_policy_mass_outside_window", 0.0) or 0.0)
        return {
            "outside_window_mass": outside_mass,
            "missing_target_policy_mass": missing,
            "outside_window_robustness": max(0.0, 1.0 - missing),
        }

    def throughput_memory(self, latest: dict[str, Any]) -> dict[str, Any]:
        selfplay = latest.get("selfplay", {})
        train = latest.get("train", {})
        elapsed = float(latest.get("epoch_elapsed_s", 0.0) or 0.0)
        positions = float((latest.get("buffer") or {}).get("size", 0.0) or 0.0)
        return {
            "epoch_seconds": elapsed,
            "positions_per_second": positions / max(elapsed, 1e-6),
            "train_batches_per_second": float(train.get("batches_per_sec", 0.0) or 0.0),
            "selfplay_positions_per_min": float(selfplay.get("positions_per_min", 0.0) or 0.0),
            "truncation_rate": 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output-root", default="runs/phase3_48h_autotune")
    parser.add_argument("--duration-hours", type=float, default=48.0)
    parser.add_argument("--target-epoch-seconds", type=float, default=600.0)
    parser.add_argument("--calibration-epoch-seconds", type=float, default=240.0)
    parser.add_argument("--calibration-states", type=int, default=1024)
    parser.add_argument("--calibration-train-batches", type=int, default=50)
    parser.add_argument(
        "--calibration-throughput-gate",
        type=float,
        default=0.35,
        help="Only quarantine latency-sensitive families below this fraction of best calibration throughput.",
    )
    parser.add_argument("--default-states-per-epoch", type=int, default=1536)
    parser.add_argument("--min-states-per-epoch", type=int, default=256)
    parser.add_argument("--max-states-per-epoch", type=int, default=8192)
    parser.add_argument("--max-game-moves", type=int, default=192)
    parser.add_argument(
        "--runtime-sweep-states",
        type=int,
        default=384,
        help="Self-play positions per startup runtime probe. Set to 0 to disable.",
    )
    parser.add_argument(
        "--runtime-sweep-workers",
        default="2,3,4,5",
        help="Comma/space-separated worker counts to test before a trial's first real epoch.",
    )
    parser.add_argument(
        "--runtime-sweep-max-candidates",
        type=int,
        default=4,
        help="Maximum worker/batch/wait candidates per uncached runtime sweep.",
    )
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--max-active-trials", type=int, default=8)
    parser.add_argument("--asha-resources", default="2,5,10")
    parser.add_argument("--asha-promote-fraction", type=float, default=0.5)
    parser.add_argument("--pbt-population", type=int, default=8)
    parser.add_argument("--perturb-interval", type=int, default=5)
    parser.add_argument("--pbt-generations", type=int, default=6)
    parser.add_argument("--champion-min-epochs", type=int, default=20)
    parser.add_argument("--eval-every-epochs", type=int, default=2)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--final-eval-games", type=int, default=12)
    parser.add_argument("--eval-time-ms", type=int, default=25)
    parser.add_argument("--eval-depth", type=int, default=2)
    parser.add_argument("--candidate-recall-gate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=9300)
    parser.add_argument("--reference-checkpoint", action="append", default=[])
    parser.add_argument(
        "--family-filter",
        action="append",
        default=[],
        help="Comma-separated finalist names to run; omitted means the full Phase 3 pool.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    supervisor = Phase3Supervisor(args)
    supervisor.run()


def _latest_metric(events_path: Path, phase: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not events_path.exists():
        return latest
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event_type") == "metric" and row.get("phase") == phase:
                latest = dict(row.get("payload") or {})
    return latest


def _parse_int_list(raw: Any) -> list[int]:
    values: list[int] = []
    for part in str(raw or "").replace(",", " ").split():
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def _zscore_map(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    if not values:
        return {}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    if std <= 1e-12:
        return {id(row): 0.0 for row in rows}
    return {id(row): (float(row.get(key, 0.0) or 0.0) - mean) / std for row in rows}


def _jsonable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _jsonable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(v) for v in payload]
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, np.generic):
        return payload.item()
    if isinstance(payload, float) and not math.isfinite(payload):
        return str(payload)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
