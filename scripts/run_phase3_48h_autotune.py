"""Phase 3 48-hour autotuning supervisor.

This implements the plan in Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md:

* Phase 3A finalist import/calibration.
* Phase 3B ASHA/BOHB static narrowing.
* Phase 3C PB2 schedule search, with explicit PBT fallback when requested.
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
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from hexorl.buffer.ring import RingBuffer, replay_feature_flags
from hexorl.config import Config, load_config
from hexorl.dashboard.recorder import RunRecorder
from hexorl.epoch import run_epoch
from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.eval.checkpoint_league import CheckpointLeague
from hexorl.eval.tactical_suite import evaluate_tactical_suite
from hexorl.models.registry import (
    architecture_display_summary,
    global_graph_architecture_ids,
    is_global_graph_architecture as registry_is_global_graph_architecture,
    resolve_model_spec,
)
from hexorl.runtime import autotune_config, configure_torch_runtime, detect_host
from hexorl.selfplay.records import BOARD_AREA
from hexorl.tuning import (
    ASHARungTable,
    BOHBSampler,
    PB2Observation,
    PB2Scheduler,
    SearchSpace,
    TrialObservation,
)


LOGGER = logging.getLogger("phase3_autotune")
FULL_GLOBAL_POLICY_ROWS = BOARD_AREA
REPLAY_POLICY_WIDTH_CAP = 512
# Stage 3 deliberately scouts the four pre-champion global graph candidates.
# Membership and capability authority still come from hexorl.models.registry.
GLOBAL_GRAPH_SCOUT_FAMILIES = (
    "global_xattn_0",
    "global_line_window_0",
    "global_pair_twostage_0",
    "global_graph_full_0",
)
_GLOBAL_GRAPH_SCOUT_FAMILY_SET = frozenset(GLOBAL_GRAPH_SCOUT_FAMILIES)
_GLOBAL_GRAPH_REGISTRY_IDS = frozenset(global_graph_architecture_ids())
if not _GLOBAL_GRAPH_SCOUT_FAMILY_SET <= _GLOBAL_GRAPH_REGISTRY_IDS:
    missing = sorted(_GLOBAL_GRAPH_SCOUT_FAMILY_SET - _GLOBAL_GRAPH_REGISTRY_IDS)
    raise RuntimeError(f"global graph scout families missing from model registry: {missing}")
GLOBAL_GRAPH_PAIR_HEADS = {"policy_pair_first", "policy_pair_second", "policy_pair_joint"}
LOW_MEMORY_GLOBAL_GRAPH_MAX_SIMS = 128
LOW_MEMORY_GLOBAL_GRAPH_RUNTIME_SWEEP_TIMEOUT_S = 240.0


class RuntimeSweepTimeout(TimeoutError):
    pass


def _short_items(items: Iterable[Any], *, limit: int = 8) -> str:
    values = [str(item) for item in items]
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", +{len(values) - limit} more"


def _candidate_log_summary(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return "{}"
    fields = []
    for key in ("workers", "batch_size_per_worker", "max_batch_size", "max_wait_us"):
        if key in candidate:
            fields.append(f"{key}={candidate[key]}")
    return "{" + ", ".join(fields) + "}"


def _float_text(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return str(number)
    return f"{number:.{digits}f}"


def _event_log_summary(event: str, row: dict[str, Any]) -> str:
    try:
        if event == "run_start":
            families = [
                item.get("name", "?")
                for item in row.get("families", [])
                if isinstance(item, dict)
            ]
            args = row.get("args", {}) if isinstance(row.get("args"), dict) else {}
            return (
                "Autotune run start: "
                f"scope=[{_short_items(families)}], "
                f"max_game_moves={args.get('max_game_moves')}, "
                f"asha_resources={args.get('asha_resources')}, "
                f"reference_checkpoints={len(args.get('reference_checkpoint') or [])}, "
                f"fallback='{row.get('fallback_branch', '')}'"
            )
        if event == "stage_start":
            return f"Stage start: stage={row.get('stage')} details={{{', '.join(k + '=' + str(v) for k, v in row.items() if k not in {'time', 'event', 'stage'})}}}"
        if event == "trial_created":
            family = row.get("family", {})
            if isinstance(family, dict):
                family_name = family.get("name")
                architecture = family.get("architecture")
            else:
                family_name = family
                architecture = "?"
            static = row.get("static", {}) if isinstance(row.get("static"), dict) else {}
            contract = row.get("model_contract", {}) if isinstance(row.get("model_contract"), dict) else {}
            pair_strategy = row.get("pair_strategy", {}) if isinstance(row.get("pair_strategy"), dict) else {}
            return (
                "Trial created: "
                f"id={row.get('trial_id')} stage={row.get('stage')} "
                f"family={family_name} arch={architecture} "
                f"heads=[{_short_items(row.get('heads', []), limit=12)}] "
                f"runtime_outputs=[{_short_items(contract.get('outputs', []), limit=12)}] "
                f"pair_capabilities=[{_short_items(contract.get('pair_capabilities', []), limit=6)}] "
                f"pair_strategy={pair_strategy.get('strategy')} max_pairs={pair_strategy.get('max_pairs')} "
                f"sims={static.get('full_sims')} pcr_low={static.get('pcr_low_sims')} "
                f"graph_tokens={static.get('graph_token_budget')} graph_layers={static.get('graph_layers')} "
                f"candidate_budget={static.get('candidate_budget')}"
            )
        if event == "calibration_trial_start":
            return f"Calibration trial start: trial={row.get('trial_id')} family={row.get('family')}"
        if event == "runtime_sweep_start":
            candidates = row.get("candidates", [])
            return (
                "Runtime sweep start: "
                f"trial={row.get('trial_id')} stage={row.get('stage')} states={row.get('states')} "
                f"candidates=[{_short_items((_candidate_log_summary(c) for c in candidates), limit=4)}]"
            )
        if event == "runtime_sweep_result":
            selfplay = row.get("selfplay", {}) if isinstance(row.get("selfplay"), dict) else {}
            memory = row.get("memory", {}) if isinstance(row.get("memory"), dict) else {}
            gpu_after = row.get("gpu_after", {}) if isinstance(row.get("gpu_after"), dict) else {}
            return (
                "Runtime sweep result: "
                f"trial={row.get('trial_id')} stage={row.get('stage')} ok={row.get('ok')} "
                f"candidate={_candidate_log_summary(row.get('candidate'))} "
                f"positions={row.get('positions')} pos_per_min={_float_text(row.get('positions_per_min'))} "
                f"elapsed_s={_float_text(row.get('elapsed_s'))} score={_float_text(row.get('score'))} "
                f"games={selfplay.get('games_done')} trunc_rate={_float_text(selfplay.get('truncation_rate'))} "
                f"max_move_games={selfplay.get('terminal_reason_max_game_moves')} "
                f"gpu_util={gpu_after.get('gpu_util_pct')} gpu_mem={gpu_after.get('memory_used_mb')}/{gpu_after.get('memory_total_mb')}MiB "
                f"mem_min_available_gb={_float_text(memory.get('min_available_gb'))} unsafe_memory={memory.get('unsafe')} "
                f"error={row.get('error', '')}"
            )
        if event == "runtime_sweep_selected":
            return (
                "Runtime sweep selected: "
                f"trial={row.get('trial_id')} stage={row.get('stage')} "
                f"selected={_candidate_log_summary(row.get('selected'))} "
                f"positions_per_min={_float_text(row.get('selected_positions_per_min'))} "
                f"candidates={row.get('candidate_count')} cached={row.get('cached')}"
            )
        if event == "trial_epoch_complete":
            train = row.get("train", {}) if isinstance(row.get("train"), dict) else {}
            selfplay = row.get("selfplay", {}) if isinstance(row.get("selfplay"), dict) else {}
            buffer = row.get("buffer", {}) if isinstance(row.get("buffer"), dict) else {}
            pair_strategy = row.get("pair_strategy", {}) if isinstance(row.get("pair_strategy"), dict) else {}
            losses = {
                key: train.get(key)
                for key in (
                    "loss_total",
                    "total_loss",
                    "policy_loss",
                    "value_loss",
                    "sparse_policy_loss",
                    "policy_pair_first_loss",
                    "policy_pair_joint_loss",
                    "policy_pair_second_loss",
                )
                if key in train
            }
            return (
                "Trial epoch complete: "
                f"trial={row.get('trial_id')} family={row.get('family')} epoch={row.get('epoch')} "
                f"heads=[{_short_items(row.get('heads', []), limit=16)}] "
                f"runtime_outputs=[{_short_items(row.get('runtime_outputs', []), limit=16)}] "
                f"pair_capabilities=[{_short_items(row.get('pair_capabilities', []), limit=8)}] "
                f"pair_strategy={pair_strategy.get('strategy')} max_pairs={pair_strategy.get('max_pairs')} "
                f"elapsed_s={_float_text(row.get('epoch_elapsed_s'))} checkpoint={row.get('checkpoint_path')} "
                f"selfplay_positions={selfplay.get('positions_done')} games={selfplay.get('games_done')} "
                f"trunc_rate={_float_text(selfplay.get('truncation_rate'))} buffer_size={buffer.get('size')} "
                f"throughput_pos_min={_float_text(selfplay.get('positions_per_min'))} "
                f"target_mass_missing={_float_text(buffer.get('avg_missing_target_policy_mass'))} "
                f"target_mass_outside={_float_text(buffer.get('avg_target_policy_mass_outside_window'))} "
                f"losses={losses}"
            )
        if event == "trial_pruned":
            return f"Trial pruned: trial={row.get('trial_id')} stage={row.get('stage')} reason={row.get('reason')}"
        if event == "trial_evaluated" or event == "score_updated":
            return (
                f"{event}: trial={row.get('trial_id')} stage={row.get('stage') or row.get('score_stage')} "
                f"scheduler_score={_float_text(row.get('scheduler_score'))} "
                f"score_mode={row.get('score_mode', '')}"
            )
        if event in {"family_quarantined", "family_throughput_below_gate", "runtime_sweep_failed", "run_failed"}:
            return f"{event}: " + json.dumps({k: v for k, v in row.items() if k not in {"time", "event"}}, sort_keys=True)
    except Exception as exc:
        return f"{event}: summary_failed={type(exc).__name__}:{exc}"
    return ""


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
    ],
}


STATIC_SPACE = {
    "full_sims": [512, 800, 1200, 1600],
    "pcr_low_sims": [128, 192, 256, 384],
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
    "3C_schedule_search": 16.0 / 48.0,
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
    global_graph: bool = False
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
            self.global_graph,
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
    train_batch_size: int = 256


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
        summary = _event_log_summary(event, row)
        if summary:
            LOGGER.info(summary)


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
        asha_resources = tuple(self._asha_resources())
        self.asha_table = ASHARungTable(
            resources=asha_resources,
            promotion_fraction=float(args.asha_promote_fraction),
        )
        self.bohb_sampler = BOHBSampler(
            self._bohb_search_space(),
            min_resource=min(asha_resources),
            max_resource=max(asha_resources),
            eta=2,
            warmup_points=max(3, min(6, int(args.max_active_trials))),
            random_fraction=float(args.bohb_random_fraction),
        )
        self.pb2_scheduler = PB2Scheduler(
            {name: tuple(bounds) for name, bounds in DYNAMIC_RANGES.items()},
            uncertainty_weight=float(args.pb2_uncertainty_weight),
            parameter_conditions={
                "sparse_policy_loss": {"key": "sparse_policy", "values": [True]},
                "pair_policy_loss": {"key": "pair_policy", "values": [True]},
                "graph_aux_multiplier": {"key": "graph", "values": [True]},
                "regret_fraction": {"key": "regret_heads", "values": [True]},
            },
        )
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
            self.phase_3c_schedule_search()
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
            while self._within_stage(stage):
                progressed = False
                for trial in current:
                    if trial.epoch >= resource or trial.pruned:
                        continue
                    self._train_trial_epoch(
                        trial,
                        stage=stage,
                        target_epoch_seconds=self.args.target_epoch_seconds,
                    )
                    progressed = True
                if not progressed:
                    break
            for trial in current:
                if not trial.pruned and trial.epoch >= resource:
                    self._evaluate_trial(trial, stage=stage, force=True)
            self._score_population(current, stage=stage)
            self._record_asha_rung(current, resource)
            decision = self.asha_table.decision_for(resource)
            self.asha_table.save(self.output_root / "asha_rungs.json")
            current = self._apply_asha_decision(current, decision, stage=stage)
        self._save_state()

    def phase_3c_schedule_search(self) -> None:
        deadline_stage = "3C_schedule_search"
        stage = "3C_pb2" if self.args.schedule_method == "pb2" else "3C_pbt_fallback"
        self.log.write("stage_start", {"stage": stage, "schedule_method": self.args.schedule_method})
        population = self._seed_pbt_population()
        generation = 0
        while self._within_stage(deadline_stage) and generation < self.args.pbt_generations and population:
            generation += 1
            self.log.write(
                "schedule_generation_start",
                {
                    "generation": generation,
                    "source_method": self.args.schedule_method,
                    "population": [t.trial_id for t in population],
                },
            )
            for trial in population:
                if trial.pruned:
                    continue
                for _ in range(self.args.perturb_interval):
                    if trial.pruned or not self._within_stage(deadline_stage):
                        break
                    self._train_trial_epoch(trial, stage=stage, target_epoch_seconds=self.args.target_epoch_seconds)
                if not trial.pruned:
                    self._evaluate_trial(trial, stage=stage, force=True)
            self._score_population(population, stage=stage)
            if self.args.schedule_method == "pb2":
                self._pb2_exploit_explore(population, generation)
                self.pb2_scheduler.save(self.output_root / "pb2_scheduler.json")
            else:
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
        while not champion.pruned and self._within_stage(stage) and time.monotonic() < self.deadline:
            self._train_trial_epoch(champion, stage=stage, target_epoch_seconds=self.args.target_epoch_seconds)
            if champion.epoch % 2 == 0:
                self._evaluate_trial(champion, stage=stage, force=True)
            self._save_state()
        if champion.pruned:
            self.log.write("champion_pruned", {"stage": stage, "trial_id": champion.trial_id, "reason": champion.prune_reason})

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

    def _replay_policy_v2_width(self, cfg: Config, family: FamilySpec) -> int:
        base_width = max(
            1,
            int(cfg.selfplay.policy_target_top_k),
            int(getattr(cfg.model, "candidate_budget", 256)),
        )
        return min(base_width, REPLAY_POLICY_WIDTH_CAP)

    def _replay_feature_flags(self, cfg: Config, family: FamilySpec) -> dict[str, bool]:
        return replay_feature_flags(
            getattr(cfg.model, "heads", []),
            architecture=getattr(cfg.model, "architecture", "cnn"),
            sparse_policy=bool(getattr(cfg.model, "sparse_policy", False)),
            graph=bool(family.graph),
        )

    def _make_replay_buffer(
        self,
        cfg: Config,
        family: FamilySpec,
        *,
        capacity: int | None = None,
    ) -> RingBuffer:
        return RingBuffer(
            capacity=int(capacity or cfg.buffer.capacity),
            max_policy_entries=int(cfg.selfplay.policy_target_top_k),
            max_policy_v2_entries=self._replay_policy_v2_width(cfg, family),
            recency_decay=cfg.buffer.recency_decay,
            num_lookahead=len(cfg.buffer.lookahead_horizons),
            **self._replay_feature_flags(cfg, family),
        )

    def _replay_memory_estimate(self, replay: RingBuffer, family: FamilySpec) -> dict[str, Any]:
        estimate = replay.memory_estimate() if hasattr(replay, "memory_estimate") else {}
        estimate["policy_width_mode"] = "compact_candidate_capped"
        estimate["full_global_policy_rows"] = FULL_GLOBAL_POLICY_ROWS
        estimate["non_global_policy_width_cap"] = REPLAY_POLICY_WIDTH_CAP
        return estimate

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
        replay = self._make_replay_buffer(cfg, family)
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
        self._save_trial_state(trial)
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
        if trial.pruned:
            return
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
            self._save_trial_state(trial)
            self._save_state()
            return

        trial.trainer = result.trainer
        trial.checkpoint_path = result.checkpoint_path
        if result.checkpoint_path:
            trial.checkpoint_history.append(result.checkpoint_path)
        trial.epoch = int(result.train_stats.get("epoch", trial.epoch + 1))
        trial.wall_time_s += time.monotonic() - started

        selfplay = _latest_metric(trial.run_dir / "events.jsonl", "selfplay")
        public_state = self._trial_public_state(trial)
        model_contract = public_state.get("model_contract", {})
        pair_strategy = public_state.get("pair_strategy", {})
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
            "heads": list(trial.cfg.model.heads),
            "runtime_outputs": list(model_contract.get("outputs") or []),
            "pair_capabilities": list(model_contract.get("pair_capabilities") or []),
            "pair_strategy": pair_strategy,
            "loss_weights": dict(
                getattr(getattr(trial.cfg, "train", None), "loss_weights", {}) or {}
            ),
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

        self._save_trial_state(trial)
        self._save_state()
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
        self._save_trial_state(trial)
        self._save_state()
        return components

    def _clone_compatible_trial(self, src: TrialState, dst: TrialState, generation: int) -> bool:
        if src.compatible_key != dst.compatible_key or src.checkpoint_path is None:
            self.log.write(
                "scheduler_clone_rejected",
                {
                    "generation": generation,
                    "source_method": self.args.schedule_method,
                    "from": src.trial_id,
                    "to": dst.trial_id,
                    "reason": "incompatible_or_missing_checkpoint",
                },
            )
            return False
        dst.replay = self._fresh_replay_for_trial(dst)
        dst.checkpoint_path = src.checkpoint_path
        if dst.trainer is not None:
            dst.trainer.load_checkpoint(src.checkpoint_path)
        dst.epoch = src.epoch
        event = {
            "generation": generation,
            "event": "exploit",
            "source_method": self.args.schedule_method,
            "from": src.trial_id,
            "to": dst.trial_id,
            "checkpoint_path": str(src.checkpoint_path),
            "compatible_key": str(src.compatible_key),
            "shared_mutable_replay": False,
            "replay_transfer": "fresh_empty_after_checkpoint_exploit",
        }
        dst.mutation_history.append(event)
        self.log.write("scheduler_exploit", event)
        return True

    def _mutate_trial(self, trial: TrialState, generation: int) -> None:
        old = asdict(trial.dynamic)
        clamped: dict[str, bool] = {}
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
                clamped[field_name] = False
            else:
                new_value = float(value) * self.rng.choice([0.8, 1.2])
                unclamped = new_value
                new_value = max(lo, min(hi, new_value))
                clamped[field_name] = new_value != unclamped
            setattr(trial.dynamic, field_name, new_value)
        self._apply_dynamic_to_config(trial)
        self._apply_dynamic_to_trainer(trial)
        event = {
            "generation": generation,
            "event": "explore",
            "source_method": "pbt_baseline",
            "trial_id": trial.trial_id,
            "old": old,
            "new": asdict(trial.dynamic),
            "clamped": clamped,
        }
        trial.mutation_history.append(event)
        self.log.write("pbt_baseline_explore", event)

    def _fresh_replay_for_trial(self, trial: TrialState) -> RingBuffer:
        return self._make_replay_buffer(trial.cfg, trial.family)

    def _release_trial_runtime(self, trial: TrialState, *, reason: str) -> None:
        replay_capacity = int(getattr(trial.replay, "capacity", 0) or 0)
        if trial.trainer is None and replay_capacity <= 1:
            return
        trial.trainer = None
        try:
            trial.replay = self._make_replay_buffer(trial.cfg, trial.family, capacity=1)
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
        # LR range is still explored by PB2/PBT schedule mutations, but bootstrapping these
        # families at the CNN center rate can corrupt weights before the first
        # scorecard exists. Start them at the known-stable safety rail and let
        # the population exploit/explore upward only after finite metrics land.
        if family.sparse_policy or family.graph or family.architecture == "restnet":
            dynamic.lr = min(dynamic.lr, 3e-4)
        return dynamic

    @staticmethod
    def _is_global_graph_architecture(architecture: str) -> bool:
        return registry_is_global_graph_architecture(str(architecture))

    def _heads_for_recipe(self, family: FamilySpec, recipe: StaticRecipe) -> list[str]:
        if not family.global_graph:
            return list(HEAD_BUNDLES[recipe.head_bundle])
        heads = ["policy_place", "value", "lookahead_4", "lookahead_12", "lookahead_36", "legal_token_quality"]
        if recipe.head_bundle in {"prediction", "full_aux_light", "graph_tactical"}:
            heads.extend(["opp_policy", "moves_left"])
        if recipe.head_bundle in {"regret", "full_aux_light", "graph_tactical"}:
            heads.extend(["regret_rank", "regret_value"])
        if recipe.head_bundle == "graph_tactical" or family.architecture in {"global_pair_twostage_0", "global_graph_full_0"}:
            heads.extend(["policy_pair_first", "policy_pair_second", "policy_pair_joint", "tactical"])
        elif family.architecture in {"global_line_window_0", "global_graph_full_0"}:
            heads.append("tactical")
        return list(dict.fromkeys(heads))

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
        cfg.model.sparse_policy = bool(family.sparse_policy or (family.graph and not family.global_graph))
        cfg.model.graph_token_set = recipe.graph_token_set
        cfg.model.graph_token_budget = recipe.graph_token_budget
        cfg.model.graph_layers = recipe.graph_layers
        cfg.model.sparse_prior_stage = 0 if family.global_graph else int(recipe.sparse_prior_stage)
        cfg.model.sparse_prior_mix = 0.0 if family.global_graph else 0.25
        cfg.model.candidate_budget = recipe.candidate_budget if (family.sparse_policy or family.graph) else 256
        cfg.model.heads = self._heads_for_recipe(family, recipe)
        if family.global_graph:
            if set(cfg.model.heads) & GLOBAL_GRAPH_PAIR_HEADS:
                cfg.model.pair_strategy = "diagnostic_full_pair"
                cfg.model.pair_strategy_max_pairs = min(256, max(1, int(recipe.graph_token_budget)))
            else:
                cfg.model.pair_prior_mix = 0.0
                cfg.model.pair_strategy = "none"
                cfg.model.pair_strategy_max_pairs = 0
        cfg.buffer.lookahead_horizons = [4, 12, 36]
        cfg.buffer.lookahead_lambdas = [0.75, 0.90, 0.97]
        cfg.selfplay.mcts_simulations = recipe.full_sims
        cfg.selfplay.pcr_low_sims = recipe.pcr_low_sims
        cfg.selfplay.policy_target_top_k = recipe.policy_top_k
        cfg.selfplay.subtree_reuse = recipe.subtree_reuse
        cfg.selfplay.train_policy_on_full_search_only = True
        cfg.selfplay.train_on_truncated_games = True
        cfg.selfplay.max_game_moves = self._max_game_moves_for_stage(stage)
        cfg.buffer.capacity = self._host_safe_buffer_capacity()
        cfg.runtime.autotune = True
        cfg.selfplay.num_workers = 0
        cfg.selfplay.batch_size_per_worker = 0
        cfg.inference.max_batch_size = 0
        cfg.train.batch_size = int(recipe.train_batch_size)
        cfg.train.batches_per_epoch = self.args.train_batches
        cfg.train.lr_schedule = "constant"
        cfg.runtime.compile_inference = False
        cfg.runtime.compile_model = False
        if self._low_memory_cuda_host() and (family.graph or family.sparse_policy or family.architecture == "restnet"):
            cfg.runtime.inference_start_timeout_s = max(float(cfg.runtime.inference_start_timeout_s), 90.0)
        self._apply_head_bundle_weights(cfg, recipe, dynamic)
        self._apply_dynamic_values(cfg, recipe, dynamic, family)
        autotune_config(cfg, self.host, selfplay_enabled=True)
        high_search_non_graph = bool(not family.graph and not family.sparse_policy and recipe.full_sims >= 512)
        if self.host.cuda_available and self.host.cuda_memory_gb < 16.0 and not high_search_non_graph:
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), 3)
            cfg.selfplay.batch_size_per_worker = min(int(cfg.selfplay.batch_size_per_worker), 8)
            cfg.inference.max_batch_size = min(
                int(cfg.inference.max_batch_size),
                max(64, cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        elif self.host.cuda_available and self.host.cuda_memory_gb < 16.0 and high_search_non_graph:
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), 3)
            cfg.selfplay.batch_size_per_worker = min(max(int(cfg.selfplay.batch_size_per_worker), 16), 16)
            cfg.inference.max_batch_size = min(
                max(int(cfg.inference.max_batch_size), cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
                128,
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        if family.graph or family.sparse_policy or (family.architecture == "restnet" and not high_search_non_graph):
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
        elif stage != "3A_calibration" and recipe.full_sims >= 512:
            cfg.selfplay.num_workers = min(int(cfg.selfplay.num_workers), 3)
            cfg.selfplay.batch_size_per_worker = min(max(int(cfg.selfplay.batch_size_per_worker), 16), 16)
            cfg.inference.max_batch_size = min(
                max(int(cfg.inference.max_batch_size), cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64),
                128,
            )
            cfg.inference.max_wait_us = max(int(cfg.inference.max_wait_us), 500)
        configure_torch_runtime(cfg, self.host)
        return Config.model_validate(cfg.model_dump())

    def _apply_head_bundle_weights(self, cfg: Config, recipe: StaticRecipe, dynamic: DynamicParams) -> None:
        heads = set(getattr(cfg.model, "heads", []))
        is_global_graph = self._is_global_graph_architecture(getattr(cfg.model, "architecture", "cnn"))
        aux = (
            dynamic.graph_aux_multiplier
            if getattr(cfg.model, "architecture", "cnn") in {"graph", "graph_hybrid_0"} or is_global_graph
            else dynamic.aux_multiplier
        )
        if is_global_graph:
            weights = {
                "policy_place": 1.0,
                "value": dynamic.value_loss_weight,
                "lookahead_4": 0.2 * aux,
                "lookahead_12": 0.2 * aux,
                "lookahead_36": 0.1 * aux,
                "legal_token_quality": 0.05 * aux,
            }
        else:
            weights = {
                "policy": 1.0,
                "value": dynamic.value_loss_weight,
                "lookahead_4": 0.2 * aux,
                "lookahead_12": 0.2 * aux,
                "lookahead_36": 0.1 * aux,
                "axis": 0.05 * aux,
            }
        if "opp_policy" in heads:
            weights["opp_policy"] = 0.15 * aux
        if "moves_left" in heads:
            weights["moves_left"] = 0.05 * aux
        if "regret_rank" in heads:
            weights["regret_rank"] = 0.1 * aux
        if "regret_value" in heads:
            weights["regret_value"] = 0.1 * aux
        if cfg.model.sparse_policy:
            weights["sparse_policy"] = dynamic.sparse_policy_loss
        if "pair_policy" in heads:
            weights["pair_policy"] = dynamic.pair_policy_loss
        if heads & GLOBAL_GRAPH_PAIR_HEADS:
            weights["policy_pair_first"] = dynamic.pair_policy_loss
            weights["policy_pair_second"] = dynamic.pair_policy_loss
            weights["policy_pair_joint"] = dynamic.pair_policy_loss
        if "tactical" in heads:
            weights["tactical"] = 0.05 * aux
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
            if family.global_graph:
                root_width = FULL_GLOBAL_POLICY_ROWS
            else:
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
        if cached and cached.get("selected") and self._runtime_sweep_cached_selection_safe(trial, cached):
            selected = dict(cached["selected"])
            self._apply_runtime_candidate(trial.cfg, selected)
            trial.runtime_sweep = {
                "applied": True,
                "cached": True,
                "key": key,
                "selected": selected,
            }
            self.log.write("runtime_sweep_cached", {"trial_id": trial.trial_id, "stage": stage, **trial.runtime_sweep})
            self._save_trial_state(trial)
            return
        if cached and cached.get("selected"):
            self.log.write(
                "runtime_sweep_cache_ignored",
                {
                    "trial_id": trial.trial_id,
                    "stage": stage,
                    "key": key,
                    "reason": "unsafe_or_suboptimal_cached_selection",
                    "cached_selected": cached.get("selected"),
                    "cached_selected_record": cached.get("selected_record"),
                },
            )

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
        memory_unsafe_worker_floor: int | None = None
        for idx, candidate in enumerate(candidates):
            if not self._within_stage(stage):
                results.append({"candidate": candidate, "error": "stage_deadline_reached"})
                break
            workers = int(candidate.get("workers", 0) or 0)
            if memory_unsafe_worker_floor is not None and workers >= memory_unsafe_worker_floor:
                row = {
                    "candidate": candidate,
                    "ok": False,
                    "positions": 0,
                    "positions_per_min": 0.0,
                    "score": 0.0,
                    "error": "skipped_after_memory_unsafe_candidate",
                    "memory_unsafe_worker_floor": memory_unsafe_worker_floor,
                }
                results.append(row)
                self.log.write("runtime_sweep_result", {"trial_id": trial.trial_id, "stage": stage, **row})
                _append_jsonl(self.output_root / "runtime_sweep_results.jsonl", {"trial_id": trial.trial_id, "stage": stage, **row})
                continue
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
            if self._runtime_sweep_memory_unsafe(row):
                memory_unsafe_worker_floor = min(memory_unsafe_worker_floor or workers, workers)
            if self._runtime_sweep_zero_progress_timeout(row):
                self.log.write(
                    "runtime_sweep_remaining_candidates_skipped",
                    {
                        "trial_id": trial.trial_id,
                        "stage": stage,
                        "candidate_index": idx,
                        "reason": "zero_progress_timeout",
                        "skipped": max(0, len(candidates) - idx - 1),
                    },
                )
                break

        valid = [
            row
            for row in results
            if row.get("ok")
            and float(row.get("positions_per_min", 0.0) or 0.0) > 0.0
            and not self._runtime_sweep_memory_unsafe(row)
        ]
        if not valid:
            reason = "runtime_sweep_failed:all_probe_candidates_failed_or_memory_unsafe"
            trial.pruned = True
            trial.prune_reason = reason
            trial.runtime_sweep = {
                "applied": False,
                "key": key,
                "candidate_count": len(candidates),
                "results": results,
                "reason": reason,
            }
            self.log.write("runtime_sweep_failed", {"trial_id": trial.trial_id, "stage": stage, **trial.runtime_sweep})
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "stage": stage, "reason": reason})
            self._release_trial_runtime(trial, reason=reason)
            self._save_trial_state(trial)
            self._save_state()
            return

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
        self._save_trial_state(trial)

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
        probe_buffer = self._make_replay_buffer(probe_cfg, trial.family)
        model = trial.trainer.model if trial.trainer is not None else None
        gpu_before = self._nvidia_smi_snapshot()
        mem_before = self._system_memory_snapshot()
        mem_samples: list[dict[str, float]] = []
        stop_memory_poll = threading.Event()

        def poll_memory() -> None:
            while not stop_memory_poll.is_set():
                sample = self._system_memory_snapshot()
                if sample:
                    mem_samples.append(sample)
                stop_memory_poll.wait(1.0)

        memory_thread = threading.Thread(target=poll_memory, daemon=True)
        memory_thread.start()
        started = time.monotonic()
        timeout_s = self._runtime_sweep_timeout_s(trial, sweep_states, candidate)
        old_alarm_handler = None
        alarm_armed = False

        def _runtime_sweep_timeout(_signum, _frame) -> None:
            raise RuntimeSweepTimeout(f"runtime sweep exceeded {timeout_s:.0f}s")

        try:
            if timeout_s > 0 and threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):
                old_alarm_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _runtime_sweep_timeout)
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
                alarm_armed = True
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
            mem_after = self._system_memory_snapshot()
            memory = self._summarize_runtime_memory(mem_before, mem_after, mem_samples)
            gpu_util = float(gpu_after.get("gpu_util_pct", 0.0) or 0.0)
            memory_penalty = 0.35 if memory.get("unsafe") else 1.0
            score = positions_per_min * (1.0 + min(gpu_util, 95.0) / 2000.0) * memory_penalty
            return {
                "ok": positions > 0,
                "candidate": candidate,
                "elapsed_s": elapsed_s,
                "positions": positions,
                "positions_per_min": positions_per_min,
                "score": score,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "memory": memory,
                "replay_memory": self._replay_memory_estimate(probe_buffer, trial.family),
                "selfplay": selfplay,
            }
        except RuntimeSweepTimeout as exc:
            elapsed_s = time.monotonic() - started
            gpu_after = self._nvidia_smi_snapshot()
            mem_after = self._system_memory_snapshot()
            memory = self._summarize_runtime_memory(mem_before, mem_after, mem_samples)
            selfplay = _latest_metric(probe_dir / "events.jsonl", "selfplay")
            positions = int(selfplay.get("positions_done") or 0)
            positions_per_min = float(selfplay.get("positions_per_min") or (positions / max(elapsed_s, 1e-6) * 60.0))
            return {
                "ok": False,
                "candidate": candidate,
                "elapsed_s": elapsed_s,
                "positions": positions,
                "positions_per_min": positions_per_min,
                "score": 0.0,
                "error": f"runtime_sweep_timeout:{timeout_s:.0f}s:{exc}",
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "memory": memory,
                "replay_memory": self._replay_memory_estimate(probe_buffer, trial.family),
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
                "memory": self._summarize_runtime_memory(mem_before, self._system_memory_snapshot(), mem_samples),
                "replay_memory": self._replay_memory_estimate(probe_buffer, trial.family),
            }
        finally:
            if alarm_armed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_alarm_handler)
            stop_memory_poll.set()
            memory_thread.join(timeout=2.0)
            self._cleanup_shared_memory()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()

    def _runtime_sweep_candidates(self, trial: TrialState) -> list[dict[str, int]]:
        cfg = trial.cfg
        base_workers = max(1, int(cfg.selfplay.num_workers))
        base_bpw = max(1, int(cfg.selfplay.batch_size_per_worker))
        base_wait = max(1, int(cfg.inference.max_wait_us))
        worker_values = _parse_int_list(getattr(self.args, "runtime_sweep_workers", ""))
        if not worker_values:
            worker_values = [max(1, base_workers - 1), base_workers, base_workers + 1]
        graph_low_memory = bool(trial.family.graph and self._low_memory_cuda_host())
        if graph_low_memory:
            # Probe a narrow 1/2-worker lane after replay-width fixes, then let
            # the runtime memory summary reject unsafe candidates.
            base_workers = 1
            worker_values = [1, 2]
        high_search_non_graph = bool(not trial.family.graph and int(trial.static.full_sims) >= 512)
        if high_search_non_graph:
            # Compact replay removes the main fixed-width RAM sink. Probe back
            # toward the historical worker counts and let measured RAM/swap
            # telemetry reject only candidates that are actually unsafe.
            worker_values = sorted(set(worker_values + [1, 2, 3, 4]))
            if self._low_memory_cuda_host():
                base_workers = 1
                if getattr(trial.family, "architecture", "") == "restnet":
                    worker_values = [1, 2, 3]
                else:
                    worker_values = [1, 2, 3, 4]
        worker_budget = max(1, int(self.host.logical_cpus) - max(1, int(cfg.runtime.selfplay_cpu_reserve)))
        max_workers = min(worker_budget, max(worker_values + [base_workers]))
        if high_search_non_graph and self._low_memory_cuda_host() and worker_values:
            max_workers = min(max_workers, max(worker_values))
        if graph_low_memory:
            max_workers = min(max_workers, 2)

        candidates: list[dict[str, int]] = []

        def add(workers: int, wait_us: int) -> None:
            workers = max(1, min(int(workers), max_workers))
            batch_per_worker = base_bpw
            if not trial.family.graph and int(trial.static.full_sims) >= 512:
                batch_per_worker = max(batch_per_worker, 16)
                if self._low_memory_cuda_host():
                    batch_per_worker = min(batch_per_worker, 8)
            max_batch = max(64, workers * batch_per_worker + 64)
            if self.host.cuda_available and self.host.cuda_memory_gb < 16.0:
                max_batch = min(max_batch, 192 if (not trial.family.graph and int(trial.static.full_sims) >= 512) else 128)
            candidate = {
                "workers": workers,
                "batch_size_per_worker": batch_per_worker,
                "max_batch_size": max_batch,
                "max_wait_us": max(1, int(wait_us)),
            }
            if candidate not in candidates:
                candidates.append(candidate)

        if not high_search_non_graph and not graph_low_memory:
            add(base_workers, base_wait)
        for workers in sorted(set(worker_values)):
            add(workers, base_wait)
        if len(candidates) < int(self.args.runtime_sweep_max_candidates):
            add(base_workers, max(base_wait, 800))
        if len(candidates) < int(self.args.runtime_sweep_max_candidates):
            add(max_workers, max(base_wait, 800))
        return candidates[: max(1, int(self.args.runtime_sweep_max_candidates))]

    def _runtime_sweep_timeout_s(self, trial: TrialState, sweep_states: int, candidate: dict[str, int]) -> float:
        if not (trial.family.global_graph and self._low_memory_cuda_host()):
            return 0.0
        worker_scale = max(1.0, float(candidate.get("workers", 1) or 1))
        state_scale = max(1.0, float(sweep_states) / 384.0)
        timeout_s = LOW_MEMORY_GLOBAL_GRAPH_RUNTIME_SWEEP_TIMEOUT_S * state_scale / min(worker_scale, 2.0)
        return max(90.0, min(timeout_s, LOW_MEMORY_GLOBAL_GRAPH_RUNTIME_SWEEP_TIMEOUT_S))

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

    def _runtime_sweep_cached_selection_safe(self, trial: TrialState, cached: dict[str, Any]) -> bool:
        record = cached.get("selected_record")
        # Older cache entries did not include system-memory telemetry. They are
        # not safe for high-search dense/ResTNet runs because a candidate can
        # look fast while pushing WSL into OOM reset territory.
        high_search_non_graph = bool(not trial.family.graph and not trial.family.sparse_policy and trial.static.full_sims >= 512)
        if isinstance(record, dict):
            candidate = record.get("candidate") or {}
            selected = cached.get("selected") or {}
            if isinstance(candidate, dict) and isinstance(selected, dict) and candidate != selected:
                return False
            workers = int(candidate.get("workers", 0) or 0) if isinstance(candidate, dict) else 0
            if high_search_non_graph and self._low_memory_cuda_host() and workers > 4:
                return False
            if "memory" not in record:
                return not high_search_non_graph
            if self._runtime_sweep_memory_unsafe(record):
                return False
            results = cached.get("results")
            if isinstance(results, list):
                valid = [
                    row
                    for row in results
                    if isinstance(row, dict)
                    and row.get("ok")
                    and float(row.get("positions_per_min", 0.0) or 0.0) > 0.0
                    and not self._runtime_sweep_memory_unsafe(row)
                ]
                if valid:
                    best = max(valid, key=lambda row: float(row.get("score", row.get("positions_per_min", 0.0)) or 0.0))
                    best_candidate = best.get("candidate") or {}
                    if isinstance(candidate, dict) and isinstance(best_candidate, dict) and candidate != best_candidate:
                        return False
            return True
        return not high_search_non_graph

    def _runtime_sweep_memory_unsafe(self, row: dict[str, Any]) -> bool:
        memory = row.get("memory") or {}
        if not isinstance(memory, dict):
            return False
        return bool(memory.get("unsafe"))

    @staticmethod
    def _runtime_sweep_zero_progress_timeout(row: dict[str, Any]) -> bool:
        error = str(row.get("error") or "")
        return error.startswith("runtime_sweep_timeout:") and int(row.get("positions") or 0) <= 0

    def _system_memory_snapshot(self) -> dict[str, float]:
        try:
            with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
                values: dict[str, float] = {}
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].endswith(":"):
                        values[parts[0][:-1]] = float(parts[1]) / (1024.0 * 1024.0)
        except Exception:
            return {}
        total = float(values.get("MemTotal", 0.0))
        available = float(values.get("MemAvailable", values.get("MemFree", 0.0)))
        swap_total = float(values.get("SwapTotal", 0.0))
        swap_free = float(values.get("SwapFree", 0.0))
        return {
            "total_gb": total,
            "available_gb": available,
            "used_gb": max(0.0, total - available),
            "swap_total_gb": swap_total,
            "swap_used_gb": max(0.0, swap_total - swap_free),
        }

    def _summarize_runtime_memory(
        self,
        before: dict[str, float],
        after: dict[str, float],
        samples: list[dict[str, float]],
    ) -> dict[str, Any]:
        all_samples = [sample for sample in [before, *samples, after] if sample]
        if not all_samples:
            return {}
        total_gb = max(float(sample.get("total_gb", 0.0) or 0.0) for sample in all_samples)
        min_available_gb = min(float(sample.get("available_gb", total_gb) or 0.0) for sample in all_samples)
        max_used_gb = max(float(sample.get("used_gb", 0.0) or 0.0) for sample in all_samples)
        max_swap_used_gb = max(float(sample.get("swap_used_gb", 0.0) or 0.0) for sample in all_samples)
        swap_before_gb = float(before.get("swap_used_gb", 0.0) or 0.0) if before else 0.0
        min_available_fraction = min_available_gb / max(total_gb, 1e-6)
        unsafe = bool(
            min_available_gb < max(4.0, total_gb * 0.15)
            or min_available_fraction < 0.14
            or max_swap_used_gb > max(swap_before_gb + 0.50, 1.0)
        )
        return {
            "before": before,
            "after": after,
            "sample_count": len(all_samples),
            "total_gb": total_gb,
            "min_available_gb": min_available_gb,
            "min_available_fraction": min_available_fraction,
            "max_used_gb": max_used_gb,
            "max_swap_used_gb": max_swap_used_gb,
            "unsafe": unsafe,
        }

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
        seen: set[str] = set()

        def add_candidate(family: FamilySpec, recipe: StaticRecipe) -> bool:
            key = json.dumps({"family": family.name, "recipe": asdict(recipe)}, sort_keys=True)
            if key in seen:
                return False
            seen.add(key)
            combos.append((family, recipe))
            return True

        for family in available:
            recipe = self._recommended_recipe(family)
            add_candidate(family, recipe)
            if len(combos) >= max_trials:
                break
        attempts = 0
        while len(combos) < max_trials and attempts < max_trials * 8 * max(1, len(available)) and available:
            made_progress = False
            for family in available:
                if len(combos) >= max_trials:
                    break
                for _ in range(max_trials * 8):
                    attempts += 1
                    rng = random.Random(self.args.seed + attempts)
                    config = self.bohb_sampler.search_space.sample_random(rng, {"model_family": family.name})
                    recipe = self._static_recipe_from_bohb_config(family, config)
                    if not add_candidate(family, recipe):
                        continue
                    self.bohb_sampler.samples.append(
                        {
                            "config": config,
                            "seed": self.args.seed + attempts,
                            "source": "family_balanced_random",
                            "density_model": None,
                            "candidate_scores": [],
                            "brackets": [asdict(bracket) for bracket in self.bohb_sampler.create_brackets()],
                        }
                    )
                    made_progress = True
                    break
            if not made_progress:
                break
        if len(combos) < min(max_trials, len(available)):
            for family in available:
                recipe = self._recommended_recipe(family)
                add_candidate(family, recipe)
                if len(combos) >= max_trials:
                    break
        counts: dict[str, int] = {}
        for family, _recipe in combos:
            counts[family.name] = counts.get(family.name, 0) + 1
        self.log.write(
            "static_candidates_generated",
            {
                "max_trials": max_trials,
                "family_counts": counts,
                "family_balanced": bool(available and max(counts.values(), default=0) - min(counts.values(), default=0) <= 1),
            },
        )
        self.bohb_sampler.save(self.output_root / "bohb_sampler.json")
        _write_json(
            self.output_root / "static_search_space.json",
            {
                "planned": STATIC_SPACE,
                "actual_bohb_space": self.bohb_sampler.search_space.parameters,
                "blocked_families": self.blocked_families,
            },
        )
        return combos[:max_trials]

    def _bohb_search_space(self) -> SearchSpace:
        family_names = [family.name for family in self._eligible_families()]
        if not family_names:
            family_names = [family.name for family in self.families]
        graph_family_names = [family.name for family in self._eligible_families() if family.graph]
        if not graph_family_names:
            graph_family_names = ["graph_hybrid_0", *GLOBAL_GRAPH_SCOUT_FAMILIES]
        parameters = {
            "model_family": {"type": "categorical", "choices": family_names},
            "full_sims": {"type": "categorical", "choices": STATIC_SPACE["full_sims"]},
            "candidate_budget": {"type": "categorical", "choices": [128, 256, 384]},
            "policy_top_k": {"type": "categorical", "choices": [64, 96, 128]},
            "head_bundle": {"type": "categorical", "choices": list(HEAD_BUNDLES)},
            "temperature_family": {"type": "categorical", "choices": STATIC_SPACE["temperature_family"]},
            "train_batch_size": {"type": "categorical", "choices": [128, 256, 384]},
            "graph_token_budget": {
                "type": "categorical",
                "choices": [256, 384, 512],
                "condition": {"key": "model_family", "values": graph_family_names},
            },
            "graph_layers": {
                "type": "categorical",
                "choices": [1, 2, 3],
                "condition": {"key": "model_family", "values": graph_family_names},
            },
        }
        if "graph_hybrid_0" in family_names:
            parameters["sparse_prior_stage"] = {
                "type": "categorical",
                "choices": [0, 1],
                "condition": {"key": "model_family", "values": ["graph_hybrid_0"]},
            }
        return SearchSpace(parameters)

    def _static_recipe_from_bohb_config(self, family: FamilySpec, config: dict[str, Any]) -> StaticRecipe:
        default_global_sims = LOW_MEMORY_GLOBAL_GRAPH_MAX_SIMS if self._low_memory_cuda_host() else 384
        full_sims = int(config.get("full_sims", default_global_sims if family.global_graph else 256 if family.graph else 800))
        full_sims = self._host_safe_full_sims(family, full_sims)
        graph_budget = int(config.get("graph_token_budget", 256))
        graph_layers = int(config.get("graph_layers", 1))
        sparse_stage = int(config.get("sparse_prior_stage", 0 if not family.graph else 0))
        if not family.graph or family.global_graph:
            sparse_stage = 0
        if family.graph:
            token_set = {
                256: "graph256_cells",
                384: "graph384_windows",
                512: "graph512_cover",
            }.get(graph_budget, "graph256_cells")
            if family.global_graph and family.architecture in {"global_pair_twostage_0", "global_graph_full_0"} and graph_budget >= 512:
                token_set = "graph512_turn_pair_prior"
        else:
            token_set = "graph256_cells"
            graph_budget = 256
            graph_layers = 1
        head_bundle = str(config.get("head_bundle", "structural"))
        if family.graph and not family.global_graph and head_bundle not in {"graph_tactical", "full_aux_light", "structural"}:
            head_bundle = "graph_tactical"
        if family.global_graph and family.architecture in {"global_pair_twostage_0", "global_graph_full_0"}:
            head_bundle = "graph_tactical"
        if not family.graph and head_bundle == "graph_tactical":
            head_bundle = "full_aux_light"
        candidate_budget = graph_budget if family.graph else int(config.get("candidate_budget", 256))
        return StaticRecipe(
            full_sims=full_sims,
            pcr_low_sims=self._pcr_low_sims_for_full_sims(full_sims),
            policy_top_k=int(config.get("policy_top_k", 96)),
            candidate_budget=candidate_budget,
            head_bundle=head_bundle,
            temperature_family=str(config.get("temperature_family", "slow_cool")),
            subtree_reuse=True,
            graph_token_set=token_set,
            graph_token_budget=graph_budget,
            graph_layers=graph_layers,
            sparse_prior_stage=sparse_stage,
            train_batch_size=self._host_safe_train_batch_size(family, int(config.get("train_batch_size", 256))),
        )

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
        if family.global_graph:
            pair_scout = family.architecture in {"global_pair_twostage_0", "global_graph_full_0"}
            token_budget = 384 if family.architecture == "global_line_window_0" else 256
            token_set = "graph384_windows" if token_budget == 384 else "graph256_cells"
            if family.architecture == "global_graph_full_0":
                token_budget = 512
                token_set = "graph512_turn_pair_prior"
            requested_sims = LOW_MEMORY_GLOBAL_GRAPH_MAX_SIMS if self._low_memory_cuda_host() else 384
            full_sims = self._host_safe_full_sims(family, requested_sims)
            return StaticRecipe(
                full_sims=full_sims,
                pcr_low_sims=self._pcr_low_sims_for_full_sims(full_sims),
                policy_top_k=96,
                candidate_budget=min(token_budget, 512),
                head_bundle="graph_tactical" if pair_scout else "structural",
                temperature_family="slow_cool",
                subtree_reuse=True,
                graph_token_set=token_set,
                graph_token_budget=token_budget,
                graph_layers=1 if token_budget <= 384 else 2,
                sparse_prior_stage=0,
                train_batch_size=self._host_safe_train_batch_size(family, 128),
            )
        if family.graph:
            return StaticRecipe(
                full_sims=512,
                pcr_low_sims=128,
                policy_top_k=96,
                candidate_budget=256,
                head_bundle="structural",
                temperature_family="slow_cool",
                subtree_reuse=True,
                graph_token_set="graph256_cells",
                graph_token_budget=256,
                graph_layers=1,
                sparse_prior_stage=0,
                train_batch_size=self._host_safe_train_batch_size(family, 256),
            )
        full_sims = self._host_safe_full_sims(family, 800)
        if family.sparse_policy:
            return StaticRecipe(
                full_sims=full_sims,
                pcr_low_sims=self._pcr_low_sims_for_full_sims(full_sims),
                policy_top_k=96,
                candidate_budget=256,
                head_bundle="structural",
                temperature_family="slow_cool",
                subtree_reuse=True,
                train_batch_size=self._host_safe_train_batch_size(family, 256),
            )
        return StaticRecipe(
            full_sims=full_sims,
            pcr_low_sims=self._pcr_low_sims_for_full_sims(full_sims),
            policy_top_k=96,
            candidate_budget=256,
            head_bundle="structural",
            temperature_family="slow_cool",
            subtree_reuse=True,
            train_batch_size=self._host_safe_train_batch_size(family, 256),
        )

    def _host_safe_full_sims(self, family: FamilySpec, requested: int) -> int:
        full_sims = max(1, int(requested))
        if not self._low_memory_cuda_host():
            return full_sims
        if family.architecture == "restnet":
            return min(full_sims, 512)
        if family.global_graph:
            return min(full_sims, LOW_MEMORY_GLOBAL_GRAPH_MAX_SIMS)
        return full_sims

    def _host_safe_train_batch_size(self, family: FamilySpec, requested: int) -> int:
        batch_size = max(1, int(requested))
        if not self._low_memory_cuda_host():
            return batch_size
        memory_hungry = bool(family.graph or family.sparse_policy or family.architecture == "restnet")
        if memory_hungry:
            return min(batch_size, 128)
        return min(batch_size, 256)

    def _host_safe_buffer_capacity(self) -> int:
        capacity = max(1, int(self.base_cfg.buffer.capacity))
        if not self._low_memory_cuda_host():
            return capacity
        # The full policy-v2 replay schema preallocates many board-area-wide
        # arrays. A 100k-capacity buffer is useful for long standalone training,
        # but even an 8k buffer touched enough memory to fill RAM+swap during
        # 384-move ASHA screening on this WSL host. Keep enough room for the
        # observed ~2.8k-position dense ASHA epochs with margin, without
        # carrying a buffer that can starve the dashboard and inference server.
        return min(capacity, 4096)

    @staticmethod
    def _pcr_low_sims_for_full_sims(full_sims: int) -> int:
        if full_sims <= 64:
            return min(32, max(1, full_sims))
        if full_sims <= 128:
            return 64
        if full_sims <= 256:
            return 96
        if full_sims <= 512:
            return 128
        if full_sims <= 800:
            return 192
        if full_sims <= 1200:
            return 256
        return 384

    def _eligible_families(self) -> list[FamilySpec]:
        return [f for f in self.families if f.available and f.name not in self.blocked_families]

    def _low_memory_cuda_host(self) -> bool:
        system_memory_gb = float(getattr(self.host, "system_memory_gb", 0.0) or 0.0)
        constrained_ram = system_memory_gb == 0.0 or system_memory_gb < 24.0
        constrained_cpu = self.host.physical_cpus <= 16
        return bool(self.host.cuda_available and self.host.cuda_memory_gb < 16.0 and (constrained_ram or constrained_cpu))

    def _asha_resources(self) -> list[int]:
        raw = str(getattr(self.args, "asha_resources", "10,20,30") or "10,20,30")
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
            self.log.write(
                "family_throughput_below_gate",
                {
                    "stage": stage,
                    "family": family.name,
                    "reason": reason,
                    "effect": "kept_for_fair_comparison_but_throughput_penalized_in_scores",
                },
            )

    def _quarantine_family(self, family: FamilySpec, reason: str, *, stage: str) -> None:
        hard_reasons = (
            "policy_target_mass_silently_dropped",
            "decisive_candidate_discovery_below_gate",
            "non_finite_train_metric",
            "illegal_or_crash_rate",
            "runtime_sweep_failed",
            "runtime_sweep_timeout",
            "host_guard:",
        )
        if not any(reason.startswith(prefix) for prefix in hard_reasons):
            self.log.write(
                "family_quarantine_skipped",
                {
                    "stage": stage,
                    "family": family.name,
                    "reason": reason,
                    "effect": "trial_pruned_but_family_kept_for_fair_comparison",
                },
            )
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

    def _record_asha_rung(self, trials: list[TrialState], resource: int) -> None:
        for trial in trials:
            hard_failure = bool(trial.pruned and trial.prune_reason)
            score = trial.last_score if math.isfinite(trial.last_score) else float("-inf")
            latest = trial.metrics_history[-1] if trial.metrics_history else {}
            self.asha_table.record(
                TrialObservation(
                    trial_id=trial.trial_id,
                    resource=int(resource),
                    score=float(score),
                    completed_epochs=int(trial.epoch),
                    wall_time_seconds=float(trial.wall_time_s),
                    selfplay_positions=int(
                        (latest.get("selfplay") or {}).get("positions_done")
                        or (latest.get("buffer") or {}).get("size")
                        or 0
                    ),
                    hard_failure=hard_failure,
                    failure_reason=trial.prune_reason or None,
                    metrics={
                        "scheduler_score": float(score) if math.isfinite(score) else -1e9,
                        "epoch_seconds": float(latest.get("epoch_elapsed_s", 0.0) or 0.0),
                    },
                )
            )

    def _apply_asha_decision(
        self,
        trials: list[TrialState],
        decision: dict[str, Any],
        *,
        stage: str,
    ) -> list[TrialState]:
        promoted = set(decision["promoted"])
        quarantined = set(decision["quarantined"])
        for trial in trials:
            if trial.trial_id in promoted:
                continue
            if trial.trial_id in quarantined:
                trial.pruned = True
                trial.prune_reason = trial.prune_reason or f"asha_quarantined_resource_{decision['resource']}"
            else:
                trial.pruned = True
                trial.prune_reason = f"asha_not_promoted_resource_{decision['resource']}"
            self.log.write(
                "trial_pruned",
                {"trial_id": trial.trial_id, "stage": stage, "reason": trial.prune_reason},
            )
            self._release_trial_runtime(trial, reason=trial.prune_reason)
        self.log.write("asha_promoted", {"stage": stage, **decision})
        return [trial for trial in trials if trial.trial_id in promoted and not trial.pruned]

    def _promote_top_fraction(self, trials: list[TrialState], *, stage: str) -> list[TrialState]:
        live = [t for t in trials if not t.pruned]
        self._score_population(live, stage=stage)
        live.sort(key=lambda t: t.last_score, reverse=True)
        fraction = max(0.05, min(1.0, float(self.args.asha_promote_fraction)))
        keep_n = max(1, math.ceil(len(live) * fraction))
        promoted_by_id: dict[str, TrialState] = {t.trial_id: t for t in live[:keep_n]}
        by_family: dict[str, list[TrialState]] = {}
        for trial in live:
            by_family.setdefault(trial.family.name, []).append(trial)
        for family_trials in by_family.values():
            best = max(family_trials, key=lambda t: t.last_score)
            promoted_by_id.setdefault(best.trial_id, best)
        promoted = sorted(promoted_by_id.values(), key=lambda t: t.last_score, reverse=True)
        pruned = live[keep_n:]
        for trial in pruned:
            if trial.trial_id in promoted_by_id:
                continue
            trial.pruned = True
            trial.prune_reason = f"asha_not_promoted_{stage}"
            self.log.write("trial_pruned", {"trial_id": trial.trial_id, "reason": trial.prune_reason})
            self._release_trial_runtime(trial, reason=trial.prune_reason)
        self.log.write(
            "asha_promoted",
            {
                "stage": stage,
                "trial_ids": [t.trial_id for t in promoted],
                "family_floor": True,
            },
        )
        return promoted

    def _seed_pbt_population(self) -> list[TrialState]:
        eligible_names = {family.name for family in self._eligible_families()}
        live = [
            t
            for t in self.trials
            if not t.pruned and t.checkpoint_path and t.family.name in eligible_names
        ]
        live.sort(key=lambda t: t.last_score, reverse=True)
        asha_survivors = [t for t in live if t.trial_id.startswith("asha_")]
        if asha_survivors:
            # PBT is a refinement stage for ASHA survivors. Calibration-only
            # controls are useful baselines, but pulling them into long 192-move
            # PBT can resurrect slow families that ASHA never promoted.
            live = asha_survivors
            self.log.write(
                "pbt_seed_restricted_to_asha_survivors",
                {"trial_ids": [t.trial_id for t in live]},
            )
        population = live[: self.args.pbt_population]
        eligible = self._eligible_families()
        while not asha_survivors and len(population) < min(self.args.pbt_population, len(eligible)):
            family = eligible[len(population)]
            recipe = self._recommended_recipe(family)
            stage = "3C_pb2" if self.args.schedule_method == "pb2" else "3C_pbt_fallback"
            trial = self._create_trial(
                f"{self.args.schedule_method}_seed_{len(population):02d}_{family.name}",
                family,
                recipe,
                self._initial_dynamic(family),
                stage,
            )
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

    def _pb2_exploit_explore(self, population: list[TrialState], generation: int) -> None:
        live = [t for t in population if not t.pruned and t.score_history]
        if len(live) < 2:
            return
        for trial in live:
            self.pb2_scheduler.observe(
                PB2Observation(
                    trial_id=trial.trial_id,
                    epoch=int(trial.epoch),
                    params=self._dynamic_params_for_pb2(trial),
                    score=float(trial.last_score),
                    compatible_group=self._pb2_group(trial),
                )
            )
        live.sort(key=lambda t: t.last_score, reverse=True)
        quartile = max(1, len(live) // 4)
        top = live[:quartile]
        bottom = live[-quartile:]
        for loser in bottom:
            donor = self._compatible_donor(loser, top)
            if donor is None:
                self.log.write(
                    "pb2_clone_rejected",
                    {
                        "generation": generation,
                        "trial_id": loser.trial_id,
                        "reason": "no_compatible_donor",
                    },
                )
                continue
            if not self._clone_compatible_trial(donor, loser, generation):
                continue
            try:
                event = self.pb2_scheduler.propose(
                    self._dynamic_params_for_pb2(loser),
                    seed=self.args.seed + generation * 1009 + len(loser.mutation_history),
                    compatible_group=self._pb2_group(loser),
                    candidates=int(self.args.pb2_candidates),
                    context=self._pb2_context(loser),
                    epoch=int(loser.epoch + self.args.perturb_interval),
                )
            except Exception as exc:
                self.log.write(
                    "pb2_mutation_rejected",
                    {
                        "generation": generation,
                        "trial_id": loser.trial_id,
                        "reason": f"{type(exc).__name__}:{exc}",
                    },
                )
                continue
            self._apply_pb2_values(loser, event["final_values"])
            mutation_event = {
                "generation": generation,
                "event": "pb2_explore",
                "trial_id": loser.trial_id,
                **event,
            }
            loser.mutation_history.append(mutation_event)
            self.log.write("pb2_explore", mutation_event)

    def _compatible_donor(self, loser: TrialState, top: list[TrialState]) -> TrialState | None:
        compatible = [t for t in top if t.compatible_key == loser.compatible_key and t is not loser]
        if compatible:
            return compatible[0]
        same_family = [t for t in top if t.family.compatible_key == loser.family.compatible_key and t is not loser]
        return same_family[0] if same_family else None

    def _dynamic_params_for_pb2(self, trial: TrialState) -> dict[str, float]:
        values = asdict(trial.dynamic)
        return {name: float(values[name]) for name in DYNAMIC_RANGES if name in values}

    def _apply_pb2_values(self, trial: TrialState, values: dict[str, float]) -> None:
        for name, value in values.items():
            if name not in DYNAMIC_RANGES:
                continue
            lo, hi = DYNAMIC_RANGES[name]
            setattr(trial.dynamic, name, max(float(lo), min(float(hi), float(value))))
        self._apply_dynamic_to_config(trial)
        self._apply_dynamic_to_trainer(trial)

    def _pb2_group(self, trial: TrialState) -> str:
        return "|".join(str(item) for item in trial.compatible_key)

    def _pb2_context(self, trial: TrialState) -> dict[str, Any]:
        return {
            "sparse_policy": bool(trial.family.sparse_policy),
            "pair_policy": "pair_policy" in trial.cfg.model.heads,
            "graph": bool(trial.family.graph),
            "regret_heads": any(head.startswith("regret_") for head in trial.cfg.model.heads),
        }

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
            "fallback_prior_use_on_mcts_topk",
            "pair_fallback_prior_use_on_mcts_topk",
        ]
        z = {key: _zscore_map(rows, key) for key in keys}
        for trial in trials:
            if not trial.score_history or trial.pruned:
                continue
            row = trial.score_history[-1]
            epoch = int(row.get("epoch", 0) or 0)
            if epoch < int(self.args.strategy_score_min_epochs):
                strength = (
                    0.45 * z["policy_target_quality"].get(id(row), 0.0)
                    + 0.35 * z["value_calibration_score"].get(id(row), 0.0)
                    + 0.20 * z["outside_window_robustness"].get(id(row), 0.0)
                )
                score_mode = "health_warmup"
            elif epoch < int(self.args.classical_score_min_epochs):
                strength = (
                    0.30 * z["tactical_suite_score"].get(id(row), 0.0)
                    + 0.25 * z["outside_window_robustness"].get(id(row), 0.0)
                    + 0.25 * z["policy_target_quality"].get(id(row), 0.0)
                    + 0.20 * z["value_calibration_score"].get(id(row), 0.0)
                )
                score_mode = "pre_classical_strategy"
            else:
                strength = (
                    0.40 * z["league_lcb"].get(id(row), 0.0)
                    + 0.20 * z["outside_window_robustness"].get(id(row), 0.0)
                    + 0.15 * z["tactical_suite_score"].get(id(row), 0.0)
                    + 0.10 * z["classical_survival_score"].get(id(row), 0.0)
                    + 0.10 * z["value_calibration_score"].get(id(row), 0.0)
                    + 0.05 * z["policy_target_quality"].get(id(row), 0.0)
                )
                score_mode = "classical_strategy"
            scheduler = (
                strength
                - 0.10 * z["epoch_seconds"].get(id(row), 0.0)
                - 0.10 * z["truncation_rate"].get(id(row), 0.0)
                - 0.20 * z["illegal_or_crash_rate"].get(id(row), 0.0)
                - 0.10 * z["fallback_prior_use_on_mcts_topk"].get(id(row), 0.0)
                - 0.10 * z["pair_fallback_prior_use_on_mcts_topk"].get(id(row), 0.0)
            )
            row["strength_score"] = strength
            row["scheduler_score"] = scheduler
            row["score_stage"] = stage
            row["score_mode"] = score_mode
            self.log.write("score_updated", {"trial_id": trial.trial_id, **row})
            if stage == "3B_static_asha":
                self.bohb_sampler.observe(
                    self._bohb_config_from_trial(trial),
                    scheduler,
                    valid=not trial.pruned,
                    budget=max(1, int(trial.epoch)),
                    status="completed" if not trial.pruned else "pruned",
                    reason=trial.prune_reason or None,
                )
        if stage == "3B_static_asha":
            self.bohb_sampler.save(self.output_root / "bohb_sampler.json")

    def _bohb_config_from_trial(self, trial: TrialState) -> dict[str, Any]:
        payload = {
            "model_family": trial.family.name,
            "full_sims": int(trial.static.full_sims),
            "candidate_budget": int(trial.static.candidate_budget),
            "policy_top_k": int(trial.static.policy_top_k),
            "head_bundle": trial.static.head_bundle,
            "temperature_family": trial.static.temperature_family,
            "train_batch_size": int(trial.static.train_batch_size),
        }
        if trial.family.graph:
            payload.update(
                {
                    "graph_token_budget": int(trial.static.graph_token_budget),
                    "graph_layers": int(trial.static.graph_layers),
                    "sparse_prior_stage": int(trial.static.sparse_prior_stage),
                }
            )
        return payload

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
            discovery = float(buffer.get("candidate_discovery_top8", 0.0) or 0.0)
            decisive = min(
                float(buffer.get("candidate_discovery_winning_move", 0.0) or 0.0),
                float(buffer.get("candidate_discovery_forced_block", 0.0) or 0.0),
                float(buffer.get("candidate_discovery_two_placement_cover", 0.0) or 0.0),
            )
            if record.get("buffer", {}).get("size", 0) > 0 and decisive < 0.995:
                return f"decisive_candidate_discovery_below_gate:{decisive:.4f}"
            if (
                record.get("buffer", {}).get("size", 0) > 0
                and discovery < self.args.candidate_recall_gate
                and record.get("stage") != "3A_calibration"
            ):
                self.log.write(
                    "candidate_discovery_below_gate",
                    {
                        "trial_id": trial.trial_id,
                        "stage": record.get("stage"),
                        "candidate_discovery_top8": discovery,
                        "gate": self.args.candidate_recall_gate,
                        "effect": "score_penalty_not_calibration_hard_prune",
                    },
                )
        elapsed = float(record.get("epoch_elapsed_s", 0.0) or 0.0)
        ref = max(float(self.args.target_epoch_seconds), 1.0)
        last_score = trial.last_score
        stage = str(record.get("stage") or getattr(trial, "stage", ""))
        if elapsed > 3.0 * ref and (not math.isfinite(last_score) or last_score < 0.0):
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
        league = CheckpointLeague()
        for color, record in (arena.get("by_color") or {}).items():
            games = int(record.get("wins", 0)) + int(record.get("losses", 0)) + int(record.get("draws", 0))
            if games > 0:
                league.record_match(
                    label,
                    color=color,
                    wins=int(record.get("wins", 0)),
                    losses=int(record.get("losses", 0)),
                    draws=int(record.get("draws", 0)),
                )
        league_rating = league.ratings.get(label)
        outside = 0.0
        tactical = classical_winrate
        league_lcb = float(league_rating.lcb) if league_rating is not None else classical_winrate - float(arena.get("winrate_std", 0.0))
        final_score = (
            league_lcb
            + 20.0 * outside
            + 15.0 * tactical
            + 5.0 * float(arena.get("classical_survival_score", 0.0))
            - 100.0 * illegal
        )
        return {
            "label": label,
            "checkpoint": str(ckpt),
            "final_score": final_score,
            "final_league_lcb": league_lcb,
            "final_league_rating": asdict(league_rating) if league_rating is not None else None,
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
        by_color = {
            "black": {"wins": 0, "losses": 0, "draws": 0},
            "white": {"wins": 0, "losses": 0, "draws": 0},
        }
        for result in stats.results:
            color_key = "black" if result.opening_is_black else "white"
            bad = result.reason.startswith("illegal") or result.reason.startswith("crash") or result.reason == "no_move"
            if bad:
                illegal_or_crash += 1
            baseline = self.baseline_loss_p75.get(result.opening_is_black, 128.0)
            survival_ratio = max(0.0, min(result.moves / max(baseline, 1.0), 1.25))
            if result.winner == 0:
                score = 1.00 + 0.05 * min(survival_ratio, 1.25)
                by_color[color_key]["wins"] += 1
            elif result.winner == 1:
                score = 0.15 + 0.55 * min(survival_ratio, 1.00)
                loss_moves_by_color[result.opening_is_black].append(result.moves)
                by_color[color_key]["losses"] += 1
            elif result.reason == "max_moves":
                score = 0.25 + 0.20 * min(survival_ratio, 1.00)
                by_color[color_key]["draws"] += 1
            else:
                score = -0.50
                by_color[color_key]["losses"] += 1
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
            "by_color": by_color,
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
            FamilySpec(
                "global_xattn_0",
                "Staged global cross-attention legal-row graph scout.",
                "global_xattn_0",
                graph=True,
                global_graph=True,
                available=True,
            ),
            FamilySpec(
                "global_line_window_0",
                "Staged global line/window legal-row graph scout.",
                "global_line_window_0",
                graph=True,
                global_graph=True,
                available=True,
            ),
            FamilySpec(
                "global_pair_twostage_0",
                "Staged global two-stage pair-policy graph scout.",
                "global_pair_twostage_0",
                graph=True,
                global_graph=True,
                available=True,
            ),
            FamilySpec(
                "global_graph_full_0",
                "Staged full global graph scout below champion scale.",
                "global_graph_full_0",
                graph=True,
                global_graph=True,
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
        available_families = [family for family in self.families if family.available]
        graph_available = any(family.graph for family in available_families)
        non_graph_available = any(not family.graph for family in available_families)
        available_names = {family.name for family in available_families}
        if graph_available and not non_graph_available and available_names <= _GLOBAL_GRAPH_SCOUT_FAMILY_SET:
            family_scope = "no-hybrid global graph run; tuning global graph scouts only"
        elif graph_available and not non_graph_available:
            family_scope = "graph-only run; tuning configured graph finalists"
        elif graph_available:
            family_scope = "graph available; tuning graph finalists with configured comparison families"
        else:
            family_scope = "graph unavailable; no graph-family finalists available after filters"
        return {
            "doc": "Docs/AUTOTUNING_METHODS_AND_48H_PLAN_20260427.md",
            "duration_hours": self.args.duration_hours,
            "phase_fractions": PHASE_FRACTIONS,
            "static_space": STATIC_SPACE,
            "bohb_space": self.bohb_sampler.search_space.parameters,
            "dynamic_ranges": DYNAMIC_RANGES,
            "dynamic_center": DYNAMIC_CENTER,
            "schedule_method": self.args.schedule_method,
            "families": [asdict(f) for f in self.families],
            "blocked_families": self.blocked_families,
            "fallback_branch": family_scope,
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
            "asha_decisions": self.asha_table.replay_decisions(),
            "bohb_samples": self.bohb_sampler.samples[-10:],
            "pb2_events": self.pb2_scheduler.events[-10:],
            "trials": [self._trial_public_state(t) for t in self.trials],
        }
        _write_json(self.output_root / "state.json", state)

    def _save_trial_state(self, trial: TrialState) -> None:
        _write_json(trial.run_dir / "trial.json", self._trial_public_state(trial))

    def _trial_public_state(self, trial: TrialState) -> dict[str, Any]:
        resolved = resolve_model_spec(trial.cfg)
        model_dump = (
            trial.cfg.model.model_dump()
            if hasattr(trial.cfg.model, "model_dump")
            else dict(getattr(trial.cfg, "model", {}))
        )
        family_dump = asdict(trial.family)
        output_contracts = {
            name: {
                "kind": contract.kind,
                "prediction_key": contract.prediction_key,
                "row_family": contract.row_family,
                "state_row": contract.state_row,
                "runtime_consumed": contract.runtime_consumed,
                "required_for_selfplay": contract.required_for_selfplay,
                "optional": contract.optional,
            }
            for name, contract in resolved.output_contracts.items()
        }
        return {
            "trial_id": trial.trial_id,
            "family": family_dump,
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
            "model_summary": architecture_display_summary(model_dump, family_dump),
            "model_contract": {
                "architecture_id": resolved.architecture_id,
                "family_id": resolved.family_id,
                "recipe_id": resolved.recipe_id,
                "input_contract_id": resolved.spec.input_contract_id,
                "training_adapter_id": resolved.spec.training_adapter_id,
                "inference_adapter_id": resolved.spec.inference_adapter_id,
                "policy_provider_id": resolved.spec.policy_provider_id,
                "value_provider_id": resolved.spec.value_provider_id,
                "outputs": list(resolved.outputs),
                "selfplay_required_outputs": list(resolved.selfplay_required_outputs),
                "pair_capabilities": list(resolved.pair_capabilities),
                "output_contracts": output_contracts,
                "row_tables": {
                    name: {
                        "family": row.family,
                        "schema_version": row.schema_version,
                        "ordering_rule": row.ordering_rule,
                        "mask_semantics": row.mask_semantics,
                    }
                    for name, row in resolved.row_table_definitions.items()
                },
                "value_decoder": {
                    "name": resolved.value_decoder.name,
                    "logits_key": resolved.value_decoder.logits_key,
                    "n_bins": resolved.value_decoder.n_bins,
                    "output_range": list(resolved.value_decoder.output_range),
                    "perspective": resolved.value_decoder.perspective,
                },
            },
            "loss_weights": dict(trial.cfg.train.loss_weights),
            "pair_strategy": {
                "strategy": getattr(trial.cfg.model, "pair_strategy", "none"),
                "max_pairs": int(getattr(trial.cfg.model, "pair_strategy_max_pairs", 0) or 0),
                "prior_mix": float(getattr(trial.cfg.model, "pair_prior_mix", 0.0) or 0.0),
            },
            "replay_memory": self._replay_memory_estimate(trial.replay, trial.family),
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
            f"- schedule_method: `{self.args.schedule_method}`",
            f"- fallback branch: `{self._manifest_payload()['fallback_branch']}`",
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
                f"- graph family scorecards: `tracked when graph trials pass hard gates; quarantines={self.blocked_families}`",
                "- BOHB static samples: persisted in `bohb_sampler.json`",
                "- ASHA rung decisions: persisted in `asha_rungs.json`",
                "- PB2/PBT schedule events: persisted in `pb2_scheduler.json` for PB2 or trial mutation history for PBT fallback",
                "- 1200 vs 800 per wall-clock: recorded in ASHA/BOHB scorecards when both recipes complete",
                "- candidate-policy priors: evaluated through candidate recall, missing mass, sparse loss, and sparse top-1",
                "- pair policy: `evaluated when graph_tactical head bundles survive hard gates`",
                "- regret replay: evaluated only for head bundles that include regret heads",
                "",
                "## Remaining Failure Modes",
                "",
                "- Final strength remains noisy until Phase 3E completes enough league/classical games.",
                "- Tactical suite uses authored replayable fixtures; failed fixture rows are included in trial scorecards.",
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
        self.args = supervisor.args

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
        raw_league_lcb = float(arena.get("model_win_rate", 0.0)) - float(arena.get("winrate_std", 0.0))
        raw_classical_survival = float(arena.get("classical_survival_score", 0.0) or 0.0)
        classical_score_active = int(trial.epoch) >= int(self.args.classical_score_min_epochs)
        league_lcb = raw_league_lcb if classical_score_active else 0.0
        classical_survival = raw_classical_survival if classical_score_active else 0.0
        row = {
            "stage": stage,
            "trial_id": trial.trial_id,
            "epoch": trial.epoch,
            "league_lcb": league_lcb,
            "raw_league_lcb": raw_league_lcb,
            "outside_window_robustness": outside["outside_window_robustness"],
            "tactical_suite_score": tactical["tactical_suite_score"],
            "classical_survival_score": classical_survival,
            "raw_classical_survival_score": raw_classical_survival,
            "classical_score_active": classical_score_active,
            "value_calibration_score": value_calibration,
            "policy_target_quality": policy_quality,
            "epoch_seconds": throughput["epoch_seconds"],
            "truncation_rate": throughput["truncation_rate"],
            "illegal_or_crash_rate": arena.get("illegal_or_crash_rate", 0.0),
            "fallback_prior_use_on_mcts_topk": float(
                buffer.get(
                    "fallback_prior_use_on_mcts_topk",
                    buffer.get("fallback_prior_use_on_mcts_top4", buffer.get("fallback_prior_use", 0.0)),
                )
                or 0.0
            ),
            "pair_fallback_prior_use_on_mcts_topk": float(
                buffer.get(
                    "pair_fallback_prior_use_on_mcts_topk",
                    buffer.get("pair_fallback_prior_use_on_mcts_top4", buffer.get("pair_fallback_prior_use", 0.0)),
                )
                or 0.0
            ),
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
        candidate_score = float(candidate.get("score", 1.0))
        row["candidate_recall_penalty"] = (
            0.15 * max(0.0, 1.0 - candidate_score)
            if bool(candidate.get("applicable")) and not bool(candidate.get("gate_pass"))
            else 0.0
        )
        row["scheduler_score"] = (
            row["strength_score"]
            - 0.10 * row["epoch_seconds"] / max(self.args.target_epoch_seconds, 1.0)
            - 0.10 * row["truncation_rate"]
            - 0.20 * row["illegal_or_crash_rate"]
            - row["candidate_recall_penalty"]
            - 0.10 * row["fallback_prior_use_on_mcts_topk"]
            - 0.10 * row["pair_fallback_prior_use_on_mcts_topk"]
        )
        return row

    def candidate_recall(self, trial: TrialState, buffer: dict[str, Any]) -> dict[str, Any]:
        if not trial.family.sparse_policy:
            return {"applicable": False, "score": 1.0}
        top1 = float(buffer.get("candidate_discovery_top1", 0.0) or 0.0)
        top4 = float(buffer.get("candidate_discovery_top4", 0.0) or 0.0)
        top8 = float(buffer.get("candidate_discovery_top8", 0.0) or 0.0)
        winning = float(buffer.get("candidate_discovery_winning_move", 0.0) or 0.0)
        forced = float(buffer.get("candidate_discovery_forced_block", 0.0) or 0.0)
        cover = float(buffer.get("candidate_discovery_two_placement_cover", 0.0) or 0.0)
        missing = float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
        decisive = min(winning, forced, cover)
        return {
            "applicable": True,
            "candidate_discovery_top1": top1,
            "candidate_discovery_top4": top4,
            "candidate_discovery_top8": top8,
            "candidate_discovery_winning_move": winning,
            "candidate_discovery_forced_block": forced,
            "candidate_discovery_two_placement_cover": cover,
            "missing_target_policy_mass": missing,
            "gate_pass": top8 >= self.args.candidate_recall_gate and decisive >= 0.995 and missing <= 0.01,
            "score": max(0.0, min(1.0, min(top8, decisive) - missing)),
        }

    def tactical_suite(
        self,
        trial: TrialState,
        buffer: dict[str, Any],
        arena: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            model = load_checkpoint_model(Path(trial.checkpoint_path), trial.cfg)
            player = model_move_fn(
                model,
                temperature=0.05,
                top_p=1.0,
                seed=self.s.args.seed + int(trial.epoch),
                near_radius=8,
                constrain_threats=True,
            )
            suite = evaluate_tactical_suite(
                player,
                time_ms=self.s.args.eval_time_ms,
            )
            component_scores = {
                str(row["suite"]): 1.0 if row["passed"] else 0.0
                for row in suite.positions
            }
            return {
                "components": component_scores,
                "tactical_suite_score": suite.score,
                "fixture_mode": "authored_replayable_positions",
                "passed": suite.passed,
                "total": suite.total,
                "positions": suite.positions,
            }
        except Exception as exc:
            missing = float(buffer.get("avg_missing_target_policy_mass", 0.0) or 0.0)
            legal = 1.0 - float(arena.get("illegal_or_crash_rate", 0.0) or 0.0)
            recall = float(candidate.get("score", 1.0))
            diagnostic_proxy = max(0.0, min(1.0, legal * recall * (1.0 - missing)))
            return {
                "components": {name: 0.0 for name in TACTICAL_COMPONENTS},
                "tactical_suite_score": 0.0,
                "fixture_mode": "diagnostic_fallback_after_fixture_error",
                "diagnostic_proxy_score": diagnostic_proxy,
                "error": repr(exc),
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
        truncation_rate = float(selfplay.get("truncation_rate", 0.0) or 0.0)
        if "truncation_rate" not in selfplay and "truncated_games" in selfplay:
            games = float(selfplay.get("games_done", 0.0) or 0.0)
            truncation_rate = float(selfplay.get("truncated_games", 0.0) or 0.0) / max(games, 1.0)
        return {
            "epoch_seconds": elapsed,
            "positions_per_second": positions / max(elapsed, 1e-6),
            "train_batches_per_second": float(train.get("batches_per_sec", 0.0) or 0.0),
            "selfplay_positions_per_min": float(selfplay.get("positions_per_min", 0.0) or 0.0),
            "truncation_rate": truncation_rate,
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
    parser.add_argument("--max-game-moves", type=int, default=500)
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
    parser.add_argument("--max-active-trials", type=int, default=6)
    parser.add_argument("--asha-resources", default="10,20,30")
    parser.add_argument("--asha-promote-fraction", type=float, default=0.5)
    parser.add_argument("--bohb-random-fraction", type=float, default=0.25)
    parser.add_argument("--schedule-method", choices=["pb2", "pbt"], default="pb2")
    parser.add_argument("--pb2-candidates", type=int, default=64)
    parser.add_argument("--pb2-uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--pbt-population", type=int, default=8)
    parser.add_argument("--perturb-interval", type=int, default=10)
    parser.add_argument("--pbt-generations", type=int, default=6)
    parser.add_argument("--champion-min-epochs", type=int, default=20)
    parser.add_argument("--strategy-score-min-epochs", type=int, default=10)
    parser.add_argument("--classical-score-min-epochs", type=int, default=12)
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
