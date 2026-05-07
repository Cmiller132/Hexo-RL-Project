"""Phase 3 per-family Optuna study helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import optuna as _OPTUNA
except ModuleNotFoundError:
    _OPTUNA = None

_BasePruner = _OPTUNA.pruners.BasePruner if _OPTUNA is not None else object


PAIR_MODES = {"none", "root_pair_mcts", "full_pair_mcts"}


@dataclass(frozen=True)
class Phase3StudySpec:
    architecture_id: str
    pair_mode: str
    study_name: str
    storage: str
    direction: str = "maximize"
    sampler: dict[str, Any] = field(default_factory=dict)
    pruner: dict[str, Any] = field(default_factory=dict)
    search_scope: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_user_attrs(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hexo_phase"] = "phase3_per_family_tuning"
        payload["architecture_family"] = self.architecture_id
        payload["pair_mode"] = self.pair_mode
        return payload


def phase3_study_name(architecture_id: str, pair_mode: str, *, version: int = 1) -> str:
    _validate_pair_mode(pair_mode)
    return f"study_{architecture_id}__{pair_mode}__schedule_v{version}"


def phase3_study_spec(
    *,
    architecture_id: str,
    pair_mode: str,
    storage: str,
    seed: int | None = None,
    signal_floor_epoch: int = 12,
    reduction_factor: int = 2,
    n_startup_trials: int = 8,
    version: int = 1,
) -> Phase3StudySpec:
    _validate_pair_mode(pair_mode)
    sampler = {
        "type": "TPESampler",
        "multivariate": True,
        "group": True,
        "n_startup_trials": int(n_startup_trials),
        "seed": seed,
    }
    pruner = {
        "type": "HexoScorecardFloorPruner",
        "signal_floor_epoch": int(signal_floor_epoch),
        "requires_scorecard_written": True,
        "delegate": {
            "type": "SuccessiveHalvingPruner",
            "min_resource": int(signal_floor_epoch),
            "reduction_factor": int(reduction_factor),
        },
    }
    knobs = [
        "lr_multiplier",
        "weight_decay",
        "c_puct",
        "c_puct_init",
        "dirichlet_fraction",
        "scaled_alpha_total",
        "pcr_low_sim_prob",
        "recency_decay",
        "value_loss_weight",
        "auxiliary_loss_weight",
    ]
    if pair_mode != "none":
        knobs.extend(["pair_loss_weight", "pair_prior_mix"])
    return Phase3StudySpec(
        architecture_id=architecture_id,
        pair_mode=pair_mode,
        study_name=phase3_study_name(architecture_id, pair_mode, version=version),
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        search_scope={
            "mode": "per_family_pair_mode_schedule_search",
            "architecture_mutations_allowed": False,
            "knobs": knobs,
        },
        metadata={
            "score_scalar": "classical_survival_lcb",
            "hexo_hard_gates_before_optuna": True,
            "runtime_quarantine_separate_from_model_score": True,
        },
    )


def create_phase3_sampler(*, seed: int | None = None, n_startup_trials: int = 8) -> Any:
    optuna = _import_optuna()
    return optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=int(n_startup_trials),
        seed=seed,
    )


def create_phase3_floor_pruner(
    *,
    signal_floor_epoch: int = 12,
    reduction_factor: int = 2,
    scorecard_written_attr: str = "scorecards_written",
) -> Any:
    optuna = _import_optuna()
    delegate = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=int(signal_floor_epoch),
        reduction_factor=int(reduction_factor),
    )
    return HexoScorecardFloorPruner(
        signal_floor_epoch=signal_floor_epoch,
        delegate=delegate,
        scorecard_written_attr=scorecard_written_attr,
    )


def create_phase3_study(
    *,
    architecture_id: str,
    pair_mode: str,
    storage: str,
    seed: int | None = None,
    signal_floor_epoch: int = 12,
    reduction_factor: int = 2,
    n_startup_trials: int = 8,
    load_if_exists: bool = True,
) -> Any:
    optuna = _import_optuna()
    spec = phase3_study_spec(
        architecture_id=architecture_id,
        pair_mode=pair_mode,
        storage=storage,
        seed=seed,
        signal_floor_epoch=signal_floor_epoch,
        reduction_factor=reduction_factor,
        n_startup_trials=n_startup_trials,
    )
    study = optuna.create_study(
        study_name=spec.study_name,
        storage=storage,
        direction=spec.direction,
        sampler=create_phase3_sampler(seed=seed, n_startup_trials=n_startup_trials),
        pruner=create_phase3_floor_pruner(
            signal_floor_epoch=signal_floor_epoch,
            reduction_factor=reduction_factor,
        ),
        load_if_exists=load_if_exists,
    )
    for key, value in spec.to_user_attrs().items():
        study.set_user_attr(key, value)
    return study


def phase3_study_specs_from_phase2_report(
    report: Any,
    *,
    storage_template: str,
    seed: int | None = None,
    signal_floor_epoch: int = 12,
    reduction_factor: int = 2,
    n_startup_trials: int = 8,
    max_promoted: int | None = None,
) -> tuple[Phase3StudySpec, ...]:
    specs: list[Phase3StudySpec] = []
    ranked = list(getattr(report, "ranked", ()))
    if max_promoted is not None:
        ranked = ranked[: int(max_promoted)]
    for row in ranked:
        architecture_id, pair_mode = _parse_candidate_family_pair_mode(str(row.candidate_id))
        storage = (
            storage_template
            .replace("<candidate_id>", str(row.candidate_id))
            .replace("<family>", architecture_id)
            .replace("<pair_mode>", pair_mode)
        )
        spec = phase3_study_spec(
            architecture_id=architecture_id,
            pair_mode=pair_mode,
            storage=storage,
            seed=seed,
            signal_floor_epoch=signal_floor_epoch,
            reduction_factor=reduction_factor,
            n_startup_trials=n_startup_trials,
        )
        metadata = dict(spec.metadata)
        metadata.update(
            {
                "promoted_candidate_id": str(row.candidate_id),
                "phase2_rank": int(row.rank),
                "phase2_classical_survival_lcb": float(row.classical_survival_lcb),
                "phase2_checkpoint_lineage": dict(row.checkpoint_lineage),
                "phase2_evidence_paths": list(row.evidence_paths),
                "phase2_scorecard_path": str(row.scorecard_path),
            }
        )
        specs.append(Phase3StudySpec(**{**asdict(spec), "metadata": metadata}))
    return tuple(specs)


def mark_trial_scorecard_written(trial: Any, scorecard_path: str | Path) -> None:
    trial.set_user_attr("scorecards_written", True)
    trial.set_user_attr("final_scorecard_path", str(scorecard_path))


def mark_trial_hexo_artifacts(
    trial: Any,
    *,
    scorecard_path: str | Path,
    checkpoint_path: str | Path | None = None,
    evidence_paths: list[str | Path] | tuple[str | Path, ...] = (),
    hard_gates: dict[str, Any] | None = None,
    debug_bundle_paths: list[str | Path] | tuple[str | Path, ...] = (),
    extra_attrs: dict[str, Any] | None = None,
) -> None:
    """Record Hexo artifact authority on an Optuna trial.

    These attrs make the Optuna ledger traceable without making Optuna the
    authority for champion selection or hard-gate semantics.
    """

    mark_trial_scorecard_written(trial, scorecard_path)
    if checkpoint_path is not None:
        trial.set_user_attr("hexo_checkpoint_path", str(checkpoint_path))
    trial.set_user_attr("hexo_evidence_paths", [str(path) for path in evidence_paths])
    trial.set_user_attr("hexo_debug_bundle_paths", [str(path) for path in debug_bundle_paths])
    gates = dict(hard_gates or {})
    trial.set_user_attr("hexo_hard_gates", gates)
    trial.set_user_attr("hexo_hard_gate_passed", _hard_gates_pass(gates))
    trial.set_user_attr("hexo_artifacts_recorded", True)
    for key, value in dict(extra_attrs or {}).items():
        trial.set_user_attr(str(key), value)


class HexoScorecardFloorPruner(_BasePruner):
    """Delegate metric pruning only after Hexo has crossed the scorecard floor."""

    def __init__(
        self,
        *,
        signal_floor_epoch: int = 12,
        delegate: Any | None = None,
        scorecard_written_attr: str = "scorecards_written",
    ) -> None:
        if int(signal_floor_epoch) <= 0:
            raise ValueError("signal_floor_epoch must be positive")
        self.signal_floor_epoch = int(signal_floor_epoch)
        self.delegate = delegate
        self.scorecard_written_attr = scorecard_written_attr

    def prune(self, study: Any, trial: Any) -> bool:
        step = _trial_last_step(trial)
        if step is None or step < self.signal_floor_epoch:
            return False
        if not self._scorecard_written(trial):
            return False
        if self.delegate is None:
            return False
        return bool(self.delegate.prune(study, trial))

    def _scorecard_written(self, trial: Any) -> bool:
        attrs = getattr(trial, "user_attrs", {}) or {}
        if bool(attrs.get(self.scorecard_written_attr)):
            return True
        path = attrs.get("final_scorecard_path")
        if not path:
            return False
        return Path(str(path)).exists()


def _trial_last_step(trial: Any) -> int | None:
    last_step = getattr(trial, "last_step", None)
    if last_step is not None:
        return int(last_step)
    intermediate = getattr(trial, "intermediate_values", None) or {}
    if not intermediate:
        return None
    return int(max(intermediate))


def _validate_pair_mode(pair_mode: str) -> None:
    if pair_mode not in PAIR_MODES:
        raise ValueError(f"pair_mode must be one of {sorted(PAIR_MODES)}")


def _parse_candidate_family_pair_mode(candidate_id: str) -> tuple[str, str]:
    parts = candidate_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"cannot parse promoted candidate id {candidate_id!r}")
    architecture_id, pair_mode = parts[0], parts[1]
    _validate_pair_mode(pair_mode)
    return architecture_id, pair_mode


def _hard_gates_pass(gates: dict[str, Any]) -> bool:
    failures = gates.get("failures", ())
    if failures:
        return False
    hard_pass = gates.get("hard_pass")
    if hard_pass is not None:
        return bool(hard_pass)
    for key, value in gates.items():
        if key.endswith("_pass") and value is False:
            return False
    return True


def _import_optuna() -> Any:
    if _OPTUNA is None:
        raise RuntimeError("Phase 3 Optuna helpers require optuna to be installed")
    return _OPTUNA
