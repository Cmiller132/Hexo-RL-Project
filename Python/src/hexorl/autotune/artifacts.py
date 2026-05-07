"""Candidate-first artifact writers for scout autotuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexorl.autotune.hashing import config_hash
from hexorl.autotune.recipes import CandidateRecipe
from hexorl.config import Config


@dataclass(frozen=True)
class CandidateArtifactPaths:
    run_dir: Path
    candidate_dir: Path
    candidate_manifest: Path
    recipe_json: Path
    full_config_toml: Path
    full_config_json: Path
    optuna_trial_json: Path
    runtime_spec_json: Path
    events_jsonl: Path
    scorecards_jsonl: Path
    checkpoints_dir: Path
    debug_bundles_dir: Path


class CandidateArtifactWriter:
    """Create and append to the candidate-first artifact layout."""

    def __init__(self, runs_root: str | Path, run_id: str) -> None:
        self.runs_root = Path(runs_root)
        self.run_id = str(run_id)
        self.run_dir = self.runs_root / self.run_id

    def write_candidate(
        self,
        candidate: CandidateRecipe,
        config: Config,
        *,
        optuna_trial: dict[str, Any] | None = None,
        git_sha: str | None = None,
        host_profile: dict[str, Any] | None = None,
        study_name: str | None = None,
        trial_number: int | None = None,
    ) -> CandidateArtifactPaths:
        paths = self.paths_for(candidate.candidate_id)
        paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        paths.debug_bundles_dir.mkdir(parents=True, exist_ok=True)

        cfg_hash = config_hash(config)
        manifest = {
            "candidate_id": candidate.candidate_id,
            "architecture_id": candidate.model.architecture_id,
            "pair_strategy_mode": candidate.pair_strategy.mode,
            "recipe_schema_version": candidate.schema_version,
            "config_hash": cfg_hash,
            "git_sha": git_sha,
            "host_profile": host_profile,
            "optuna_study_name": study_name,
            "optuna_trial_number": trial_number,
        }
        trial_payload = dict(optuna_trial or {})
        trial_payload.setdefault("study_name", study_name)
        trial_payload.setdefault("trial_number", trial_number)
        trial_payload.setdefault("candidate_id", candidate.candidate_id)

        _write_json(paths.candidate_manifest, manifest)
        _write_json(paths.recipe_json, candidate.model_dump(mode="json"))
        _write_json(paths.full_config_json, config.model_dump(mode="json"))
        _write_json(paths.optuna_trial_json, trial_payload)
        _write_json(paths.runtime_spec_json, candidate.runtime.model_dump(mode="json"))
        _write_toml(paths.full_config_toml, config.model_dump(mode="json", exclude_none=True))
        paths.events_jsonl.touch(exist_ok=True)
        paths.scorecards_jsonl.touch(exist_ok=True)
        return paths

    def paths_for(self, candidate_id: str) -> CandidateArtifactPaths:
        candidate_dir = self.run_dir / "candidates" / candidate_id
        return CandidateArtifactPaths(
            run_dir=self.run_dir,
            candidate_dir=candidate_dir,
            candidate_manifest=candidate_dir / "candidate_manifest.json",
            recipe_json=candidate_dir / "recipe.json",
            full_config_toml=candidate_dir / "full_config.toml",
            full_config_json=candidate_dir / "full_config.json",
            optuna_trial_json=candidate_dir / "optuna_trial.json",
            runtime_spec_json=candidate_dir / "runtime_spec.json",
            events_jsonl=candidate_dir / "events.jsonl",
            scorecards_jsonl=candidate_dir / "scorecards.jsonl",
            checkpoints_dir=candidate_dir / "checkpoints",
            debug_bundles_dir=candidate_dir / "debug_bundles",
        )

    @staticmethod
    def append_event(paths: CandidateArtifactPaths, event: dict[str, Any]) -> None:
        _append_jsonl(paths.events_jsonl, event)

    @staticmethod
    def append_scorecard(paths: CandidateArtifactPaths, scorecard: dict[str, Any]) -> None:
        _append_jsonl(paths.scorecards_jsonl, scorecard)


def write_candidate_artifacts(
    runs_root: str | Path,
    run_id: str,
    candidate: CandidateRecipe,
    config: Config,
    **kwargs: Any,
) -> CandidateArtifactPaths:
    return CandidateArtifactWriter(runs_root, run_id).write_candidate(candidate, config, **kwargs)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import tomli_w
    except ModuleNotFoundError:
        text = _minimal_toml_dumps(payload)
    else:
        text = tomli_w.dumps(payload)
    path.write_text(text, encoding="utf-8")


def _minimal_toml_dumps(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for section, values in payload.items():
        if not isinstance(values, dict):
            lines.append(f"{section} = {_toml_value(values)}")
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_toml_value(val)}" for key, val in value.items())
        return "{ " + items + " }"
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")
