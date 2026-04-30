"""Autotune trial manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hexorl.contracts.identity import stable_digest
from hexorl.models.specs import GLOBAL_MODEL_KINDS
from hexorl.tuning.recipes import ModelRecipe
from hexorl.tuning.runtime_sweep import HostProfile, RuntimeSpec
from hexorl.tuning.scoring import ScoreComponents


@dataclass(frozen=True)
class TrialManifest:
    trial_id: str
    recipe: ModelRecipe
    runtime: RuntimeSpec
    host_profile: HostProfile
    git_sha: str
    command: str
    seeds: tuple[int, ...]
    validation_results: tuple[dict[str, Any], ...]
    scheduler_decisions: tuple[dict[str, Any], ...]
    trace_ids: tuple[str, ...]
    artifacts: tuple[str, ...]
    final_score: ScoreComponents | None = None

    @property
    def manifest_hash(self) -> str:
        return stable_digest(("TrialManifest", self.to_manifest(include_hash=False)))

    def to_manifest(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "trial_id": self.trial_id,
            "recipe": self.recipe.to_manifest(),
            "model_family": self.recipe.model_family,
            "model_spec_version": self.recipe.model_spec_version,
            "input_contract": "global_graph_v1" if self.recipe.model_family in GLOBAL_MODEL_KINDS else "crop_tensor_v1",
            "output_contract": "policy_provider_output_v1",
            "action_contract": "legal_action_table_v1",
            "runtime_spec": self.runtime.to_manifest(),
            "host_profile": self.host_profile.to_manifest(),
            "git_sha": self.git_sha,
            "command": self.command,
            "seeds": list(self.seeds),
            "validation_results": list(self.validation_results),
            "scheduler_decisions": list(self.scheduler_decisions),
            "trace_ids": list(self.trace_ids),
            "artifacts": list(self.artifacts),
            "final_score_components": None if self.final_score is None else self.final_score.to_manifest(),
        }
        if include_hash:
            payload["manifest_hash"] = self.manifest_hash
        return payload
