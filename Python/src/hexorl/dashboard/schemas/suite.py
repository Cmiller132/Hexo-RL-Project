from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SuiteTrialDetailV2(BaseModel):
    trial_id: str
    trial_dir: str
    trial: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    latest: dict[str, Any] = Field(default_factory=dict)
    checkpoint_metadata: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    architecture: dict[str, Any] = Field(default_factory=dict)
    architecture_summary: str = ""
    current_activity: dict[str, Any] = Field(default_factory=dict)


class SuiteManifestResponse(BaseModel):
    run_root: str
    manifest: dict[str, Any] = Field(default_factory=dict)
    manifest_path: str


class SuiteFamilySpaceResponse(BaseModel):
    families: list[dict[str, Any]]
    recipes: list[dict[str, Any]]
    spawned_trials: dict[str, list[dict[str, Any]]]


class SuiteSchedulerResponse(BaseModel):
    current_stage: str
    planned_stages: list[Any]
    scheduler: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    decisions: list[dict[str, Any]]
    state: dict[str, Any] = Field(default_factory=dict)


class SuiteRuntimeSweepResponse(BaseModel):
    probes: list[dict[str, Any]]
    selected: list[dict[str, Any]]

