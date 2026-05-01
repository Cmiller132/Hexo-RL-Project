from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.dashboard.schemas.suite import (
    SuiteFamilySpaceResponse,
    SuiteManifestResponse,
    SuiteRuntimeSweepResponse,
    SuiteSchedulerResponse,
)
from hexorl.dashboard.services.suite import (
    suite_family_space,
    suite_manifest,
    suite_runtime_sweep,
    suite_scheduler,
)

router = APIRouter()


def _root(request: Request):
    suite_root = request.app.state.suite_root
    if suite_root is None:
        raise HTTPException(404, "Suite run root is not configured")
    return suite_root


@router.get("/api/suite/manifest", response_model=SuiteManifestResponse)
def manifest(request: Request) -> dict[str, Any]:
    return suite_manifest(_root(request))


@router.get("/api/suite/family-space", response_model=SuiteFamilySpaceResponse)
def family_space(request: Request) -> dict[str, Any]:
    return suite_family_space(_root(request))


@router.get("/api/suite/scheduler", response_model=SuiteSchedulerResponse)
def scheduler(request: Request) -> dict[str, Any]:
    return suite_scheduler(_root(request))


@router.get("/api/suite/runtime-sweep", response_model=SuiteRuntimeSweepResponse)
def runtime_sweep(request: Request) -> dict[str, Any]:
    return suite_runtime_sweep(_root(request))
