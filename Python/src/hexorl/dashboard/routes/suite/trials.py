from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.dashboard.schemas.suite import SuiteTrialDetailV2
from hexorl.dashboard.services.suite import (
    suite_best_checkpoints,
    suite_trial_detail,
    suite_trial_events,
    suite_trial_loss_curve,
    suite_trial_runtime_sweep,
    suite_trial_scores,
    suite_trials,
)

router = APIRouter()


def _root(request: Request):
    suite_root = request.app.state.suite_root
    if suite_root is None:
        raise HTTPException(404, "Suite run root is not configured")
    return suite_root


@router.get("/api/suite/trials")
def trials(request: Request) -> list[dict[str, Any]]:
    suite_root = request.app.state.suite_root
    return [] if suite_root is None else suite_trials(suite_root)


@router.get("/api/suite/trials/{trial_id}", response_model=SuiteTrialDetailV2)
def trial_detail(request: Request, trial_id: str) -> dict[str, Any]:
    detail = suite_trial_detail(_root(request), trial_id)
    if not detail:
        raise HTTPException(404, f"Trial not found: {trial_id}")
    return detail


@router.get("/api/suite/trials/{trial_id}/scores")
def trial_scores(request: Request, trial_id: str) -> list[dict[str, Any]]:
    return suite_trial_scores(_root(request), trial_id)


@router.get("/api/suite/trials/{trial_id}/events")
def trial_events(request: Request, trial_id: str, limit: int = 1000) -> list[dict[str, Any]]:
    return suite_trial_events(_root(request), trial_id, limit=max(1, min(limit, 5000)))


@router.get("/api/suite/trials/{trial_id}/loss-curve")
def trial_loss_curve(request: Request, trial_id: str) -> list[dict[str, Any]]:
    return suite_trial_loss_curve(_root(request), trial_id)


@router.get("/api/suite/trials/{trial_id}/runtime-sweep")
def trial_runtime_sweep(request: Request, trial_id: str) -> dict[str, Any]:
    return suite_trial_runtime_sweep(_root(request), trial_id)


@router.get("/api/suite/best-checkpoints")
def best_checkpoints(request: Request, limit: int = 50) -> list[dict[str, Any]]:
    suite_root = request.app.state.suite_root
    return [] if suite_root is None else suite_best_checkpoints(suite_root, limit=max(1, min(limit, 500)))
