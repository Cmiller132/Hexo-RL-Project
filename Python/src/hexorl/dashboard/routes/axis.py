from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.axis_policy.registry import describe_prototypes, evaluate_all, get_prototype
from hexorl.dashboard.fixtures import ClassicalFixtureConfig, generate_classical_fixtures, list_axis_fixtures
from hexorl.dashboard.schemas.axis import AxisEvaluateRequest, AxisFixtureGenerateRequest, AxisPresetRequest
from hexorl.dashboard.services.common import axis_input_from_request

router = APIRouter()


@router.get("/api/axis/prototypes")
def axis_prototypes() -> list[dict[str, Any]]:
    return describe_prototypes()


@router.post("/api/axis/evaluate")
def axis_evaluate(request: Request, req: AxisEvaluateRequest) -> dict[str, Any]:
    position = axis_input_from_request(request.app.state.store, req)
    if req.prototype_id:
        return get_prototype(req.prototype_id).compute(position, req.parameters).to_json()
    return {"results": evaluate_all(position, req.parameter_overrides)}


@router.post("/api/axis/presets")
def axis_preset(request: Request, req: AxisPresetRequest) -> dict[str, Any]:
    preset_id = request.app.state.store.save_axis_preset(
        name=req.name,
        prototype_id=req.prototype_id,
        parameters=req.parameters,
        payload=req.payload,
    )
    return {"preset_id": preset_id}


@router.get("/api/axis/presets")
def axis_presets(request: Request) -> list[dict[str, Any]]:
    return request.app.state.store.rows("SELECT * FROM axis_presets ORDER BY created_at DESC")


@router.get("/api/axis/fixtures")
def axis_fixtures(request: Request, limit: int = 200) -> list[dict[str, Any]]:
    return list_axis_fixtures(request.app.state.store, limit=limit)


@router.post("/api/axis/fixtures/generate")
def axis_fixtures_generate(request: Request, req: AxisFixtureGenerateRequest) -> dict[str, Any]:
    try:
        fixtures = generate_classical_fixtures(
            request.app.state.store,
            ClassicalFixtureConfig(
                count=max(1, min(req.count, 64)),
                examples_per_move_count=max(0, min(req.examples_per_move_count, 16)),
                move_counts=tuple(req.move_counts),
                time_ms=max(1, min(req.time_ms, 500)),
                max_depth=max(1, min(req.max_depth, 8)),
                near_radius=max(1, min(req.near_radius, 32)),
                noise_level=max(0.0, min(req.noise_level, 2.0)),
                random_move_prob=max(0.0, min(req.random_move_prob, 1.0)),
                opening_random_moves=max(0, min(req.opening_random_moves, 12)),
                seed=req.seed,
                workers=max(1, min(req.workers, 8)),
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"fixtures": fixtures}

