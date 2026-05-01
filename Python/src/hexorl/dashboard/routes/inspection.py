from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hexorl.dashboard.contract_inspector import contract_catalog, required_view_names
from hexorl.dashboard.db import decode_bytes
from hexorl.dashboard.schemas.model import InferRequest
from hexorl.dashboard.services.common import parse_pair_policy_target_v2, parse_policy_target_v2

router = APIRouter()


@router.get("/api/debug/contracts")
def debug_contracts() -> dict[str, Any]:
    return contract_catalog()


@router.get("/api/inspect/views")
def inspect_views(request: Request) -> dict[str, Any]:
    return {"required": list(required_view_names()), "registered": list(request.app.state.contract_inspector.views())}


@router.post("/api/inspect/{view_name}")
def inspect_contract_view(request: Request, view_name: str, req: InferRequest) -> dict[str, Any]:
    try:
        return request.app.state.contract_inspector.inspect(
            view_name,
            history=decode_bytes(req.history_b64),
            policy_target=tuple(parse_policy_target_v2(req.policy_target_v2)),
            pair_policy_target=tuple(parse_pair_policy_target_v2(req.pair_policy_target_v2)),
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/api/debug/graph")
def debug_graph(request: Request, req: InferRequest) -> dict[str, Any]:
    return request.app.state.contract_inspector.inspect(
        "graph",
        history=decode_bytes(req.history_b64),
        policy_target=tuple(parse_policy_target_v2(req.policy_target_v2)),
        pair_policy_target=tuple(parse_pair_policy_target_v2(req.pair_policy_target_v2)),
    )


@router.post("/api/debug/d6")
def debug_d6(request: Request, req: InferRequest) -> dict[str, Any]:
    return request.app.state.contract_inspector.inspect(
        "d6",
        history=decode_bytes(req.history_b64),
        policy_target=tuple(parse_policy_target_v2(req.policy_target_v2)),
        pair_policy_target=tuple(parse_pair_policy_target_v2(req.pair_policy_target_v2)),
    )

