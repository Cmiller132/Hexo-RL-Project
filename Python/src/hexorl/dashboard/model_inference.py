"""Provider-backed dashboard model inference service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hexorl.config import Config
from hexorl.dashboard.contract_inspector import contract_catalog
from hexorl.eval.position_services import build_search_context
from hexorl.inference.local import LocalEvaluator
from hexorl.inference.protocol import protocol_manifest_from_contract
from hexorl.models.factory import inference_contract
from hexorl.models.specs import ModelSpec, model_spec_from_config
from hexorl.search.policy_provider import create_policy_provider


@dataclass(frozen=True)
class DashboardModelInferenceService:
    model: torch.nn.Module
    device: torch.device
    cfg: Config
    model_spec: ModelSpec

    def infer_history(self, model_id: str, history: bytes) -> dict[str, Any]:
        manifest = protocol_manifest_from_contract(inference_contract(self.cfg), timeout_ms=float(getattr(self.cfg.inference, "timeout_ms", 1000.0)))
        client = LocalEvaluator(self.model, manifest=manifest, device=self.device)
        provider = create_policy_provider(model_spec=self.model_spec, client=client)
        context = build_search_context(
            history,
            model_spec=self.model_spec,
            recipe_id=f"dashboard:{self.model_spec.kind}",
            candidate_budget=int(getattr(self.cfg.model, "candidate_budget", 256)),
            near_radius=int(getattr(self.cfg.selfplay, "near_radius", 8)),
            constrain_threats=bool(getattr(self.cfg.selfplay, "constrain_threats", True)),
            inference_protocol=getattr(client.manifest, "transport", "local_model_eval_v1"),
        )
        evaluation = provider.evaluate_root(context)
        ranked = _ranked_policy_rows(context.legal_table.rows, evaluation.row_priors)
        return {
            "model_id": model_id,
            "model_family": evaluation.model_family,
            "provider_type": evaluation.policy_provider,
            "trace_id": context.trace_id,
            "inference_protocol": evaluation.inference_protocol,
            "legal_moves": [{"q": int(q), "r": int(r)} for q, r in context.legal_table.rows],
            "outside_window_legal_count": 0,
            "outside_window_legal_moves": [],
            "candidate_contract": contract_catalog()["candidate"],
            "contracts": context.identity_payload(),
            "heads": {
                "policy_provider_rows": ranked,
                "value": float(evaluation.value),
            },
            "raw_metadata": dict(evaluation.raw_metadata),
            "warnings": list(evaluation.warnings),
        }


def _ranked_policy_rows(rows: np.ndarray, priors: np.ndarray, *, limit: int = 16) -> list[dict[str, float | int]]:
    priors = np.asarray(priors, dtype=np.float32).reshape(-1)
    if priors.size == 0:
        return []
    order = np.argsort(-priors)[:limit]
    return [
        {
            "q": int(rows[row, 0]),
            "r": int(rows[row, 1]),
            "prior": float(priors[row]),
        }
        for row in order
    ]
