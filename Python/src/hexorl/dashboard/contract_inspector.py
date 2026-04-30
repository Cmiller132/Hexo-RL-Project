"""Read-only dashboard contract inspection services."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history
from hexorl.axis_policy.core import AxisPolicyInput
from hexorl.axis_policy.registry import evaluate_all
from hexorl.contracts.candidates import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    CandidateContractBuilder,
)
from hexorl.contracts.history import MoveHistory
from hexorl.contracts.identity import ndarray_digest, stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.contracts.symmetry import (
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
)
from hexorl.dashboard.replay import get_replay_position, position_payload
from hexorl.engine.encoding import encode_board_and_legal
from hexorl.graph.semantic_builder import (
    GRAPH_CAPACITY_STRATEGY,
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
    GraphTokenType,
    RelationType,
)
from hexorl.graph.tensorize import build_graph_batch_from_history, graph_capacity_report
from hexorl.models.factory import get_model_registry, inference_manifest
from hexorl.models.specs import MODEL_SPEC_VERSION, ModelSpec
from hexorl.replay.codec import REPLAY_RECORD_SCHEMA_VERSION


class InspectorService(Protocol):
    name: str

    def inspect(self, request: "InspectionRequest", inspector: "ContractInspector") -> dict[str, Any]: ...


@dataclass(frozen=True)
class InspectionRequest:
    history: bytes = b""
    policy_target: tuple[tuple[int, int, float], ...] = ()
    pair_policy_target: tuple[tuple[tuple[int, int], tuple[int, int], float], ...] = ()
    model_family: str = "dense_cnn"
    recipe_id: str = "dashboard-default"
    recipe_hash: str = ""
    checkpoint_manifest: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None
    model_output: dict[str, Any] | None = None
    replay_identity: dict[str, Any] | None = None
    autotune_report: dict[str, Any] | None = None
    compare_to: dict[str, Any] | None = None


@dataclass(frozen=True)
class DashboardModelInputs:
    tensor: np.ndarray
    offset_q: int
    offset_r: int
    legal_table: LegalActionTable
    candidate_table: Any | None
    pair_table: Any | None


class ContractInspector:
    """Dispatcher over focused read-only dashboard inspector services."""

    def __init__(self) -> None:
        self._services: dict[str, InspectorService] = {}
        for service in (
            HistoryInspector(),
            LegalTableInspector(),
            TacticalInspector(),
            CandidatesInspector(),
            PairsInspector(),
            GraphInspector(),
            D6Inspector(),
            ModelInputInspector(),
            ModelOutputInspector(),
            TraceInspector(),
            ReplayInspector(),
            CheckpointInspector(),
            RecipeInspector(),
            AutotuneInspector(),
            DebugBundleInspector(),
            MismatchInspector(),
        ):
            self.register(service.name, service)

    def register(self, name: str, service: InspectorService) -> None:
        if name in self._services:
            raise ValueError(f"dashboard inspector already registered: {name}")
        self._services[name] = service

    def views(self) -> tuple[str, ...]:
        return tuple(sorted(self._services))

    def inspect(self, view: str, **kwargs: Any) -> dict[str, Any]:
        request = InspectionRequest(**kwargs)
        try:
            service = self._services[view]
        except KeyError as exc:
            raise KeyError(f"unknown dashboard inspector view: {view}") from exc
        started = time.monotonic()
        payload = service.inspect(request, self)
        payload.setdefault("view", view)
        payload.setdefault("facts", _facts(request.history, request))
        payload.setdefault("trace", _trace_payload(request.trace))
        payload["inspector"] = {
            "dispatcher": "ContractInspector",
            "service": service.__class__.__name__,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        }
        return payload


class HistoryInspector:
    name = "history"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        rows = MoveHistory.decode(request.history, source="rust").rows if request.history else []
        return {
            "history_b64": base64.b64encode(request.history).decode("ascii"),
            "history_hash": _history_hash(request.history),
            "move_count": len(rows),
            "rows": [{"player": int(p), "q": int(q), "r": int(r)} for p, q, r in rows],
        }


class LegalTableInspector:
    name = "legal-table"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        inputs = build_dashboard_model_inputs(request.history, include_pair_rows=False)
        return {
            "legal_table": inputs.legal_table.debug_payload(),
            "offset_q": inputs.offset_q,
            "offset_r": inputs.offset_r,
        }


class TacticalInspector:
    name = "tactical"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        inputs = build_dashboard_model_inputs(request.history, include_pair_rows=False)
        rows = [(int(q), int(r)) for q, r in inputs.legal_table.rows.tolist()]
        oracle = scan_tactical_oracle_from_history(
            request.history,
            rows,
            offset_q=inputs.offset_q,
            offset_r=inputs.offset_r,
        )
        return {
            "owner": "engine/legal+tactical_oracle",
            "status_kind": getattr(oracle, "status_kind", "ok"),
            "win_now_cells": [list(row) for row in oracle.win_now_cells],
            "forced_block_cells": [list(row) for row in oracle.forced_block_cells],
            "cover_cells": [list(row) for row in oracle.cover_cells],
            "open_four_cells": [list(row) for row in oracle.open_four_cells],
            "open_five_cells": [list(row) for row in oracle.open_five_cells],
        }


class CandidatesInspector:
    name = "candidates"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        inputs = build_dashboard_model_inputs(request.history, policy_target=request.policy_target)
        candidates = inputs.candidate_table
        assert candidates is not None
        active = np.flatnonzero(candidates.mask)
        return {
            "contract": "CandidateTable",
            "feature_version": CANDIDATE_FEATURE_VERSION,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "table_hash": candidates.table_hash,
            "source": candidates.source,
            "candidate_count": int(active.shape[0]),
            "target_mass": float(candidates.target.sum()),
            "missing_mass": float(candidates.missing_mass),
            "rows": [
                {
                    "row": int(row),
                    "q": int(candidates.qr[row, 0]),
                    "r": int(candidates.qr[row, 1]),
                    "dense_index": int(candidates.indices[row]),
                    "target_prob": float(candidates.target[row]),
                }
                for row in active[:64]
            ],
        }


class PairsInspector:
    name = "pairs"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        inputs = build_dashboard_model_inputs(
            request.history,
            policy_target=request.policy_target,
            pair_policy_target=request.pair_policy_target,
            include_pair_rows=True,
        )
        pair = inputs.pair_table
        if pair is None:
            return {"available": False, "target_mass": 0.0, "missing_mass": 0.0, "rows": []}
        candidates = inputs.candidate_table
        assert candidates is not None
        rows = []
        for row, active in enumerate(pair.mask):
            if not bool(active) or len(rows) >= 64:
                continue
            first_idx, second_idx = pair.pair_indices[row]
            rows.append(
                {
                    "row": int(row),
                    "first": {
                        "q": int(candidates.qr[int(first_idx), 0]),
                        "r": int(candidates.qr[int(first_idx), 1]),
                    },
                    "second": {
                        "q": int(candidates.qr[int(second_idx), 0]),
                        "r": int(candidates.qr[int(second_idx), 1]),
                    },
                    "target_prob": float(pair.target[row]),
                }
            )
        return {
            "available": True,
            "contract": "PairActionTable",
            "table_hash": pair.table_hash,
            "source": pair.source,
            "generation": getattr(pair, "generation", 1),
            "target_mass": float(pair.target.sum()),
            "missing_mass": float(pair.missing_mass),
            "pair_count": int(np.asarray(pair.mask, dtype=np.bool_).sum()),
            "rows": rows,
        }


class GraphInspector:
    name = "graph"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        graph = build_graph_batch_from_history(
            request.history,
            policy_target=request.policy_target,
            pair_policy_target=request.pair_policy_target,
            include_pair_rows=bool(request.pair_policy_target),
        )
        capacity = graph_capacity_report(graph)
        token_counts = {token.name: int((graph.token_type == int(token)).sum()) for token in GraphTokenType}
        relation_counts = {
            relation.name: int((graph.relation_type == int(relation)).sum())
            for relation in RelationType
            if int((graph.relation_type == int(relation)).sum()) > 0
        }
        return {
            "schema_version": graph.schema_version,
            "relation_schema_version": graph.relation_schema_version,
            "feature_dim": int(graph.token_features.shape[-1]),
            "graph_hash": graph.graph_semantic_hash,
            "source": "graph/semantic_builder.py",
            "capacity": {
                "fits_ipc": capacity.fits_ipc,
                "strategy": capacity.strategy,
                "failures": list(capacity.failures()),
                "max_tokens": capacity.max_tokens,
                "max_actions": capacity.max_actions,
                "max_pairs": capacity.max_pairs,
            },
            "token_count": int(graph.token_features.shape[0]),
            "token_counts": token_counts,
            "legal_count": int(graph.legal_qr.shape[0]),
            "legal_qr": [{"q": int(q), "r": int(r)} for q, r in graph.legal_qr[:64]],
            "opp_legal_count": int(graph.opp_legal_qr.shape[0]),
            "pair_count": int(graph.pair_token_indices.shape[0]),
            "relation_counts": relation_counts,
            "relation_bias_shape": [int(dim) for dim in graph.relation_bias.shape],
            "placements_remaining": int(graph.placements_remaining),
            "current_player": int(graph.current_player),
            "target_masses": {
                "policy": float(graph.policy_target.sum()),
                "pair": float(graph.pair_policy_target.sum()),
                "pair_first": float(graph.pair_first_policy_target.sum()),
                "opp_policy": float(graph.opp_policy_target.sum()),
            },
        }


class D6Inspector:
    name = "d6"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        transforms = []
        for sym_idx in range(12):
            history = transform_history(request.history, sym_idx)
            policy = transform_policy_target(list(request.policy_target), sym_idx)
            pair = transform_pair_policy_target(list(request.pair_policy_target), sym_idx)
            position = position_payload(get_replay_position(history, constrain_threats=False))
            graph = inspector.inspect("graph", history=history, policy_target=tuple(policy), pair_policy_target=tuple(pair))
            candidates = inspector.inspect("candidates", history=history, policy_target=tuple(policy))
            pairs = inspector.inspect("pairs", history=history, policy_target=tuple(policy), pair_policy_target=tuple(pair))
            transforms.append(
                {
                    "symmetry_index": sym_idx,
                    "history_b64": base64.b64encode(history).decode("ascii"),
                    "current_player": position["current_player"],
                    "placements_remaining": position["placements_remaining"],
                    "legal_count": len(position["legal_moves"]),
                    "graph": graph,
                    "contracts": {
                        "dense_legal_mask": {
                            **dict(position["encoding"]),
                            "legal_count": len(position["legal_moves"]),
                        },
                        "sparse_candidates": candidates,
                        "pair_rows": pairs,
                        "axis": _axis_payload(position, history),
                        "graph_targets": {
                            "legal_count": graph["legal_count"],
                            "pair_count": graph["pair_count"],
                            "opp_legal_count": graph["opp_legal_count"],
                            "token_counts": graph["token_counts"],
                            "target_masses": graph["target_masses"],
                        },
                    },
                }
            )
        return {
            "symmetry_count": 12,
            "source_history_b64": base64.b64encode(request.history).decode("ascii"),
            "transforms": transforms,
            "target_checks": _d6_target_checks(transforms),
        }


class ModelInputInspector:
    name = "model-input"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        inputs = build_dashboard_model_inputs(request.history, policy_target=request.policy_target)
        tensor_hash = ndarray_digest(inputs.tensor, schema_version=1, source="dashboard:model-input")
        return {
            "contract": "crop_tensor_v1",
            "source": "engine/encoding.py",
            "tensor_shape": [int(dim) for dim in inputs.tensor.shape],
            "tensor_hash": tensor_hash,
            "training_input_hash": tensor_hash,
            "dashboard_training_parity": True,
            "legal_table_hash": inputs.legal_table.table_hash,
            "candidate_table_hash": "" if inputs.candidate_table is None else inputs.candidate_table.table_hash,
        }


class ModelOutputInspector:
    name = "model-output"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        output = dict(request.model_output or {})
        heads = output.get("heads", output)
        return {
            "contract": "policy_provider_output_v1",
            "source": "policy_provider",
            "model_family": request.model_family,
            "head_names": sorted(str(key) for key in heads.keys()) if isinstance(heads, dict) else [],
            "output_hash": stable_digest(("model-output", heads)),
        }


class TraceInspector:
    name = "trace"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        return _trace_payload(request.trace)


class ReplayInspector:
    name = "replay"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        return {
            "contract": "ReplayGameRecord",
            "schema_version": REPLAY_RECORD_SCHEMA_VERSION,
            "source": "replay/codec.py",
            "identity": dict(request.replay_identity or {}),
        }


class CheckpointInspector:
    name = "checkpoint"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        manifest = dict(request.checkpoint_manifest or {})
        return {
            "contract": "CheckpointManifest",
            "schema_version": manifest.get("schema_version", 1),
            "manifest_version": manifest.get("manifest_version", manifest.get("schema_version", 1)),
            "model_family": manifest.get("model_family", request.model_family),
            "source": "models/checkpoint.py",
            "manifest_hash": stable_digest(("checkpoint", manifest)),
            "payload": manifest,
        }


class RecipeInspector:
    name = "recipe"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        return {
            "contract": "ModelRecipe",
            "recipe_id": request.recipe_id,
            "config_hash": request.recipe_hash or stable_digest(("recipe", request.recipe_id, request.model_family)),
            "model_family": request.model_family,
            "model_spec_version": MODEL_SPEC_VERSION,
            "source": "tuning/recipes.py",
        }


class AutotuneInspector:
    name = "autotune"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        report = dict(request.autotune_report or {})
        return {
            "contract": "AutotuneReport",
            "source": "tuning/reporting.py",
            "recipe_id": request.recipe_id,
            "score_components": report.get("score_components", {}),
            "scheduler_decisions": report.get("scheduler_decisions", []),
            "watchdogs": report.get("watchdogs", {}),
            "trace_ids": report.get("trace_ids", []),
        }


class DebugBundleInspector:
    name = "debug-bundle"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        return {
            "engine": inspector.inspect("history", history=request.history),
            "contracts": {
                "legal": inspector.inspect("legal-table", history=request.history),
                "candidates": inspector.inspect("candidates", history=request.history, policy_target=request.policy_target),
                "pairs": inspector.inspect(
                    "pairs",
                    history=request.history,
                    policy_target=request.policy_target,
                    pair_policy_target=request.pair_policy_target,
                ),
                "graph": inspector.inspect(
                    "graph",
                    history=request.history,
                    policy_target=request.policy_target,
                    pair_policy_target=request.pair_policy_target,
                ),
                "d6": {"available": True, "view": "d6"},
            },
            "targets": {"policy_target": list(request.policy_target), "pair_policy_target": list(request.pair_policy_target)},
            "model_outputs": inspector.inspect("model-output", history=request.history, model_output=request.model_output),
            "policy_priors": request.model_output or {},
            "mcts": {
                "token_lifecycle": (request.trace or {}).get("mcts_token_lifecycle", "not_present"),
                "token_ids": (request.trace or {}).get("mcts_token_ids", []),
            },
            "replay": inspector.inspect("replay", history=request.history, replay_identity=request.replay_identity),
            "rust_suspicion": {
                "engine_source": "rust",
                "ffi_protocol_source": (request.trace or {}).get("ffi_protocol_source", "not_present"),
                "invariant_probe_status": (request.trace or {}).get("invariant_probe_status", "not_present"),
                "tactical_status_kind": (request.trace or {}).get("tactical_status_kind", "not_present"),
                "structured_rust_error_owner": (request.trace or {}).get("structured_rust_error_owner", "not_present"),
            },
        }


class MismatchInspector:
    name = "mismatch"

    def inspect(self, request: InspectionRequest, inspector: ContractInspector) -> dict[str, Any]:
        del inspector
        left = _facts(request.history, request)
        right = dict(request.compare_to or {})
        mismatches = []
        owner_map = {
            "legal_table_hash": "engine/legal",
            "candidate_contract_hash": "candidate builder",
            "pair_table_hash": "pair table builder",
            "graph_contract_hash": "graph builder",
            "model_input_hash": "train adapter",
            "replay_schema_version": "replay projector",
            "checkpoint_manifest_version": "checkpoint manifest",
        }
        for key, owner in owner_map.items():
            if key in right and right[key] != left.get(key):
                mismatches.append({"field": key, "left": left.get(key), "right": right[key], "likely_owner": owner})
        return {"mismatches": mismatches, "likely_owner": mismatches[0]["likely_owner"] if mismatches else "none"}


def build_dashboard_model_inputs(
    history: bytes,
    *,
    policy_target: tuple[tuple[int, int, float], ...] = (),
    pair_policy_target: tuple[tuple[tuple[int, int], tuple[int, int], float], ...] = (),
    include_pair_rows: bool = False,
) -> DashboardModelInputs:
    tensor, offset_q, offset_r, legal_rows, _legal_bytes = encode_board_and_legal(history)
    rows = [(int(q), int(r)) for q, r in legal_rows.tolist()]
    legal_table = LegalActionTable.from_rows(
        rows,
        source="rust:legal",
        history_hash=_history_hash(history),
        current_player=(len(history) // 12) % 2,
        placements_remaining=1,
    )
    oracle = scan_tactical_oracle_from_history(history, rows, offset_q=offset_q, offset_r=offset_r)
    candidates = CandidateContractBuilder().build(
        rows,
        list(policy_target),
        offset_q=offset_q,
        offset_r=offset_r,
        budget=min(max(len(rows), 1), 512),
        storage_width=min(max(len(rows), 1), 512),
        winning_moves=oracle.win_now_cells,
        forced_block_moves=oracle.forced_block_cells,
        cover_cells=oracle.cover_cells,
        open_four_cells=oracle.open_four_cells,
        open_five_cells=oracle.open_five_cells,
    )
    pair_table = None
    if include_pair_rows and len(rows) >= 2:
        pair_table = PairActionTableBuilder().build(
            candidates,
            list(pair_policy_target),
            strategy=PairStrategy(mode="capped_fill", max_pairs=min(512, max(1, len(rows) * (len(rows) - 1) // 2))),
            legal_moves=rows,
        )
    return DashboardModelInputs(
        tensor=np.asarray(tensor, dtype=np.float32).reshape(1, 13, 33, 33),
        offset_q=int(offset_q),
        offset_r=int(offset_r),
        legal_table=legal_table,
        candidate_table=candidates,
        pair_table=pair_table,
    )


def required_view_names() -> tuple[str, ...]:
    return (
        "history",
        "legal-table",
        "tactical",
        "candidates",
        "pairs",
        "graph",
        "d6",
        "model-input",
        "model-output",
        "trace",
        "replay",
        "checkpoint",
        "recipe",
        "autotune",
    )


def contract_catalog() -> dict[str, Any]:
    return {
        "candidate": {
            "feature_version": CANDIDATE_FEATURE_VERSION,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "feature_width": len(CANDIDATE_FEATURE_NAMES),
        },
        "graph": {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "relation_schema_version": RELATION_SCHEMA_VERSION,
            "feature_dim": GRAPH_FEATURE_DIM,
            "capacity_strategy": GRAPH_CAPACITY_STRATEGY,
            "token_types": {token.name: int(token) for token in GraphTokenType},
            "relation_types": {relation.name: int(relation) for relation in RelationType},
        },
        "registered_model_families": list(get_model_registry().names()),
    }


def _facts(history: bytes, request: InspectionRequest) -> dict[str, Any]:
    facts = {
        "history_hash": _history_hash(history),
        "source": "rust",
        "trace_id": (request.trace or {}).get("trace_id", ""),
        "model_family": request.model_family,
        "model_spec_version": MODEL_SPEC_VERSION,
        "checkpoint_manifest_version": (request.checkpoint_manifest or {}).get("schema_version", 1),
        "recipe_id": request.recipe_id,
        "recipe_config_hash": request.recipe_hash or stable_digest(("recipe", request.recipe_id, request.model_family)),
        "inference_protocol_version": 1,
        "replay_schema_version": REPLAY_RECORD_SCHEMA_VERSION,
    }
    if history:
        try:
            inputs = build_dashboard_model_inputs(history, policy_target=request.policy_target)
            facts.update(
                {
                    "legal_table_hash": inputs.legal_table.table_hash,
                    "legal_table_source": inputs.legal_table.source,
                    "candidate_contract_hash": "" if inputs.candidate_table is None else inputs.candidate_table.table_hash,
                    "candidate_contract_source": "" if inputs.candidate_table is None else inputs.candidate_table.source,
                    "model_input_hash": ndarray_digest(inputs.tensor, schema_version=1, source="dashboard:model-input"),
                }
            )
            graph = build_graph_batch_from_history(history, policy_target=request.policy_target)
            facts.update(
                {
                    "graph_contract_hash": graph.graph_semantic_hash,
                    "graph_schema_version": graph.schema_version,
                    "graph_relation_schema_version": graph.relation_schema_version,
                }
            )
        except Exception as exc:
            facts["facts_error"] = {"owner": "dashboard inspector", "message": str(exc)}
    return facts


def _trace_payload(trace: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(trace or {})
    payload.setdefault("trace_id", "")
    payload.setdefault("span_timings", payload.get("timings_ms", {}))
    return payload


def _history_hash(history: bytes) -> str:
    return stable_digest(("history", bytes(history)))


def _axis_payload(position: dict[str, Any], history: bytes) -> dict[str, Any]:
    axis_input = AxisPolicyInput(
        stones=list(position.get("stones", [])),
        legal_moves=list(position.get("legal_moves", [])),
        current_player=int(position.get("current_player", 0)),
        offset_q=int(position.get("encoding", {}).get("offset_q", -16)),
        offset_r=int(position.get("encoding", {}).get("offset_r", -16)),
        metadata={
            "source": "dashboard_contract_inspector",
            "placements_remaining": int(position.get("placements_remaining", 1)),
            "history_b64": base64.b64encode(history).decode("ascii"),
        },
    )
    results = evaluate_all(axis_input, {})
    return {"prototype_count": len(results), "results": results[:8]}


def _d6_target_checks(transforms: list[dict[str, Any]]) -> dict[str, Any]:
    policy_masses = [float(item["contracts"]["sparse_candidates"].get("target_mass", 0.0)) for item in transforms]
    pair_masses = [float(item["contracts"]["pair_rows"].get("target_mass", 0.0)) for item in transforms]
    graph_policy_masses = [
        float(item["contracts"]["graph_targets"].get("target_masses", {}).get("policy", 0.0))
        for item in transforms
    ]
    graph_pair_masses = [
        float(item["contracts"]["graph_targets"].get("target_masses", {}).get("pair", 0.0))
        for item in transforms
    ]

    def stable(values: list[float]) -> bool:
        return not values or all(abs(value - values[0]) <= 1e-5 for value in values)

    return {
        "policy_target_mass_preserved": stable(policy_masses) and stable(graph_policy_masses),
        "pair_target_mass_preserved": stable(pair_masses) and stable(graph_pair_masses),
        "policy_target_masses": policy_masses,
        "pair_target_masses": pair_masses,
        "graph_policy_target_masses": graph_policy_masses,
        "graph_pair_target_masses": graph_pair_masses,
    }
