"""PolicyProvider registry and row-mapped provider implementations."""

from __future__ import annotations

import time
from typing import Callable, Protocol

import numpy as np

from hexorl.contracts.validation import ContractValidationError
from hexorl.inference.evaluator import Evaluator
from hexorl.models.capabilities import (
    CROP_INPUT,
    DENSE_PLACE_POLICY,
    GLOBAL_GRAPH_INPUT,
    GLOBAL_PLACE_POLICY,
    SPARSE_PLACE_POLICY,
)
from hexorl.models.factory import get_model_registry
from hexorl.models.inference_contracts import OP_GRAPH_PLACE_VALUE, OP_PLACE_VALUE, OP_SPARSE_PLACE_VALUE
from hexorl.models.specs import ModelSpec
from hexorl.search.context import SearchContext
from hexorl.search.priors import (
    PRIOR_SOURCE_DENSE,
    PRIOR_SOURCE_GLOBAL,
    PRIOR_SOURCE_SPARSE,
    SearchEvaluation,
    priors_from_logits,
)


class PolicyProvider(Protocol):
    name: str

    def evaluate_root(self, context: SearchContext) -> SearchEvaluation: ...

    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]: ...


PolicyProviderFactory = Callable[..., PolicyProvider]

_POLICY_PROVIDER_FACTORIES: dict[frozenset[str], type["_BasePolicyProvider"]] = {}
_MODEL_FAMILY_PROVIDER_FACTORIES: dict[str, type["_BasePolicyProvider"]] = {}


def register_policy_provider(required_capabilities: set[str], provider_cls: type["_BasePolicyProvider"]) -> None:
    if not required_capabilities:
        raise ValueError("policy provider registration requires at least one capability")
    _POLICY_PROVIDER_FACTORIES[frozenset(required_capabilities)] = provider_cls


def register_model_family_policy_provider(model_family: str, provider_cls: type["_BasePolicyProvider"]) -> None:
    if not model_family:
        raise ValueError("model-family policy provider registration requires a family name")
    _MODEL_FAMILY_PROVIDER_FACTORIES[str(model_family)] = provider_cls


class _BasePolicyProvider:
    source_label = PRIOR_SOURCE_DENSE

    def __init__(self, *, client: Evaluator | None, model_spec: ModelSpec):
        self.client = client
        self.model_spec = model_spec
        self.name = self.__class__.__name__

    def evaluate_root(self, context: SearchContext) -> SearchEvaluation:
        return self._evaluate_one(context)

    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]:
        return [self._evaluate_one(ctx) for ctx in contexts]

    def _evaluation(
        self,
        context: SearchContext,
        *,
        logits: np.ndarray,
        value: float,
        source: int,
        timing_ms: float,
        raw_metadata: dict[str, object] | None = None,
        fallback_reason: str | None = None,
    ) -> SearchEvaluation:
        priors, fallback = priors_from_logits(logits, fallback_reason=fallback_reason)
        if priors.shape[0] != context.legal_table.rows.shape[0]:
            raise ContractValidationError(
                "provider logits do not map to the legal table row count",
                owner=self.name,
            )
        return SearchEvaluation(
            context=context,
            value=float(value),
            legal_row_ids=np.arange(priors.shape[0], dtype=np.int64),
            legal_dense_indices=context.legal_table.dense_indices,
            row_priors=priors,
            prior_source=np.full(priors.shape[0], int(source), dtype=np.uint8),
            policy_provider=self.name,
            model_family=self.model_spec.kind,
            model_spec_version=str(self.model_spec.version),
            inference_protocol=getattr(getattr(self.client, "manifest", None), "transport", "offline"),
            timings={"policy_provider_ms": float(timing_ms)},
            raw_metadata=raw_metadata or {},
            fallback_reason=fallback,
        )


class DensePolicyProvider(_BasePolicyProvider):
    source_label = PRIOR_SOURCE_DENSE

    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]:
        if not contexts:
            return []
        if self.client is None:
            raise ContractValidationError("dense provider requires an inference client", owner=self.name)
        tensors = []
        for ctx in contexts:
            if ctx.tensor is None:
                raise ContractValidationError("dense provider requires tensor in SearchContext", owner=self.name)
            tensors.append(ctx.tensor.reshape(13, 33, 33))
        t0 = time.monotonic()
        response = self.client.evaluate(
            OP_PLACE_VALUE,
            {"tensor": np.asarray(tensors, dtype=np.float32)},
        )
        policies = response.head_outputs["policy"]
        values = response.head_outputs["value"]
        elapsed = (time.monotonic() - t0) * 1000.0
        dense = np.asarray(policies, dtype=np.float32).reshape(len(contexts), -1)
        value_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return [
            self._evaluation(
                ctx,
                logits=dense[row, ctx.legal_table.dense_indices],
                value=float(value_arr[row]),
                source=self.source_label,
                timing_ms=elapsed,
                raw_metadata={"dense_policy_shape": tuple(dense[row].shape), "batch_size": len(contexts)},
            )
            for row, ctx in enumerate(contexts)
        ]

    def _evaluate_one(self, context: SearchContext) -> SearchEvaluation:
        if context.tensor is None:
            raise ContractValidationError("dense provider requires tensor in SearchContext", owner=self.name)
        if self.client is None:
            raise ContractValidationError("dense provider requires an inference client", owner=self.name)
        t0 = time.monotonic()
        response = self.client.evaluate(
            OP_PLACE_VALUE,
            {"tensor": context.tensor.reshape(1, 13, 33, 33)},
        )
        policy = response.head_outputs["policy"]
        value = response.head_outputs["value"]
        elapsed = (time.monotonic() - t0) * 1000.0
        dense = np.asarray(policy, dtype=np.float32).reshape(-1)
        logits = dense[context.legal_table.dense_indices]
        return self._evaluation(
            context,
            logits=logits,
            value=float(np.asarray(value, dtype=np.float32).reshape(-1)[0]),
            source=self.source_label,
            timing_ms=elapsed,
            raw_metadata={"dense_policy_shape": tuple(dense.shape)},
        )


class RestNetPolicyProvider(DensePolicyProvider):
    pass


class GraphHybridPolicyProvider(_BasePolicyProvider):
    source_label = PRIOR_SOURCE_SPARSE

    def _evaluate_one(self, context: SearchContext) -> SearchEvaluation:
        if context.tensor is None:
            raise ContractValidationError("graph hybrid provider requires tensor in SearchContext", owner=self.name)
        if context.candidate_table is None:
            raise ContractValidationError("graph hybrid provider requires canonical candidate_table in SearchContext", owner=self.name)
        active = np.flatnonzero(context.candidate_table.mask)
        if active.shape[0] == 0:
            raise ContractValidationError("graph hybrid provider requires active candidate rows", owner=self.name)
        if self.client is None:
            raise ContractValidationError("graph hybrid provider requires an inference client", owner=self.name)
        candidate_rows = context.candidate_table.rows[active]
        legal_index = {(int(q), int(r)): idx for idx, (q, r) in enumerate(context.legal_table.rows.tolist())}
        candidate_to_legal = np.asarray(
            [legal_index.get((int(q), int(r)), -1) for q, r in candidate_rows.tolist()],
            dtype=np.int64,
        )
        if np.any(candidate_to_legal < 0):
            raise ContractValidationError("candidate rows are not traceable to legal rows", owner=self.name)
        t0 = time.monotonic()
        response = self.client.evaluate(
            OP_SPARSE_PLACE_VALUE,
            {
                "tensor": context.tensor.reshape(1, 13, 33, 33),
                "candidate_indices": context.candidate_table.dense_indices[active].reshape(1, -1),
                "candidate_features": context.candidate_table.features[active].reshape(1, active.shape[0], context.candidate_table.features.shape[1]),
                "candidate_mask": context.candidate_table.mask[active].reshape(1, -1),
            },
        )
        dense_policy = response.head_outputs["policy"]
        value = response.head_outputs["value"]
        sparse = response.head_outputs["sparse_policy"]
        elapsed = (time.monotonic() - t0) * 1000.0
        logits = np.full(context.legal_table.rows.shape[0], -80.0, dtype=np.float32)
        logits[candidate_to_legal] = np.asarray(sparse, dtype=np.float32).reshape(1, -1)[0, : active.shape[0]]
        if not np.isfinite(logits).all():
            raise ContractValidationError("sparse provider produced non-finite logits", owner=self.name)
        return self._evaluation(
            context,
            logits=logits,
            value=float(np.asarray(value, dtype=np.float32).reshape(-1)[0]),
            source=self.source_label,
            timing_ms=elapsed,
            raw_metadata={
                "candidate_table_hash": context.candidate_table.table_hash,
                "dense_policy_shape": tuple(np.asarray(dense_policy).shape),
            },
        )


class GlobalGraphPolicyProvider(_BasePolicyProvider):
    source_label = PRIOR_SOURCE_GLOBAL

    def _evaluate_one(self, context: SearchContext) -> SearchEvaluation:
        if context.graph_batch is None:
            raise ContractValidationError("global graph provider requires graph_batch in SearchContext", owner=self.name)
        if self.client is None:
            raise ContractValidationError("global graph provider requires an inference client", owner=self.name)
        t0 = time.monotonic()
        response = self.client.evaluate(OP_GRAPH_PLACE_VALUE, _flat_graph_payload(context.graph_batch))
        out = response.head_outputs
        elapsed = (time.monotonic() - t0) * 1000.0
        graph_legal = np.asarray(context.graph_batch.legal_qr, dtype=np.int32).reshape(-1, 2)
        logits = np.asarray(out["policy_place"], dtype=np.float32).reshape(-1)
        legal = context.legal_table.rows
        if graph_legal.shape != legal.shape:
            raise ContractValidationError("global graph legal row count mismatch", owner=self.name)
        graph_index = {(int(q), int(r)): idx for idx, (q, r) in enumerate(graph_legal.tolist())}
        order = []
        for q, r in legal.tolist():
            key = (int(q), int(r))
            if key not in graph_index:
                raise ContractValidationError("global graph logits contain unmapped legal rows", owner=self.name)
            order.append(graph_index[key])
        mapped = logits[np.asarray(order, dtype=np.int64)]
        return self._evaluation(
            context,
            logits=mapped,
            value=float(np.asarray(out["value"], dtype=np.float32).reshape(-1)[0]),
            source=self.source_label,
            timing_ms=elapsed,
            raw_metadata={
                "graph_semantic_hash": getattr(context.graph_batch, "graph_semantic_hash", ""),
                "policy_pair_first_rows": int(np.asarray(out.get("policy_pair_first", [])).shape[0]),
            },
        )


def create_policy_provider(*, model_spec: ModelSpec, client: Evaluator | None) -> PolicyProvider:
    descriptor = get_model_registry().resolve(model_spec)
    family_cls = _MODEL_FAMILY_PROVIDER_FACTORIES.get(descriptor.name)
    if family_cls is not None:
        return family_cls(client=client, model_spec=model_spec)
    capabilities = set(descriptor.capabilities.to_manifest())
    matches: list[tuple[int, type[_BasePolicyProvider]]] = []
    for required, provider_cls in _POLICY_PROVIDER_FACTORIES.items():
        if required.issubset(capabilities):
            matches.append((len(required), provider_cls))
    if not matches:
        raise ValueError(
            "no PolicyProvider registered for model family "
            f"{descriptor.name!r} with capabilities {sorted(capabilities)}"
        )
    _score, cls = max(matches, key=lambda item: item[0])
    return cls(client=client, model_spec=model_spec)


register_policy_provider({GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY}, GlobalGraphPolicyProvider)
register_policy_provider({CROP_INPUT, SPARSE_PLACE_POLICY}, GraphHybridPolicyProvider)
register_policy_provider({CROP_INPUT, DENSE_PLACE_POLICY}, DensePolicyProvider)
register_model_family_policy_provider("restnet", RestNetPolicyProvider)


def _flat_graph_payload(graph_batch) -> dict[str, np.ndarray]:
    return {
        "token_features": np.asarray(graph_batch.token_features, dtype=np.float32),
        "token_type": np.asarray(graph_batch.token_type, dtype=np.int64),
        "token_qr": np.asarray(graph_batch.token_qr, dtype=np.int32),
        "token_mask": np.asarray(graph_batch.token_mask, dtype=np.uint8),
        "legal_token_indices": np.asarray(graph_batch.legal_token_indices, dtype=np.int64),
        "legal_mask": np.asarray(graph_batch.legal_mask, dtype=np.uint8),
        "opp_legal_qr": np.asarray(graph_batch.opp_legal_qr, dtype=np.int32),
        "opp_legal_mask": np.asarray(graph_batch.opp_legal_mask, dtype=np.uint8),
        "pair_token_indices": np.asarray(graph_batch.pair_token_indices, dtype=np.int64),
        "pair_first_indices": np.asarray(graph_batch.pair_first_indices, dtype=np.int64),
        "pair_second_indices": np.asarray(graph_batch.pair_second_indices, dtype=np.int64),
        "relation_type": np.asarray(graph_batch.relation_type, dtype=np.int64),
        "relation_bias": np.asarray(graph_batch.relation_bias, dtype=np.float32),
    }
