"""PolicyProvider registry and row-mapped provider implementations."""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from hexorl.contracts.validation import ContractValidationError
from hexorl.inference.client import InferenceClient
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


class _BasePolicyProvider:
    source_label = PRIOR_SOURCE_DENSE

    def __init__(self, *, client: InferenceClient | None, model_spec: ModelSpec):
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

    def _uniform_fallback(self, context: SearchContext, *, reason: str) -> SearchEvaluation:
        width = int(context.legal_table.rows.shape[0])
        return self._evaluation(
            context,
            logits=np.zeros(width, dtype=np.float32),
            value=0.0,
            source=self.source_label,
            timing_ms=0.0,
            fallback_reason=reason,
        )


class DensePolicyProvider(_BasePolicyProvider):
    source_label = PRIOR_SOURCE_DENSE

    def evaluate_leaves(self, contexts: list[SearchContext]) -> list[SearchEvaluation]:
        if not contexts:
            return []
        if self.client is None:
            return [self._uniform_fallback(ctx, reason="no_inference_client") for ctx in contexts]
        tensors = []
        for ctx in contexts:
            if ctx.tensor is None:
                raise ContractValidationError("dense provider requires tensor in SearchContext", owner=self.name)
            tensors.append(ctx.tensor.reshape(13, 33, 33))
        t0 = time.monotonic()
        policies, values = self.client.evaluate_dense(np.asarray(tensors, dtype=np.float32), len(contexts))
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
            return self._uniform_fallback(context, reason="no_inference_client")
        t0 = time.monotonic()
        policy, value = self.client.evaluate_dense(context.tensor.reshape(1, 13, 33, 33), 1)
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
        if context.tensor is None or context.candidate_table is None:
            return DensePolicyProvider(client=self.client, model_spec=self.model_spec)._evaluate_one(context)
        active = np.flatnonzero(context.candidate_table.mask)
        if active.shape[0] == 0 or self.client is None:
            return self._uniform_fallback(context, reason="empty_candidate_table")
        candidate_rows = context.candidate_table.rows[active]
        legal_index = {(int(q), int(r)): idx for idx, (q, r) in enumerate(context.legal_table.rows.tolist())}
        candidate_to_legal = np.asarray(
            [legal_index.get((int(q), int(r)), -1) for q, r in candidate_rows.tolist()],
            dtype=np.int64,
        )
        if np.any(candidate_to_legal < 0):
            raise ContractValidationError("candidate rows are not traceable to legal rows", owner=self.name)
        t0 = time.monotonic()
        dense_policy, value, sparse = self.client.evaluate_sparse(
            context.tensor.reshape(1, 13, 33, 33),
            1,
            context.candidate_table.dense_indices[active].reshape(1, -1),
            context.candidate_table.features[active].reshape(1, active.shape[0], context.candidate_table.features.shape[1]),
            context.candidate_table.mask[active].reshape(1, -1),
        )
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
            return self._uniform_fallback(context, reason="no_inference_client")
        t0 = time.monotonic()
        out = self.client.evaluate_global_graph(context.graph_batch)
        elapsed = (time.monotonic() - t0) * 1000.0
        graph_legal = np.asarray(out["metadata"]["legal_qr"], dtype=np.int32).reshape(-1, 2)
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


def create_policy_provider(*, model_spec: ModelSpec, client: InferenceClient | None) -> PolicyProvider:
    providers = {
        "dense_cnn": DensePolicyProvider,
        "restnet": RestNetPolicyProvider,
        "graph_hybrid": GraphHybridPolicyProvider,
        "global_xattn": GlobalGraphPolicyProvider,
        "global_line_window": GlobalGraphPolicyProvider,
        "global_relation_graph": GlobalGraphPolicyProvider,
    }
    try:
        cls = providers[model_spec.kind]
    except KeyError as exc:
        raise ValueError(f"no PolicyProvider registered for model kind {model_spec.kind!r}") from exc
    return cls(client=client, model_spec=model_spec)
