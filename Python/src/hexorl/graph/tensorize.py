"""Pure tensor projection from graph semantic contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hexorl.graph.semantic_builder import (
    GRAPH_FEATURE_DIM,
    GRAPH_IPC_ACTION_CAPACITY,
    GRAPH_IPC_PAIR_CAPACITY,
    GRAPH_IPC_TOKEN_CAPACITY,
    GRAPH_CAPACITY_STRATEGY,
    GRAPH_SCHEMA_VERSION,
    RELATION_SCHEMA_VERSION,
    GraphSemanticBuilder,
    GraphSemanticContract,
    GraphTokenType,
)
from hexorl.contracts.pairs import PairActionTable


@dataclass(frozen=True)
class GraphBatch:
    token_features: np.ndarray
    token_type: np.ndarray
    token_qr: np.ndarray
    token_mask: np.ndarray
    legal_token_indices: np.ndarray
    legal_qr: np.ndarray
    legal_mask: np.ndarray
    pair_token_indices: np.ndarray
    pair_first_indices: np.ndarray
    pair_second_indices: np.ndarray
    relation_bias: np.ndarray
    relation_type: np.ndarray
    policy_target: np.ndarray
    opp_legal_qr: np.ndarray
    opp_legal_mask: np.ndarray
    opp_policy_target: np.ndarray
    pair_first_policy_target: np.ndarray
    pair_policy_target: np.ndarray
    tactical_target: np.ndarray
    placements_remaining: int
    current_player: int
    schema_version: int = GRAPH_SCHEMA_VERSION
    relation_schema_version: int = RELATION_SCHEMA_VERSION
    graph_semantic_hash: str = ""


@dataclass(frozen=True)
class GraphCapacityReport:
    token_count: int
    legal_count: int
    opp_legal_count: int
    pair_count: int
    max_tokens: int = GRAPH_IPC_TOKEN_CAPACITY
    max_actions: int = GRAPH_IPC_ACTION_CAPACITY
    max_pairs: int = GRAPH_IPC_PAIR_CAPACITY
    strategy: str = GRAPH_CAPACITY_STRATEGY

    @property
    def fits_ipc(self) -> bool:
        return (
            self.token_count <= self.max_tokens
            and self.legal_count <= self.max_actions
            and self.opp_legal_count <= self.max_actions
            and self.pair_count <= self.max_pairs
        )

    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.token_count > self.max_tokens:
            failures.append("graph_token_capacity")
        if self.legal_count > self.max_actions:
            failures.append("graph_legal_action_capacity")
        if self.opp_legal_count > self.max_actions:
            failures.append("graph_opp_legal_action_capacity")
        if self.pair_count > self.max_pairs:
            failures.append("graph_pair_chunk_capacity")
        return tuple(failures)


class GraphTensorizer:
    """Project graph semantics to model-facing NumPy tensors without rebuilding rows."""

    def tensorize(self, semantic: GraphSemanticContract) -> GraphBatch:
        _validate_semantic_shapes(semantic)
        return GraphBatch(
            token_features=np.array(semantic.token_features, dtype=np.float32, copy=True),
            token_type=np.array(semantic.token_type, dtype=np.int64, copy=True),
            token_qr=np.array(semantic.token_qr, dtype=np.int32, copy=True),
            token_mask=np.array(semantic.token_mask, dtype=np.bool_, copy=True),
            legal_token_indices=np.array(semantic.legal_token_indices, dtype=np.int64, copy=True),
            legal_qr=np.array(semantic.legal_qr, dtype=np.int32, copy=True),
            legal_mask=np.array(semantic.legal_mask, dtype=np.bool_, copy=True),
            pair_token_indices=np.array(semantic.pair_token_indices, dtype=np.int64, copy=True),
            pair_first_indices=np.array(semantic.pair_first_indices, dtype=np.int64, copy=True),
            pair_second_indices=np.array(semantic.pair_second_indices, dtype=np.int64, copy=True),
            relation_bias=np.array(semantic.relation_bias, dtype=np.float32, copy=True),
            relation_type=np.array(semantic.relation_type, dtype=np.int64, copy=True),
            policy_target=np.array(semantic.policy_target, dtype=np.float32, copy=True),
            opp_legal_qr=np.array(semantic.opp_legal_qr, dtype=np.int32, copy=True),
            opp_legal_mask=np.array(semantic.opp_legal_mask, dtype=np.bool_, copy=True),
            opp_policy_target=np.array(semantic.opp_policy_target, dtype=np.float32, copy=True),
            pair_first_policy_target=np.array(semantic.pair_first_policy_target, dtype=np.float32, copy=True),
            pair_policy_target=np.array(semantic.pair_policy_target, dtype=np.float32, copy=True),
            tactical_target=np.array(semantic.tactical_target, dtype=np.float32, copy=True),
            placements_remaining=int(semantic.placements_remaining),
            current_player=int(semantic.current_player),
            schema_version=int(semantic.schema_version),
            relation_schema_version=int(semantic.relation_schema_version),
            graph_semantic_hash=graph_semantic_hash(semantic),
        )


def build_graph_batch_from_history(
    history: bytes,
    *,
    policy_target: Sequence[tuple[int, int, float]] = (),
    opp_legal_moves: Sequence[tuple[int, int]] | None = None,
    opp_policy_target: Sequence[tuple[int, int, float]] = (),
    pair_policy_target: Sequence[tuple[tuple[int, int], tuple[int, int], float]] = (),
    radius: int = 8,
    max_pair_rows: int = 4096,
    allow_pair_truncation: bool = False,
    include_pair_rows: bool = False,
) -> GraphBatch:
    semantic = GraphSemanticBuilder().build(
        history,
        policy_target=policy_target,
        opp_legal_moves=opp_legal_moves,
        opp_policy_target=opp_policy_target,
        pair_policy_target=pair_policy_target,
        radius=radius,
        max_pair_rows=max_pair_rows,
        allow_pair_truncation=allow_pair_truncation,
        include_pair_rows=include_pair_rows,
    )
    return GraphTensorizer().tensorize(semantic)


def graph_batch_with_pair_table(graph_batch: GraphBatch, pair_table: PairActionTable) -> GraphBatch:
    """Project canonical pair rows onto an existing graph tensor batch."""
    legal_tokens = np.asarray(graph_batch.legal_token_indices, dtype=np.int64)
    first_refs = np.asarray(pair_table.first_candidate_rows, dtype=np.int64)
    second_refs = np.asarray(pair_table.second_candidate_rows, dtype=np.int64)
    mask = np.asarray(pair_table.mask, dtype=np.bool_)
    active = np.flatnonzero(mask)
    if pair_table.phase == "second_placement_known_first":
        known_first = pair_table.known_first
        stone_tokens = [
            idx
            for idx, (tt, qr) in enumerate(zip(np.asarray(graph_batch.token_type).tolist(), np.asarray(graph_batch.token_qr).tolist()))
            if int(tt) == int(GraphTokenType.STONE) and tuple(int(x) for x in qr) == known_first
        ]
        if not stone_tokens:
            raise ValueError("pair table known_first row is not present as a graph stone token")
        first_token = int(stone_tokens[-1])
        pair_first = np.full(active.shape[0], first_token, dtype=np.int64)
        pair_second = np.asarray([legal_tokens[int(second_refs[idx]) - 1] for idx in active], dtype=np.int64)
    else:
        pair_first = np.asarray([legal_tokens[int(first_refs[idx])] for idx in active], dtype=np.int64)
        pair_second = np.asarray([legal_tokens[int(second_refs[idx])] for idx in active], dtype=np.int64)
    pair_target = np.asarray(pair_table.target[active], dtype=np.float32)
    pair_first_target = np.zeros(graph_batch.legal_qr.shape[0], dtype=np.float32)
    if pair_table.phase == "first_placement":
        source_first = np.asarray(pair_table.first_policy_target, dtype=np.float32)
        pair_first_target[: min(pair_first_target.shape[0], source_first.shape[0])] = source_first[: pair_first_target.shape[0]]
    return GraphBatch(
        token_features=graph_batch.token_features,
        token_type=graph_batch.token_type,
        token_qr=graph_batch.token_qr,
        token_mask=graph_batch.token_mask,
        legal_token_indices=graph_batch.legal_token_indices,
        legal_qr=graph_batch.legal_qr,
        legal_mask=graph_batch.legal_mask,
        pair_token_indices=np.zeros(pair_first.shape[0], dtype=np.int64),
        pair_first_indices=pair_first,
        pair_second_indices=pair_second,
        relation_bias=graph_batch.relation_bias,
        relation_type=graph_batch.relation_type,
        policy_target=graph_batch.policy_target,
        opp_legal_qr=graph_batch.opp_legal_qr,
        opp_legal_mask=graph_batch.opp_legal_mask,
        opp_policy_target=graph_batch.opp_policy_target,
        pair_first_policy_target=pair_first_target,
        pair_policy_target=pair_target,
        tactical_target=graph_batch.tactical_target,
        placements_remaining=graph_batch.placements_remaining,
        current_player=graph_batch.current_player,
        schema_version=graph_batch.schema_version,
        relation_schema_version=graph_batch.relation_schema_version,
        graph_semantic_hash=graph_batch.graph_semantic_hash,
    )


def graph_capacity_report(graph_batch: GraphBatch) -> GraphCapacityReport:
    return GraphCapacityReport(
        token_count=int(np.asarray(graph_batch.token_features).shape[-2]),
        legal_count=int(np.asarray(graph_batch.legal_qr).shape[-2]),
        opp_legal_count=int(np.asarray(graph_batch.opp_legal_qr).shape[-2]),
        pair_count=int(np.asarray(graph_batch.pair_token_indices).shape[-1]),
    )


def validate_graph_ipc_capacity(graph_batch: GraphBatch) -> GraphCapacityReport:
    report = graph_capacity_report(graph_batch)
    if not report.fits_ipc:
        raise ValueError(
            "global graph IPC capacity exceeded; "
            f"tokens={report.token_count}/{report.max_tokens}, "
            f"legal={report.legal_count}/{report.max_actions}, "
            f"opp_legal={report.opp_legal_count}/{report.max_actions}, "
            f"pairs={report.pair_count}/{report.max_pairs}. "
            f"strategy={report.strategy}; lower max-game length, bucket/microbatch this position, "
            "or score pair rows through chunks. Semantic legal, stone, and tactical rows were not dropped."
        )
    return report


def graph_semantic_hash(semantic: GraphSemanticContract) -> str:
    import hashlib

    hasher = hashlib.sha256()
    for arr in (
        semantic.token_type,
        semantic.token_qr,
        semantic.legal_token_indices,
        semantic.legal_qr,
        semantic.pair_token_indices,
        semantic.pair_first_indices,
        semantic.pair_second_indices,
        semantic.relation_type,
    ):
        data = np.ascontiguousarray(arr).tobytes()
        hasher.update(len(data).to_bytes(8, "little", signed=False))
        hasher.update(data)
    hasher.update(str(int(semantic.schema_version)).encode("utf-8"))
    hasher.update(str(int(semantic.relation_schema_version)).encode("utf-8"))
    return hasher.hexdigest()


def _validate_semantic_shapes(semantic: GraphSemanticContract) -> None:
    token_count = int(np.asarray(semantic.token_type).shape[0])
    legal_count = int(np.asarray(semantic.legal_qr).shape[0])
    pair_count = int(np.asarray(semantic.pair_first_indices).shape[0])
    if np.asarray(semantic.token_features).shape != (token_count, GRAPH_FEATURE_DIM):
        raise ValueError("graph tensorizer received token feature shape that does not match semantic token count")
    if np.asarray(semantic.token_qr).shape != (token_count, 2):
        raise ValueError("graph tensorizer received token_qr shape that does not match semantic token count")
    if np.asarray(semantic.legal_token_indices).shape != (legal_count,):
        raise ValueError("graph tensorizer received legal token references that do not match legal rows")
    if np.asarray(semantic.pair_second_indices).shape != (pair_count,):
        raise ValueError("graph tensorizer received pair reference length mismatch")
    if np.asarray(semantic.relation_type).shape != (token_count, token_count):
        raise ValueError("graph tensorizer received relation_type shape that does not match token count")
    if np.asarray(semantic.relation_bias).shape != (1, token_count, token_count):
        raise ValueError("graph tensorizer received relation_bias shape that does not match token count")
