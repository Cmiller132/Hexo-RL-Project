"""Compatibility exports for graph tensor constants and capacity helpers.

Phase 02 semantic construction lives in `graph.semantic_builder`; tensor
projection lives in `graph.tensorize`; batching lives in `graph.collate`.
Runtime consumers should import those modules directly.
"""

from hexorl.graph.collate import collate_graph_batches
from hexorl.graph.semantic_builder import (
    GRAPH_CAPACITY_STRATEGY,
    GRAPH_FEATURE_DIM,
    GRAPH_IPC_ACTION_CAPACITY,
    GRAPH_IPC_PAIR_CAPACITY,
    GRAPH_IPC_TOKEN_CAPACITY,
    GRAPH_SCHEMA_VERSION,
    PAIR_CHUNK_LIMIT,
    RELATION_SCHEMA_VERSION,
    GraphSemanticBuilder,
    GraphSemanticContract,
    GraphTokenType,
    RelationType,
)
from hexorl.graph.tensorize import (
    GraphBatch,
    GraphCapacityReport,
    GraphTensorizer,
    build_graph_batch_from_history,
    graph_capacity_report,
    graph_batch_with_pair_table,
    validate_graph_ipc_capacity,
)

__all__ = [
    "GRAPH_CAPACITY_STRATEGY",
    "GRAPH_FEATURE_DIM",
    "GRAPH_IPC_ACTION_CAPACITY",
    "GRAPH_IPC_PAIR_CAPACITY",
    "GRAPH_IPC_TOKEN_CAPACITY",
    "GRAPH_SCHEMA_VERSION",
    "PAIR_CHUNK_LIMIT",
    "RELATION_SCHEMA_VERSION",
    "GraphBatch",
    "GraphCapacityReport",
    "GraphSemanticBuilder",
    "GraphSemanticContract",
    "GraphTensorizer",
    "GraphTokenType",
    "RelationType",
    "build_graph_batch_from_history",
    "collate_graph_batches",
    "graph_batch_with_pair_table",
    "graph_capacity_report",
    "validate_graph_ipc_capacity",
]
