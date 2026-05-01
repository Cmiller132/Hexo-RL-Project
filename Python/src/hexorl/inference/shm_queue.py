"""Compatibility exports for the contract-derived shared-memory arena."""

from hexorl.inference.arena import (
    ArenaLayout,
    InferenceQueue,
    SharedEvent,
    TensorViewSpec,
    WorkerSlots,
    arena_layout_from_manifest,
    connect_inference_queue,
    create_inference_queue,
)
from hexorl.models.inference_contracts import (
    BOARD_AREA,
    BOARD_SIZE,
    CANDIDATE_FEATURES,
    GRAPH_FEATURE_DIM,
    GRAPH_SCHEMA_VERSION,
    MAX_CANDIDATES,
    MAX_GRAPH_ACTIONS,
    MAX_GRAPH_PAIRS,
    MAX_GRAPH_TOKENS,
    MAX_PAIR_CANDIDATES,
    RELATION_SCHEMA_VERSION,
)

NUM_CHANNELS = 13

__all__ = [
    "ArenaLayout",
    "BOARD_AREA",
    "BOARD_SIZE",
    "CANDIDATE_FEATURES",
    "GRAPH_FEATURE_DIM",
    "GRAPH_SCHEMA_VERSION",
    "InferenceQueue",
    "MAX_CANDIDATES",
    "MAX_GRAPH_ACTIONS",
    "MAX_GRAPH_PAIRS",
    "MAX_GRAPH_TOKENS",
    "MAX_PAIR_CANDIDATES",
    "NUM_CHANNELS",
    "RELATION_SCHEMA_VERSION",
    "SharedEvent",
    "TensorViewSpec",
    "WorkerSlots",
    "arena_layout_from_manifest",
    "connect_inference_queue",
    "create_inference_queue",
]
