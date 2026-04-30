"""Built-in model family descriptors."""

from hexorl.models.families.dense_cnn import descriptor as dense_cnn_descriptor
from hexorl.models.families.global_line_window import descriptor as global_line_window_descriptor
from hexorl.models.families.global_relation_graph import descriptor as global_relation_graph_descriptor
from hexorl.models.families.global_xattn import descriptor as global_xattn_descriptor
from hexorl.models.families.graph_hybrid import descriptor as graph_hybrid_descriptor
from hexorl.models.families.restnet import descriptor as restnet_descriptor
from hexorl.models.registry import ModelFamilyDescriptor


def builtin_descriptors() -> tuple[ModelFamilyDescriptor, ...]:
    return (
        dense_cnn_descriptor(),
        restnet_descriptor(),
        graph_hybrid_descriptor(),
        global_xattn_descriptor(),
        global_line_window_descriptor(),
        global_relation_graph_descriptor(),
    )


__all__ = [
    "builtin_descriptors",
    "dense_cnn_descriptor",
    "global_line_window_descriptor",
    "global_relation_graph_descriptor",
    "global_xattn_descriptor",
    "graph_hybrid_descriptor",
    "restnet_descriptor",
]
