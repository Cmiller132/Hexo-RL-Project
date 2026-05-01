"""Global graph trunk exports."""

from hexorl.models.trunks.global_line_window import GLOBAL_LINE_WINDOW_TRUNK, GlobalLineWindowTrunk, build_global_line_window_model
from hexorl.models.trunks.global_relation_graph import GLOBAL_RELATION_GRAPH_TRUNK, GlobalRelationGraphTrunk, build_global_relation_graph_model
from hexorl.models.trunks.global_xattn import GLOBAL_XATTN_TRUNK, GlobalXAttnTrunk, build_global_xattn_model

GLOBAL_GRAPH_TRUNK = "global_graph"

__all__ = [
    "GLOBAL_GRAPH_TRUNK",
    "GLOBAL_LINE_WINDOW_TRUNK",
    "GLOBAL_RELATION_GRAPH_TRUNK",
    "GLOBAL_XATTN_TRUNK",
    "GlobalLineWindowTrunk",
    "GlobalRelationGraphTrunk",
    "GlobalXAttnTrunk",
    "build_global_line_window_model",
    "build_global_relation_graph_model",
    "build_global_xattn_model",
]
