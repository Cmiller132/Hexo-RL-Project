"""Global graph IPC capacity constants.

Token capacity is intentionally separate from action-row capacity.  Raising
tokens also raises relation matrix storage quadratically, while action rows are
linear shared-memory tables used for legal and opponent-policy logits.
"""

GRAPH_IPC_TOKEN_CAPACITY = 4096
GRAPH_IPC_ACTION_CAPACITY = 8192
GRAPH_IPC_PAIR_CAPACITY = 4096
GRAPH_IPC_BATCH_CAPACITY = 8
GRAPH_IPC_RELATION_EDGE_CAPACITY = 524288
PAIR_CHUNK_LIMIT = GRAPH_IPC_PAIR_CAPACITY
GRAPH_CAPACITY_STRATEGY = "preserve_legal_stone_tactical_rows_fail_or_chunk_context"
