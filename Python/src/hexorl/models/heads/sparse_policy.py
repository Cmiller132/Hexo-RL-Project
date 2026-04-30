"""Sparse row-policy head names and implementation binding."""

from hexorl.models.network import SparsePolicyHead

SPARSE_POLICY_HEAD = "sparse_policy"
GRAPH_HYBRID_POLICY_HEADS = ("policy", SPARSE_POLICY_HEAD)


__all__ = ["GRAPH_HYBRID_POLICY_HEADS", "SPARSE_POLICY_HEAD", "SparsePolicyHead"]
