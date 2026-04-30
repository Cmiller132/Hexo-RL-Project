"""Typed inference adapters."""

from hexorl.inference.adapters.dense import DensePolicyValueAdapter
from hexorl.inference.adapters.global_graph import GlobalGraphPolicyValueAdapter
from hexorl.inference.adapters.pair_scoring import PairScoringAdapter
from hexorl.inference.adapters.sparse import SparsePolicyValueAdapter

__all__ = [
    "DensePolicyValueAdapter",
    "GlobalGraphPolicyValueAdapter",
    "PairScoringAdapter",
    "SparsePolicyValueAdapter",
]
