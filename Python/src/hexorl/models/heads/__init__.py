"""Named model head facets used by family descriptors and builders."""

from hexorl.models.heads.pair_policy import CROP_PAIR_HEAD, GLOBAL_PAIR_HEADS, PairPolicyHead
from hexorl.models.heads.policy import DENSE_POLICY_HEADS, GLOBAL_PLACE_HEAD, PolicyHead
from hexorl.models.heads.regret import REGRET_HEADS, RegretRankHead
from hexorl.models.heads.sparse_policy import GRAPH_HYBRID_POLICY_HEADS, SPARSE_POLICY_HEAD, SparsePolicyHead
from hexorl.models.heads.tactical import GLOBAL_GRAPH_OUTPUT_HEADS
from hexorl.models.heads.value import VALUE_HEAD, ValueBinnedHead

__all__ = [
    "CROP_PAIR_HEAD",
    "DENSE_POLICY_HEADS",
    "GLOBAL_GRAPH_OUTPUT_HEADS",
    "GLOBAL_PAIR_HEADS",
    "GLOBAL_PLACE_HEAD",
    "GRAPH_HYBRID_POLICY_HEADS",
    "PairPolicyHead",
    "PolicyHead",
    "REGRET_HEADS",
    "RegretRankHead",
    "SPARSE_POLICY_HEAD",
    "SparsePolicyHead",
    "VALUE_HEAD",
    "ValueBinnedHead",
]
