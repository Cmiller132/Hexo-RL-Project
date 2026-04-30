"""Pair-policy head names and implementation binding."""

from hexorl.models.network import PairPolicyHead

CROP_PAIR_HEAD = "pair_policy"
GLOBAL_PAIR_HEADS = ("policy_pair_first", "policy_pair_second", "policy_pair_joint")


def pair_policy_heads(*, global_graph: bool) -> tuple[str, ...]:
    return GLOBAL_PAIR_HEADS if global_graph else (CROP_PAIR_HEAD,)


__all__ = ["CROP_PAIR_HEAD", "GLOBAL_PAIR_HEADS", "PairPolicyHead", "pair_policy_heads"]
