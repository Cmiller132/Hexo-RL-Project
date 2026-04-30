"""Place-policy head names and implementation binding."""

from hexorl.models.network import PolicyHead

DENSE_POLICY_HEAD = "policy"
GLOBAL_PLACE_HEAD = "policy_place"
DENSE_POLICY_HEADS = (DENSE_POLICY_HEAD,)


__all__ = ["DENSE_POLICY_HEAD", "DENSE_POLICY_HEADS", "GLOBAL_PLACE_HEAD", "PolicyHead"]
