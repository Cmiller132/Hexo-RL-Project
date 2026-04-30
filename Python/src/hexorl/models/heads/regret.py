"""Regret/value-ranking head names and implementation bindings."""

from hexorl.models.network import RegretRankHead, ValueBinnedHead

REGRET_HEADS = ("regret_value", "regret_rank")


__all__ = ["REGRET_HEADS", "RegretRankHead", "ValueBinnedHead"]
