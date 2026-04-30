"""Global graph tactical head declarations."""

from hexorl.models.heads.pair_policy import GLOBAL_PAIR_HEADS
from hexorl.models.heads.policy import GLOBAL_PLACE_HEAD
from hexorl.models.heads.value import VALUE_HEAD

GLOBAL_GRAPH_OUTPUT_HEADS = (GLOBAL_PLACE_HEAD, *GLOBAL_PAIR_HEADS, VALUE_HEAD)


__all__ = ["GLOBAL_GRAPH_OUTPUT_HEADS"]
