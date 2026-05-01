"""Built-in pair strategy descriptors."""

from __future__ import annotations

from hexorl.contracts.pair_strategy.registry import PairStrategyDescriptor, PairStrategyRegistry

MAX_PAIR_CANDIDATES = 512
MAX_GRAPH_PAIRS = 4096

_ALL_RECIPE_FAMILIES = frozenset({"dense_cnn", "restnet", "graph_hybrid", "global"})
_PAIR_CAPABLE_RECIPE_FAMILIES = frozenset({"graph_hybrid", "global"})

NONE_DESCRIPTOR = PairStrategyDescriptor(
    name="none",
    aliases=frozenset(),
    generation_mode="none",
    root_enabled=False,
    leaf_enabled=False,
    diagnostic=False,
    max_pair_rows_field="max_root_pair_rows",
    chunk_cap=0,
    allow_full=False,
    requires_pair_head=False,
    recipe_family_tags=_ALL_RECIPE_FAMILIES,
)

TWO_STAGE_ROOT_ONLY_DESCRIPTOR = PairStrategyDescriptor(
    name="two_stage_root_only",
    aliases=frozenset({"two_stage_root"}),
    generation_mode="capped_fill",
    root_enabled=True,
    leaf_enabled=False,
    diagnostic=False,
    max_pair_rows_field="max_root_pair_rows",
    chunk_cap=MAX_GRAPH_PAIRS,
    allow_full=False,
    requires_pair_head=True,
    recipe_family_tags=frozenset(),
)

TACTICAL_ONLY_DESCRIPTOR = PairStrategyDescriptor(
    name="tactical_only",
    aliases=frozenset({"tactical"}),
    generation_mode="capped_fill",
    root_enabled=True,
    leaf_enabled=False,
    diagnostic=False,
    max_pair_rows_field="max_root_pair_rows",
    chunk_cap=MAX_PAIR_CANDIDATES,
    allow_full=False,
    requires_pair_head=True,
    recipe_family_tags=frozenset(),
)

DIAGNOSTIC_FULL_ROOT_DESCRIPTOR = PairStrategyDescriptor(
    name="diagnostic_full_root",
    aliases=frozenset({"diagnostic_full_pair"}),
    generation_mode="full_capped",
    root_enabled=True,
    leaf_enabled=False,
    diagnostic=True,
    max_pair_rows_field="max_full_pair_rows",
    chunk_cap=MAX_GRAPH_PAIRS,
    allow_full=True,
    requires_pair_head=True,
    recipe_family_tags=_PAIR_CAPABLE_RECIPE_FAMILIES,
)

PAIR_STRATEGY_REGISTRY = PairStrategyRegistry(
    {
        descriptor.name: descriptor
        for descriptor in (
            NONE_DESCRIPTOR,
            TWO_STAGE_ROOT_ONLY_DESCRIPTOR,
            TACTICAL_ONLY_DESCRIPTOR,
            DIAGNOSTIC_FULL_ROOT_DESCRIPTOR,
        )
    }
)
