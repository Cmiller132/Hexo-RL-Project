"""Public pair strategy registry API."""

from hexorl.contracts.pair_strategy.descriptors import PAIR_STRATEGY_REGISTRY
from hexorl.contracts.pair_strategy.registry import (
    PairStrategyDescriptor,
    PairStrategyName,
    PairStrategyRegistry,
    PairStrategySpec,
)


def resolve(name: str) -> PairStrategyDescriptor:
    return PAIR_STRATEGY_REGISTRY.resolve(name)


def names() -> tuple[str, ...]:
    return PAIR_STRATEGY_REGISTRY.names()


__all__ = [
    "PAIR_STRATEGY_REGISTRY",
    "PairStrategyDescriptor",
    "PairStrategyName",
    "PairStrategyRegistry",
    "PairStrategySpec",
    "names",
    "resolve",
]
