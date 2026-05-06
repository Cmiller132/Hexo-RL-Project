"""Explicit pair-prior strategies for search runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAIR_STRATEGY_NONE = "none"
PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR = "diagnostic_full_pair"


@dataclass(frozen=True)
class PairStrategyConfig:
    name: str = PAIR_STRATEGY_NONE
    max_pairs: int = 0
    prior_mix: float = 0.0


@dataclass(frozen=True)
class PairStrategy:
    """Declared pair behavior used by search providers and self-play."""

    config: PairStrategyConfig
    required_output_contracts: tuple[str, ...] = ()
    pair_rows_owned: bool = False
    leaf_pair_scoring_enabled: bool = False

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def enabled(self) -> bool:
        return self.name != PAIR_STRATEGY_NONE

    @property
    def max_pairs(self) -> int:
        return int(self.config.max_pairs)

    @property
    def prior_mix(self) -> float:
        return float(self.config.prior_mix)

    def require_enabled(self, *, context: str) -> None:
        if not self.enabled:
            raise ValueError(f"{context}: pair behavior requires an explicit pair strategy")
        if self.max_pairs <= 0:
            raise ValueError(f"{context}: pair_strategy_max_pairs must be > 0")
        if self.prior_mix <= 0.0:
            raise ValueError(f"{context}: pair_prior_mix must be > 0")

    def require_pair_phase(self, *, second_placement: bool, known_first: bool, context: str) -> None:
        self.require_enabled(context=context)
        if second_placement and not known_first:
            raise ValueError(f"{context}: second-placement pair strategy requires known first action")

    def summary(self) -> dict[str, object]:
        return {
            "pair_strategy": self.name,
            "pair_prior_mix": self.prior_mix,
            "pair_strategy_max_pairs": self.max_pairs,
            "required_output_contracts": list(self.required_output_contracts),
            "pair_rows_owned": bool(self.pair_rows_owned),
            "leaf_pair_scoring_enabled": bool(self.leaf_pair_scoring_enabled),
        }

    @staticmethod
    def has_graph_pair_first(outputs: dict[str, object]) -> bool:
        return "policy_pair_first" in outputs

    @staticmethod
    def graph_pair_first_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        if "policy_pair_first" not in outputs:
            raise ValueError("graph pair strategy requires policy_pair_first output")
        return np.asarray(outputs["policy_pair_first"], dtype=np.float32)[: int(width)]

    @staticmethod
    def graph_pair_joint_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        if "policy_pair_joint" not in outputs:
            raise ValueError("graph pair strategy requires policy_pair_joint output")
        return np.asarray(outputs["policy_pair_joint"], dtype=np.float32)[: int(width)]

    @staticmethod
    def graph_pair_second_logits(outputs: dict[str, object], width: int) -> np.ndarray:
        if "policy_pair_second" not in outputs:
            raise ValueError("graph pair strategy requires policy_pair_second output")
        return np.asarray(outputs["policy_pair_second"], dtype=np.float32)[: int(width)]


def build_pair_strategy(
    name: str,
    *,
    max_pairs: int,
    prior_mix: float,
) -> PairStrategy:
    normalized = str(name).lower()
    if normalized == PAIR_STRATEGY_NONE:
        if int(max_pairs) != 0:
            raise ValueError("pair_strategy_max_pairs must be 0 when pair_strategy='none'")
        return PairStrategy(PairStrategyConfig(normalized, 0, 0.0))
    if normalized == PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR:
        strategy = PairStrategy(
            PairStrategyConfig(normalized, int(max_pairs), float(prior_mix)),
            required_output_contracts=(
                "pair_policy",
                "policy_pair_first",
                "policy_pair_joint",
                "policy_pair_second",
            ),
            pair_rows_owned=True,
            leaf_pair_scoring_enabled=True,
        )
        strategy.require_enabled(context="diagnostic_full_pair")
        return strategy
    raise ValueError(
        f"model.pair_strategy must be one of "
        f"{[PAIR_STRATEGY_NONE, PAIR_STRATEGY_DIAGNOSTIC_FULL_PAIR]}"
    )
