"""Model family capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DENSE_PLACE_POLICY = "DENSE_PLACE_POLICY"
SPARSE_PLACE_POLICY = "SPARSE_PLACE_POLICY"
GLOBAL_PLACE_POLICY = "GLOBAL_PLACE_POLICY"
PAIR_FIRST_POLICY = "PAIR_FIRST_POLICY"
PAIR_SECOND_POLICY = "PAIR_SECOND_POLICY"
JOINT_PAIR_POLICY = "JOINT_PAIR_POLICY"
REGRET_HEAD = "REGRET_HEAD"
GLOBAL_GRAPH_INPUT = "GLOBAL_GRAPH_INPUT"
CROP_INPUT = "CROP_INPUT"


@dataclass(frozen=True)
class CapabilitySet:
    """Immutable capability set owned by a registered model family."""

    names: frozenset[str]

    @classmethod
    def of(cls, values: Iterable[str]) -> "CapabilitySet":
        return cls(frozenset(str(value) for value in values))

    def has(self, name: str) -> bool:
        return name in self.names

    def require(self, name: str) -> None:
        if name not in self.names:
            raise ValueError(f"model family does not expose required capability {name}")

    def to_manifest(self) -> list[str]:
        return sorted(self.names)
