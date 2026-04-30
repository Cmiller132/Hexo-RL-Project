"""Graph contract identity types for Phase 01."""

from __future__ import annotations

from dataclasses import dataclass


GRAPH_CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GraphContractIdentity:
    schema_version: int
    source: str
    graph_hash: str
