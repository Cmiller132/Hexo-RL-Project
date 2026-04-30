"""Contract trace payloads."""

from __future__ import annotations

from dataclasses import dataclass, field


TRACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ContractTrace:
    trace_id: str
    history_hash: str
    model_family: str
    phase: str
    legal_count: int = 0
    candidate_count: int = 0
    pair_rows_total: int = 0
    pair_rows_scored: int = 0
    graph_token_count: int = 0
    graph_relation_count: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: int = TRACE_SCHEMA_VERSION

    def debug_payload(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "history_hash": self.history_hash,
            "model_family": self.model_family,
            "phase": self.phase,
            "legal_count": self.legal_count,
            "candidate_count": self.candidate_count,
            "pair_rows_total": self.pair_rows_total,
            "pair_rows_scored": self.pair_rows_scored,
            "graph_token_count": self.graph_token_count,
            "graph_relation_count": self.graph_relation_count,
            "timings_ms": dict(self.timings_ms),
            "warnings": list(self.warnings),
            "schema_version": self.schema_version,
        }
