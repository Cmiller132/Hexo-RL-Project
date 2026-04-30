"""Single-position debug payloads for contract validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SinglePositionDebugPayload:
    trace_id: str
    history: dict[str, Any]
    legal_table: dict[str, Any]
    d6: dict[str, Any]
    validation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "history": self.history,
            "legal_table": self.legal_table,
            "d6": self.d6,
            "validation": self.validation,
        }
