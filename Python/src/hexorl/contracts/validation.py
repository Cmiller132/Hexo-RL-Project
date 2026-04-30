"""Validation errors and source policy for V2 contracts."""

from __future__ import annotations


PRODUCTION_SOURCES = frozenset({"rust", "rust:legal", "rust:history", "rust:encoding", "rust:synthetic"})
FIXTURE_SOURCES = frozenset({"fixture"})
FORBIDDEN_SOURCES = frozenset({"fallback"})


class ContractValidationError(ValueError):
    """Raised when a contract rejects malformed or wrong-source data."""

    def __init__(self, message: str, *, owner: str = "contract_validation", source: str | None = None):
        self.owner = owner
        self.source = source
        suffix = f" [owner={owner}" + (f", source={source}" if source is not None else "") + "]"
        super().__init__(message + suffix)


def validate_source(source: str, *, allow_fixture: bool = False, owner: str = "source_policy") -> str:
    normalized = str(source).strip().lower()
    if normalized in FORBIDDEN_SOURCES:
        raise ContractValidationError("fallback source is not allowed", owner=owner, source=normalized)
    if normalized in FIXTURE_SOURCES and not allow_fixture:
        raise ContractValidationError("fixture source requires explicit opt-in", owner=owner, source=normalized)
    if normalized not in PRODUCTION_SOURCES and normalized not in FIXTURE_SOURCES:
        raise ContractValidationError("unknown contract source", owner=owner, source=normalized)
    return normalized


def require(condition: bool, message: str, *, owner: str = "contract_validation", source: str | None = None) -> None:
    if not condition:
        raise ContractValidationError(message, owner=owner, source=source)
