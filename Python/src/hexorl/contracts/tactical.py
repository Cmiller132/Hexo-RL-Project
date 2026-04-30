"""Tactical status contract payloads."""

from __future__ import annotations

from dataclasses import dataclass


TACTICAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TacticalPayload:
    status: str
    source: str = "rust"
    schema_version: int = TACTICAL_SCHEMA_VERSION
