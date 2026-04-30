"""Replay contract identity types for Phase 01."""

from __future__ import annotations

from dataclasses import dataclass


REPLAY_RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReplayRecordIdentity:
    schema_version: int
    source: str
    record_hash: str
