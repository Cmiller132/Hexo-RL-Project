"""Action contract aliases used by later V2 phases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ActionRow:
    q: int
    r: int
