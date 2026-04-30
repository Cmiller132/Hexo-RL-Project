"""Compact move-history contract."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from hexorl.contracts.coordinates import HISTORY_STRIDE, PLACEMENT_RADIUS
from hexorl.contracts.identity import ContractIdentity, stable_digest
from hexorl.contracts.validation import ContractValidationError, validate_source


HISTORY_SCHEMA_VERSION = 1
MoveRow = tuple[int, int, int]


@dataclass(frozen=True)
class MoveHistory:
    rows: tuple[MoveRow, ...]
    source: str = "rust"
    radius: int = PLACEMENT_RADIUS
    schema_version: int = HISTORY_SCHEMA_VERSION
    allow_fixture: bool = False

    def __post_init__(self) -> None:
        source = validate_source(self.source, allow_fixture=self.allow_fixture, owner="MoveHistory")
        rows = tuple((int(p), int(q), int(r)) for p, q, r in self.rows)
        _validate_rows(rows, radius=int(self.radius), source=source)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "radius", int(self.radius))

    @classmethod
    def decode(
        cls,
        payload: bytes,
        *,
        source: str = "rust",
        radius: int = PLACEMENT_RADIUS,
        allow_fixture: bool = False,
    ) -> "MoveHistory":
        if len(payload) % HISTORY_STRIDE != 0:
            raise ContractValidationError(
                "compact move history length must be a multiple of 12",
                owner="MoveHistory.decode",
                source=source,
            )
        rows = [
            struct.unpack_from("<iii", payload, offset)
            for offset in range(0, len(payload), HISTORY_STRIDE)
        ]
        return cls(tuple(rows), source=source, radius=radius, allow_fixture=allow_fixture)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[MoveRow],
        *,
        source: str = "rust",
        radius: int = PLACEMENT_RADIUS,
        allow_fixture: bool = False,
    ) -> "MoveHistory":
        return cls(tuple(rows), source=source, radius=radius, allow_fixture=allow_fixture)

    def encode(self) -> bytes:
        out = bytearray()
        for player, q, r in self.rows:
            out.extend(struct.pack("<iii", int(player), int(q), int(r)))
        return bytes(out)

    @property
    def move_count(self) -> int:
        return len(self.rows)

    @property
    def stones(self) -> dict[tuple[int, int], int]:
        return {(q, r): player for player, q, r in self.rows}

    @property
    def current_player(self) -> int:
        current_player, _placements_remaining = turn_state_after(self.rows)
        return current_player

    @property
    def placements_remaining(self) -> int:
        _current_player, placements_remaining = turn_state_after(self.rows)
        return placements_remaining

    @property
    def history_hash(self) -> str:
        return stable_digest(("MoveHistory", self.schema_version, self.source, self.radius, self.encode()))

    @property
    def identity(self) -> ContractIdentity:
        return ContractIdentity("MoveHistory", self.schema_version, self.source, self.history_hash)

    def debug_payload(self) -> dict[str, object]:
        return {
            "contract": "MoveHistory",
            "schema_version": self.schema_version,
            "source": self.source,
            "history_hash": self.history_hash,
            "move_count": self.move_count,
            "current_player": self.current_player,
            "placements_remaining": self.placements_remaining,
            "rows": [list(row) for row in self.rows],
        }


def encode_move_history(rows: Sequence[MoveRow]) -> bytes:
    return MoveHistory.from_rows(rows, source="fixture", allow_fixture=True).encode()


def decode_move_history(payload: bytes, *, source: str = "rust", allow_fixture: bool = False) -> list[MoveRow]:
    return list(MoveHistory.decode(payload, source=source, allow_fixture=allow_fixture).rows)


def turn_state_after(rows: Sequence[MoveRow]) -> tuple[int, int]:
    current_player = 0
    placements_remaining = 1
    for player, _q, _r in rows:
        if int(player) != current_player:
            raise ContractValidationError(
                f"invalid player order: expected {current_player}, got {player}",
                owner="MoveHistory.turn_state",
            )
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2
    return current_player, placements_remaining


def _validate_rows(rows: tuple[MoveRow, ...], *, radius: int, source: str) -> None:
    if int(radius) <= 0:
        raise ContractValidationError("radius must be positive", owner="MoveHistory", source=source)
    occupied: set[tuple[int, int]] = set()
    current_player = 0
    placements_remaining = 1
    for idx, (player, q, r) in enumerate(rows):
        if player not in (0, 1):
            raise ContractValidationError(f"invalid player at move {idx}", owner="MoveHistory", source=source)
        if player != current_player:
            raise ContractValidationError(
                f"invalid player order at move {idx}: expected {current_player}, got {player}",
                owner="MoveHistory",
                source=source,
            )
        cell = (int(q), int(r))
        if cell in occupied:
            raise ContractValidationError(f"duplicate occupied cell at move {idx}: {cell}", owner="MoveHistory", source=source)
        if idx == 0 and cell != (0, 0) and source != "fixture":
            raise ContractValidationError("opening move must be at origin", owner="MoveHistory", source=source)
        occupied.add(cell)
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2
