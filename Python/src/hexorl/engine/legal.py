"""Rust-backed legal table provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hexorl.contracts.history import MoveHistory
from hexorl.contracts.legal import LegalActionTable
from hexorl.engine.history import game_from_history
from hexorl.engine.rust import hex_game_class


def decode_legal_bytes(legal_bytes: bytes) -> np.ndarray:
    if len(legal_bytes) % 8 != 0:
        raise ValueError("legal_bytes length must be a multiple of 8")
    arr = np.frombuffer(legal_bytes, dtype=np.int32)
    if arr.size % 2 != 0:
        raise ValueError("legal row byte payload must contain int32 pairs")
    out = np.array(arr.reshape(-1, 2), dtype=np.int32, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class LegalTableProvider:
    near_radius: int = 8
    constrain_threats: bool = True

    def from_history(self, history: bytes | MoveHistory) -> LegalActionTable:
        contract = history if isinstance(history, MoveHistory) else MoveHistory.decode(history, source="rust")
        game = game_from_history(contract)
        return self.from_game(game, history=contract)

    def from_game(self, game: object, *, history: MoveHistory | None = None) -> LegalActionTable:
        if not hasattr(game, "encode_board_and_legal"):
            raise RuntimeError(
                "LegalTableProvider requires the Rust/PyO3 encode_board_and_legal legal-byte protocol; "
                "arbitrary legal-move objects are not accepted as rust legal rows"
            )
        _tensor, _offset_q, _offset_r, legal_bytes = game.encode_board_and_legal(
            int(self.near_radius),
            bool(self.constrain_threats),
        )
        rows = decode_legal_bytes(legal_bytes)
        current_player = _attr_int(game, "current_player", history.current_player if history else 0)
        placements_remaining = _attr_int(game, "placements_remaining", history.placements_remaining if history else 1)
        terminal = bool(_attr_value(game, "is_over", False))
        table = LegalActionTable.from_rows(
            [(int(q), int(r)) for q, r in rows.tolist()],
            source="rust:legal",
            radius=int(self.near_radius),
            occupied_count=len(_stones_from_game(game, history)),
            current_player=current_player,
            placements_remaining=placements_remaining,
            history_hash=history.history_hash if history else "",
        )
        table.assert_semantic_consistency(occupied=set(_stones_from_game(game, history)), terminal=terminal)
        return table


def legal_table_from_stones(
    stones: dict[tuple[int, int], int],
    *,
    current_player: int = 0,
    near_radius: int = 8,
    constrain_threats: bool = True,
) -> LegalActionTable:
    cls = hex_game_class(required=True)
    game = cls()
    if hasattr(game, "set_position"):
        game.set_position([(int(q), int(r), int(player)) for (q, r), player in stones.items()], int(current_player))
    else:
        for (_q, _r), _player in sorted(stones.items()):
            raise RuntimeError("Rust game does not expose set_position for synthetic legal table construction")
    return LegalTableProvider(near_radius=near_radius, constrain_threats=constrain_threats).from_game(game)


def legal_rows_from_history(history: bytes | MoveHistory, *, near_radius: int = 8, constrain_threats: bool = True) -> np.ndarray:
    return LegalTableProvider(near_radius=near_radius, constrain_threats=constrain_threats).from_history(history).rows


def legal_rows_from_stones(stones: dict[tuple[int, int], int], *, radius: int = 8, current_player: int = 0) -> list[tuple[int, int]]:
    table = legal_table_from_stones(stones, current_player=current_player, near_radius=radius)
    return [(int(q), int(r)) for q, r in table.rows.tolist()]


def _attr_value(obj: object, name: str, default):
    value = getattr(obj, name, default)
    return value() if callable(value) else value


def _attr_int(obj: object, name: str, default: int) -> int:
    return int(_attr_value(obj, name, default))


def _stones_from_game(game: object, history: MoveHistory | None) -> dict[tuple[int, int], int]:
    if history is not None:
        return history.stones
    if hasattr(game, "board_pieces"):
        return {(int(q), int(r)): int(player) for q, r, player in game.board_pieces()}
    return {}
