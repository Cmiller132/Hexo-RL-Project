"""Rust-backed move-history helpers."""

from __future__ import annotations

from typing import Iterable

from hexorl.contracts.history import MoveHistory, MoveRow
from hexorl.engine.rust import hex_game_class


def encode_history(rows: Iterable[MoveRow], *, source: str = "rust") -> bytes:
    return MoveHistory.from_rows(tuple(rows), source=source, allow_fixture=source == "fixture").encode()


def decode_history(payload: bytes, *, source: str = "rust", allow_fixture: bool = False) -> MoveHistory:
    return MoveHistory.decode(payload, source=source, allow_fixture=allow_fixture)


def game_from_history(history: bytes | MoveHistory, *, allow_fixture: bool = False):
    contract = history if isinstance(history, MoveHistory) else MoveHistory.decode(history, source="rust", allow_fixture=allow_fixture)
    cls = hex_game_class(required=True)
    game = cls()
    if hasattr(game, "load_history"):
        game.load_history(contract.encode())
        return game
    for player, q, r in contract.rows:
        current = getattr(game, "current_player", player)
        current_player = current() if callable(current) else current
        if int(current_player) != int(player):
            raise ValueError(f"history player mismatch: expected {current_player}, got {player}")
        game.place(int(q), int(r))
    return game


def history_from_game(game: object) -> MoveHistory:
    if not hasattr(game, "move_history_bytes"):
        raise TypeError("Rust game object does not expose move_history_bytes")
    return MoveHistory.decode(game.move_history_bytes(), source="rust")
