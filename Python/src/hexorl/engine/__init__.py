"""V2 engine boundary package."""

from hexorl.engine.encoding import apply_d6_symmetry, encode_board_and_legal, encode_compact_record
from hexorl.engine.history import decode_history, encode_history, game_from_history, history_from_game
from hexorl.engine.legal import (
    LegalTableProvider,
    decode_legal_bytes,
    legal_rows_from_history,
    legal_rows_from_stones,
    legal_table_from_stones,
)
from hexorl.engine.rust import EngineUnavailableError, engine_available, engine_module, hex_game_class, mcts_engine_class

__all__ = [
    "EngineUnavailableError",
    "LegalTableProvider",
    "apply_d6_symmetry",
    "decode_history",
    "decode_legal_bytes",
    "encode_board_and_legal",
    "encode_compact_record",
    "encode_history",
    "engine_available",
    "engine_module",
    "game_from_history",
    "hex_game_class",
    "history_from_game",
    "legal_rows_from_history",
    "legal_rows_from_stones",
    "legal_table_from_stones",
    "mcts_engine_class",
]
