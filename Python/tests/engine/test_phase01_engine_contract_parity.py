import numpy as np
import pytest

_engine = pytest.importorskip("_engine")

from hexorl.contracts.history import MoveHistory, encode_move_history
from hexorl.engine.encoding import encode_board_and_legal
from hexorl.engine.history import game_from_history, history_from_game
from hexorl.engine.legal import LegalTableProvider, decode_legal_bytes


def test_engine_history_round_trips_through_contract():
    payload = encode_move_history([(0, 0, 0), (1, 1, 0), (1, 0, 1)])

    game = game_from_history(payload)
    round_tripped = history_from_game(game)

    assert round_tripped == MoveHistory.decode(payload, source="rust")
    assert int(game.current_player) == round_tripped.current_player
    assert int(game.placements_remaining) == round_tripped.placements_remaining


def test_engine_legal_provider_matches_rust_encoding_bytes_and_freezes_rows():
    payload = encode_move_history([(0, 0, 0), (1, 1, 0)])
    game = game_from_history(payload)
    _tensor, _offset_q, _offset_r, legal_bytes = game.encode_board_and_legal(8, False)
    expected = decode_legal_bytes(legal_bytes)

    table = LegalTableProvider(near_radius=8, constrain_threats=False).from_history(payload)

    assert np.array_equal(table.rows, expected)
    assert table.source == "rust:legal"
    assert table.rows.flags.writeable is False
    assert table.debug_payload()["table_hash"] == table.table_hash
    table.assert_semantic_consistency(occupied=MoveHistory.decode(payload, source="rust").stones)


def test_engine_encoding_returns_validated_legal_rows_and_bytes_identity():
    payload = encode_move_history([(0, 0, 0)])

    tensor, offset_q, offset_r, rows, legal_bytes = encode_board_and_legal(payload, 8, False)

    assert tensor.shape == (13, 33, 33)
    assert isinstance(offset_q, int)
    assert isinstance(offset_r, int)
    assert np.array_equal(rows, decode_legal_bytes(legal_bytes))
    assert rows.flags.writeable is False


def test_decode_legal_bytes_rejects_malformed_protocol_width():
    with pytest.raises(ValueError, match="multiple of 8"):
        decode_legal_bytes(b"\x00\x00\x00\x00")
