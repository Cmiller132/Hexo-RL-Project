"""Smoke tests for the Rust-compiled Python extension."""
import numpy as np
import pytest

_engine = pytest.importorskip("_engine")


def test_constants_exported():
    assert hasattr(_engine, "BOARD_SIZE")
    assert hasattr(_engine, "NUM_CHANNELS")
    assert _engine.NUM_CHANNELS == 13
    assert _engine.BOARD_SIZE == 33


def test_game_basic():
    g = _engine.PyHexGame()
    g.unplace()
    g.place(0, 0)
    assert not g.is_over
    g.unplace()
    assert not g.is_over


def test_load_history_replays_chronological_turns_transactionally():
    g = _engine.PyHexGame()
    g.load_history([(0, 0, 0), (1, 0, 1), (0, 1, 1), (-1, 1, 0), (2, -1, 0)])

    assert g.move_history() == [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, -1, 1), (0, 2, -1)]
    assert g.current_player == 1
    assert g.placements_remaining == 2

    before = g.move_history()
    with pytest.raises(ValueError, match="Wrong player"):
        g.load_history([(0, 0, 0), (1, 0, 0)])
    assert g.move_history() == before


def test_encode_shape():
    g = _engine.PyHexGame()
    g.place(0, 0)
    tensor, oq, or_, legal_bytes = g.encode_board_and_legal(near_radius=2, constrain_threats=False)
    assert tensor.shape == (_engine.NUM_CHANNELS, _engine.BOARD_SIZE, _engine.BOARD_SIZE)
    assert tensor.dtype == np.float32


def test_v1_legal_row_table_metadata_and_bytes():
    g = _engine.PyHexGame()
    opening = g.legal_row_table_v1()

    assert opening["schema_version"] == _engine.LEGAL_ROW_SCHEMA_VERSION_V1
    assert opening["schema_hash"] == _engine.LEGAL_ROW_SCHEMA_HASH_V1
    assert opening["phase"] == "opening_single"
    assert opening["query_phase"] == "turn_start"
    assert opening["row_count"] == 1

    g.place(0, 0)
    table = g.legal_row_table_v1()
    rows = np.frombuffer(table["rows_bytes"], dtype=np.int32).reshape(-1, 3)
    legal = np.frombuffer(table["legal_bytes"], dtype=np.int32).reshape(-1, 2)

    assert table["phase"] == "normal_two_placement"
    assert table["placements_remaining"] == 2
    assert table["current_placements_remaining"] == 2
    assert table["row_count"] == len(g.legal_moves()) == rows.shape[0]
    assert table["row_width_bytes"] == 12
    assert table["table_hash"] != 0
    assert np.array_equal(rows[:, 0], np.arange(rows.shape[0], dtype=np.int32))
    assert np.array_equal(rows[:, 1:], legal)

    first_q, first_r = rows[0, 1:].tolist()
    g.place(int(first_q), int(first_r))
    continuation = g.legal_row_table_v1()
    assert continuation["phase"] == "normal_two_placement"
    assert continuation["query_phase"] == "turn_continuation"
    assert continuation["table_hash"] == table["table_hash"]
    assert continuation["first_placement_row_id"] == 0


def test_v1_pair_rows_canonicalize_and_are_deterministic():
    g = _engine.PyHexGame()
    g.place(0, 0)
    legal = g.legal_row_table_v1()
    rows = np.frombuffer(legal["rows_bytes"], dtype=np.int32).reshape(-1, 3)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()

    canonical = g.canonical_pair_rows_v1(
        np.array([[b[0], b[1], a[0], a[1]]], dtype=np.int32)
    )
    pair_rows = np.frombuffer(canonical["rows_bytes"], dtype=np.int32).reshape(-1, 7)
    assert canonical["schema_version"] == _engine.PAIR_ROW_SCHEMA_VERSION_V1
    assert canonical["schema_hash"] == _engine.PAIR_ROW_SCHEMA_HASH_V1
    assert canonical["phase"] == "normal_two_placement"
    assert canonical["legal_row_table_hash"] == legal["table_hash"]
    assert pair_rows.tolist() == [[0, 0, 1, a[0], a[1], b[0], b[1]]]

    full_a = g.pair_row_table_v1()
    full_b = g.pair_row_table_v1()
    full_rows = np.frombuffer(full_a["rows_bytes"], dtype=np.int32).reshape(-1, 7)
    assert full_a["row_count"] == legal["row_count"] * (legal["row_count"] - 1) // 2
    assert full_a["table_hash"] == full_b["table_hash"]
    assert full_a["rows_bytes"] == full_b["rows_bytes"]
    assert full_rows[0].tolist() == [0, 0, 1, a[0], a[1], b[0], b[1]]


def test_v1_pair_rows_reject_duplicate_illegal_and_wrong_phase():
    opening = _engine.PyHexGame()
    with pytest.raises(ValueError, match="normal_two_placement"):
        opening.pair_row_table_v1()

    g = _engine.PyHexGame()
    g.place(0, 0)
    rows = np.frombuffer(g.legal_row_table_v1()["rows_bytes"], dtype=np.int32).reshape(-1, 3)
    a = rows[0, 1:].tolist()
    b = rows[1, 1:].tolist()

    with pytest.raises(ValueError, match="duplicate cell"):
        g.canonical_pair_rows_v1(np.array([[a[0], a[1], a[0], a[1]]], dtype=np.int32))
    with pytest.raises(ValueError, match="duplicate canonical pair"):
        g.canonical_pair_rows_v1(
            np.array(
                [[a[0], a[1], b[0], b[1]], [b[0], b[1], a[0], a[1]]],
                dtype=np.int32,
            )
        )
    with pytest.raises(ValueError, match="illegal cell"):
        g.canonical_pair_rows_v1(np.array([[a[0], a[1], 999, 999]], dtype=np.int32))


def test_v1_terminal_tactical_payload_shape_and_statuses():
    required = {
        "status",
        "winning_single_cells",
        "hot_completion_pairs",
        "terminal_equivalent_pairs",
        "opponent_win_requirements",
        "hot_cover_pairs",
        "impossible_to_cover",
    }
    quiet = _engine.PyHexGame().terminal_tactical_v1()
    assert required <= set(quiet)
    assert quiet["status"] == "quiet"
    assert quiet["impossible_to_cover"] is False

    completion = _engine.PyHexGame()
    completion.set_position([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 0, 2)
    payload = completion.terminal_tactical_v1()
    assert payload["status"] == "hot_completion_available"
    assert any(row[3:7] == (-1, 0, 4, 0) for row in payload["hot_completion_pairs"])

    cover = _engine.PyHexGame()
    cover.set_position([(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)], 0, 2)
    payload = cover.terminal_tactical_v1()
    assert payload["status"] == "hot_cover_required"
    assert (-2, 0) in payload["opponent_win_requirements"]
    assert payload["hot_cover_pairs"]

    impossible = _engine.PyHexGame()
    impossible.set_position(
        [
            (0, 0, 1),
            (1, 0, 1),
            (2, 0, 1),
            (3, 0, 1),
            (4, 0, 1),
            (10, 0, 1),
            (11, 0, 1),
            (12, 0, 1),
            (13, 0, 1),
            (14, 0, 1),
        ],
        0,
        2,
    )
    payload = impossible.terminal_tactical_v1()
    assert payload["status"] == "hot_cover_impossible"
    assert payload["impossible_to_cover"] is True
    assert payload["opponent_win_requirements"]


def test_mcts_runs_to_completion():
    g = _engine.PyHexGame()
    g.place(0, 0)
    engine = _engine.PyMCTSEngine(g, num_simulations=20, c_puct=1.5,
                                   near_radius=2, constrain_threats=False,
                                   c_puct_init=19652.0)
    result = engine.init_root()
    assert result is not None
    tensor, oq, or_, legal_bytes, root_token = result
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    policy /= policy.sum()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_token)

    while not engine.done():
        tensor_batch, count, batch_token = engine.select_leaves(4)
        policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32)
        policies /= policies.sum(axis=1, keepdims=True)
        values = np.zeros(count, dtype=np.float32)
        engine.expand_and_backprop(policies.flatten(), values, batch_token)

    moves_q, moves_r, visits, root_q = engine.get_results()
    assert len(visits) > 0
    assert sum(visits) == 20
    assert -1.0 <= root_q <= 1.0


def _initialized_mcts_root():
    g = _engine.PyHexGame()
    g.place(0, 0)
    engine = _engine.PyMCTSEngine(g, num_simulations=4, c_puct=1.5,
                                   near_radius=2, constrain_threats=False,
                                   c_puct_init=19652.0)
    tensor, oq, or_, legal_bytes, root_token = engine.init_root()
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    return engine, policy, oq, or_, legal_bytes, root_token


def test_mcts_rejects_shifted_root_offset():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    with pytest.raises(ValueError, match="offset mismatch"):
        engine.expand_root(policy, 0.0, oq + 1, or_, legal_bytes, root_token)


def test_mcts_rejects_mutated_root_legal_bytes():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    rows = np.frombuffer(legal_bytes, dtype=np.int32).copy().reshape(-1, 2)
    assert len(rows) >= 2
    rows[[0, 1]] = rows[[1, 0]]
    with pytest.raises(ValueError, match="root legal row mismatch"):
        engine.expand_root(policy, 0.0, oq, or_, rows.tobytes(), root_token)


def test_mcts_rejects_malformed_root_legal_bytes():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    with pytest.raises(ValueError, match="legal_bytes length .* multiple of 8"):
        engine.expand_root(policy, 0.0, oq, or_, bytes(legal_bytes) + b"\x00", root_token)


def test_mcts_rejects_stale_root_token():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    with pytest.raises(ValueError, match="root token mismatch"):
        engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_token + 1)


def test_mcts_rejects_stale_batch_token():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_token)
    _tensor_batch, count, batch_token = engine.select_leaves(2)
    policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32).reshape(-1)
    values = np.zeros(count, dtype=np.float32)

    with pytest.raises(ValueError, match="batch token mismatch"):
        engine.expand_and_backprop(policies, values, batch_token + 1)


@pytest.mark.parametrize("method", ["apply_root_pair_priors", "apply_root_pair_second_priors"])
def test_mcts_rejects_malformed_pair_rows(method):
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_token)
    pair_qr = np.zeros((1, 3), dtype=np.int32)
    pair_logits = np.zeros(1, dtype=np.float32)

    with pytest.raises(ValueError, match="pair_qr must have shape"):
        getattr(engine, method)(pair_qr, pair_logits, 0.5)


def test_mcts_rejects_non_finite_root_policy():
    engine, policy, oq, or_, legal_bytes, root_token = _initialized_mcts_root()
    policy[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_token)


def test_encode_compact_record_rejects_malformed_history_bytes():
    with pytest.raises(ValueError, match="history_bytes length .* multiple of 12"):
        _engine.encode_compact_record(b"\x00", 2)
