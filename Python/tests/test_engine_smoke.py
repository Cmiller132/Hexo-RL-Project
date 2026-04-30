"""Smoke tests for the Rust-compiled Python extension."""
import numpy as np
import pytest

_engine = pytest.importorskip("_engine")

from hexorl.contracts.history import encode_move_history
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.engine.history import game_from_history
from hexorl.engine.legal import decode_legal_bytes
from hexorl.search.context import SearchContext
from hexorl.search.engine_adapter import create_engine_adapter
from hexorl.search.priors import SearchEvaluation


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
    g = game_from_history(encode_move_history([(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, -1, 1), (0, 2, -1)]))

    assert g.move_history() == [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, -1, 1), (0, 2, -1)]
    assert g.current_player == 1
    assert g.placements_remaining == 2

    before = g.move_history()
    with pytest.raises(ValueError, match="duplicate occupied|duplicate cell"):
        game_from_history(encode_move_history([(0, 0, 0), (1, 0, 0)]))
    assert g.move_history() == before


def test_encode_shape():
    g = _engine.PyHexGame()
    g.place(0, 0)
    tensor, oq, or_, legal_bytes = g.encode_board_and_legal(near_radius=2, constrain_threats=False)
    assert tensor.shape == (_engine.NUM_CHANNELS, _engine.BOARD_SIZE, _engine.BOARD_SIZE)
    assert tensor.dtype == np.float32


def test_mcts_runs_to_completion():
    g = _engine.PyHexGame()
    g.place(0, 0)
    engine = _engine.PyMCTSEngine(g, num_simulations=20, c_puct=1.5,
                                   near_radius=2, constrain_threats=False,
                                   c_puct_init=19652.0)
    result = engine.init_root()
    assert result is not None
    tensor, oq, or_, legal_bytes, root_generation = result
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    policy /= policy.sum()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_generation)

    while not engine.done():
        tensor_batch, count, batch_generation = engine.select_leaves(4)
        policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32)
        policies /= policies.sum(axis=1, keepdims=True)
        values = np.zeros(count, dtype=np.float32)
        engine.expand_and_backprop(policies.flatten(), values, batch_generation)

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
    tensor, oq, or_, legal_bytes, root_generation = engine.init_root()
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    return engine, policy, oq, or_, legal_bytes, root_generation


def _initialized_validating_mcts_root():
    g = _engine.PyHexGame()
    g.place(0, 0)
    engine = create_engine_adapter(
        game=g,
        num_simulations=4,
        c_puct=1.5,
        near_radius=2,
        constrain_threats=False,
        c_puct_init=19652.0,
        seed=0,
    )
    tensor, oq, or_, legal_bytes, root_generation = engine.init_root()
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    return engine, tensor, policy, oq, or_, legal_bytes, root_generation


def _root_search_evaluation(engine, tensor, policy, oq, or_, legal_bytes, root_generation):
    rows = decode_legal_bytes(legal_bytes)
    legal = LegalActionTable.from_rows(
        [(int(q), int(r)) for q, r in rows.tolist()],
        source="rust:legal",
        history_hash=stable_digest(("smoke-root", bytes(legal_bytes))),
    )
    row_priors = np.asarray(policy, dtype=np.float32).reshape(-1)[legal.dense_indices]
    context = SearchContext.create(
        phase="root",
        legal_table=legal,
        model_family="dense_cnn",
        tensor=tensor.reshape(1, _engine.NUM_CHANNELS, _engine.BOARD_SIZE, _engine.BOARD_SIZE),
        root_generation=root_generation,
        extra={"offset_q": oq, "offset_r": or_, "legal_bytes": legal_bytes},
    )
    return SearchEvaluation(
        context=context,
        value=0.0,
        legal_row_ids=np.arange(legal.rows.shape[0], dtype=np.int64),
        legal_dense_indices=legal.dense_indices,
        row_priors=row_priors,
        prior_source=np.ones(legal.rows.shape[0], dtype=np.uint8),
        policy_provider="smoke",
        model_family="dense_cnn",
        model_spec_version="v2",
        inference_protocol="smoke",
    )


def test_mcts_rejects_shifted_root_offset():
    engine, tensor, policy, oq, or_, legal_bytes, root_generation = _initialized_validating_mcts_root()
    evaluation = _root_search_evaluation(engine, tensor, policy, oq + 1, or_, legal_bytes, root_generation)
    with pytest.raises(ValueError, match="offset mismatch"):
        engine.expand_root(evaluation)


def test_mcts_rejects_mutated_root_legal_bytes():
    engine, tensor, policy, oq, or_, legal_bytes, root_generation = _initialized_validating_mcts_root()
    rows = np.frombuffer(legal_bytes, dtype=np.int32).copy().reshape(-1, 2)
    assert len(rows) >= 2
    rows[[0, 1]] = rows[[1, 0]]
    with pytest.raises(ValueError, match="root legal row mismatch"):
        engine.expand_root(_root_search_evaluation(engine, tensor, policy, oq, or_, rows.tobytes(), root_generation))


def test_mcts_rejects_malformed_root_legal_bytes():
    engine, tensor, policy, oq, or_, legal_bytes, root_generation = _initialized_validating_mcts_root()
    with pytest.raises(ValueError, match="legal_bytes length .* multiple of 8"):
        engine.expand_root(_root_search_evaluation(engine, tensor, policy, oq, or_, bytes(legal_bytes) + b"\x00", root_generation))


def test_mcts_root_protocol_exposes_current_five_field_shape():
    engine, _policy, _oq, _or_, _legal_bytes, _root_generation = _initialized_mcts_root()
    assert len(engine.init_root()) == 5


def test_mcts_leaf_protocol_exposes_current_three_field_shape():
    engine, policy, oq, or_, legal_bytes, root_generation = _initialized_mcts_root()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_generation)
    _tensor_batch, count, batch_generation = engine.select_leaves(2)
    policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32).reshape(-1)
    values = np.zeros(count, dtype=np.float32)

    engine.expand_and_backprop(policies, values, batch_generation)


@pytest.mark.parametrize("method", ["apply_root_pair_priors", "apply_root_pair_second_priors"])
def test_mcts_rejects_malformed_pair_rows(method):
    engine, policy, oq, or_, legal_bytes, root_generation = _initialized_mcts_root()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes, root_generation)
    pair_qr = np.zeros((1, 3), dtype=np.int32)
    pair_logits = np.zeros(1, dtype=np.float32)

    with pytest.raises(ValueError, match="pair_qr must have shape"):
        getattr(engine, method)(pair_qr, pair_logits, 0.5)


def test_mcts_rejects_non_finite_root_policy():
    engine, tensor, policy, oq, or_, legal_bytes, root_generation = _initialized_validating_mcts_root()
    first_legal = LegalActionTable.from_rows(
        [(int(q), int(r)) for q, r in decode_legal_bytes(legal_bytes).tolist()],
        source="rust:legal",
        history_hash=stable_digest(("smoke-root", bytes(legal_bytes))),
    ).dense_indices[0]
    policy[first_legal] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        engine.expand_root(_root_search_evaluation(engine, tensor, policy, oq, or_, legal_bytes, root_generation))


def test_encode_compact_record_rejects_malformed_history_bytes():
    with pytest.raises(ValueError, match="history_bytes length .* multiple of 12"):
        _engine.encode_compact_record(b"\x00", 2)
