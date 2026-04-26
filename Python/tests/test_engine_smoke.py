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
    tensor, oq, or_, legal_bytes = result
    policy = np.ones(_engine.BOARD_SIZE ** 2, dtype=np.float32)
    policy /= policy.sum()
    engine.expand_root(policy, 0.0, oq, or_, legal_bytes)

    while not engine.done():
        tensor_batch, count = engine.select_leaves(4)
        policies = np.ones((count, _engine.BOARD_SIZE ** 2), dtype=np.float32)
        policies /= policies.sum(axis=1, keepdims=True)
        values = np.zeros(count, dtype=np.float32)
        engine.expand_and_backprop(policies.flatten(), values)

    moves_q, moves_r, visits, root_q = engine.get_results()
    assert len(visits) > 0
    assert sum(visits) == 20
    assert -1.0 <= root_q <= 1.0
