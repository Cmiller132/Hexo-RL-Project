import struct

import numpy as np
import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import _py_decode_compact_record
from hexorl.buffer.targets import process_game_record
from hexorl.selfplay.records import GameRecord, PositionRecord, BOARD_SIZE, action_to_board_index
from hexorl.train.losses import compute_losses


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def test_python_decoder_returns_final_position_for_history():
    history = _move(0, 0, 0)
    decoded = _py_decode_compact_record(history)

    assert decoded.shape == (2, 13, BOARD_SIZE, BOARD_SIZE)
    assert decoded[0, 0].sum() == 0.0
    assert decoded[-1, 0, BOARD_SIZE // 2, BOARD_SIZE // 2] == 1.0


def test_process_game_record_populates_auxiliary_targets():
    game = GameRecord(
        positions=[
            PositionRecord(b"", {1: 1.0}, 0.2, player=0, turn_index=0),
            PositionRecord(_move(0, 0, 0), {2: 1.0}, -0.1, player=1, turn_index=1),
        ],
        outcome=1.0,
        final_move_history=_move(0, 0, 0) + _move(1, 1, 0),
    )

    process_game_record(game, lookahead_horizons=[1], lookahead_lambdas=[0.5])

    assert game.positions[0].opp_policy_target == {2: 1.0}
    assert game.positions[0].moves_left == 2.0
    assert game.positions[1].moves_left == 1.0
    assert game.positions[0].regret_rank >= 0.0
    assert len(game.positions[0].lookahead_values) == 1


def test_ring_buffer_preserves_auxiliary_targets():
    rec = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.0,
        player=0,
        outcome=1.0,
        opp_policy_target={action_to_board_index(1, 0): 1.0},
        regret_rank=0.25,
        regret_value=-0.5,
        axis_label=2,
        moves_left=7.0,
    )
    buffer = RingBuffer(capacity=4)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert out.opp_policy_target == rec.opp_policy_target
    assert out.regret_rank == rec.regret_rank
    assert out.axis_label == rec.axis_label
    assert out.moves_left == rec.moves_left


def test_compute_losses_skips_missing_targets_and_handles_batch_one():
    predictions = {
        "policy": torch.zeros(1, 1089),
        "value": torch.zeros(1, 65),
        "regret_rank": torch.zeros(1, 1),
        "axis": torch.zeros(1, 3),
        "moves_left": torch.ones(1, 1),
    }
    targets = {
        "policy": torch.nn.functional.one_hot(torch.tensor([0]), 1089).float(),
        "value": torch.tensor([1.0]),
        "axis": torch.tensor([-1]),
    }

    total, per_head = compute_losses(
        predictions,
        targets,
        {"policy": 1.0, "value": 1.0, "regret_rank": 1.0, "axis": 1.0, "moves_left": 1.0},
    )

    assert torch.isfinite(total)
    assert "regret_rank" not in per_head
    assert "moves_left" not in per_head
    assert per_head["axis"].item() == 0.0
