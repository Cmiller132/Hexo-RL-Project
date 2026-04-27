import struct

import numpy as np
import pytest
import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import (
    _py_apply_d6_symmetry,
    _py_decode_compact_record,
    _transform_axis_label,
    _transform_dense_policy,
    ReplayDataset,
)
from hexorl.buffer.targets import process_game_record
from hexorl.config import Config
from hexorl.epoch import pipeline
from hexorl.selfplay.records import GameRecord, PositionRecord, BOARD_SIZE, action_to_board_index
from hexorl.train.losses import compute_losses
from hexorl.model.network import HexNet


def _move(player: int, q: int, r: int) -> bytes:
    return struct.pack("<iii", player, q, r)


def test_python_decoder_returns_final_position_for_history():
    history = _move(0, 0, 0)
    decoded = _py_decode_compact_record(history)

    assert decoded.shape == (2, 13, BOARD_SIZE, BOARD_SIZE)
    assert decoded[0, 0].sum() == 0.0
    assert decoded[0, 2].sum() == BOARD_SIZE * BOARD_SIZE
    assert decoded[0, 3, BOARD_SIZE // 2, BOARD_SIZE // 2] == 1.0
    assert decoded[0, 6].sum() == BOARD_SIZE * BOARD_SIZE
    assert decoded[-1, 1, BOARD_SIZE // 2, BOARD_SIZE // 2] == 1.0
    assert decoded[-1, 6].sum() == 0.0


def test_policy_symmetry_transform_tracks_dense_target():
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    src_i = BOARD_SIZE // 2 + 1
    src_j = BOARD_SIZE // 2
    policy[src_i * BOARD_SIZE + src_j] = 1.0

    transformed = _transform_dense_policy(policy, sym_idx=3)

    dst_i = BOARD_SIZE // 2 - 1
    dst_j = BOARD_SIZE // 2
    assert transformed[dst_i * BOARD_SIZE + dst_j] == 1.0
    assert transformed.sum() == 1.0


def test_tensor_and_policy_symmetry_match_for_all_transforms():
    src_i = BOARD_SIZE // 2 + 2
    src_j = BOARD_SIZE // 2 - 1
    tensor = np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    tensor[0, src_i, src_j] = 1.0
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    policy[src_i * BOARD_SIZE + src_j] = 1.0

    for sym_idx in range(12):
        transformed_tensor = _py_apply_d6_symmetry(tensor, sym_idx)
        transformed_policy = _transform_dense_policy(policy, sym_idx)
        tensor_idx = int(transformed_tensor[0].argmax())
        policy_idx = int(transformed_policy.argmax())
        assert tensor_idx == policy_idx


def test_axis_label_symmetry_transform_remains_valid():
    for axis in range(3):
        for sym_idx in range(12):
            assert _transform_axis_label(axis, sym_idx) in {0, 1, 2}


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


def test_run_epoch_appends_selfplay_to_existing_replay(monkeypatch, tmp_path):
    generated = RingBuffer(capacity=8)
    generated.append(
        PositionRecord(
            move_history=_move(0, 0, 0),
            policy_target={action_to_board_index(1, 0): 1.0},
            root_value=0.0,
            player=1,
            game_id=0,
        )
    )

    class FakeOrchestrator:
        buffer = generated
        stats = {"games_done": 1, "positions_done": 1}

    monkeypatch.setattr(pipeline, "run_orchestrator", lambda *args, **kwargs: FakeOrchestrator())
    cfg = Config()
    cfg.model.channels = 4
    cfg.model.blocks = 1

    empty_existing = RingBuffer(capacity=8)
    empty_result = pipeline.run_epoch(
        cfg,
        buffer=empty_existing,
        output_dir=tmp_path / "empty",
        use_selfplay=True,
        train=False,
    )

    assert empty_result.buffer_stats["size"] == 1
    assert len(empty_existing) == 1

    existing = RingBuffer(capacity=8)
    existing.append(
        PositionRecord(
            move_history=b"",
            policy_target={action_to_board_index(0, 0): 1.0},
            root_value=0.0,
            player=0,
            game_id=4,
        )
    )
    result = pipeline.run_epoch(
        cfg,
        buffer=existing,
        output_dir=tmp_path,
        use_selfplay=True,
        train=False,
    )

    assert result.buffer_stats["size"] == 2
    assert [record.game_id for record in existing.records()] == [4, 5]


def test_compute_losses_skips_missing_targets_and_handles_batch_one():
    predictions = {
        "policy": torch.zeros(1, 1089),
        "value": torch.zeros(1, 65),
        "regret_rank": torch.zeros(1, 1),
        "axis": torch.zeros(1, 3),
        "axis_delta_norm": torch.zeros(1, 6, 33, 33),
        "moves_left": torch.ones(1, 1),
    }
    targets = {
        "policy": torch.nn.functional.one_hot(torch.tensor([0]), 1089).float(),
        "value": torch.tensor([1.0]),
        "axis": torch.tensor([-1]),
        "axis_delta_norm": torch.ones(1, 6, 33, 33),
    }

    total, per_head = compute_losses(
        predictions,
        targets,
        {
            "policy": 1.0,
            "value": 1.0,
            "regret_rank": 1.0,
            "axis": 1.0,
            "axis_delta_norm": 1.0,
            "moves_left": 1.0,
        },
    )

    assert torch.isfinite(total)
    assert "regret_rank" not in per_head
    assert "moves_left" not in per_head
    assert per_head["axis"].item() == 0.0
    assert per_head["axis_delta_norm"].item() > 0.0


def test_axis_delta_norm_head_shape():
    model = HexNet(channels=4, blocks=1, heads=["axis_delta_norm"])
    out = model(torch.zeros(2, 13, 33, 33))
    assert out["axis_delta_norm"].shape == (2, 6, 33, 33)


def test_replay_dataset_can_emit_axis_delta_norm_target():
    pytest.importorskip("_engine")
    buffer = RingBuffer(capacity=8)
    buffer.append(
        PositionRecord(
            move_history=_move(0, 0, 0),
            policy_target={action_to_board_index(1, 0): 1.0},
            root_value=0.0,
            player=1,
        )
    )
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
        include_axis_delta_norm=True,
    )

    *_rest, aux_targets = next(iter(dataset))

    assert aux_targets["axis_delta_norm"].shape == (1, 6, 33, 33)
    assert aux_targets["axis_delta_norm"].sum() > 0.0
