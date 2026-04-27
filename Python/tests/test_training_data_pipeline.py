import struct

import numpy as np
import pytest
import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import (
    _py_apply_d6_symmetry,
    _py_decode_compact_record,
    _transform_axis_maps,
    _transform_axis_label,
    _transform_dense_policy,
    ReplayDataset,
)
from hexorl.buffer.targets import process_game_record
from hexorl.config import Config
from hexorl.epoch import pipeline
from hexorl.selfplay.orchestrator import SelfPlayOrchestrator
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    BOARD_SIZE,
    action_to_board_index,
    sparsify_policy,
)
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


def test_each_symmetry_permutates_axes_one_to_one():
    for sym_idx in range(12):
        mapped = [_transform_axis_label(axis, sym_idx) for axis in range(3)]
        assert sorted(mapped) == [0, 1, 2]


def test_axis_delta_maps_symmetry_transforms_space_and_axis_planes():
    maps = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    src_i = BOARD_SIZE // 2 + 1
    src_j = BOARD_SIZE // 2
    maps[0, src_i, src_j] = 2.0
    maps[3, src_i, src_j] = 3.0

    transformed = _transform_axis_maps(maps, sym_idx=1)

    dst_axis = _transform_axis_label(0, 1)
    dst_i = BOARD_SIZE // 2
    dst_j = BOARD_SIZE // 2 + 1
    assert transformed[dst_axis, dst_i, dst_j] == 2.0
    assert transformed[dst_axis + 3, dst_i, dst_j] == 3.0
    assert transformed.sum() == 5.0


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
        value_weight=0.0,
    )
    buffer = RingBuffer(capacity=4)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert out.opp_policy_target == rec.opp_policy_target
    assert out.regret_rank == rec.regret_rank
    assert out.axis_label == rec.axis_label
    assert out.moves_left == rec.moves_left
    assert out.value_weight == rec.value_weight


def test_policy_target_top64_is_preserved_when_configured():
    dense = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    dense[:80] = np.arange(80, 0, -1, dtype=np.float32)
    policy = sparsify_policy(dense, top_k=64)
    rec = PositionRecord(
        move_history=b"",
        policy_target=policy,
        root_value=0.0,
        player=0,
    )
    buffer = RingBuffer(capacity=2, max_policy_entries=64)
    buffer.append(rec)

    out = buffer[0]
    assert out is not None
    assert len(out.policy_target) == 64
    assert abs(sum(out.policy_target.values()) - 1.0) < 1e-6


def test_run_epoch_appends_selfplay_to_existing_replay(monkeypatch, tmp_path):
    generated = RingBuffer(capacity=8)
    generated.append(
        PositionRecord(
            move_history=_move(0, 0, 0),
            policy_target={action_to_board_index(1, 0): 1.0},
            root_value=0.0,
            player=1,
            game_id=1 << 24,
        )
    )
    generated.append(
        PositionRecord(
            move_history=_move(0, 0, 0) + _move(1, 1, 0),
            policy_target={action_to_board_index(2, 0): 1.0},
            root_value=0.0,
            player=0,
            game_id=1 << 24,
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

    assert empty_result.buffer_stats["size"] == 2
    assert len(empty_existing) == 2
    assert [record.game_id for record in empty_existing.records()] == [0, 0]

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

    assert result.buffer_stats["size"] == 3
    assert [record.game_id for record in existing.records()] == [4, 5, 5]


def test_selfplay_epoch_completion_requires_games_and_states():
    cfg = Config()
    cfg.selfplay.games_per_epoch = 2
    cfg.selfplay.states_per_epoch = 10
    orchestrator = SelfPlayOrchestrator(cfg, buffer_capacity=16)

    orchestrator._games_done = 2
    orchestrator._positions_done = 9
    assert not orchestrator.epoch_complete
    assert orchestrator.progress == pytest.approx(0.9)

    orchestrator._positions_done = 10
    assert orchestrator.epoch_complete
    assert orchestrator.progress == 1.0


def test_orchestrator_masks_truncated_game_value_targets(tmp_path):
    from hexorl.dashboard.recorder import RunRecorder
    from hexorl.dashboard.db import DashboardStore

    cfg = Config()
    cfg.selfplay.train_on_truncated_games = False
    store = DashboardStore(tmp_path / "dashboard.sqlite3")
    recorder = RunRecorder(store, "trunc-test")
    orchestrator = SelfPlayOrchestrator(cfg, buffer_capacity=16, recorder=recorder)
    game = GameRecord(
        positions=[
            PositionRecord(
                move_history=b"",
                policy_target={action_to_board_index(0, 0): 1.0},
                root_value=0.0,
                player=0,
                outcome=0.0,
            )
        ],
        outcome=0.0,
        game_id=9,
        game_length=1,
        final_move_history=b"",
        truncated=True,
        terminal_reason="max_game_moves",
    )

    orchestrator._ingest_game(game)

    assert len(orchestrator.buffer) == 1
    assert orchestrator.buffer[0].value_weight == 0.0
    assert orchestrator.stats["positions_done"] == 1
    rows = store.rows("SELECT payload_json FROM games")
    assert rows[0]["payload_json"]["truncated"] is True
    assert rows[0]["payload_json"]["terminal_reason"] == "max_game_moves"


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


def test_policy_loss_can_be_masked_to_full_search_samples():
    predictions = {
        "policy": torch.zeros(2, 1089, requires_grad=True),
    }
    targets = {
        "policy": torch.nn.functional.one_hot(torch.tensor([0, 1]), 1089).float(),
        "policy_weight": torch.tensor([1.0, 0.0]),
    }

    total, per_head = compute_losses(predictions, targets, {"policy": 1.0})

    expected = torch.log(torch.tensor(1089.0))
    assert torch.allclose(total.detach(), expected, atol=1e-5)
    assert torch.allclose(per_head["policy"].detach(), expected, atol=1e-5)


def test_value_loss_can_be_masked_for_truncated_games():
    predictions = {"value": torch.zeros(2, 65, requires_grad=True)}
    targets = {
        "value": torch.tensor([1.0, -1.0]),
        "value_weight": torch.tensor([0.0, 0.0]),
    }

    total, per_head = compute_losses(predictions, targets, {"value": 1.0})

    assert total.item() == 0.0
    assert per_head["value"].item() == 0.0


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


def test_replay_dataset_marks_low_sim_policy_weight_zero():
    buffer = RingBuffer(capacity=2)
    buffer.append(
        PositionRecord(
            move_history=b"",
            policy_target={action_to_board_index(0, 0): 1.0},
            root_value=0.0,
            player=0,
            is_full_search=False,
        )
    )
    dataset = ReplayDataset(
        buffer,
        batch_size=1,
        use_symmetry=False,
    )
    *_rest, aux_targets = next(iter(dataset))

    assert aux_targets["policy_weight"].shape == (1,)
    assert aux_targets["policy_weight"][0].item() == 0.0
