import struct

import pytest
import torch

from hexorl.graph.batch import build_graph_batch_from_history, collate_graph_batches
from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.config import Config
from hexorl.models.assembly import build_model_from_config
from hexorl.selfplay.records import PositionRecord, action_to_board_index
from hexorl.replay.training_batch import (
    graph_batch_training_targets,
    prepare_dense_training_batch,
    prepare_global_graph_training_batch,
)
from hexorl.train.loss_plan import LossContractError, build_loss_plan
from hexorl.train.losses import compute_losses
from hexorl.train.trainer import Trainer


def _hist(*moves):
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", player, q, r))
    return bytes(data)


def _policy_rows(*rows):
    return torch.tensor([[list(row) for row in rows]], dtype=torch.long)


def _compute_losses(predictions, targets, loss_weights, **kwargs):
    return compute_losses(
        predictions,
        targets,
        loss_weights,
        loss_plan=build_loss_plan(tuple(predictions.keys()), loss_weights),
        **kwargs,
    )


def test_compute_losses_requires_explicit_loss_plan():
    with pytest.raises(LossContractError, match="explicit loss_plan"):
        compute_losses(
            {"value": torch.zeros(1, 65)},
            {"value": torch.zeros(1), "value_weight": torch.ones(1)},
            {"value": 1.0},
        )


def test_loss_plan_rejects_missing_required_target():
    with pytest.raises(LossContractError, match="requires target 'value'"):
        _compute_losses(
            {"value": torch.zeros(1, 65)},
            {"value_weight": torch.ones(1)},
            {"value": 1.0},
        )


def test_loss_plan_rejects_missing_required_mask():
    with pytest.raises(LossContractError, match="requires mask 'candidate_mask'"):
        _compute_losses(
            {"sparse_policy": torch.zeros(1, 2)},
            {
                "sparse_policy_target": torch.tensor([[1.0, 0.0]]),
                "sparse_policy_weight": torch.ones(1),
            },
            {"sparse_policy": 1.0},
        )


def test_loss_plan_rejects_missing_required_weight():
    with pytest.raises(LossContractError, match="requires weight 'policy_weight'"):
        _compute_losses(
            {"policy": torch.zeros(1, 3)},
            {"policy": torch.tensor([[1.0, 0.0, 0.0]])},
            {"policy": 1.0},
        )


def test_loss_plan_rejects_missing_required_phase():
    with pytest.raises(LossContractError, match="requires phase 'pair_second_known_first'"):
        _compute_losses(
            {"policy_pair_second": torch.zeros(1, 2)},
            {
                "pair_second_policy_target": torch.tensor([[0.5, 0.5]]),
                "pair_second_row_mask": torch.ones(1, 2, dtype=torch.bool),
                "pair_policy_weight": torch.ones(1),
                "pair_row_mask": torch.ones(1, 2, dtype=torch.bool),
                "pair_first_indices": torch.tensor([[0, 0]]),
                "pair_second_indices": torch.tensor([[1, 2]]),
            },
            {"policy_pair_second": 1.0},
        )


def test_loss_plan_rejects_duplicate_active_rows():
    with pytest.raises(LossContractError, match="duplicate active rows"):
        _compute_losses(
            {"policy_place": torch.zeros(1, 2)},
            {
                "policy_target": torch.tensor([[1.0, 0.0]]),
                "legal_mask": torch.ones(1, 2, dtype=torch.bool),
                "legal_qr": _policy_rows((0, 0), (0, 0)),
                "policy_weight": torch.ones(1),
            },
            {"policy_place": 1.0},
        )


def test_loss_plan_rejects_zero_mass_active_policy():
    with pytest.raises(LossContractError, match="positive target mass"):
        _compute_losses(
            {"policy_place": torch.zeros(1, 2)},
            {
                "policy_target": torch.zeros(1, 2),
                "legal_mask": torch.ones(1, 2, dtype=torch.bool),
                "legal_qr": _policy_rows((0, 0), (1, 0)),
                "policy_weight": torch.ones(1),
            },
            {"policy_place": 1.0},
        )


def test_pair_second_rejects_positive_targets_outside_known_first_phase():
    with pytest.raises(LossContractError, match="outside required phase"):
        _compute_losses(
            {"policy_pair_second": torch.zeros(1, 2)},
            {
                "pair_second_policy_target": torch.tensor([[0.25, 0.75]]),
                "pair_second_row_mask": torch.ones(1, 2, dtype=torch.bool),
                "pair_policy_weight": torch.ones(1),
                "pair_second_known_first": torch.tensor([False]),
                "pair_row_mask": torch.ones(1, 2, dtype=torch.bool),
                "pair_first_indices": torch.tensor([[0, 0]]),
                "pair_second_indices": torch.tensor([[1, 2]]),
            },
            {"policy_pair_second": 1.0},
        )


def test_lookahead_head_requires_exact_horizon_target():
    with pytest.raises(LossContractError, match="requires target 'lookahead_4'"):
        _compute_losses(
            {"lookahead_4": torch.zeros(1, 65)},
            {
                "value": torch.tensor([0.5]),
                "value_weight": torch.ones(1),
            },
            {"lookahead_4": 1.0},
        )


def test_ring_buffer_does_not_synthesize_missing_lookahead_targets():
    full = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(0, 0): 1.0},
        root_value=0.0,
        player=0,
        outcome=0.5,
        lookahead_values=[0.1, 0.2],
    )
    short = PositionRecord(
        move_history=b"",
        policy_target={action_to_board_index(1, 0): 1.0},
        root_value=0.0,
        player=0,
        outcome=0.5,
        lookahead_values=[0.25],
    )
    replay = RingBuffer(capacity=1, num_lookahead=2)
    replay.append(full)
    replay.append(short)

    stored = replay[0]

    assert stored is not None
    assert stored.lookahead_values == pytest.approx([0.25])

    dataset = ReplayDataset(
        replay,
        batch_size=1,
        use_symmetry=False,
        lookahead_horizons=[4, 12],
    )

    with pytest.raises(ValueError, match="lookahead target missing"):
        next(iter(dataset))


def test_global_graph_policy_place_does_not_consume_dense_policy_target():
    with pytest.raises(LossContractError, match="requires target 'policy_target'"):
        _compute_losses(
            {"policy_place": torch.zeros(1, 2)},
            {
                "policy": torch.tensor([[1.0, 0.0]]),
                "policy_weight": torch.ones(1),
            },
            {"policy_place": 1.0},
        )


def test_global_graph_adapter_emits_phase_metadata_without_dense_policy_target():
    graph = build_graph_batch_from_history(
        _hist((0, 0, 0)),
        radius=8,
        policy_target=[(1, 0, 1.0)],
        include_pair_rows=False,
    )

    prepared = prepare_global_graph_training_batch(
        values=torch.zeros(1),
        lookahead_list=[],
        aux_targets={
            **graph_batch_training_targets(collate_graph_batches([graph])),
            "policy_weight": torch.ones(1),
        },
        lookahead_keys=[],
        device=torch.device("cpu"),
        train_policy_on_full_search_only=True,
    )

    assert "policy" not in prepared.targets
    assert "policy_target" in prepared.targets
    assert "legal" in prepared.row_tables
    assert "pair_second_known_first" in prepared.targets


def test_training_adapter_rejects_target_namespace_overwrite():
    with pytest.raises(ValueError, match="overwrite prepared training target"):
        prepare_dense_training_batch(
            tensors=torch.zeros(1, 13, 33, 33),
            policies=torch.zeros(1, 1089),
            values=torch.zeros(1),
            lookahead_list=[],
            aux_targets={"policy": torch.ones(1, 1089)},
            lookahead_keys=[],
            device=torch.device("cpu"),
            channels_last=False,
            train_policy_on_full_search_only=True,
        )


def test_dense_adapter_masks_zero_mass_policy_rows_when_training_all_policy():
    prepared = prepare_dense_training_batch(
        tensors=torch.zeros(2, 13, 33, 33),
        policies=torch.tensor([[0.0, 0.0], [0.25, 0.75]]),
        values=torch.zeros(2),
        lookahead_list=[],
        aux_targets={},
        lookahead_keys=[],
        device=torch.device("cpu"),
        channels_last=False,
        train_policy_on_full_search_only=False,
    )

    assert prepared.targets["policy_weight"].tolist() == [0.0, 1.0]


def test_dense_adapter_existing_policy_weight_cannot_activate_zero_mass_target():
    prepared = prepare_dense_training_batch(
        tensors=torch.zeros(2, 13, 33, 33),
        policies=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        values=torch.zeros(2),
        lookahead_list=[],
        aux_targets={"policy_weight": torch.ones(2)},
        lookahead_keys=[],
        device=torch.device("cpu"),
        channels_last=False,
        train_policy_on_full_search_only=True,
    )

    assert prepared.targets["policy_weight"].tolist() == [0.0, 1.0]


def test_crop_pair_replay_masks_pair_head_when_target_mass_is_empty():
    pytest.importorskip("_engine")
    rec = PositionRecord(
        move_history=_hist((0, 0, 0)),
        policy_target={action_to_board_index(1, 0): 1.0},
        policy_target_v2=[(1, 0, 1.0)],
        pair_policy_target_v2=[],
        root_value=0.0,
        player=1,
        outcome=1.0,
    )
    replay = RingBuffer(capacity=4, max_policy_v2_entries=8)
    replay.append(rec)
    dataset = ReplayDataset(
        replay,
        batch_size=1,
        use_symmetry=False,
        include_sparse_policy=True,
        include_pair_policy=True,
        candidate_budget=8,
    )

    _tensors, _policies, _values, _lookahead, aux = next(iter(dataset))

    assert aux["pair_policy_target"].sum() == pytest.approx(0.0)
    assert aux["pair_policy_weight"][0] == pytest.approx(0.0)
    assert aux["pair_candidate_missing_mass"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("architecture", ["cnn", "restnet_crop_scout", "graph_hybrid_0"])
def test_dense_sparse_graph_hybrid_batches_train_through_trainer_adapter(architecture):
    pytest.importorskip("_engine")
    rec = PositionRecord(
        move_history=_hist((0, 0, 0)),
        policy_target={action_to_board_index(1, 0): 1.0},
        policy_target_v2=[(1, 0, 1.0)],
        root_value=0.0,
        player=1,
        outcome=1.0,
    )
    replay = RingBuffer(capacity=4, max_policy_v2_entries=8)
    replay.append(rec)
    dataset = ReplayDataset(
        replay,
        batch_size=1,
        use_symmetry=False,
        include_sparse_policy=True,
        candidate_budget=8,
    )
    batch = next(iter(dataset))
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": architecture,
                "channels": 4,
                "blocks": 1,
                "attention_heads": 1,
                "heads": ["policy", "value", "sparse_policy"],
                "sparse_policy": True,
            },
            "buffer": {
                "lookahead_horizons": [],
                "lookahead_lambdas": [],
            },
            "train": {
                "batches_per_epoch": 1,
                "loss_weights": {
                    "policy": 1.0,
                    "value": 1.0,
                    "sparse_policy": 0.1,
                },
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    trainer = Trainer(model, cfg, dataloader=[], device=torch.device("cpu"))

    losses = trainer._train_step(batch, 0)

    assert torch.isfinite(torch.tensor(losses["total"]))
    assert "sparse_policy" in losses
