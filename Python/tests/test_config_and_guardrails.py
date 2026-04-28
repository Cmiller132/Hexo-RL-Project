import pytest
import torch

from hexorl.buffer import RingBuffer
from hexorl.config import Config
from hexorl.runtime import HostProfile, autotune_config, _estimate_train_peak_gb
from hexorl.selfplay.worker import SelfPlayWorker
from hexorl.train.ema import ModelEMA
from hexorl.train.losses import compute_losses
from hexorl.model.network import build_model_from_config


def test_config_rejects_lookahead_head_without_matching_horizon():
    with pytest.raises(ValueError, match="lookahead"):
        Config.model_validate(
            {
                "model": {"heads": ["policy", "value", "lookahead_6"]},
                "buffer": {
                    "lookahead_horizons": [4, 12, 36],
                    "lookahead_lambdas": [0.75, 0.9, 0.97],
                },
            }
        )


def test_config_rejects_mismatched_lookahead_horizon_and_lambda_counts():
    with pytest.raises(ValueError, match="same length"):
        Config.model_validate(
            {
                "buffer": {
                    "lookahead_horizons": [4, 12, 36],
                    "lookahead_lambdas": [0.75],
                },
            }
        )


def test_compute_losses_raises_when_no_loss_can_be_computed():
    with pytest.raises(ValueError, match="No trainable losses"):
        compute_losses(
            {"regret_rank": torch.zeros(2, 1)},
            {"policy": torch.zeros(2, 1089)},
            {"regret_rank": 1.0},
        )


def test_ring_buffer_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="capacity"):
        RingBuffer(capacity=0)
    with pytest.raises(ValueError, match="max_policy_entries"):
        RingBuffer(capacity=4, max_policy_entries=0)
    with pytest.raises(ValueError, match="num_lookahead"):
        RingBuffer(capacity=4, num_lookahead=-1)


def test_model_ema_decay_keeps_most_shadow_weight():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.9)

    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update()
    first = ema.state_dict()["shadow"]["weight"].item()
    ema.update()
    second = ema.state_dict()["shadow"]["weight"].item()

    assert first == pytest.approx(0.5)
    assert second == pytest.approx(2.0 / 3.0)


def test_selfplay_worker_game_ids_are_unique_across_workers():
    cfg = Config()
    worker0 = SelfPlayWorker(0, cfg, record_queue=None)
    worker1 = SelfPlayWorker(1, cfg, record_queue=None)

    assert worker0._game_id() != worker1._game_id()


def test_autotune_train_batch_avoids_memory_cliff_for_production_model():
    cfg = Config()
    cfg.model.channels = 128
    cfg.model.blocks = 16
    cfg.model.heads = ["policy", "value", "lookahead_4", "lookahead_12", "lookahead_36", "axis"]
    cfg.selfplay.num_workers = 0
    cfg.train.batch_size = 0
    cfg.train.batches_per_epoch = 100
    host = HostProfile(
        logical_cpus=32,
        physical_cpus=16,
        system="linux",
        cuda_available=True,
        cuda_name="test-gpu",
        cuda_memory_gb=12.0,
    )

    autotune_config(cfg, host)

    assert cfg.selfplay.num_workers == 8
    assert cfg.train.batch_size < 384
    assert _estimate_train_peak_gb(cfg, cfg.train.batch_size) <= 12.0 * cfg.runtime.train_memory_fraction
    assert cfg.runtime.compile_model is False
    assert cfg.runtime.compile_inference is False


def test_autotune_compile_model_for_long_cuda_training():
    cfg = Config()
    cfg.train.batches_per_epoch = 1000
    host = HostProfile(
        logical_cpus=32,
        physical_cpus=16,
        system="linux",
        cuda_available=True,
        cuda_name="test-gpu",
        cuda_memory_gb=12.0,
    )

    autotune_config(cfg, host, selfplay_enabled=False)

    assert cfg.runtime.compile_model is True
    assert cfg.runtime.compile_inference is False


def test_restnet_config_validation_and_forward_shapes():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 16,
                "blocks": 3,
                "architecture": "restnet",
                "attention_positions": [2],
                "attention_heads": 4,
                "heads": ["policy", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    out = model(torch.zeros(2, 13, 33, 33))
    assert out["policy"].shape == (2, 1089)
    assert out["value"].shape == (2, 65)


def test_graph_config_validation_and_action_keyed_forward_shapes():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 16,
                "blocks": 4,
                "architecture": "graph",
                "attention_heads": 4,
                "graph_token_set": "graph256_cells",
                "graph_token_budget": 64,
                "graph_layers": 1,
                "sparse_policy": True,
                "candidate_budget": 8,
                "heads": ["policy", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    x = torch.zeros(2, 13, 33, 33)
    x[:, 2] = 1.0
    x[:, 3, 16, 16] = 1.0
    candidate_indices = torch.tensor([[544, -1, -1], [544, 545, -1]], dtype=torch.long)
    candidate_features = torch.zeros(2, 3, 12)
    candidate_mask = candidate_indices >= 0
    out = model(
        x,
        candidate_indices=candidate_indices,
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
    )
    assert out["policy"].shape == (2, 1089)
    assert out["value"].shape == (2, 65)
    assert out["sparse_policy"].shape == (2, 3)


def test_pair_policy_head_forward_and_default_weight():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 16,
                "blocks": 2,
                "heads": ["policy", "value", "pair_policy"],
                "candidate_budget": 8,
            },
            "inference": {"fp16": False},
        }
    )
    assert cfg.model.sparse_policy is True
    assert cfg.train.loss_weights["pair_policy"] == pytest.approx(0.05)
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    x = torch.zeros(2, 13, 33, 33)
    candidate_indices = torch.tensor([[544, 545, -1], [544, 545, 546]], dtype=torch.long)
    candidate_features = torch.zeros(2, 3, 12)
    candidate_mask = candidate_indices >= 0
    pair_candidate_indices = torch.tensor([[[0, 1], [-1, -1]], [[0, 1], [1, 2]]], dtype=torch.long)
    pair_candidate_mask = pair_candidate_indices[..., 0] >= 0

    out = model(
        x,
        candidate_indices=candidate_indices,
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
        pair_candidate_indices=pair_candidate_indices,
        pair_candidate_mask=pair_candidate_mask,
    )

    assert out["pair_policy"].shape == (2, 2)
    assert torch.isfinite(out["pair_policy"][pair_candidate_mask]).all()


def test_restnet_config_rejects_invalid_attention_position():
    with pytest.raises(ValueError, match="attention_positions"):
        Config.model_validate(
            {
                "model": {
                    "blocks": 2,
                    "architecture": "restnet",
                    "attention_positions": [3],
                }
            }
        )


def test_sparse_policy_config_adds_default_loss_weight():
    cfg = Config.model_validate({"model": {"sparse_policy": True}})
    assert cfg.train.loss_weights["sparse_policy"] == pytest.approx(0.25)


def test_sparse_policy_head_enables_sparse_data_contract():
    cfg = Config.model_validate({"model": {"heads": ["policy", "value", "sparse_policy"]}})

    assert cfg.model.sparse_policy is True
    assert cfg.train.loss_weights["sparse_policy"] == pytest.approx(0.25)


def test_cnn_config_does_not_require_attention_head_divisibility():
    cfg = Config.model_validate({"model": {"channels": 10, "blocks": 1}})
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    out = model(torch.zeros(1, 13, 33, 33))
    assert out["policy"].shape == (1, 1089)


def test_config_rejects_reserved_or_invalid_attention_options():
    with pytest.raises(ValueError, match="relative_bias"):
        Config.model_validate({"model": {"architecture": "restnet", "relative_bias": True}})
    with pytest.raises(ValueError, match="dropout"):
        Config.model_validate({"model": {"dropout": 1.0}})
    with pytest.raises(ValueError, match="attention_dropout"):
        Config.model_validate({"model": {"attention_dropout": -0.1}})


def test_sparse_policy_effective_candidate_width_capped_by_shm():
    with pytest.raises(ValueError, match="candidate width"):
        Config.model_validate(
            {
                "model": {"sparse_policy": True, "candidate_budget": 128},
                "selfplay": {"policy_target_top_k": 513},
            }
        )
