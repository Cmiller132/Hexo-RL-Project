import multiprocessing as mp
from types import SimpleNamespace

import pytest
import torch

from hexorl.buffer import RingBuffer
from hexorl.config import Config
from hexorl.runtime import HostProfile, autotune_config, _estimate_train_peak_gb
from hexorl.selfplay.worker import SelfPlayWorker, _current_turn_first_qr
from hexorl.search.pair_strategy import build_pair_strategy
from hexorl.train.ema import ModelEMA
from hexorl.train.loss_plan import build_loss_plan
from hexorl.train.losses import compute_losses
from hexorl.models.assembly import build_model_from_config
from hexorl.models.families.network import HexConv2d, GatedResBlock
from hexorl.models.loading import restore_model_weights


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


def test_config_requires_active_loss_for_matching_lookahead_head():
    with pytest.raises(ValueError, match="active train.loss_weights"):
        Config.model_validate(
            {
                "model": {"heads": ["policy", "value", "lookahead_4"]},
                "train": {"loss_weights": {"policy": 1.0, "value": 1.0}},
            }
        )

    cfg = Config.model_validate(
        {
            "model": {"heads": ["policy", "value", "lookahead_4"]},
            "train": {"loss_weights": {"policy": 1.0, "value": 1.0, "lookahead_4": 0.15}},
        }
    )
    assert cfg.train.loss_weights["lookahead_4"] == pytest.approx(0.15)


def test_config_forbids_unknown_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Config.model_validate({"buffer": {"regret_fracton": 0.1}})


def test_regret_fraction_requires_weighted_regret_heads_or_replay_only():
    with pytest.raises(ValueError, match="regret_fraction"):
        Config.model_validate(
            {
                "buffer": {"regret_fraction": 0.1, "regret_replay_only": False},
                "model": {"heads": ["policy", "value"]},
            }
        )

    cfg = Config.model_validate(
        {
            "buffer": {"regret_fraction": 0.1, "regret_replay_only": False},
            "model": {"heads": ["policy", "value", "regret_rank", "regret_value"]},
        }
    )
    assert cfg.buffer.regret_replay_only is False


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


def test_compute_losses_raises_when_required_contract_target_is_missing():
    with pytest.raises(ValueError, match="requires target 'regret_rank'"):
        predictions = {"regret_rank": torch.zeros(2, 1)}
        loss_weights = {"regret_rank": 1.0}
        compute_losses(
            predictions,
            {"policy": torch.zeros(2, 1089)},
            loss_weights,
            loss_plan=build_loss_plan(tuple(predictions.keys()), loss_weights),
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


def test_selfplay_pcr_schedule_keeps_full_search_rows_inside_long_games():
    cfg = Config()
    cfg.selfplay.pcr_low_sim_prob = 0.70
    cfg.selfplay.pcr_low_sims = 64
    cfg.selfplay.mcts_simulations = 128
    worker = SelfPlayWorker(0, cfg, record_queue=None)

    schedule = [worker._search_mode_for_turn(game_seed=1234, move_idx=i) for i in range(40)]
    full_rows = [idx for idx, (use_pcr, sims) in enumerate(schedule) if not use_pcr]
    pcr_rows = [idx for idx, (use_pcr, sims) in enumerate(schedule) if use_pcr]

    assert len(full_rows) == 12
    assert pcr_rows
    assert all(sims == 128 for _idx, (use_pcr, sims) in enumerate(schedule) if not use_pcr)
    assert all(sims == 64 for _idx, (use_pcr, sims) in enumerate(schedule) if use_pcr)
    assert any(idx % 2 == 0 for idx in full_rows)
    assert any(idx % 2 == 1 for idx in full_rows)
    assert any(not use_pcr for use_pcr, _sims in schedule[:20])
    assert any(not use_pcr for use_pcr, _sims in schedule[20:])


def test_selfplay_pcr_schedule_honors_extreme_probabilities():
    cfg = Config()
    cfg.selfplay.pcr_low_sims = 64
    cfg.selfplay.mcts_simulations = 128
    worker = SelfPlayWorker(0, cfg, record_queue=None)

    worker.pcr_low_sim_prob = 0.0
    assert [worker._search_mode_for_turn(1234, i) for i in range(4)] == [
        (False, 128),
        (False, 128),
        (False, 128),
        (False, 128),
    ]

    worker.pcr_low_sim_prob = 1.0
    assert [worker._search_mode_for_turn(1234, i) for i in range(4)] == [
        (True, 64),
        (True, 64),
        (True, 64),
        (True, 64),
    ]


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
                "architecture": "graph_hybrid_0",
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
    assert cfg.model.architecture == "graph_hybrid_0"
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


def test_graph_architecture_alias_maps_to_graph_hybrid_0():
    with pytest.warns(UserWarning, match="deprecated crop-compatible alias"):
        cfg = Config.model_validate({"model": {"architecture": "graph"}})
    assert cfg.model.architecture == "graph_hybrid_0"


def test_hex_conv_invalid_axial_corners_stay_zero_after_optimizer_step():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 8,
                "blocks": 2,
                "heads": ["policy", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    hex_convs = [m for m in model.modules() if isinstance(m, HexConv2d)]
    assert hex_convs
    for conv in hex_convs:
        assert torch.count_nonzero(conv.weight[:, :, 0, 0]) == 0
        assert torch.count_nonzero(conv.weight[:, :, 2, 2]) == 0

    opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)
    out = model(torch.randn(2, 13, 33, 33))
    loss = out["policy"].sum() + out["value"].sum()
    loss.backward()
    opt.step()

    for conv in hex_convs:
        assert torch.count_nonzero(conv.weight[:, :, 0, 0]) == 0
        assert torch.count_nonzero(conv.weight[:, :, 2, 2]) == 0

    with torch.no_grad():
        hex_convs[0].weight[:, :, 0, 0].fill_(1.0)
        hex_convs[0].weight[:, :, 2, 2].fill_(1.0)
    model.apply_hex_masks_()
    assert torch.count_nonzero(hex_convs[0].weight[:, :, 0, 0]) == 0
    assert torch.count_nonzero(hex_convs[0].weight[:, :, 2, 2]) == 0


def test_hex_conv_masks_are_reapplied_after_loading_state_dict():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 8,
                "blocks": 1,
                "heads": ["policy", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for key, value in state.items():
        if value.ndim == 4 and value.shape[-2:] == (3, 3):
            value[:, :, 0, 0].fill_(1.0)
            value[:, :, 2, 2].fill_(1.0)

    restore_model_weights(model, state)

    for conv in [m for m in model.modules() if isinstance(m, HexConv2d)]:
        assert torch.count_nonzero(conv.weight[:, :, 0, 0]) == 0
        assert torch.count_nonzero(conv.weight[:, :, 2, 2]) == 0


@pytest.mark.parametrize("architecture", ["cnn", "restnet", "graph_hybrid_0"])
def test_trunks_use_hex_conv_for_architecture_names(architecture):
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 8,
                "blocks": 3,
                "architecture": architecture,
                "attention_heads": 4,
                "dropout": 0.25,
                "graph_token_set": "graph256_cells",
                "graph_token_budget": 64,
                "graph_layers": 1,
                "heads": ["policy", "value"],
            },
            "inference": {"fp16": False},
        }
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"))

    assert isinstance(model.conv_in, HexConv2d)
    gated_blocks = [m for m in model.res_blocks if isinstance(m, GatedResBlock)]
    assert gated_blocks
    for block in gated_blocks:
        assert isinstance(block.conv1, HexConv2d)
        assert isinstance(block.conv2, HexConv2d)
        assert isinstance(block.conv_gate, HexConv2d)
        assert isinstance(block.dropout, torch.nn.Dropout2d)

    head_convs = [m.conv for m in model.heads.values() if hasattr(m, "conv")]
    assert head_convs
    assert all(not isinstance(conv, HexConv2d) and conv.kernel_size == (1, 1) for conv in head_convs)


def test_pair_policy_head_forward_and_default_weight():
    cfg = Config.model_validate(
        {
            "model": {
                "channels": 16,
                "blocks": 2,
                "heads": ["policy", "value", "pair_policy"],
                "candidate_budget": 8,
                "sparse_policy": True,
            },
            "train": {
                "loss_weights": {"policy": 1.0, "value": 1.5, "pair_policy": 0.05}
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


def test_global_xattn_pair_strategy_defaults_to_none():
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
            },
            "inference": {"fp16": False},
        }
    )
    queue = mp.Queue()
    try:
        worker = SelfPlayWorker(0, cfg, queue, num_workers=1, max_batch_size=1)
        assert worker.global_graph_enabled is True
        assert worker.constrain_threats is True
        assert worker.pair_strategy == "none"
        assert worker.pair_policy_enabled is False
        assert worker.pair_strategy_summary()["pair_rows_scored"] == 0
    finally:
        queue.close()


def test_global_xattn_pair_heads_do_not_enable_pair_scoring_without_strategy():
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": ["value", "policy_place", "policy_pair_first", "policy_pair_joint"],
                "pair_prior_mix": 0.75,
            },
            "inference": {"fp16": False},
        }
    )
    queue = mp.Queue()
    try:
        worker = SelfPlayWorker(0, cfg, queue, num_workers=1, max_batch_size=1)
        assert worker.pair_policy_enabled is False
        assert worker.pair_strategy_summary(pair_rows_possible=100)["pair_rows_scored"] == 0
    finally:
        queue.close()


def test_pair_second_scoring_only_has_known_current_turn_first_placement():
    def move_bytes(player: int, q: int, r: int) -> bytes:
        return (
            int(player).to_bytes(4, "little", signed=True)
            + int(q).to_bytes(4, "little", signed=True)
            + int(r).to_bytes(4, "little", signed=True)
        )

    assert (
        _current_turn_first_qr(
            b"",
            current_player=0,
            placements_remaining=1,
        )
        is None
    )
    assert (
        _current_turn_first_qr(
            move_bytes(1, 0, 0),
            current_player=0,
            placements_remaining=1,
        )
        is None
    )
    assert (
        _current_turn_first_qr(
            move_bytes(0, 2, -1),
            current_player=0,
            placements_remaining=2,
        )
        is None
    )
    assert _current_turn_first_qr(
        move_bytes(0, 2, -1),
        current_player=0,
        placements_remaining=1,
    ) == (2, -1)


def test_pair_scoring_requires_explicit_diagnostic_strategy_and_cap():
    with pytest.raises(ValueError, match="pair_strategy_max_pairs"):
        Config.model_validate(
            {
                "model": {
                    "architecture": "global_xattn_0",
                    "channels": 16,
                    "attention_heads": 4,
                    "graph_layers": 1,
                    "heads": ["value", "policy_place", "policy_pair_joint"],
                    "pair_strategy": "diagnostic_full_pair",
                },
                "inference": {"fp16": False},
            }
        )

    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
                "heads": ["value", "policy_place", "policy_pair_joint"],
                "pair_strategy": "diagnostic_full_pair",
                "pair_strategy_max_pairs": 32,
            },
            "inference": {"fp16": False},
        }
    )
    queue = mp.Queue()
    try:
        worker = SelfPlayWorker(0, cfg, queue, num_workers=1, max_batch_size=1)
        assert worker.pair_policy_enabled is True
        assert worker.pair_strategy_max_pairs == 32
    finally:
        queue.close()

    with pytest.raises(ValueError, match="pair_strategy_max_pairs"):
        build_pair_strategy(
            "diagnostic_full_pair",
            max_pairs=0,
            prior_mix=0.35,
        ).score_graph_pair_chunks(
            client=object(),  # type: ignore[arg-type]
            graph_batch=SimpleNamespace(
                legal_qr=torch.zeros(2, 2, dtype=torch.int32).numpy(),
                legal_token_indices=torch.arange(2).numpy(),
            ),
            second_placement=False,
        )


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


def test_sparse_policy_config_requires_active_loss_for_sparse_policy_head():
    with pytest.raises(ValueError, match="sparse_policy"):
        Config.model_validate({"model": {"heads": ["policy", "value", "sparse_policy"]}})

    cfg = Config.model_validate(
        {
            "model": {"heads": ["policy", "value", "sparse_policy"], "sparse_policy": True},
            "train": {
                "loss_weights": {"policy": 1.0, "value": 1.5, "sparse_policy": 0.25}
            },
        }
    )
    assert cfg.train.loss_weights["sparse_policy"] == pytest.approx(0.25)


def test_sparse_policy_head_enables_sparse_data_contract():
    cfg = Config.model_validate(
        {
            "model": {"heads": ["policy", "value", "sparse_policy"], "sparse_policy": True},
            "train": {
                "loss_weights": {"policy": 1.0, "value": 1.5, "sparse_policy": 0.25}
            },
        }
    )

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


def test_sparse_prior_stage_requires_sparse_policy_contract():
    with pytest.raises(ValueError, match="sparse_prior_stage"):
        Config.model_validate({"model": {"sparse_prior_stage": 1}})


def test_rgsc_selfplay_config_bounds():
    cfg = Config.model_validate(
        {
            "selfplay": {
                "rgsc_beta": 0.75,
                "rgsc_prb_capacity": 8,
                "rgsc_prb_temperature": 0.25,
                "rgsc_prb_ema_alpha": 0.2,
            }
        }
    )
    assert cfg.selfplay.rgsc_beta == pytest.approx(0.75)

    with pytest.raises(ValueError, match="rgsc_beta"):
        Config.model_validate({"selfplay": {"rgsc_beta": 1.1}})
    with pytest.raises(ValueError, match="rgsc_prb_capacity"):
        Config.model_validate({"selfplay": {"rgsc_prb_capacity": -1}})
    with pytest.raises(ValueError, match="rgsc_prb_temperature"):
        Config.model_validate({"selfplay": {"rgsc_prb_temperature": 0.0}})
    with pytest.raises(ValueError, match="rgsc_prb_ema_alpha"):
        Config.model_validate({"selfplay": {"rgsc_prb_ema_alpha": -0.1}})
