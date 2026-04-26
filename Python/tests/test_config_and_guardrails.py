import pytest
import torch

from hexorl.buffer import RingBuffer
from hexorl.config import Config
from hexorl.selfplay.worker import SelfPlayWorker
from hexorl.train.ema import ModelEMA
from hexorl.train.losses import compute_losses


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
