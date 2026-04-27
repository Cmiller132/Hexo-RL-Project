from __future__ import annotations

from pathlib import Path

import torch

from hexorl.buffer.ring import RingBuffer
from hexorl.config import Config
from hexorl.epoch.pipeline import run_epoch
from hexorl.model.network import HexNet


def build_config() -> Config:
    cfg = Config()
    cfg.run.output_dir = "./runs/{name}"
    cfg.model.channels = 48
    cfg.model.blocks = 4
    cfg.model.heads = ["policy", "value", "axis_delta_norm"]

    cfg.selfplay.num_workers = 4
    cfg.selfplay.games_per_epoch = 28
    cfg.selfplay.states_per_epoch = 2048
    cfg.selfplay.max_game_moves = 256
    cfg.selfplay.batch_size_per_worker = 2
    cfg.selfplay.mcts_simulations = 16
    cfg.selfplay.pcr_low_sims = 4
    cfg.selfplay.pcr_low_sim_prob = 0.75
    cfg.selfplay.subtree_reuse = True
    cfg.selfplay.near_radius = 6
    cfg.selfplay.constrain_threats = True
    cfg.selfplay.temperature_schedule = [[0, 1.0], [20, 0.5], [60, 0.15], [120, 0.05]]
    cfg.selfplay.dirichlet_alpha = 0.3
    cfg.selfplay.dirichlet_fraction = 0.25
    cfg.selfplay.resign_threshold = -0.98
    cfg.selfplay.resign_disable_prob = 0.25
    cfg.selfplay.train_on_truncated_games = False

    cfg.inference.max_batch_size = 32
    cfg.inference.max_wait_us = 500
    cfg.inference.fp16 = False

    cfg.buffer.capacity = 16384
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.buffer.regret_fraction = 0.0

    cfg.train.batch_size = 32
    cfg.train.batches_per_epoch = 64
    cfg.train.lr_schedule = "constant"
    cfg.train.peak_lr = 8e-4
    cfg.train.loss_weights = {
        "policy": 1.0,
        "value": 1.0,
        "axis_delta_norm": 0.05,
        "entropy": 0.001,
    }
    return cfg


def main() -> None:
    cfg = build_config()
    out = Path("runs/fresh_d6_noise_48x4")
    out.mkdir(parents=True, exist_ok=True)
    model = HexNet(
        channels=cfg.model.channels,
        blocks=cfg.model.blocks,
        heads=cfg.model.heads,
    )
    buffer = RingBuffer(capacity=cfg.buffer.capacity, num_lookahead=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"TRAIN_DEVICE {device}", flush=True)
    print(
        "RUN_CONFIG",
        {
            "channels": cfg.model.channels,
            "blocks": cfg.model.blocks,
            "heads": cfg.model.heads,
            "workers": cfg.selfplay.num_workers,
            "games_per_epoch": cfg.selfplay.games_per_epoch,
            "states_per_epoch": cfg.selfplay.states_per_epoch,
            "max_game_moves": cfg.selfplay.max_game_moves,
            "mcts_simulations": cfg.selfplay.mcts_simulations,
            "pcr_low_sims": cfg.selfplay.pcr_low_sims,
            "pcr_low_sim_prob": cfg.selfplay.pcr_low_sim_prob,
            "subtree_reuse": cfg.selfplay.subtree_reuse,
            "dirichlet_alpha": cfg.selfplay.dirichlet_alpha,
            "dirichlet_fraction": cfg.selfplay.dirichlet_fraction,
            "batch_size": cfg.train.batch_size,
            "batches_per_epoch": cfg.train.batches_per_epoch,
        },
        flush=True,
    )

    trainer = None
    for epoch in range(1, 31):
        result = run_epoch(
            cfg,
            model=model,
            trainer=trainer,
            buffer=buffer,
            output_dir=out,
            bootstrap_games=0,
            use_selfplay=True,
            train=True,
            device=device,
        )
        trainer = result.trainer
        if trainer is not None:
            model = trainer.model
        print(
            "FRESH_EPOCH",
            epoch,
            result.train_stats,
            result.buffer_stats,
            result.checkpoint_path,
            "elapsed",
            result.elapsed_s,
            flush=True,
        )


if __name__ == "__main__":
    main()
