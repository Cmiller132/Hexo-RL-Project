from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.config import Config
from hexorl.epoch.pipeline import run_epoch
from hexorl.model.network import HexNet
from hexorl.train.trainer import Trainer


def main() -> None:
    cfg = Config()
    cfg.run.output_dir = "./runs/{name}"
    cfg.model.channels = 32
    cfg.model.blocks = 3
    cfg.model.heads = ["policy", "value", "axis_delta_norm"]
    cfg.selfplay.num_workers = 1
    cfg.selfplay.games_per_epoch = 1
    cfg.selfplay.states_per_epoch = 32
    cfg.selfplay.max_game_moves = 24
    cfg.selfplay.batch_size_per_worker = 1
    cfg.selfplay.mcts_simulations = 4
    cfg.selfplay.pcr_low_sims = 2
    cfg.selfplay.pcr_low_sim_prob = 1.0
    cfg.selfplay.subtree_reuse = True
    cfg.selfplay.near_radius = 6
    cfg.selfplay.constrain_threats = True
    cfg.inference.max_batch_size = 16
    cfg.inference.max_wait_us = 500
    cfg.inference.fp16 = False
    cfg.buffer.capacity = 4096
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.buffer.regret_fraction = 0.0
    cfg.train.batch_size = 16
    cfg.train.batches_per_epoch = 8
    cfg.train.lr_schedule = "constant"
    cfg.train.peak_lr = 1e-3
    cfg.train.loss_weights = {
        "policy": 1.0,
        "value": 1.0,
        "axis_delta_norm": 0.05,
        "entropy": 0.001,
    }

    out = Path("runs/axis_delta_prod_probe")
    out.mkdir(parents=True, exist_ok=True)
    model = HexNet(
        channels=cfg.model.channels,
        blocks=cfg.model.blocks,
        heads=cfg.model.heads,
    )
    buffer = RingBuffer(capacity=cfg.buffer.capacity, num_lookahead=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"TRAIN_DEVICE {device}", flush=True)
    trainer = None
    latest = sorted(out.glob("epoch_*.pt"))[-1:] or []
    if latest:
        # The real self-play dataloader is installed by run_epoch before training.
        placeholder = ReplayDataset(buffer, batch_size=cfg.train.batch_size)
        trainer = Trainer(model, cfg, DataLoader(placeholder, batch_size=None, num_workers=0), device=device)
        trainer.load_checkpoint(latest[0])
        print(
            f"RESUME_CHECKPOINT {latest[0]} epoch={trainer.epoch} global_step={trainer.global_step}",
            flush=True,
        )
    for epoch in range(1, 6):
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
            "PROD_EPOCH",
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
