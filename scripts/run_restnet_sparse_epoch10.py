"""Run the Phase 1 ResTNet+sparse-policy scout for a fixed epoch count.

This is an experiment runner, not a general training CLI. It keeps the trainer
and replay buffer alive across epochs so optimizer state, EMA, and replay
growth match a normal long-running job.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from hexorl.config import Config, load_config
from hexorl.dashboard.recorder import RunRecorder
from hexorl.epoch import run_epoch
from hexorl.replay.storage import ReplayStorage
from hexorl.runtime import autotune_config, configure_torch_runtime, detect_host


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("Configs/wsl_speed_probe.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/restnet_sparse_stage0_epoch10"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--attention", default="5,10,14")
    parser.add_argument("--sparse-stage", type=int, default=0)
    parser.add_argument("--candidate-budget", type=int, default=256)
    parser.add_argument("--games-per-epoch", type=int, default=128)
    parser.add_argument("--states-per-epoch", type=int, default=1024)
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    return parser.parse_args()


def configure_experiment(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.run.output_dir = str(args.output_dir)
    cfg.run.log_level = "INFO"

    cfg.model.architecture = "restnet"
    cfg.model.attention_positions = [
        int(part.strip()) for part in args.attention.split(",") if part.strip()
    ]
    cfg.model.attention_heads = 8
    cfg.model.attention_mlp_ratio = 2.0
    cfg.model.sparse_policy = True
    cfg.model.sparse_prior_stage = int(args.sparse_stage)
    cfg.model.sparse_prior_mix = 0.25
    cfg.model.candidate_budget = int(args.candidate_budget)
    if "sparse_policy" not in cfg.train.loss_weights:
        cfg.train.loss_weights["sparse_policy"] = 0.25

    cfg.selfplay.games_per_epoch = int(args.games_per_epoch)
    cfg.selfplay.states_per_epoch = int(args.states_per_epoch)
    cfg.selfplay.mcts_simulations = int(args.mcts_sims)
    cfg.selfplay.train_on_truncated_games = True

    cfg.train.batches_per_epoch = int(args.train_batches)
    cfg.train.peak_lr = float(args.peak_lr)
    cfg.runtime.compile_model = False
    cfg.runtime.compile_inference = False

    return Config.model_validate(cfg.model_dump())


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = configure_experiment(load_config(args.config), args)
    host = autotune_config(cfg, selfplay_enabled=True)
    runtime = configure_torch_runtime(cfg, host)

    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder.for_run_dir(output_dir)

    replay = ReplayStorage(
        capacity=cfg.buffer.capacity,
        prefetch_records=cfg.train.prefetch_batches,
    )

    trainer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_t0 = time.monotonic()

    header = {
        "event": "run_start",
        "epochs": args.epochs,
        "output_dir": str(output_dir),
        "device": str(device),
        "host": detect_host().__dict__,
        "runtime": runtime,
        "model": cfg.model.model_dump(),
        "selfplay": cfg.selfplay.model_dump(),
        "inference": cfg.inference.model_dump(),
        "train": cfg.train.model_dump(),
    }
    print(json.dumps(header, sort_keys=True), flush=True)

    for epoch_idx in range(1, args.epochs + 1):
        result = run_epoch(
            cfg,
            trainer=trainer,
            buffer=replay,
            output_dir=output_dir,
            bootstrap_games=0,
            use_selfplay=True,
            train=True,
            device=device,
            recorder=recorder,
        )
        trainer = result.trainer
        row = {
            "event": "epoch_complete",
            "requested_epoch": epoch_idx,
            "elapsed_s": result.elapsed_s,
            "run_elapsed_s": time.monotonic() - run_t0,
            "checkpoint": str(result.checkpoint_path) if result.checkpoint_path else None,
            "train": result.train_stats,
            "buffer": result.buffer_stats,
        }
        print(json.dumps(row, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "event": "run_complete",
                "epochs": args.epochs,
                "elapsed_s": time.monotonic() - run_t0,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
