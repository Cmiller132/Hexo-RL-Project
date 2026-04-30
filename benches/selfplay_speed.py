"""Benchmark real self-play throughput with Rust MCTS and GPU inference."""

from __future__ import annotations

import argparse
import time

from hexorl.config import load_config
from hexorl.models.network import HexNet
from hexorl.runtime import autotune_config, configure_torch_runtime
from hexorl.selfplay.orchestrator import run_orchestrator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--leaf-batch", type=int, default=0)
    parser.add_argument("--games", type=int, default=28)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--max-moves", type=int, default=128)
    parser.add_argument("--low-sim-prob", type=float, default=0.0)
    parser.add_argument("--max-wait-us", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=True)
    if args.workers > 0:
        cfg.selfplay.num_workers = args.workers
    if args.leaf_batch > 0:
        cfg.selfplay.batch_size_per_worker = args.leaf_batch
    cfg.inference.max_batch_size = max(
        cfg.inference.max_batch_size,
        cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64,
    )
    cfg.selfplay.games_per_epoch = args.games
    cfg.selfplay.states_per_epoch = 0
    cfg.selfplay.mcts_simulations = args.sims
    cfg.selfplay.max_game_moves = args.max_moves
    cfg.selfplay.pcr_low_sim_prob = args.low_sim_prob
    cfg.selfplay.pcr_low_sims = max(1, args.sims // 2)
    if args.max_wait_us is not None:
        cfg.inference.max_wait_us = args.max_wait_us
    configure_torch_runtime(cfg)

    model = HexNet(cfg.model.channels, cfg.model.blocks, cfg.model.heads)
    start = time.monotonic()
    orchestrator = run_orchestrator(
        cfg,
        buffer_capacity=max(100000, args.games * args.max_moves),
        initial_model_state=model.state_dict(),
    )
    elapsed = time.monotonic() - start
    stats = orchestrator.stats
    print(
        {
            "workers": cfg.selfplay.num_workers,
            "leaf_batch": cfg.selfplay.batch_size_per_worker,
            "max_batch": cfg.inference.max_batch_size,
            "games": stats["games_done"],
            "positions": stats["positions_done"],
            "elapsed_s": round(elapsed, 2),
            "games_min": round(stats["games_done"] * 60.0 / max(elapsed, 1e-9), 2),
            "positions_min": round(stats["positions_done"] * 60.0 / max(elapsed, 1e-9), 1),
        }
    )


if __name__ == "__main__":
    main()
