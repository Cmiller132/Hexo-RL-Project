"""Time one self-play game with a fake evaluator.

This isolates Rust MCTS/game-loop cost from the shared-memory GPU server.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from hexorl.config import load_config
from hexorl.runtime import autotune_config
from hexorl.selfplay.worker import SelfPlayWorker


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.positions = 0

    def submit(self, tensor, count):
        self.calls += 1
        self.positions += int(count)
        return (
            np.zeros(int(count) * 1089, dtype=np.float32),
            np.zeros(int(count), dtype=np.float32),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--max-moves", type=int, default=128)
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=False)
    cfg.selfplay.num_workers = 1
    cfg.selfplay.mcts_simulations = args.sims
    cfg.selfplay.pcr_low_sim_prob = 0.0
    cfg.selfplay.max_game_moves = args.max_moves

    worker = SelfPlayWorker(
        0,
        cfg,
        mp.Queue(),
        num_workers=1,
        max_batch_size=cfg.inference.max_batch_size,
    )
    client = FakeClient()
    t0 = time.monotonic()
    game = worker._play_one_game(client)
    elapsed = time.monotonic() - t0
    print(
        {
            "elapsed_s": round(elapsed, 3),
            "positions": len(game.positions) if game else None,
            "eval_calls": client.calls,
            "eval_positions": client.positions,
            "eval_positions_s": round(client.positions / max(elapsed, 1e-9), 1),
            "truncated": getattr(game, "truncated", None),
            "terminal_reason": getattr(game, "terminal_reason", None),
            "outcome": getattr(game, "outcome", None),
        }
    )


if __name__ == "__main__":
    main()
