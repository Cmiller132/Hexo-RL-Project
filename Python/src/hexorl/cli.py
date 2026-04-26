"""CLI entry point for hexorl."""

import argparse
import logging
from pathlib import Path

from hexorl.config import load_config
from hexorl.epoch import run_epoch, run_tiny_training_smoke
from hexorl.eval.arena import run_arena
from hexorl.eval.classical import classical_opponent_fn


def main():
    parser = argparse.ArgumentParser(description="Hexo-RL training pipeline")
    subparsers = parser.add_subparsers(dest="command")

    epoch_p = subparsers.add_parser("epoch", help="Run one training epoch")
    epoch_p.add_argument("--config", type=Path, default=None)
    epoch_p.add_argument("--output-dir", type=Path, default=None)
    epoch_p.add_argument("--bootstrap-games", type=int, default=16)
    epoch_p.add_argument("--selfplay", action="store_true")
    epoch_p.add_argument("--no-train", action="store_true")

    smoke_p = subparsers.add_parser("smoke-train", help="Run a tiny multi-epoch training smoke")
    smoke_p.add_argument("--config", type=Path, default=None)
    smoke_p.add_argument("--output-dir", type=Path, default=Path("./runs/smoke"))
    smoke_p.add_argument("--epochs", type=int, default=3)

    arena_p = subparsers.add_parser("arena", help="Run classical-vs-classical arena smoke")
    arena_p.add_argument("--games", type=int, default=4)
    arena_p.add_argument("--time-ms", type=int, default=25)
    arena_p.add_argument("--depth", type=int, default=2)

    subparsers.add_parser("bench", help="Run the tiny training benchmark smoke")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "epoch":
        cfg = load_config(args.config)
        result = run_epoch(
            cfg,
            output_dir=args.output_dir,
            bootstrap_games=args.bootstrap_games,
            use_selfplay=args.selfplay,
            train=not args.no_train,
        )
        print(_format_result(result))
    elif args.command == "smoke-train":
        cfg = load_config(args.config) if args.config else None
        results = run_tiny_training_smoke(cfg, epochs=args.epochs, output_dir=args.output_dir)
        for result in results:
            print(_format_result(result))
    elif args.command == "bench":
        results = run_tiny_training_smoke(epochs=1, output_dir=Path("./runs/bench_smoke"))
        print(_format_result(results[-1]))
    elif args.command == "arena":
        side_a = classical_opponent_fn(time_ms=args.time_ms, max_depth=args.depth)
        side_b = classical_opponent_fn(time_ms=args.time_ms, max_depth=args.depth)
        stats = run_arena(side_a, side_b, num_games=args.games)
        print(
            f"games={stats.total_games} win_rate_a={stats.win_rate_a:.3f} "
            f"avg_moves={stats.avg_moves:.1f} elo_diff={stats.elo_diff:.1f}"
        )


def _format_result(result) -> str:
    loss = result.train_stats.get("loss_total", 0.0)
    ckpt = result.checkpoint_path or "-"
    return (
        f"epoch={result.train_stats.get('epoch', 0)} "
        f"loss_total={loss:.5f} "
        f"buffer={result.buffer_stats.get('size', 0)} "
        f"checkpoint={ckpt} "
        f"elapsed_s={result.elapsed_s:.2f}"
    )


if __name__ == "__main__":
    main()
