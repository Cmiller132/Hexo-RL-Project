"""CLI entry point for hexorl."""

import argparse
import logging
from pathlib import Path

from hexorl.config import load_config
from hexorl.dashboard.checkpoints import scan_checkpoints
from hexorl.dashboard.db import DashboardStore
from hexorl.epoch import run_epoch, run_tiny_training_smoke
from hexorl.eval.arena import run_arena
from hexorl.eval.arena import load_checkpoint_model, model_move_fn
from hexorl.eval.classical import classical_opponent_fn
from hexorl.runtime import autotune_config, configure_torch_runtime


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

    eval_p = subparsers.add_parser("eval", help="Run noisy checkpoint-vs-classical eval")
    eval_p.add_argument("checkpoint", type=Path)
    eval_p.add_argument("--config", type=Path, default=None)
    eval_p.add_argument("--games", type=int, default=20)
    eval_p.add_argument("--temperature", type=float, default=0.35)
    eval_p.add_argument("--top-p", type=float, default=0.98)
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("--time-ms", type=int, default=50)
    eval_p.add_argument("--depth", type=int, default=3)

    dash_p = subparsers.add_parser("dashboard", help="Serve the FastAPI/React dashboard")
    dash_p.add_argument("--db", type=Path, default=Path("./runs/dashboard.sqlite3"))
    dash_p.add_argument("--host", default="127.0.0.1")
    dash_p.add_argument("--port", type=int, default=8765)

    idx_p = subparsers.add_parser("index-checkpoints", help="Import/index checkpoints into dashboard DB")
    idx_p.add_argument("path", type=Path)
    idx_p.add_argument("--db", type=Path, default=Path("./runs/dashboard.sqlite3"))
    idx_p.add_argument("--run-id", default=None)

    subparsers.add_parser("bench", help="Run the tiny training benchmark smoke")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "epoch":
        cfg = load_config(args.config)
        host = autotune_config(cfg, selfplay_enabled=args.selfplay)
        runtime = configure_torch_runtime(cfg, host)
        logging.info("Runtime: %s", runtime)
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
        if cfg is not None:
            host = autotune_config(cfg, selfplay_enabled=False)
            runtime = configure_torch_runtime(cfg, host)
            logging.info("Runtime: %s", runtime)
        results = run_tiny_training_smoke(cfg, epochs=args.epochs, output_dir=args.output_dir)
        for result in results:
            print(_format_result(result))
    elif args.command == "bench":
        cfg = load_config()
        host = autotune_config(cfg, selfplay_enabled=False)
        runtime = configure_torch_runtime(cfg, host)
        logging.info("Runtime: %s", runtime)
        results = run_tiny_training_smoke(cfg, epochs=1, output_dir=Path("./runs/bench_smoke"))
        print(_format_result(results[-1]))
    elif args.command == "arena":
        side_a = classical_opponent_fn(time_ms=args.time_ms, max_depth=args.depth)
        side_b = classical_opponent_fn(time_ms=args.time_ms, max_depth=args.depth)
        stats = run_arena(side_a, side_b, num_games=args.games)
        print(
            f"games={stats.total_games} win_rate_a={stats.win_rate_a:.3f} "
            f"avg_moves={stats.avg_moves:.1f} elo_diff={stats.elo_diff:.1f}"
        )
    elif args.command == "eval":
        cfg = load_config(args.config)
        model = load_checkpoint_model(args.checkpoint, cfg)
        side_a = model_move_fn(
            model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        side_b = classical_opponent_fn(time_ms=args.time_ms, max_depth=args.depth)
        stats = run_arena(side_a, side_b, num_games=args.games)
        print(
            f"games={stats.total_games} model_win_rate={stats.win_rate_a:.3f} "
            f"avg_moves={stats.avg_moves:.1f} elo_diff={stats.elo_diff:.1f}"
        )
    elif args.command == "dashboard":
        import uvicorn

        from hexorl.dashboard.app import create_app

        app = create_app(args.db)
        print(f"Serving dashboard at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "index-checkpoints":
        store = DashboardStore(args.db)
        results = scan_checkpoints(args.path, store, run_id=args.run_id)
        for result in results:
            status = "loadable" if result.is_loadable else "metadata-only"
            print(
                f"{result.checkpoint_id}: {result.path} "
                f"epoch={result.epoch} step={result.global_step} {status}"
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
