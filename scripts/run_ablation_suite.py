"""Run long full-pipeline training ablations.

Each ablation runs self-play, replay insertion, training, checkpointing, and
telemetry for multiple epochs in one process so optimizer/EMA state and replay
buffer state carry forward naturally across epochs.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import Config, load_config
from hexorl.dashboard.recorder import RunRecorder
from hexorl.epoch import run_epoch
from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.replay.storage import ReplayStorage
from hexorl.runtime import autotune_config, configure_torch_runtime


@dataclass(frozen=True)
class Ablation:
    name: str
    description: str
    overrides: dict[str, Any]


HEADS_FULL = ["policy", "value", "lookahead_4", "lookahead_12", "lookahead_36", "axis"]


ABLATIONS: list[Ablation] = [
    Ablation(
        "baseline_128x16_noise025",
        "Current tuned 128x16 baseline.",
        {},
    ),
    Ablation(
        "model_64x8",
        "Compact model to measure throughput/learning tradeoff.",
        {"model.channels": 64, "model.blocks": 8},
    ),
    Ablation(
        "model_96x12",
        "Mid-size model between compact and baseline.",
        {"model.channels": 96, "model.blocks": 12},
    ),
    Ablation(
        "model_160x20",
        "Larger model to test whether extra capacity is worth slower search/inference.",
        {"model.channels": 160, "model.blocks": 20},
    ),
    Ablation(
        "noise_low_a015_f015",
        "Lower root Dirichlet noise.",
        {"selfplay.dirichlet_alpha": 0.15, "selfplay.dirichlet_fraction": 0.15},
    ),
    Ablation(
        "noise_high_a050_f035",
        "Higher root Dirichlet noise.",
        {"selfplay.dirichlet_alpha": 0.50, "selfplay.dirichlet_fraction": 0.35},
    ),
    Ablation(
        "search_96sims",
        "Lower MCTS budget for faster but noisier targets.",
        {"selfplay.mcts_simulations": 96, "selfplay.pcr_low_sims": 48},
    ),
    Ablation(
        "search_192sims",
        "Higher MCTS budget for stronger targets.",
        {"selfplay.mcts_simulations": 192, "selfplay.pcr_low_sims": 96},
    ),
    Ablation(
        "pcr_low_more",
        "More frequent low-simulation PCR games.",
        {"selfplay.pcr_low_sim_prob": 0.75, "selfplay.pcr_low_sims": 64},
    ),
    Ablation(
        "pcr_low_less",
        "Less frequent low-simulation PCR games.",
        {"selfplay.pcr_low_sim_prob": 0.25, "selfplay.pcr_low_sims": 64},
    ),
    Ablation(
        "cpuct_100",
        "Lower exploration constant.",
        {"selfplay.c_puct": 1.0},
    ),
    Ablation(
        "cpuct_200",
        "Higher exploration constant.",
        {"selfplay.c_puct": 2.0},
    ),
    Ablation(
        "lr_0015",
        "Lower peak learning rate.",
        {"train.peak_lr": 0.0015},
    ),
    Ablation(
        "lr_0050",
        "Higher peak learning rate.",
        {"train.peak_lr": 0.0050},
    ),
    Ablation(
        "train_compile",
        "Torch compile for multi-epoch training.",
        {"runtime.compile_model": True},
    ),
]


SUITES = {
    "priority": [
        "baseline_128x16_noise025",
        "model_64x8",
        "model_96x12",
        "model_160x20",
        "noise_low_a015_f015",
        "noise_high_a050_f035",
        "search_96sims",
        "search_192sims",
        "pcr_low_more",
        "pcr_low_less",
        "cpuct_100",
        "cpuct_200",
        "lr_0015",
        "lr_0050",
        "train_compile",
    ],
    "models": ["baseline_128x16_noise025", "model_64x8", "model_96x12", "model_160x20"],
    "noise": ["baseline_128x16_noise025", "noise_low_a015_f015", "noise_high_a050_f035"],
    "search": ["baseline_128x16_noise025", "search_96sims", "search_192sims", "pcr_low_more", "pcr_low_less"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output-root", default="runs/ablations_priority")
    parser.add_argument("--suite", choices=sorted(SUITES), default="priority")
    parser.add_argument("--only", default="", help="Comma-separated ablation names to run.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--bootstrap-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-time-ms", type=int, default=25)
    parser.add_argument("--eval-depth", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.epochs < 10:
        raise ValueError("--epochs must be at least 10 for long-term ablations")

    base_cfg = load_config(Path(args.config))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    suite = _select_ablations(args.suite, args.only)
    _write_json(
        output_root / "suite_manifest.json",
        {
            "config": args.config,
            "epochs": args.epochs,
            "bootstrap_games": args.bootstrap_games,
            "seed": args.seed,
            "suite": args.suite,
            "ablations": [
                {"name": item.name, "description": item.description, "overrides": item.overrides}
                for item in suite
            ],
        },
    )
    logging.info("Selected %d ablations: %s", len(suite), ", ".join(a.name for a in suite))
    if args.dry_run:
        return

    suite_summary = output_root / "suite_summary.jsonl"
    for index, ablation in enumerate(suite):
        run_dir = output_root / ablation.name
        done_path = run_dir / "DONE"
        if done_path.exists():
            logging.info("Skipping completed ablation %s", ablation.name)
            continue
        _cleanup_shared_memory()
        cfg = _make_config(base_cfg, ablation, run_dir, args.seed + index * 1009)
        host = autotune_config(cfg, selfplay_enabled=True)
        runtime = configure_torch_runtime(cfg, host)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "variant.json", {
            "name": ablation.name,
            "description": ablation.description,
            "overrides": ablation.overrides,
            "runtime": runtime,
        })
        _write_json(run_dir / "config.resolved.json", cfg.model_dump(mode="json"))

        logging.info("Starting %s for %d epochs | runtime=%s", ablation.name, args.epochs, runtime)
        status = _run_ablation(cfg, ablation, run_dir, args, suite_summary)
        if args.eval_games > 0 and status.get("checkpoint_path"):
            status["eval"] = _run_eval(cfg, Path(status["checkpoint_path"]), args)
            _append_jsonl(suite_summary, {"event": "ablation_eval", **status})
        _write_json(done_path, status)
        logging.info("Completed %s", ablation.name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _select_ablations(suite_name: str, only: str) -> list[Ablation]:
    by_name = {item.name: item for item in ABLATIONS}
    names = [name.strip() for name in only.split(",") if name.strip()] or SUITES[suite_name]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown ablations: {missing}")
    return [by_name[name] for name in names]


def _make_config(base_cfg: Config, ablation: Ablation, run_dir: Path, seed: int) -> Config:
    cfg = base_cfg.model_copy(deep=True)
    cfg.run.output_dir = str(run_dir)
    cfg.run.seed = seed
    cfg.model.heads = list(HEADS_FULL)
    cfg.buffer.lookahead_horizons = [4, 12, 36]
    cfg.buffer.lookahead_lambdas = [0.75, 0.90, 0.97]
    cfg.selfplay.num_workers = 0
    cfg.selfplay.batch_size_per_worker = 0
    cfg.inference.max_batch_size = 0
    cfg.train.batch_size = 0
    cfg.runtime.compile_inference = False
    for path, value in ablation.overrides.items():
        _set_path(cfg, path, value)
    return cfg


def _run_ablation(
    cfg: Config,
    ablation: Ablation,
    run_dir: Path,
    args: argparse.Namespace,
    suite_summary: Path,
) -> dict[str, Any]:
    _seed_everything(cfg.run.seed)
    recorder = RunRecorder.for_run_dir(run_dir, run_id=ablation.name)
    replay = ReplayStorage(
        capacity=cfg.buffer.capacity,
        prefetch_records=cfg.train.prefetch_batches,
    )
    trainer = None
    last_record: dict[str, Any] = {}
    started = time.monotonic()
    for local_epoch in range(1, args.epochs + 1):
        logging.info("%s epoch %d/%d", ablation.name, local_epoch, args.epochs)
        result = run_epoch(
            cfg,
            trainer=trainer,
            buffer=replay,
            output_dir=run_dir,
            bootstrap_games=args.bootstrap_games if local_epoch == 1 else 0,
            use_selfplay=True,
            train=True,
            recorder=recorder,
        )
        trainer = result.trainer
        selfplay = _latest_metric(run_dir / "events.jsonl", "selfplay")
        record = {
            "event": "epoch_complete",
            "ablation": ablation.name,
            "local_epoch": local_epoch,
            "train_epoch": int(result.train_stats.get("epoch", local_epoch)),
            "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
            "epoch_elapsed_s": result.elapsed_s,
            "buffer_size": result.buffer_stats.get("size"),
            "full_search_pct": result.buffer_stats.get("full_search_pct"),
            "train": result.train_stats,
            "selfplay": selfplay,
        }
        _append_jsonl(run_dir / "summary.jsonl", record)
        _append_jsonl(suite_summary, record)
        _write_json(run_dir / "LATEST.json", record)
        last_record = record
    return {
        "event": "ablation_complete",
        "ablation": ablation.name,
        "epochs": args.epochs,
        "elapsed_s": time.monotonic() - started,
        "checkpoint_path": last_record.get("checkpoint_path"),
        "latest": last_record,
    }


def _run_eval(cfg: Config, checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    logging.info("Evaluating %s with %d arena games", checkpoint, args.eval_games)
    model = load_checkpoint_model(checkpoint, cfg)
    model_player = model_move_fn(model, temperature=0.20, top_p=0.95, seed=cfg.run.seed)
    classical = classical_opponent_fn(time_ms=args.eval_time_ms, max_depth=args.eval_depth)
    stats = run_arena(model_player, classical, num_games=args.eval_games)
    reason_counts = stats.reason_counts
    return {
        "games": stats.total_games,
        "model_win_rate": stats.win_rate_a,
        "model_wins": stats.wins_a,
        "opponent_wins": stats.wins_b,
        "draws": stats.draws,
        "elo_diff": stats.elo_diff,
        "avg_moves": stats.avg_moves,
        "games_per_min": stats.games_per_min,
        "reason_counts": reason_counts,
        "crash_games": sum(v for k, v in reason_counts.items() if k.startswith("crash")),
        "illegal_games": sum(v for k, v in reason_counts.items() if k.startswith("illegal")),
        "no_move_games": reason_counts.get("no_move", 0),
    }


def _set_path(cfg: Config, dotted_path: str, value: Any) -> None:
    obj: Any = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _latest_metric(events_path: Path, phase: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not events_path.exists():
        return latest
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event_type") == "metric" and row.get("phase") == phase:
                latest = dict(row.get("payload") or {})
    return latest


def _cleanup_shared_memory() -> None:
    shm = Path("/dev/shm")
    if not shm.exists():
        return
    for path in shm.glob("hexorl_*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
