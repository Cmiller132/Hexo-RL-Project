"""Quick dense-CNN-only ablations for policy-learning diagnostics."""

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

from hexorl.buffer.ring import RingBuffer, replay_feature_flags
from hexorl.config import Config, load_config
from hexorl.dashboard.recorder import RunRecorder
from hexorl.epoch import run_epoch
from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
from hexorl.eval.classical import classical_opponent_fn
from hexorl.runtime import configure_torch_runtime


@dataclass(frozen=True)
class Variant:
    name: str
    overrides: dict[str, Any]


VARIANTS = (
    Variant("cnn32x4_lr3e4_fullonly", {"train.peak_lr": 3e-4}),
    Variant("cnn32x4_lr1e3_fullonly", {"train.peak_lr": 1e-3}),
    Variant("cnn32x4_lr3e3_const_fullonly", {"train.peak_lr": 3e-3, "train.lr_schedule": "constant"}),
    Variant(
        "cnn32x4_lr1e3_allpolicy",
        {"train.peak_lr": 1e-3, "selfplay.train_policy_on_full_search_only": False},
    ),
    Variant(
        "cnn32x4_lr1e3_fullonly_low_noise",
        {
            "train.peak_lr": 1e-3,
            "selfplay.temperature_schedule": [[0, 0.0]],
            "selfplay.dirichlet_fraction": 0.0,
        },
    ),
    Variant(
        "cnn32x4_lr1e3_allpolicy_low_noise",
        {
            "train.peak_lr": 1e-3,
            "selfplay.train_policy_on_full_search_only": False,
            "selfplay.temperature_schedule": [[0, 0.0]],
            "selfplay.dirichlet_fraction": 0.0,
        },
    ),
    Variant(
        "cnn32x4_lr1e3_allpolicy_seed8",
        {
            "train.peak_lr": 1e-3,
            "selfplay.train_policy_on_full_search_only": False,
            "selfplay.classical_seed_plies": 8,
            "selfplay.classical_seed_time_ms": 10,
            "selfplay.classical_seed_max_depth": 3,
            "selfplay.classical_seed_near_radius": 2,
        },
    ),
    Variant(
        "cnn32x4_lr1e3_allpolicy_seed8_shape",
        {
            "train.peak_lr": 1e-3,
            "selfplay.train_policy_on_full_search_only": False,
            "selfplay.classical_seed_plies": 8,
            "selfplay.classical_seed_time_ms": 10,
            "selfplay.classical_seed_max_depth": 3,
            "selfplay.classical_seed_near_radius": 2,
            "selfplay.tactical_target_mix": 0.25,
            "selfplay.open_four_target_mix": 0.25,
        },
    ),
    Variant(
        "cnn32x4_lr1e3_allpolicy_seed8_rootprior",
        {
            "train.peak_lr": 1e-3,
            "selfplay.train_policy_on_full_search_only": False,
            "selfplay.classical_seed_plies": 8,
            "selfplay.classical_seed_time_ms": 10,
            "selfplay.classical_seed_max_depth": 3,
            "selfplay.classical_seed_near_radius": 2,
            "selfplay.classical_root_prior_mix": 0.5,
            "selfplay.classical_root_prior_time_ms": 10,
            "selfplay.classical_root_prior_max_depth": 3,
            "selfplay.classical_root_prior_near_radius": 2,
        },
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output-root", default="runs/quick_dense_cnn_ablation_512sims_pcr50")
    parser.add_argument("--only", default="", help="Comma-separated variant names.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--states-per-epoch", type=int, default=768)
    parser.add_argument("--max-game-moves", type=int, default=768)
    parser.add_argument("--bootstrap-games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-time-ms", type=int, default=50)
    parser.add_argument("--eval-depth", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.states_per_epoch <= 0:
        raise ValueError("--states-per-epoch must be positive")

    base_cfg = load_config(Path(args.config))
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    variants = _select_variants(args.only)
    _write_json(
        root / "suite_manifest.json",
        {
            "config": args.config,
            "epochs": args.epochs,
            "states_per_epoch": args.states_per_epoch,
            "max_game_moves": args.max_game_moves,
            "mcts_simulations": 512,
            "pcr_low_sim_prob": 0.5,
            "pcr_low_sims": 128,
            "variants": [{"name": v.name, "overrides": v.overrides} for v in variants],
        },
    )

    suite_summary = root / "suite_summary.jsonl"
    for index, variant in enumerate(variants):
        run_dir = root / variant.name
        done = run_dir / "DONE"
        if done.exists():
            logging.info("Skipping completed variant %s", variant.name)
            continue
        cfg = _make_config(base_cfg, variant, run_dir, args, seed=args.seed + 1009 * index)
        _seed_everything(cfg.run.seed)
        runtime = configure_torch_runtime(cfg)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "variant.json", {"name": variant.name, "overrides": variant.overrides, "runtime": runtime})
        _write_json(run_dir / "config.resolved.json", cfg.model_dump(mode="json"))

        logging.info("Starting %s", variant.name)
        result = _run_variant(cfg, variant, run_dir, args, suite_summary)
        if args.eval_games > 0 and result.get("checkpoint_path"):
            result["eval"] = _run_eval(cfg, Path(str(result["checkpoint_path"])), args)
            _append_jsonl(suite_summary, {"event": "variant_eval", **result})
        _write_json(done, result)
        logging.info("Completed %s", variant.name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


def _select_variants(only: str) -> tuple[Variant, ...]:
    by_name = {v.name: v for v in VARIANTS}
    if not only.strip():
        return VARIANTS
    names = [name.strip() for name in only.split(",") if name.strip()]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown variants: {missing}")
    return tuple(by_name[name] for name in names)


def _make_config(base: Config, variant: Variant, run_dir: Path, args: argparse.Namespace, *, seed: int) -> Config:
    cfg = base.model_copy(deep=True)
    cfg.run.seed = int(seed)
    cfg.run.output_dir = str(run_dir)
    cfg.model.architecture = "cnn"
    cfg.model.channels = 32
    cfg.model.blocks = 4
    cfg.model.heads = ["policy", "value"]
    cfg.model.sparse_policy = False
    cfg.model.attention_positions = []
    cfg.selfplay.num_workers = 4
    cfg.selfplay.batch_size_per_worker = 8
    cfg.selfplay.games_per_epoch = 0
    cfg.selfplay.states_per_epoch = int(args.states_per_epoch)
    cfg.selfplay.max_game_moves = int(args.max_game_moves)
    cfg.selfplay.mcts_simulations = 512
    cfg.selfplay.pcr_low_sim_prob = 0.5
    cfg.selfplay.pcr_low_sims = 128
    cfg.selfplay.policy_target_top_k = 96
    cfg.selfplay.train_policy_on_full_search_only = True
    cfg.inference.max_batch_size = 96
    cfg.inference.max_wait_us = 200
    cfg.inference.fp16 = True
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.buffer.pcr_weight = 0.25
    cfg.train.batch_size = 128
    cfg.train.batches_per_epoch = 16
    cfg.train.loss_weights = {"policy": 1.0, "value": 1.0}
    cfg.train.lr_schedule = "cosine"
    cfg.train.peak_lr = 3e-4
    cfg.train.weight_decay = 1e-4
    cfg.runtime.autotune = False
    cfg.runtime.compile_model = False
    cfg.runtime.compile_inference = False
    cfg.runtime.selfplay_workers = 4
    cfg.runtime.dataloader_workers = 0
    for path, value in variant.overrides.items():
        _set_path(cfg, path, value)
    return Config.model_validate(cfg.model_dump(mode="json"))


def _run_variant(
    cfg: Config,
    variant: Variant,
    run_dir: Path,
    args: argparse.Namespace,
    suite_summary: Path,
) -> dict[str, Any]:
    recorder = RunRecorder.for_run_dir(run_dir, run_id=variant.name)
    replay = RingBuffer(
        capacity=cfg.buffer.capacity,
        max_policy_entries=cfg.selfplay.policy_target_top_k,
        max_policy_v2_entries=min(max(cfg.selfplay.policy_target_top_k, cfg.model.candidate_budget), 512),
        recency_decay=cfg.buffer.recency_decay,
        num_lookahead=len(cfg.buffer.lookahead_horizons),
        **replay_feature_flags(cfg.model.heads, architecture=cfg.model.architecture, sparse_policy=cfg.model.sparse_policy),
    )
    trainer = None
    latest: dict[str, Any] = {}
    started = time.monotonic()
    for epoch in range(1, int(args.epochs) + 1):
        result = run_epoch(
            cfg,
            trainer=trainer,
            buffer=replay,
            output_dir=run_dir / "checkpoints",
            bootstrap_games=int(args.bootstrap_games) if epoch == 1 else 0,
            use_selfplay=True,
            train=True,
            recorder=recorder,
        )
        trainer = result.trainer
        latest = {
            "event": "epoch_complete",
            "variant": variant.name,
            "epoch": epoch,
            "train_epoch": int(result.train_stats.get("epoch", epoch)),
            "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else "",
            "elapsed_s": float(result.elapsed_s),
            "buffer_size": int(result.buffer_stats.get("size", 0) or 0),
            "full_search_pct": float(result.buffer_stats.get("full_search_pct", 0.0) or 0.0),
            "train": result.train_stats,
            "selfplay": {
                key: result.buffer_stats.get(key)
                for key in (
                    "games_done",
                    "positions_done",
                    "truncated_games",
                    "truncation_rate",
                    "terminal_reason_win",
                    "terminal_reason_max_game_moves",
                )
                if key in result.buffer_stats
            },
        }
        _append_jsonl(run_dir / "summary.jsonl", latest)
        _append_jsonl(suite_summary, latest)
        _write_json(run_dir / "LATEST.json", latest)
    return {
        "event": "variant_complete",
        "variant": variant.name,
        "epochs": int(args.epochs),
        "elapsed_s": time.monotonic() - started,
        "checkpoint_path": latest.get("checkpoint_path", ""),
        "latest": latest,
    }


def _run_eval(cfg: Config, checkpoint: Path, args: argparse.Namespace) -> dict[str, Any]:
    model = load_checkpoint_model(checkpoint, cfg)
    model_player = model_move_fn(model, temperature=0.05, top_p=0.95, seed=cfg.run.seed)
    classical = classical_opponent_fn(time_ms=int(args.eval_time_ms), max_depth=int(args.eval_depth))
    stats = run_arena(model_player, classical, num_games=int(args.eval_games), sims=128)
    return {
        "games": stats.total_games,
        "model_wins": stats.wins_a,
        "opponent_wins": stats.wins_b,
        "draws": stats.draws,
        "model_win_rate": stats.win_rate_a,
        "avg_moves": stats.avg_moves,
        "reason_counts": stats.reason_counts,
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
