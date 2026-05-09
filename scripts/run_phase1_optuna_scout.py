#!/usr/bin/env python
"""Run the Phase 0/1 queued Optuna architecture scout controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.autotune import candidate_recipes_from_config, candidate_recipes_from_plan_entries
from hexorl.config import Config
from hexorl.tuning.optuna_scout import (
    DryRunScoutEpochRunner,
    EpochScoutEpochRunner,
    Phase1OptunaScoutController,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs", help="Root directory for scout runs.")
    parser.add_argument("--run-id", required=True, help="Durable run id under --runs-root.")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL.")
    parser.add_argument(
        "--candidate-plan",
        action="append",
        default=[],
        help="Override scout candidate plan entry '<architecture_id>:<pair_strategy>'. Repeat for each candidate.",
    )
    parser.add_argument(
        "--max-game-moves",
        type=int,
        default=None,
        help="Override selfplay.max_game_moves for every candidate in this run.",
    )
    parser.add_argument(
        "--phase1-mcts-simulations",
        type=int,
        default=None,
        help="Override selfplay.mcts_simulations for Phase 1 candidate materialization.",
    )
    parser.add_argument(
        "--phase1-states-per-epoch",
        type=int,
        default=None,
        help="Override autotune.scout.min_generated_selfplay_positions_per_epoch for Phase 1.",
    )
    parser.add_argument(
        "--phase1-train-batches-per-epoch",
        type=int,
        default=None,
        help="Override train.batches_per_epoch for Phase 1 scout epochs.",
    )
    parser.add_argument(
        "--phase1-candidate-budget",
        type=int,
        default=None,
        help="Override model.candidate_budget for Phase 1 legal/action rows.",
    )
    parser.add_argument(
        "--phase1-global-graph-leaf-eval",
        action="store_true",
        help="Enable global graph model evaluation at MCTS leaves instead of root-only neutral rollouts.",
    )
    parser.add_argument(
        "--phase1-graph-dataloader-workers",
        type=int,
        default=None,
        help="Override runtime.graph_dataloader_workers for graph training.",
    )
    parser.add_argument(
        "--phase1-dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Override runtime.dataloader_prefetch_factor for training DataLoaders.",
    )
    parser.add_argument(
        "--phase1-graph-cache-size",
        type=int,
        default=None,
        help="Override runtime.graph_cache_size for per-worker graph-base caches.",
    )
    parser.add_argument(
        "--phase1-graph-relation-rebuild-threads",
        type=int,
        default=None,
        help="Override runtime.graph_relation_rebuild_threads for sparse-to-dense training relation rebuilds.",
    )
    parser.add_argument(
        "--phase1-disable-dataloader-pin-memory",
        action="store_true",
        help="Disable CUDA DataLoader pin_memory for large graph batches.",
    )
    parser.add_argument(
        "--phase1-inference-start-timeout-s",
        type=float,
        default=None,
        help="Override runtime.inference_start_timeout_s for slower WSL/CUDA process startup.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic smoke runner.")
    parser.add_argument("--production", action="store_true", help="Run real self-play/training epoch quanta.")
    parser.add_argument(
        "--bootstrap-games",
        type=int,
        default=0,
        help="Optional bootstrap games per production epoch before training.",
    )
    parser.add_argument(
        "--no-selfplay",
        action="store_true",
        help="Production mode only: train from bootstrap/replay without running self-play.",
    )
    parser.add_argument("--max-quanta", type=int, default=None, help="Stop after this many quanta.")
    args = parser.parse_args()

    if args.dry_run == args.production:
        raise SystemExit(
            "Choose exactly one runner mode: --dry-run for smoke/resume verification "
            "or --production for real self-play/training epoch quanta."
        )
    runner = (
        DryRunScoutEpochRunner()
        if args.dry_run
        else EpochScoutEpochRunner(
            bootstrap_games=args.bootstrap_games,
            use_selfplay=not args.no_selfplay,
            train=True,
        )
    )

    base_config = _base_config_from_args(args)
    candidates = _candidates_from_args(base_config, args)
    controller = Phase1OptunaScoutController(
        runs_root=Path(args.runs_root),
        run_id=args.run_id,
        base_config=base_config,
        candidates=candidates,
        storage=args.storage,
        runner=runner,
    )
    summary = controller.run(max_quanta=args.max_quanta)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


def _base_config_from_args(args: argparse.Namespace) -> Config:
    data = Config().model_dump(mode="json")
    if args.max_game_moves is not None:
        if int(args.max_game_moves) <= 0:
            raise SystemExit("--max-game-moves must be positive")
        data["selfplay"]["max_game_moves"] = int(args.max_game_moves)
    if args.phase1_mcts_simulations is not None:
        if int(args.phase1_mcts_simulations) <= 0:
            raise SystemExit("--phase1-mcts-simulations must be positive")
        data["selfplay"]["mcts_simulations"] = int(args.phase1_mcts_simulations)
    if args.phase1_states_per_epoch is not None:
        if int(args.phase1_states_per_epoch) <= 0:
            raise SystemExit("--phase1-states-per-epoch must be positive")
        data["autotune"]["scout"]["min_generated_selfplay_positions_per_epoch"] = int(args.phase1_states_per_epoch)
    if args.phase1_train_batches_per_epoch is not None:
        if int(args.phase1_train_batches_per_epoch) <= 0:
            raise SystemExit("--phase1-train-batches-per-epoch must be positive")
        data["train"]["batches_per_epoch"] = int(args.phase1_train_batches_per_epoch)
    if args.phase1_candidate_budget is not None:
        if int(args.phase1_candidate_budget) <= 0:
            raise SystemExit("--phase1-candidate-budget must be positive")
        data["model"]["candidate_budget"] = int(args.phase1_candidate_budget)
    if bool(getattr(args, "phase1_global_graph_leaf_eval", False)):
        data["model"]["global_graph_leaf_eval"] = True
    if args.phase1_graph_dataloader_workers is not None:
        if int(args.phase1_graph_dataloader_workers) < 0:
            raise SystemExit("--phase1-graph-dataloader-workers must be non-negative")
        data["runtime"]["graph_dataloader_workers"] = int(args.phase1_graph_dataloader_workers)
    if args.phase1_dataloader_prefetch_factor is not None:
        if int(args.phase1_dataloader_prefetch_factor) <= 0:
            raise SystemExit("--phase1-dataloader-prefetch-factor must be positive")
        data["runtime"]["dataloader_prefetch_factor"] = int(args.phase1_dataloader_prefetch_factor)
    if args.phase1_graph_cache_size is not None:
        if int(args.phase1_graph_cache_size) < 0:
            raise SystemExit("--phase1-graph-cache-size must be non-negative")
        data["runtime"]["graph_cache_size"] = int(args.phase1_graph_cache_size)
    graph_relation_rebuild_threads = getattr(args, "phase1_graph_relation_rebuild_threads", None)
    if graph_relation_rebuild_threads is not None:
        if int(graph_relation_rebuild_threads) < 0:
            raise SystemExit("--phase1-graph-relation-rebuild-threads must be non-negative")
        data["runtime"]["graph_relation_rebuild_threads"] = int(graph_relation_rebuild_threads)
    if bool(getattr(args, "phase1_disable_dataloader_pin_memory", False)):
        data["runtime"]["dataloader_pin_memory"] = False
    inference_start_timeout_s = getattr(args, "phase1_inference_start_timeout_s", None)
    if inference_start_timeout_s is not None:
        if float(inference_start_timeout_s) <= 0.0:
            raise SystemExit("--phase1-inference-start-timeout-s must be positive")
        data["runtime"]["inference_start_timeout_s"] = float(inference_start_timeout_s)
    return Config.model_validate(data)


def _candidates_from_args(base_config: Config, args: argparse.Namespace):
    candidates = (
        candidate_recipes_from_plan_entries(
            args.candidate_plan,
            metadata_source="cli.candidate_plan",
        )
        if args.candidate_plan
        else candidate_recipes_from_config(base_config)
    )
    phase1_global_graph_leaf_eval = bool(getattr(args, "phase1_global_graph_leaf_eval", False))
    if (
        args.phase1_mcts_simulations is None
        and args.phase1_candidate_budget is None
        and not phase1_global_graph_leaf_eval
    ):
        return candidates
    full_sims = int(args.phase1_mcts_simulations) if args.phase1_mcts_simulations is not None else None
    candidate_budget = (
        int(args.phase1_candidate_budget)
        if args.phase1_candidate_budget is not None
        else None
    )
    updated = []
    for candidate in candidates:
        update = {}
        if full_sims is not None:
            update["search"] = candidate.search.model_copy(
                update={
                    "full_mcts_simulations": full_sims,
                    "pcr_low_sims": min(int(candidate.search.pcr_low_sims), max(1, full_sims - 32)),
                }
            )
        model_update = {}
        if candidate_budget is not None:
            model_update["candidate_budget"] = candidate_budget
        if phase1_global_graph_leaf_eval:
            model_update["global_graph_leaf_eval"] = True
        if model_update:
            update["model"] = candidate.model.model_copy(update=model_update)
        updated.append(candidate.model_copy(update=update))
    return tuple(updated)


if __name__ == "__main__":
    raise SystemExit(main())
