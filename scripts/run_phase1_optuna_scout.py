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
    if args.phase1_mcts_simulations is None:
        return candidates
    full_sims = int(args.phase1_mcts_simulations)
    return tuple(
        candidate.model_copy(
            update={
                "search": candidate.search.model_copy(
                    update={
                        "full_mcts_simulations": full_sims,
                        "pcr_low_sims": min(int(candidate.search.pcr_low_sims), max(1, full_sims - 32)),
                    }
                )
            }
        )
        for candidate in candidates
    )


if __name__ == "__main__":
    raise SystemExit(main())
