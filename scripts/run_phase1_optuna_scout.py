#!/usr/bin/env python
"""Run the Phase 0/1 queued Optuna architecture scout controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

    controller = Phase1OptunaScoutController(
        runs_root=Path(args.runs_root),
        run_id=args.run_id,
        base_config=Config(),
        storage=args.storage,
        runner=runner,
    )
    summary = controller.run(max_quanta=args.max_quanta)
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
