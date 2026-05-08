#!/usr/bin/env python
"""Continuously advance non-legacy Phase 3 Optuna TPE rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.tuning.phase3_supervisor import (
    DEFAULT_FIXED_CLASSICAL_GAMES,
    DEFAULT_FIXED_CLASSICAL_MAX_MOVES,
    DEFAULT_MAX_TRIALS_PER_STUDY,
    DEFAULT_PHASE3_TRIAL_EPOCHS,
    Phase3AutonomousSupervisor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--spec-path", type=Path, default=None)
    parser.add_argument("--start-target", default="auto")
    parser.add_argument("--max-trials-per-study", type=int, default=DEFAULT_MAX_TRIALS_PER_STUDY)
    parser.add_argument("--trial-epochs", type=int, default=DEFAULT_PHASE3_TRIAL_EPOCHS)
    parser.add_argument("--fixed-classical-games", type=int, default=DEFAULT_FIXED_CLASSICAL_GAMES)
    parser.add_argument("--fixed-classical-seed", type=int, default=20260507)
    parser.add_argument("--eval-time-ms", type=int, default=100)
    parser.add_argument("--eval-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_FIXED_CLASSICAL_MAX_MOVES)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--retry-limit", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-dashboard-mirror", action="store_true")
    args = parser.parse_args()

    start_target: int | str
    if str(args.start_target).lower() == "auto":
        start_target = "auto"
    else:
        start_target = int(args.start_target)

    summary = Phase3AutonomousSupervisor(
        run_dir=args.run_dir,
        spec_path=args.spec_path,
        start_target=start_target,
        max_trials_per_study=args.max_trials_per_study,
        trial_epochs=args.trial_epochs,
        fixed_classical_games=args.fixed_classical_games,
        fixed_classical_seed=args.fixed_classical_seed,
        eval_time_ms=args.eval_time_ms,
        eval_depth=args.eval_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        max_moves=args.max_moves,
        summary_path=args.summary,
        dry_run=args.dry_run,
        max_rounds=args.max_rounds,
        retry_limit=args.retry_limit,
        mirror_dashboards=not args.no_dashboard_mirror,
        sleep_seconds=args.sleep_seconds,
    ).run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
