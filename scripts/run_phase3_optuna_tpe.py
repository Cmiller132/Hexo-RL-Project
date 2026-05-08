#!/usr/bin/env python
"""Run production Phase 3 Optuna TPE studies from Phase 2 promotion specs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.tuning.fixed_classical_eval import (
    DEFAULT_FIXED_CLASSICAL_MAX_MOVES,
    FixedClassicalEvalSettings,
)
from hexorl.tuning.phase3_runner import (
    DryRunPhase3TrialRunner,
    EpochPhase3TrialRunner,
    Phase3OptunaTpeRunner,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Optuna sequential run directory.")
    parser.add_argument(
        "--spec-path",
        default=None,
        help="Phase 3 study specs JSON. Defaults to <run-dir>/phase2_review/phase3_study_specs.json.",
    )
    parser.add_argument("--n-trials-per-study", type=int, default=1)
    parser.add_argument("--trial-epochs", type=int, default=4)
    parser.add_argument("--max-studies", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic dry trial runner.")
    parser.add_argument("--bootstrap-games", type=int, default=0)
    parser.add_argument("--no-selfplay", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fixed-classical-games", type=int, default=64)
    parser.add_argument("--fixed-classical-seed", type=int, default=20260507)
    parser.add_argument("--eval-time-ms", type=int, default=100)
    parser.add_argument("--eval-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-moves", type=int, default=DEFAULT_FIXED_CLASSICAL_MAX_MOVES)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    spec_path = Path(args.spec_path) if args.spec_path else run_dir / "phase2_review" / "phase3_study_specs.json"
    summary_path = Path(args.summary) if args.summary else run_dir / "phase3_runner_summary.json"
    runner = (
        DryRunPhase3TrialRunner()
        if args.dry_run
        else EpochPhase3TrialRunner(
            bootstrap_games=args.bootstrap_games,
            use_selfplay=not args.no_selfplay,
            train=True,
            device=None if args.device == "auto" else args.device,
        )
    )
    fixed_settings = FixedClassicalEvalSettings(
        games_per_candidate=args.fixed_classical_games,
        seed=args.fixed_classical_seed,
        eval_time_ms=args.eval_time_ms,
        eval_depth=args.eval_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        max_moves=args.max_moves,
        device=args.device,
    )
    summary = Phase3OptunaTpeRunner(
        run_dir=run_dir,
        spec_path=spec_path,
        trial_runner=runner,
        n_trials_per_study=args.n_trials_per_study,
        trial_epochs=args.trial_epochs,
        fixed_eval_settings=fixed_settings,
        max_studies=args.max_studies,
        summary_path=summary_path,
    ).run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
