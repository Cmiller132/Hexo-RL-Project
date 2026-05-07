#!/usr/bin/env python
"""Run fixed-classical evaluation for completed Optuna scout candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.tuning.fixed_classical_eval import (
    FixedClassicalEvalSettings,
    evaluate_run_fixed_classical,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Optuna scout run directory.")
    parser.add_argument("--candidate", action="append", default=[], help="Candidate id to evaluate; omit for all.")
    parser.add_argument("--games-per-candidate", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--eval-time-ms", type=int, default=100)
    parser.add_argument("--eval-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--opponent-id", default="fixed_strong")
    parser.add_argument("--confidence-method", default="normal_95")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or torch device string.")
    parser.add_argument("--summary", default="", help="Optional JSON summary path.")
    args = parser.parse_args()

    settings = FixedClassicalEvalSettings(
        games_per_candidate=args.games_per_candidate,
        seed=args.seed,
        eval_time_ms=args.eval_time_ms,
        eval_depth=args.eval_depth,
        temperature=args.temperature,
        top_p=args.top_p,
        max_moves=args.max_moves,
        opponent_id=args.opponent_id,
        confidence_method=args.confidence_method,
        device=args.device,
    )
    summary = evaluate_run_fixed_classical(
        Path(args.run_dir),
        settings=settings,
        candidate_ids=args.candidate,
        summary_path=Path(args.summary) if args.summary else None,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
