#!/usr/bin/env python
"""Run fixed-classical evaluation for Phase 3 child trial directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hexorl.tuning.fixed_classical_eval import (
    FixedClassicalEvalSettings,
    evaluate_candidate_fixed_classical,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--games-per-candidate", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--eval-time-ms", type=int, default=100)
    parser.add_argument("--eval-depth", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--opponent-id", default="fixed_strong")
    parser.add_argument("--confidence-method", default="normal_95")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
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
    results = []
    for candidate_id in args.candidate:
        candidate_dir = run_dir / "phase3_trials" / candidate_id
        if not candidate_dir.exists():
            raise FileNotFoundError(f"Phase 3 candidate directory not found: {candidate_dir}")
        result = evaluate_candidate_fixed_classical(
            candidate_dir,
            run_dir=run_dir,
            settings=settings,
        )
        results.append(asdict(result))

    summary = {
        "run_dir": str(run_dir),
        "settings": asdict(settings),
        "candidates": results,
        "appended_games": sum(int(item.get("appended_games") or 0) for item in results),
        "appended_scorecards": sum(int(item.get("appended_scorecards") or 0) for item in results),
    }
    if args.summary:
        output = Path(args.summary)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
