"""Append corrected Phase 1 scorecards from candidate dashboard metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.tuning.scorecard_repair import repair_phase1_scorecards_from_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Optuna scout run directory.")
    parser.add_argument("--candidate", action="append", help="Candidate id to repair; defaults to all candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Report append count without writing scorecards.")
    parser.add_argument("--summary", help="Optional JSON summary path.")
    args = parser.parse_args()

    summary = repair_phase1_scorecards_from_dashboard(
        Path(args.run_dir),
        candidate_ids=args.candidate,
        dry_run=args.dry_run,
    )
    payload = summary.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
