#!/usr/bin/env python
"""Select the final Optuna scout/tuning champion from saved Hexo scorecards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexorl.tuning.champion import (
    build_champion_selection_report_from_scorecard_files,
    write_champion_selection_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", action="append", required=True, help="Candidate scorecards.jsonl path.")
    parser.add_argument("--output", required=True, help="Champion report JSON path.")
    parser.add_argument("--reproduction-command", required=True)
    parser.add_argument("--min-completed-epochs", type=int, default=12)
    args = parser.parse_args()

    report = build_champion_selection_report_from_scorecard_files(
        [Path(path) for path in args.scorecard],
        reproduction_command=args.reproduction_command,
        min_completed_epochs=args.min_completed_epochs,
    )
    output = Path(args.output)
    write_champion_selection_report(output, report)
    summary = {
        "champion_report": str(output),
        "selected": report.selected.candidate_id if report.selected is not None else None,
        "runner_up": report.runner_up.candidate_id if report.runner_up is not None else None,
        "ranked_count": len(report.ranked),
        "rejected_count": len(report.rejected),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
