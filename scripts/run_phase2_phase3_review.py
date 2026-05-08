#!/usr/bin/env python
"""Generate Phase 2 promotion reports and Phase 3 Optuna study metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hexorl.tuning.optuna_tuning import (
    create_phase3_study,
    phase3_study_specs_from_phase2_report,
)
from hexorl.tuning.review import (
    build_phase2_promotion_report_from_scorecard_files,
    write_phase2_promotion_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", action="append", required=True, help="Candidate scorecards.jsonl path.")
    parser.add_argument("--output-dir", required=True, help="Directory for review artifacts.")
    parser.add_argument("--min-epoch-floor", type=int, default=12)
    parser.add_argument("--phase3-storage-template", default="")
    parser.add_argument("--phase3-max-promoted", type=int, default=None)
    parser.add_argument("--phase3-seed", type=int, default=None)
    parser.add_argument("--phase3-tpe-startup-trials", type=int, default=8)
    parser.add_argument("--create-phase3-studies", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_phase2_promotion_report_from_scorecard_files(
        [Path(path) for path in args.scorecard],
        min_epoch_floor=args.min_epoch_floor,
    )
    phase2_path = output_dir / "phase2_promotion_report.json"
    write_phase2_promotion_report(phase2_path, report)

    specs_payload = []
    phase3_specs_path = None
    if args.phase3_storage_template:
        specs = phase3_study_specs_from_phase2_report(
            report,
            storage_template=args.phase3_storage_template,
            seed=args.phase3_seed,
            n_startup_trials=args.phase3_tpe_startup_trials,
            max_promoted=args.phase3_max_promoted,
        )
        specs_payload = [asdict(spec) for spec in specs]
        phase3_specs_path = output_dir / "phase3_study_specs.json"
        phase3_specs_path.write_text(
            json.dumps(specs_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.create_phase3_studies:
            for spec in specs:
                create_phase3_study(
                    architecture_id=spec.architecture_id,
                    pair_mode=spec.pair_mode,
                    storage=spec.storage,
                    seed=args.phase3_seed,
                    n_startup_trials=args.phase3_tpe_startup_trials,
                    load_if_exists=True,
                )

    manifest = {
        "phase2_promotion_report": str(phase2_path),
        "phase3_study_specs": str(phase3_specs_path) if phase3_specs_path is not None else None,
        "ranked_count": len(report.ranked),
        "excluded_count": len(report.excluded),
        "phase3_specs_count": len(specs_payload),
        "create_phase3_studies": bool(args.create_phase3_studies),
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
