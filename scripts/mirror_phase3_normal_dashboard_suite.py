"""Expose candidate/trial directories in the normal dashboard suite layout."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def _copy_file_if_changed(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def mirror_once(run_dir: Path, suite_dir: Path, summary: Path | None = None) -> dict[str, object]:
    source_root = run_dir / "phase3_trials"
    layout = "phase3_trials_as_dashboard_suite"
    if not source_root.exists():
        source_root = run_dir / "candidates"
        layout = "phase1_candidates_as_dashboard_suite"
    trials_dir = suite_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    mirrored: list[dict[str, object]] = []
    for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
        target = trials_dir / source.name
        if not target.exists():
            target.mkdir(parents=True)
        for child in source.iterdir():
            if child.is_file():
                _copy_file_if_changed(child, target / child.name)
            elif child.name == "checkpoints" and child.is_dir():
                (target / child.name).mkdir(exist_ok=True)
                for checkpoint in child.glob("*"):
                    if checkpoint.is_file():
                        _copy_file_if_changed(checkpoint, target / child.name / checkpoint.name)
        dashboard = source / "dashboard.sqlite3"
        source_mtime = max(
            (child.stat().st_mtime for child in source.iterdir() if child.is_file()),
            default=source.stat().st_mtime,
        )
        mirrored.append(
            {
                "trial_id": source.name,
                "source": str(source),
                "dashboard_mtime": (
                    datetime.fromtimestamp(dashboard.stat().st_mtime, timezone.utc).isoformat()
                    if dashboard.exists()
                    else None
                ),
                "source_mtime": datetime.fromtimestamp(source_mtime, timezone.utc).isoformat(),
                "has_dashboard": dashboard.exists(),
            }
        )

    manifest = {
        "run_id": run_dir.name,
        "source_run_dir": str(run_dir),
        "layout": layout,
    }
    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if (run_dir / "champion_selection_report_phase3.json").exists():
        _copy_file_if_changed(
            run_dir / "champion_selection_report_phase3.json",
            suite_dir / "champion_selection_report_phase3.json",
        )

    result = {
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "suite_dir": str(suite_dir),
        "trial_count": len(mirrored),
        "trials": mirrored,
    }
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        result = mirror_once(args.run_dir, args.suite_dir, args.summary)
        print(
            f"{result['mirrored_at']} mirrored {result['trial_count']} trials to {result['suite_dir']}",
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
