"""Mirror per-family Phase 3 Optuna studies into one dashboard SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import optuna


def _storage_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _source_dbs(run_dir: Path, output: Path) -> list[Path]:
    studies_dir = run_dir / "phase3_studies"
    output_resolved = output.resolve()
    return [
        path
        for path in sorted(studies_dir.glob("*.sqlite3"))
        if path.resolve() != output_resolved and path.name != output.name
    ]


def mirror_once(run_dir: Path, output: Path, summary: Path | None = None) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    dest_storage = _storage_url(output)

    for existing in optuna.study.get_all_study_summaries(storage=dest_storage):
        optuna.delete_study(study_name=existing.study_name, storage=dest_storage)

    mirrored: list[dict[str, object]] = []
    for source in _source_dbs(run_dir, output):
        source_storage = _storage_url(source)
        for study in optuna.study.get_all_study_summaries(storage=source_storage):
            optuna.copy_study(
                from_study_name=study.study_name,
                from_storage=source_storage,
                to_storage=dest_storage,
                to_study_name=study.study_name,
            )
            mirrored.append(
                {
                    "source": str(source),
                    "study_name": study.study_name,
                    "n_trials": study.n_trials,
                    "datetime_start": study.datetime_start.isoformat()
                    if study.datetime_start
                    else None,
                }
            )
    _annotate_dashboard_examples(run_dir, output)

    result = {
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "study_count": len(mirrored),
        "studies": mirrored,
    }
    if summary is not None:
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _annotate_dashboard_examples(run_dir: Path, output: Path) -> None:
    """Attach normal-dashboard example links to copied Optuna trials."""
    with sqlite3.connect(output) as conn:
        rows = conn.execute(
            """
            SELECT trial_id, value_json
            FROM trial_user_attributes
            WHERE key='hexo_phase3_candidate_id'
            """
        ).fetchall()
        for trial_id, value_json in rows:
            try:
                candidate_id = str(json.loads(value_json))
            except Exception:
                candidate_id = str(value_json).strip('"')
            trial_dir = run_dir / "phase3_trials" / candidate_id
            _upsert_trial_attr(conn, int(trial_id), "hexo_dashboard_examples_url", "http://192.168.68.62:8766/")
            _upsert_trial_attr(
                conn,
                int(trial_id),
                "hexo_dashboard_examples_hint",
                "Open the normal dashboard Examples tab for self-play replays and fixed-classical game examples.",
            )
            _upsert_trial_attr(
                conn,
                int(trial_id),
                "hexo_replayable_classical_examples",
                _count_replayable_classical_examples(trial_dir),
            )


def _upsert_trial_attr(conn: sqlite3.Connection, trial_id: int, key: str, value: object) -> None:
    conn.execute(
        "DELETE FROM trial_user_attributes WHERE trial_id=? AND key=?",
        (trial_id, key),
    )
    conn.execute(
        "INSERT INTO trial_user_attributes(trial_id, key, value_json) VALUES (?, ?, ?)",
        (trial_id, key, json.dumps(value)),
    )


def _count_replayable_classical_examples(trial_dir: Path) -> int:
    total = 0
    for path in trial_dir.glob("fixed_classical_epoch_*_games.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if '"final_history_b64"' in line:
                total += 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        result = mirror_once(args.run_dir, args.output, args.summary)
        print(
            f"{result['mirrored_at']} mirrored {result['study_count']} studies to {result['output']}",
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
