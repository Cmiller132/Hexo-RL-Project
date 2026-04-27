"""Run the ablation suite with one fresh Python process per variant."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_ablation_suite import SUITES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--output-root", default="runs/ablations_priority_20260427_v2")
    parser.add_argument("--suite", choices=sorted(SUITES), default="priority")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--bootstrap-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-time-ms", type=int, default=25)
    parser.add_argument("--eval-depth", type=int, default=2)
    parser.add_argument("--md", default="Docs/ABLATION_RESULTS_20260427.md")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = repo / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "isolated_supervisor.log"

    for name in SUITES[args.suite]:
        done = root / name / "DONE"
        if done.exists():
            _log(log_path, f"skip completed {name}")
            continue
        _log(log_path, f"start {name}")
        cmd = [
            sys.executable,
            "scripts/run_ablation_suite.py",
            "--config",
            args.config,
            "--output-root",
            args.output_root,
            "--suite",
            args.suite,
            "--only",
            name,
            "--epochs",
            str(args.epochs),
            "--bootstrap-games",
            str(args.bootstrap_games),
            "--seed",
            str(args.seed),
            "--eval-games",
            str(args.eval_games),
            "--eval-time-ms",
            str(args.eval_time_ms),
            "--eval-depth",
            str(args.eval_depth),
        ]
        with (root / f"{name}.isolated.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        _refresh_report(repo, args.output_root, args.md)
        if result.returncode != 0:
            _log(log_path, f"failed {name} rc={result.returncode}")
            raise SystemExit(result.returncode)
        _log(log_path, f"done {name}")

    _refresh_report(repo, args.output_root, args.md)
    _log(log_path, "suite complete")


def _refresh_report(repo: Path, suite_root: str, md: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_ablation_suite.py",
            suite_root,
            f"--md={md}",
        ],
        cwd=repo,
        check=False,
    )


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


if __name__ == "__main__":
    main()
