"""Periodically refresh the Markdown ablation report while a suite runs."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root")
    parser.add_argument("--md", default="Docs/ABLATION_RESULTS_20260427.md")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--interval-s", type=int, default=300)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    while True:
        _refresh(repo, args.suite_root, args.md)
        if args.pid and not _pid_alive(args.pid):
            _refresh(repo, args.suite_root, args.md)
            break
        time.sleep(max(30, args.interval_s))


def _refresh(repo: Path, suite_root: str, md_path: str) -> None:
    subprocess.run(
        [
            "python",
            str(repo / "scripts" / "summarize_ablation_suite.py"),
            suite_root,
            f"--md={md_path}",
        ],
        cwd=repo,
        check=False,
    )


def _pid_alive(pid: int) -> bool:
    try:
        subprocess.run(["kill", "-0", str(pid)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    main()
