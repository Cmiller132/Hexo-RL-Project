"""ASGI entrypoint for the current Optuna Phase 3 normal dashboard."""

from __future__ import annotations

import os
from pathlib import Path

from hexorl.dashboard.app import create_app


RUN_DIR = Path(os.environ.get("HEXO_DASHBOARD_RUN_DIR", "runs/optuna_sequential_scout_20260507_001212"))
SUITE_DIR = Path(os.environ.get("HEXO_DASHBOARD_SUITE_DIR", str(RUN_DIR / "phase3_normal_dashboard_suite")))

app = create_app(
    db_path=RUN_DIR / "phase3_normal_dashboard.sqlite3",
    run_root=SUITE_DIR,
)
