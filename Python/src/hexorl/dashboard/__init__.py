"""Dashboard subsystem."""

from .db import DashboardStore
from .pseudocode import DASHBOARD_PSEUDOCODE

__all__ = ["DASHBOARD_PSEUDOCODE", "DashboardStore", "RunRecorder"]


def __getattr__(name: str):
    if name == "RunRecorder":
        from .recorder import RunRecorder

        return RunRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
