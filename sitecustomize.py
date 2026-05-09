"""Local Python startup guards for Hexo development commands."""

from __future__ import annotations

import os
import platform


def _seed_windows_platform_cache() -> None:
    if os.name != "nt" or getattr(platform, "_uname_cache", None) is not None:
        return
    uname_result = getattr(platform, "uname_result", None)
    if uname_result is None:
        return
    try:
        platform._uname_cache = uname_result(  # type: ignore[attr-defined]
            "Windows",
            os.environ.get("COMPUTERNAME", ""),
            "",
            "",
            os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
        )
    except Exception:
        return


_seed_windows_platform_cache()
