"""Load configuration from TOML files."""

import sys
from pathlib import Path
from typing import Optional

from .schema import Config

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate a configuration file."""
    if path is None:
        repo_root = Path(__file__).resolve().parents[4]
        candidates = [
            repo_root / "Configs" / "default.toml",
            repo_root / "configs" / "default.toml",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    elif not isinstance(path, Path):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return Config.model_validate(raw)
