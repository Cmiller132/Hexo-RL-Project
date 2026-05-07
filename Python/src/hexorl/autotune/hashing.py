"""Canonical config hashing for autotune traceability."""

from __future__ import annotations

import hashlib
import json

from hexorl.config import Config


def config_hash(config: Config) -> str:
    """Return SHA256 over canonical sorted JSON from ``Config.model_dump``."""

    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
