"""Canonical config hashing for autotune traceability."""

from __future__ import annotations

import hashlib
import json

from hexorl.config import Config

_HASH_IGNORED_RUNTIME_KEYS = {
    "min_free_system_memory_gb",
    "v1_metadata_compression",
    "v1_selfplay_worker_probe_max",
}


def config_hash(config: Config) -> str:
    """Return SHA256 over canonical sorted JSON from ``Config.model_dump``."""

    payload = config.model_dump(mode="json")
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        for key in _HASH_IGNORED_RUNTIME_KEYS:
            runtime.pop(key, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
