"""Runtime-failure debug bundle writers for autotuning candidates."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_MISSING_PAYLOAD = {
    "available": False,
    "reason": "not captured for this runtime failure",
}


def write_runtime_failure_debug_bundle(
    root: Path | str,
    *,
    candidate_id: str,
    reason: str,
    repro_command: str | Iterable[str] | None,
    runtime_telemetry: Mapping[str, Any] | None = None,
    runtime_probe_results: Iterable[Mapping[str, Any]] | None = None,
    dashboard_links: Mapping[str, Any] | None = None,
    legal_rows: Mapping[str, Any] | None = None,
    pair_rows: Mapping[str, Any] | None = None,
    replay: Mapping[str, Any] | None = None,
    model_output_summary: Mapping[str, Any] | None = None,
    failing_history: Mapping[str, Any] | None = None,
    bundle_name: str | None = None,
) -> Path:
    """Write the standard runtime debug bundle and return its directory.

    The bundle is intentionally diagnosis-oriented: even when semantic payloads
    are unavailable during a compute-safety failure, the files exist with a
    structured unavailable marker so downstream tooling can depend on the
    contract.
    """

    bundle_dir = Path(root) / (bundle_name or _bundle_name(reason))
    bundle_dir.mkdir(parents=True, exist_ok=True)

    command_text = _repro_command_text(repro_command)
    (bundle_dir / "repro_command.txt").write_text(command_text, encoding="utf-8")
    _write_json(bundle_dir / "runtime_telemetry.json", runtime_telemetry or {})
    _write_jsonl(bundle_dir / "runtime_probe_results.jsonl", runtime_probe_results or [])
    _write_json(bundle_dir / "dashboard_links.json", dashboard_links or {})
    _write_json(bundle_dir / "legal_rows.json", legal_rows or _MISSING_PAYLOAD)
    _write_json(bundle_dir / "pair_rows.json", pair_rows or _MISSING_PAYLOAD)
    _write_json(bundle_dir / "replay.json", replay or _MISSING_PAYLOAD)
    _write_json(bundle_dir / "model_output_summary.json", model_output_summary or _MISSING_PAYLOAD)
    _write_json(bundle_dir / "failing_history.json", failing_history or _MISSING_PAYLOAD)
    _write_json(
        bundle_dir / "manifest.json",
        {
            "candidate_id": candidate_id,
            "reason": reason,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": sorted(path.name for path in bundle_dir.iterdir() if path.is_file()),
        },
    )
    return bundle_dir


def _bundle_name(reason: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", reason).strip("_") or "runtime_failure"
    return f"{timestamp}_{safe_reason[:80]}"


def _repro_command_text(repro_command: str | Iterable[str] | None) -> str:
    if repro_command is None:
        return "unavailable\n"
    if isinstance(repro_command, str):
        text = repro_command.strip()
    else:
        text = " ".join(str(part) for part in repro_command).strip()
    return (text or "unavailable") + "\n"


def _jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return _jsonable(asdict(payload))
    if isinstance(payload, Mapping):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    if isinstance(payload, Path):
        return str(payload)
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
