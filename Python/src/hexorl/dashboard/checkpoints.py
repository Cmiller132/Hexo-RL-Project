"""Checkpoint import and indexing helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hexorl.dashboard.db import DashboardStore


@dataclass(frozen=True)
class CheckpointIndexResult:
    path: Path
    sha256: str
    checkpoint_id: int
    run_id: str
    epoch: int | None
    global_step: int | None
    is_loadable: bool
    model_heads: list[str]
    error: str | None = None


def scan_checkpoints(
    root: Path | str,
    store: DashboardStore,
    *,
    run_id: str | None = None,
) -> list[CheckpointIndexResult]:
    """Index all ``.pt``/``.pth`` checkpoints under ``root``."""
    root = Path(root)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in {".pt", ".pth"}
        )
    return [index_checkpoint(path, store, run_id=run_id) for path in paths]


def index_checkpoint(
    path: Path | str,
    store: DashboardStore,
    *,
    run_id: str | None = None,
) -> CheckpointIndexResult:
    path = Path(path)
    sha = _sha256_file(path)
    metadata = _read_checkpoint_metadata(path)
    inferred_run_id = run_id or _infer_run_id(path)
    checkpoint_id = store.upsert_checkpoint(
        path=path,
        sha256=sha,
        run_id=inferred_run_id,
        epoch=metadata.get("epoch"),
        global_step=metadata.get("global_step"),
        is_loadable=metadata["is_loadable"],
        model_heads=metadata.get("model_heads", []),
        payload=metadata,
    )
    return CheckpointIndexResult(
        path=path,
        sha256=sha,
        checkpoint_id=checkpoint_id,
        run_id=inferred_run_id,
        epoch=metadata.get("epoch"),
        global_step=metadata.get("global_step"),
        is_loadable=metadata["is_loadable"],
        model_heads=metadata.get("model_heads", []),
        error=metadata.get("error"),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format": "unknown",
        "epoch": None,
        "global_step": None,
        "model_heads": [],
        "is_loadable": False,
    }
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        metadata["error"] = str(exc)
        return metadata

    if isinstance(checkpoint, dict):
        metadata["format"] = "hexorl" if "model_state_dict" in checkpoint else "state_dict"
        metadata["epoch"] = _as_int_or_none(checkpoint.get("epoch"))
        metadata["global_step"] = _as_int_or_none(checkpoint.get("global_step"))
        state = checkpoint.get("model_state_dict", checkpoint)
        cfg = checkpoint.get("cfg")
        heads = getattr(getattr(cfg, "model", None), "heads", None)
        if heads is None and isinstance(cfg, dict):
            model_cfg = cfg.get("model", {})
            if isinstance(model_cfg, dict):
                heads = model_cfg.get("heads")
        if heads is None:
            heads = _heads_from_state_dict(state)
        metadata["model_heads"] = list(heads or [])
        if checkpoint.get("action_contract_metadata") is not None:
            metadata["action_contract_metadata"] = checkpoint["action_contract_metadata"]
        if checkpoint.get("model_metadata") is not None:
            model_metadata = checkpoint["model_metadata"]
            if isinstance(model_metadata, dict) and "candidate_feature_version" in model_metadata:
                metadata["candidate_feature_version"] = model_metadata["candidate_feature_version"]
        metadata["state_keys"] = len(state) if isinstance(state, dict) else 0
        metadata["is_loadable"] = isinstance(state, dict) and (
            "model_state_dict" in checkpoint or any(k.startswith("conv_in.") for k in state)
        )
    else:
        metadata["error"] = f"Unsupported checkpoint object: {type(checkpoint).__name__}"

    return metadata


def _heads_from_state_dict(state: Any) -> list[str]:
    if not isinstance(state, dict):
        return []
    heads = set()
    for key in state:
        if key.startswith("heads."):
            parts = key.split(".")
            if len(parts) > 1:
                heads.add(parts[1])
    return sorted(heads)


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _infer_run_id(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        if part not in {"checkpoints", "models", "runs"}:
            return part
    return "unassigned"
