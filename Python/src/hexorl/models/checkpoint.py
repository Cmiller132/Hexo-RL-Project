"""Strict checkpoint save/load/inspect ownership."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
import zipfile

import torch
import torch.nn as nn

from hexorl.models.factory import REGISTRY, inference_manifest
from hexorl.models.specs import model_spec_from_config


LoadPurpose = Literal["train", "inference", "eval"]
_INSPECTION_MANIFEST_NAME = "checkpoint_manifest.json"


@dataclass
class CheckpointManifest:
    checkpoint_schema_version: int
    model_family: str
    model_spec_version: int
    model_spec: dict[str, Any]
    input_contract: str
    output_contract: str
    action_contract: str
    inference_protocol: dict[str, Any]
    heads: list[str]
    pair_strategy_used: str
    created_by: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        required = set(cls.__dataclass_fields__)
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required)
        if missing:
            raise ValueError(f"checkpoint manifest missing required fields: {missing}")
        if unknown:
            raise ValueError(f"checkpoint manifest contains unknown fields: {unknown}")
        manifest = cls(**data)
        if manifest.checkpoint_schema_version != 1:
            raise ValueError(f"stale checkpoint schema version {manifest.checkpoint_schema_version}")
        if manifest.model_spec_version != 1:
            raise ValueError(f"stale model spec version {manifest.model_spec_version}")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": self.checkpoint_schema_version,
            "model_family": self.model_family,
            "model_spec_version": self.model_spec_version,
            "model_spec": self.model_spec,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "action_contract": self.action_contract,
            "inference_protocol": self.inference_protocol,
            "heads": list(self.heads),
            "pair_strategy_used": self.pair_strategy_used,
            "created_by": dict(self.created_by),
        }


@dataclass
class CheckpointBundle:
    cfg: Any
    model: nn.Module
    optimizer_state_dict: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    ema_state_dict: dict[str, Any] | None = None
    scaler_state_dict: dict[str, Any] | None = None
    epoch: int = 0
    global_step: int = 0
    created_by: dict[str, Any] | None = None


@dataclass
class LoadedCheckpoint:
    manifest: CheckpointManifest
    payload: dict[str, Any]


class CheckpointManager:
    def save(self, bundle: CheckpointBundle, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _manifest_from_cfg(bundle.cfg, bundle.created_by or {})
        payload = {
            "checkpoint_manifest": manifest.to_dict(),
            "model_state_dict": bundle.model.state_dict(),
            "optimizer_state_dict": bundle.optimizer_state_dict,
            "scheduler_state_dict": bundle.scheduler_state_dict,
            "ema_state_dict": bundle.ema_state_dict,
            "scaler_state_dict": bundle.scaler_state_dict,
            "epoch": int(bundle.epoch),
            "global_step": int(bundle.global_step),
            "cfg": bundle.cfg,
            "cfg_json": bundle.cfg.model_dump(mode="json") if hasattr(bundle.cfg, "model_dump") else None,
        }
        torch.save(payload, path)
        _write_inspection_manifest(path, manifest)

    def inspect(self, path: Path) -> CheckpointManifest:
        raw = _read_inspection_manifest(Path(path))
        return CheckpointManifest.from_dict(raw)

    def load(self, path: Path, *, purpose: LoadPurpose, device: str | torch.device) -> LoadedCheckpoint:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        raw = checkpoint.get("checkpoint_manifest")
        if raw is None:
            raise ValueError("checkpoint is missing strict checkpoint_manifest")
        manifest = CheckpointManifest.from_dict(raw)
        expected_family = REGISTRY.resolve(str(manifest.model_spec.get("kind"))).name
        if manifest.model_family != expected_family:
            raise ValueError(
                f"checkpoint model family mismatch: manifest={manifest.model_family} "
                f"model_spec={expected_family}"
            )
        if purpose in {"inference", "eval"} and manifest.inference_protocol.get("protocol_version") != 1:
            raise ValueError("checkpoint inference protocol mismatch")
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("checkpoint missing model_state_dict")
        compiled_prefix = "_orig" "_mod."
        bad_keys = [key for key in state if str(key).startswith(compiled_prefix) or str(key).startswith("module.")]
        if bad_keys:
            raise ValueError("checkpoint requires offline conversion; prefixed model keys are forbidden")
        return LoadedCheckpoint(manifest=manifest, payload=checkpoint)

    def load_state_into_model(self, model: nn.Module, state_dict: dict[str, Any]) -> None:
        compiled_prefix = "_orig" "_mod."
        bad_keys = [key for key in state_dict if str(key).startswith(compiled_prefix) or str(key).startswith("module.")]
        if bad_keys:
            raise ValueError("checkpoint state contains runtime-forbidden prefixed keys")
        result = model.load_state_dict(state_dict, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "strict model state load failed: "
                f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
        if hasattr(model, "apply_hex_masks_"):
            model.apply_hex_masks_()


def _manifest_from_cfg(cfg: Any, created_by: dict[str, Any]) -> CheckpointManifest:
    spec = model_spec_from_config(cfg)
    descriptor = REGISTRY.resolve(spec)
    infer = inference_manifest(cfg).canonical_dict()
    return CheckpointManifest(
        checkpoint_schema_version=1,
        model_family=descriptor.name,
        model_spec_version=1,
        model_spec=spec.manifest(),
        input_contract=infer["input_contract"],
        output_contract="model_inference_contract_v3",
        action_contract=infer["action_contract"],
        inference_protocol={"protocol_version": 1, "model_inference_contract": infer},
        heads=list(descriptor.components.heads),
        pair_strategy_used=str(getattr(cfg.model, "pair_strategy", "none")),
        created_by={
            "git_sha": str(created_by.get("git_sha", "unknown")),
            "command": str(created_by.get("command", "unknown")),
            "config_hash": str(created_by.get("config_hash", "unknown")),
        },
    )


def _write_inspection_manifest(path: Path, manifest: CheckpointManifest) -> None:
    member = _inspection_manifest_member(path)
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")))


def _read_inspection_manifest(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise ValueError("checkpoint is not a PyTorch zip archive with inspectable manifest metadata")
    with zipfile.ZipFile(path, mode="r") as archive:
        candidates = [name for name in archive.namelist() if name.endswith(f"/{_INSPECTION_MANIFEST_NAME}")]
        if not candidates:
            raise ValueError(
                "checkpoint is missing inspectable checkpoint_manifest metadata; "
                "resave or convert it with the strict CheckpointManager format"
            )
        if len(candidates) != 1:
            raise ValueError(f"checkpoint contains ambiguous inspectable manifests: {sorted(candidates)}")
        raw = json.loads(archive.read(candidates[0]).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("checkpoint inspectable manifest must be a JSON object")
    return raw


def _inspection_manifest_member(path: Path) -> str:
    with zipfile.ZipFile(path, mode="r") as archive:
        prefixes = {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and name.endswith("data.pkl")
        }
    if len(prefixes) != 1:
        raise ValueError(f"checkpoint archive has ambiguous PyTorch prefixes: {sorted(prefixes)}")
    return f"{next(iter(prefixes))}/{_INSPECTION_MANIFEST_NAME}"
