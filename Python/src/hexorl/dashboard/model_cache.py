"""Small in-process model cache for dashboard inference/debug routes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hexorl.config import Config
from hexorl.dashboard.model_inference import DashboardModelInferenceService
from hexorl.models.checkpoint import CheckpointManager
from hexorl.models.factory import build_model
from hexorl.models.specs import ModelSpec, model_spec_from_config


@dataclass
class CachedModel:
    model_id: str
    path: Path
    model: torch.nn.Module
    device: torch.device
    cfg: Config
    model_spec: ModelSpec
    inference_service: DashboardModelInferenceService


class ModelCache:
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self._models: dict[str, CachedModel] = {}
        self._order: list[str] = []

    def load(self, path: Path | str, cfg: Config | None = None) -> CachedModel:
        path = Path(path)
        cfg = cfg or Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manager = CheckpointManager()
        loaded = manager.load(path, purpose="eval", device=device)
        checkpoint = loaded.payload
        ckpt_cfg = checkpoint.get("cfg")
        if not isinstance(ckpt_cfg, Config) and checkpoint.get("cfg_json") is not None:
            ckpt_cfg = Config.model_validate(checkpoint["cfg_json"])
        model_cfg = ckpt_cfg if isinstance(ckpt_cfg, Config) else cfg
        model = build_model(model_cfg, device=device, inference=True)
        manager.load_state_into_model(model, checkpoint["model_state_dict"])
        model.eval()
        model_spec = model_spec_from_config(model_cfg)
        model_id = uuid.uuid4().hex[:12]
        cached = CachedModel(
            model_id,
            path,
            model,
            device,
            model_cfg,
            model_spec,
            DashboardModelInferenceService(
                model=model,
                device=device,
                cfg=model_cfg,
                model_spec=model_spec,
            ),
        )
        self._models[model_id] = cached
        self._order.append(model_id)
        while len(self._order) > self.max_models:
            old_id = self._order.pop(0)
            self._models.pop(old_id, None)
        return cached

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": cached.model_id,
                "path": str(cached.path),
                "device": str(cached.device),
                "model_family": cached.model_spec.kind,
            }
            for cached in self._models.values()
        ]

    def unload(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self._order = [mid for mid in self._order if mid != model_id]

    def infer_history(self, model_id: str, history: bytes) -> dict[str, Any]:
        cached = self._models[model_id]
        return cached.inference_service.infer_history(model_id, history)
