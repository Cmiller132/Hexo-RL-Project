"""Small in-process model cache for dashboard inference/debug routes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import Config
from hexorl.eval.arena import load_checkpoint_model
from hexorl.eval.players import model_input_dtype
from hexorl.dashboard.replay import encode_tensor_for_history, policy_debug
from hexorl.model.network import HexNet


@dataclass
class CachedModel:
    model_id: str
    path: Path
    model: HexNet
    device: torch.device


class ModelCache:
    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self._models: dict[str, CachedModel] = {}
        self._order: list[str] = []

    def load(self, path: Path | str, cfg: Config | None = None) -> CachedModel:
        path = Path(path)
        cfg = cfg or Config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_checkpoint_model(path, cfg, device=device)
        model_id = uuid.uuid4().hex[:12]
        cached = CachedModel(model_id, path, model, device)
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
            }
            for cached in self._models.values()
        ]

    def unload(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self._order = [mid for mid in self._order if mid != model_id]

    def infer_history(self, model_id: str, history: bytes) -> dict[str, Any]:
        cached = self._models[model_id]
        tensor, _oq, _or, legal_bytes = encode_tensor_for_history(history)
        arr = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        legal_mask = []
        # encode_tensor_for_history uses the Rust offsets internally but policy
        # debugging only needs legal action indices from channel 3.
        legal_channel = tensor[3].reshape(-1)
        legal_mask = [int(i) for i in np.flatnonzero(legal_channel > 0.0)]
        x = (
            torch.from_numpy(tensor)
            .unsqueeze(0)
            .to(device=cached.device, dtype=model_input_dtype(cached.model))
        )
        with torch.no_grad():
            out = cached.model(x)
        result: dict[str, Any] = {
            "model_id": model_id,
            "legal_moves": [{"q": int(q), "r": int(r)} for q, r in arr],
            "heads": {},
        }
        if "policy" in out:
            logits = out["policy"][0].detach().cpu().numpy()
            result["heads"]["policy"] = policy_debug(logits, legal_mask)
        if "value" in out:
            value_logits = out["value"].detach().cpu()
            result["heads"]["value"] = float(cached.model.bins_to_value(value_logits)[0])
        if "axis" in out:
            probs = torch.softmax(out["axis"][0], dim=-1).detach().cpu().numpy()
            result["heads"]["axis"] = [float(v) for v in probs]
        for name, tensor_out in out.items():
            if name.startswith("lookahead_"):
                result["heads"][name] = float(cached.model.bins_to_value(tensor_out.detach().cpu())[0])
        return result
