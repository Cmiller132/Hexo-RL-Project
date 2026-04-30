"""Small in-process model cache for dashboard inference/debug routes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hexorl.config import Config
from hexorl.contracts.candidates import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    CandidateContractBuilder,
)
from hexorl.contracts.pairs import PairActionTableBuilder, PairStrategy
from hexorl.eval.arena import load_checkpoint_model
from hexorl.eval.players import model_input_dtype
from hexorl.dashboard.replay import encode_tensor_for_history, policy_debug
from hexorl.engine.legal import decode_legal_bytes
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
        tensor, offset_q, offset_r, legal_bytes = encode_tensor_for_history(history)
        arr = decode_legal_bytes(legal_bytes)
        legal_mask = []
        outside_legal = []
        for q, r in arr:
            gi = int(q) - int(offset_q)
            gj = int(r) - int(offset_r)
            if 0 <= gi < 33 and 0 <= gj < 33:
                legal_mask.append(gi * 33 + gj)
            else:
                outside_legal.append({"q": int(q), "r": int(r)})
        x = (
            torch.from_numpy(tensor)
            .unsqueeze(0)
            .to(device=cached.device, dtype=model_input_dtype(cached.model))
        )
        model_sparse = bool(getattr(cached.model, "sparse_policy_enabled", False))
        model_pair = getattr(cached.model, "pair_policy_head", None) is not None
        candidate_payload: dict[str, Any] | None = None
        forward_kwargs: dict[str, torch.Tensor] = {}
        if (model_sparse or model_pair) and len(arr) > 0:
            cand = CandidateContractBuilder().build(
                [(int(q), int(r)) for q, r in arr],
                [],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
                budget=min(max(len(arr), 1), 512),
            )
            candidate_payload = {
                "qr": cand.qr,
                "mask": cand.mask,
            }
            forward_kwargs = {
                "candidate_indices": torch.from_numpy(cand.indices.reshape(1, -1).copy()).to(cached.device),
                "candidate_features": torch.from_numpy(cand.features.reshape(1, cand.features.shape[0], cand.features.shape[1]).copy()).to(cached.device),
                "candidate_mask": torch.from_numpy(cand.mask.reshape(1, -1).copy()).to(cached.device),
            }
            if model_pair:
                pair_budget = min(max((int(cand.mask.sum()) * max(int(cand.mask.sum()) - 1, 0)) // 2, 1), 512)
                pair = PairActionTableBuilder().build(
                    cand,
                    [],
                    strategy=PairStrategy(mode="capped_fill", max_pairs=pair_budget),
                    legal_moves=[(int(q), int(r)) for q, r in arr],
                )
                candidate_payload["pair_indices"] = pair.pair_indices
                candidate_payload["pair_mask"] = pair.mask
                forward_kwargs["pair_candidate_indices"] = torch.from_numpy(
                    pair.pair_indices.reshape(1, pair.pair_indices.shape[0], 2).copy()
                ).to(cached.device)
                forward_kwargs["pair_candidate_mask"] = torch.from_numpy(
                    pair.mask.reshape(1, -1).copy()
                ).to(cached.device)
        with torch.no_grad():
            out = cached.model(x, **forward_kwargs) if forward_kwargs else cached.model(x)
        result: dict[str, Any] = {
            "model_id": model_id,
            "legal_moves": [{"q": int(q), "r": int(r)} for q, r in arr],
            "outside_window_legal_count": len(outside_legal),
            "outside_window_legal_moves": outside_legal[:32],
            "candidate_contract": {
                "feature_version": CANDIDATE_FEATURE_VERSION,
                "feature_names": list(CANDIDATE_FEATURE_NAMES),
            },
            "heads": {},
        }
        if "policy" in out:
            logits = out["policy"][0].detach().cpu().numpy()
            result["heads"]["policy"] = policy_debug(logits, legal_mask)
        if "sparse_policy" in out and candidate_payload is not None:
            sparse_logits = out["sparse_policy"][0].detach().float().cpu().numpy()
            mask = candidate_payload["mask"]
            qr = candidate_payload["qr"]
            valid_rows = np.where(mask)[0]
            if valid_rows.size:
                top_rows = valid_rows[np.argsort(-sparse_logits[valid_rows])[:16]]
                result["heads"]["sparse_policy"] = [
                    {
                        "q": int(qr[row, 0]),
                        "r": int(qr[row, 1]),
                        "logit": float(sparse_logits[row]),
                    }
                    for row in top_rows
                ]
        if "pair_policy" in out and candidate_payload is not None:
            pair_logits = out["pair_policy"][0].detach().float().cpu().numpy()
            pair_mask = candidate_payload.get("pair_mask")
            pair_indices = candidate_payload.get("pair_indices")
            qr = candidate_payload["qr"]
            if pair_mask is not None and pair_indices is not None:
                valid_rows = np.where(pair_mask)[0]
                if valid_rows.size:
                    top_rows = valid_rows[np.argsort(-pair_logits[valid_rows])[:16]]
                    result["heads"]["pair_policy"] = [
                        {
                            "first": {
                                "q": int(qr[int(pair_indices[row, 0]), 0]),
                                "r": int(qr[int(pair_indices[row, 0]), 1]),
                            },
                            "second": {
                                "q": int(qr[int(pair_indices[row, 1]), 0]),
                                "r": int(qr[int(pair_indices[row, 1]), 1]),
                            },
                            "logit": float(pair_logits[row]),
                        }
                        for row in top_rows
                    ]
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
