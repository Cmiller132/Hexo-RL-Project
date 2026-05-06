"""Evaluation model provider boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from hexorl.config import Config
from hexorl.models.loading import build_runtime_model, restore_model_weights


def load_eval_model(
    checkpoint_path: str | Path,
    fallback_cfg: Config,
    *,
    device: Optional[torch.device] = None,
    allow_partial: bool = False,
) -> torch.nn.Module:
    """Load an evaluation model without reaching into family classes directly."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("cfg")
    if not isinstance(ckpt_cfg, Config) and checkpoint.get("cfg_json") is not None:
        ckpt_cfg = Config.model_validate(checkpoint["cfg_json"])
    model_cfg = ckpt_cfg if isinstance(ckpt_cfg, Config) else fallback_cfg
    model = build_runtime_model(model_cfg, device=device, inference=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    restore_model_weights(model, state, allow_partial=allow_partial)
    model.eval()
    return model
