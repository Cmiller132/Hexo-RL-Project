"""Server-side model/device runtime setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from hexorl.config import Config
from hexorl.models.checkpoint import CheckpointManager
from hexorl.models.factory import build_inference_model
from hexorl.runtime import configure_torch_runtime


@dataclass
class ServerRuntime:
    model: torch.nn.Module
    device: torch.device
    forward_stream: object | None


def state_to_cpu(state_dict: Optional[dict]) -> Optional[dict]:
    if state_dict is None:
        return None
    return {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_runtime(cfg: Config, initial_state_dict: Optional[dict]) -> ServerRuntime:
    configure_torch_runtime(cfg)
    device = select_device()
    model = build_inference_model(cfg, device=device)
    if device.type == "cuda" and getattr(cfg.runtime, "channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    compile_inference = getattr(cfg.runtime, "compile_inference", None)
    if compile_inference is None:
        compile_inference = getattr(cfg.runtime, "compile_model", False)
    if device.type == "cuda" and compile_inference:
        try:
            model = torch.compile(
                model,
                mode=getattr(cfg.runtime, "compile_mode", "reduce-overhead"),
            )
        except Exception as exc:
            print(f"[inference-server] torch.compile disabled: {exc}", flush=True)
    if initial_state_dict is not None:
        load_state_into_model(model, initial_state_dict, device=device)
    forward_stream = torch.cuda.Stream(priority=-1) if device.type == "cuda" else None
    return ServerRuntime(model=model, device=device, forward_stream=forward_stream)


def load_state_into_model(model: torch.nn.Module, state_dict: dict, *, device: torch.device) -> None:
    latest = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in state_dict.items()
    }
    CheckpointManager().load_state_into_model(model, latest)
    model.eval()


def apply_latest_weight_update(model: torch.nn.Module, weight_queue, *, device: torch.device) -> None:
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except Exception:
            break
    if latest is not None:
        load_state_into_model(model, latest, device=device)


__all__ = [
    "ServerRuntime",
    "apply_latest_weight_update",
    "initialize_runtime",
    "select_device",
    "state_to_cpu",
]
