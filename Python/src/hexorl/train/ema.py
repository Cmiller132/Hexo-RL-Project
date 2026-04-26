"""Exponential Moving Average (EMA) of model weights.

KataGo-style: maintains a shadow copy updated via polyak averaging.
The inference server uses EMA weights rather than raw training weights
for more stable self-play evaluation.
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, Optional


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Shadow parameters are updated as:
        shadow = (1 - decay) * shadow + decay * model

    where decay is a fixed value like 0.9999 (or 0.99 for faster tracking),
    with optional adaptive warmup: decay = min(fixed, 1 - 1/(1 + num_updates)).

    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for batch in dataloader:
            loss = ...; loss.backward(); optimizer.step()
            ema.update()  # Update shadow after each step
        # Swap to EMA for inference
        ema.apply_shadow()
        server.update_model(model)
        ema.restore()
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.model = model
        self._device = device or next(model.parameters()).device
        self._shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        self._num_updates = 0

        # Initialize shadow with a copy of model parameters
        self._init_shadow()

    def _init_shadow(self):
        """Copy all model parameters and persistent buffers into shadow storage."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()
        # Also track BatchNorm running stats and other persistent buffers.
        for name, buf in self.model.named_buffers():
            if buf is not None and torch.is_floating_point(buf):
                self._shadow[f"__buf__{name}"] = buf.data.clone().detach()

    def update(self):
        """Update shadow parameters using polyak averaging."""
        self._num_updates += 1

        # Adaptive warmup: ramp up from near-0 to self.decay over first ~1000 steps.
        # Once num_updates is large enough, 1 - 1/(1+n) ≈ self.decay and clamps there.
        d = min(self.decay, 1.0 - 1.0 / (1.0 + self._num_updates))

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._shadow[name].mul_(1.0 - d).add_(param.data, alpha=d)
            # Update buffer shadows (BatchNorm stats etc.) with same decay.
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if buf is not None and key in self._shadow:
                    if torch.is_floating_point(buf):
                        self._shadow[key].mul_(1.0 - d).add_(buf.data, alpha=d)
                    else:
                        self._shadow[key].copy_(buf.data)

    def update_step(self):
        """Alias for update() — called after each optimizer step."""
        self.update()

    def apply_shadow(self):
        """Swap model weights with EMA shadow (for inference).

        Saves current model weights to backup, then loads shadow weights.
        Call restore() to revert.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._backup[name] = param.data.clone()
                    param.data.copy_(self._shadow[name])
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if buf is not None and key in self._shadow:
                    self._backup[key] = buf.data.clone()
                    buf.data.copy_(self._shadow[key])

    def restore(self):
        """Restore model weights from backup (after apply_shadow())."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup.pop(name))
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if key in self._backup:
                    buf.data.copy_(self._backup.pop(key))

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "shadow": {k: v.cpu() for k, v in self._shadow.items()},
            "decay": self.decay,
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state: dict):
        """Restore EMA state from checkpoint."""
        self.decay = state.get("decay", self.decay)
        self._num_updates = state.get("num_updates", 0)
        for name, tensor in state["shadow"].items():
            if name in self._shadow:
                self._shadow[name].copy_(tensor.to(self._shadow[name].device))

    def to(self, device: torch.device):
        """Move shadow parameters to a different device."""
        for name in self._shadow:
            self._shadow[name] = self._shadow[name].to(device)
        return self

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def effective_decay(self) -> float:
        if self._num_updates <= 0:
            return 0.0
        return min(self.decay, 1.0 - 1.0 / (1.0 + self._num_updates))
