"""Reusable arena/eval players.

The default model player samples from a legal-masked policy distribution.  This
keeps eval games varied without reintroducing Gumbel, variance selectors, or
training gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from hexorl.model.network import HexNet
from hexorl.selfplay.records import BOARD_SIZE

try:
    import _engine

    HAS_ENGINE = True
except ImportError:  # pragma: no cover - depends on local extension build
    _engine = None
    HAS_ENGINE = False


PlayerFn = Callable[[list[tuple[int, int, int]], int, int], tuple[int | None, int | None]]


@dataclass
class NoisyPolicyConfig:
    temperature: float = 0.35
    top_p: float = 0.98
    near_radius: int = 8
    constrain_threats: bool = True
    seed: int = 0


class NoisyModelPlayer:
    """Direct-policy model player with legal masking, temperature, and top-p."""

    def __init__(
        self,
        model: HexNet,
        *,
        device: Optional[torch.device] = None,
        config: NoisyPolicyConfig | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.config = config or NoisyPolicyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model.eval()

    def __call__(
        self,
        move_history: list[tuple[int, int, int]],
        time_ms_override: int,
        player: int,
    ) -> tuple[int | None, int | None]:
        if not HAS_ENGINE:
            return self._fallback_move()

        game = _new_game()
        for _p, q, r in move_history:
            game.place(int(q), int(r))
        tensor_3d, offset_q, offset_r, legal_bytes = game.encode_board_and_legal(
            self.config.near_radius,
            self.config.constrain_threats,
        )
        legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
        if len(legal) == 0:
            return None, None
        tensor = (
            torch.from_numpy(np.array(tensor_3d, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor)["policy"][0].detach().cpu().numpy()
        return self._sample_legal(logits, legal, int(offset_q), int(offset_r))

    def _sample_legal(
        self,
        logits: np.ndarray,
        legal: np.ndarray,
        offset_q: int,
        offset_r: int,
    ) -> tuple[int, int]:
        action_indices = []
        legal_logits = []
        legal_coords = []
        for q, r in legal:
            idx = _flat_index(int(q), int(r), offset_q, offset_r)
            if 0 <= idx < logits.size:
                action_indices.append(idx)
                legal_logits.append(float(logits[idx]))
                legal_coords.append((int(q), int(r)))
        if not legal_coords:
            q, r = legal[0]
            return int(q), int(r)

        temperature = max(float(self.config.temperature), 1e-4)
        scores = np.array(legal_logits, dtype=np.float64) / temperature
        scores -= np.max(scores)
        probs = np.exp(scores)
        probs /= max(float(probs.sum()), 1e-12)

        keep = _top_p_indices(probs, self.config.top_p)
        kept_probs = probs[keep]
        kept_probs /= max(float(kept_probs.sum()), 1e-12)
        chosen = int(self.rng.choice(keep, p=kept_probs))
        return legal_coords[chosen]

    def _fallback_move(self) -> tuple[int, int]:
        with torch.no_grad():
            tensor = torch.zeros(1, 13, BOARD_SIZE, BOARD_SIZE, device=self.device)
            logits = self.model(tensor)["policy"][0].detach().cpu().numpy()
        idx = int(np.argmax(logits + self.rng.normal(0.0, 1e-3, size=logits.shape)))
        return idx // BOARD_SIZE - 16, idx % BOARD_SIZE - 16


def noisy_model_player(
    model: HexNet,
    *,
    device: Optional[torch.device] = None,
    temperature: float = 0.35,
    top_p: float = 0.98,
    near_radius: int = 8,
    constrain_threats: bool = True,
    seed: int = 0,
) -> PlayerFn:
    return NoisyModelPlayer(
        model,
        device=device,
        config=NoisyPolicyConfig(
            temperature=temperature,
            top_p=top_p,
            near_radius=near_radius,
            constrain_threats=constrain_threats,
            seed=seed,
        ),
    )


def greedy_model_player(model: HexNet, *, device: Optional[torch.device] = None) -> PlayerFn:
    """Compatibility helper.  Eval defaults should use noisy_model_player."""
    return noisy_model_player(model, device=device, temperature=1e-4, top_p=1.0)


def _top_p_indices(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return np.arange(len(probs))
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, max(float(top_p), 1e-6), side="left") + 1
    return order[: max(1, cutoff)]


def _flat_index(q: int, r: int, offset_q: int, offset_r: int) -> int:
    gi = q - offset_q
    gj = r - offset_r
    if 0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE:
        return int(gi * BOARD_SIZE + gj)
    return -1


def _new_game():
    cls = getattr(_engine, "HexGame", None) or getattr(_engine, "PyHexGame")
    return cls()
