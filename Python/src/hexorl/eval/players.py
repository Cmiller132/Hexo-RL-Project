"""Reusable arena/eval players.

The default model player samples from a legal-masked policy distribution.  This
keeps eval games varied without reintroducing Gumbel, variance selectors, or
training gates.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from hexorl.graph.batch import build_graph_batch_from_history, collate_graph_batches
from hexorl.models.assembly import is_global_graph_model as _model_registry_is_global_graph_model
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
        model: torch.nn.Module,
        *,
        device: Optional[torch.device] = None,
        config: NoisyPolicyConfig | None = None,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.dtype = model_input_dtype(model)
        self.config = config or NoisyPolicyConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._global_graph_model = _is_global_graph_model(model)
        self._unwrapped_model = _unwrap_model(model)
        self.model.eval()

    def __call__(
        self,
        move_history: list[tuple[int, int, int]],
        time_ms_override: int,
        player: int,
    ) -> tuple[int | None, int | None]:
        if self._global_graph_model:
            return self._graph_move(move_history)
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
            .to(device=self.device, dtype=self.dtype)
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
            tensor = torch.zeros(
                1, 13, BOARD_SIZE, BOARD_SIZE, device=self.device, dtype=self.dtype
            )
            logits = self.model(tensor)["policy"][0].detach().cpu().numpy()
        idx = int(np.argmax(logits + self.rng.normal(0.0, 1e-3, size=logits.shape)))
        return idx // BOARD_SIZE - 16, idx % BOARD_SIZE - 16

    def _graph_move(
        self,
        move_history: list[tuple[int, int, int]],
    ) -> tuple[int | None, int | None]:
        history = _encode_move_history(move_history)
        legal_override = _rust_legal_rows_for_history(
            move_history,
            near_radius=8,
            constrain_threats=self.config.constrain_threats,
        )
        if legal_override is not None and len(legal_override) == 0:
            return None, None
        graph = build_graph_batch_from_history(
            history,
            legal_moves=legal_override,
            constrain_threats=self.config.constrain_threats,
            include_pair_rows=False,
            max_context_tokens=_positive_int_attr(
                self._unwrapped_model,
                "graph_context_tokens",
                "graph_token_budget",
                default=256,
            ),
            max_legal_rows=_positive_int_attr(
                self._unwrapped_model,
                "graph_legal_rows",
                "candidate_budget",
                default=256,
            ),
        )
        graph = collate_graph_batches([graph])
        legal = np.asarray(graph.legal_qr[0], dtype=np.int32)
        legal_mask = np.asarray(graph.legal_mask[0], dtype=np.bool_)
        if legal.shape[0] == 0 or not np.any(legal_mask):
            return None, None
        inputs = _graph_batch_to_tensors(graph, self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits_tensor = outputs.get("policy_place")
        if logits_tensor is None:
            raise ValueError("global graph arena eval requires policy_place output")
        logits = logits_tensor[0].detach().float().cpu().numpy()
        return self._sample_graph_legal(logits, legal, legal_mask)

    def _sample_graph_legal(
        self,
        logits: np.ndarray,
        legal: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[int, int]:
        valid = np.flatnonzero(legal_mask[: logits.shape[0]])
        if valid.size == 0:
            q, r = legal[0]
            return int(q), int(r)
        temperature = float(self.config.temperature)
        if temperature <= 1e-4:
            chosen = int(valid[int(np.argmax(logits[valid]))])
            q, r = legal[chosen]
            return int(q), int(r)
        scores = logits[valid].astype(np.float64) / max(temperature, 1e-4)
        scores -= np.max(scores)
        probs = np.exp(scores)
        probs /= max(float(probs.sum()), 1e-12)
        keep = _top_p_indices(probs, self.config.top_p)
        kept_probs = probs[keep]
        kept_probs /= max(float(kept_probs.sum()), 1e-12)
        chosen = int(valid[int(self.rng.choice(keep, p=kept_probs))])
        q, r = legal[chosen]
        return int(q), int(r)


def noisy_model_player(
    model: torch.nn.Module,
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


def greedy_model_player(model: torch.nn.Module, *, device: Optional[torch.device] = None) -> PlayerFn:
    """Compatibility helper.  Eval defaults should use noisy_model_player."""
    return noisy_model_player(model, device=device, temperature=1e-4, top_p=1.0)


def model_input_dtype(model: torch.nn.Module) -> torch.dtype:
    """Return the floating dtype expected by a model's parameters."""
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return torch.float32
    return dtype if dtype.is_floating_point else torch.float32


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _is_global_graph_model(model: torch.nn.Module) -> bool:
    return _model_registry_is_global_graph_model(_unwrap_model(model))


def _positive_int_attr(
    model: torch.nn.Module,
    *names: str,
    default: int,
) -> int | None:
    for name in names:
        value = getattr(model, name, None)
        if value is None:
            continue
        value = int(value)
        if value > 0:
            return value
    return int(default) if int(default) > 0 else None


def _encode_move_history(move_history: list[tuple[int, int, int]]) -> bytes:
    data = bytearray()
    for player, q, r in move_history:
        data.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(data)


def _rust_legal_rows_for_history(
    move_history: list[tuple[int, int, int]],
    *,
    near_radius: int,
    constrain_threats: bool,
) -> list[tuple[int, int]] | None:
    if not HAS_ENGINE:
        return None
    game = _new_game()
    for _player, q, r in move_history:
        game.place(int(q), int(r))
    _tensor, _offset_q, _offset_r, legal_bytes = game.encode_board_and_legal(
        int(near_radius),
        bool(constrain_threats),
    )
    legal = np.frombuffer(legal_bytes, dtype=np.int32).reshape(-1, 2)
    return [(int(q), int(r)) for q, r in legal.tolist()]


def _graph_batch_to_tensors(
    graph,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "token_features": torch.as_tensor(graph.token_features, device=device, dtype=dtype),
        "token_type": torch.as_tensor(graph.token_type, device=device),
        "token_qr": torch.as_tensor(graph.token_qr, device=device),
        "token_mask": torch.as_tensor(graph.token_mask, device=device),
        "legal_token_indices": torch.as_tensor(graph.legal_token_indices, device=device),
        "legal_mask": torch.as_tensor(graph.legal_mask, device=device),
        "opp_legal_qr": torch.as_tensor(graph.opp_legal_qr, device=device),
        "opp_legal_mask": torch.as_tensor(graph.opp_legal_mask, device=device),
        "pair_first_indices": torch.as_tensor(graph.pair_first_indices, device=device),
        "pair_second_indices": torch.as_tensor(graph.pair_second_indices, device=device),
        "pair_token_indices": torch.as_tensor(graph.pair_token_indices, device=device),
        "relation_type": torch.as_tensor(graph.relation_type, device=device),
        "relation_bias": torch.as_tensor(graph.relation_bias, device=device, dtype=dtype),
    }


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
