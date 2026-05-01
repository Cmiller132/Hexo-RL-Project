"""Reusable arena/eval players backed by V2 policy providers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from hexorl.config import Config
from hexorl.contracts.history import encode_move_history
from hexorl.engine.rust import engine_available
from hexorl.eval.position_services import build_search_context
from hexorl.inference.local import LocalEvaluator
from hexorl.inference.protocol import protocol_manifest_from_contract
from hexorl.models.factory import inference_contract
from hexorl.models.specs import ModelSpec, model_spec_from_config
from hexorl.search.context import SearchContext
from hexorl.search.policy_provider import PolicyProvider, create_policy_provider
from hexorl.search.priors import SearchEvaluation


PlayerFn = Callable[[list[tuple[int, int, int]], int, int], tuple[int | None, int | None]]
HAS_ENGINE = engine_available()


@dataclass
class NoisyPolicyConfig:
    temperature: float = 0.35
    top_p: float = 0.98
    near_radius: int = 8
    constrain_threats: bool = True
    seed: int = 0


@dataclass(frozen=True)
class EvalPolicyTrace:
    trace_id: str
    model_family: str
    provider_type: str
    input_contract: str
    output_contract: str
    legal_row_count: int
    pair_rows_scored: int
    inference_protocol: str
    warnings: tuple[str, ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "model_family": self.model_family,
            "provider_type": self.provider_type,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "legal_row_count": self.legal_row_count,
            "pair_rows_scored": self.pair_rows_scored,
            "inference_protocol": self.inference_protocol,
            "warnings": list(self.warnings),
            "timings_ms": dict(self.timings_ms),
        }


class PolicyPlayer:
    """Arena callback that samples from provider-scored legal rows."""

    def __init__(
        self,
        provider: PolicyProvider,
        *,
        model_spec: ModelSpec,
        config: NoisyPolicyConfig | None = None,
        candidate_budget: int = 256,
        recipe_id: str = "eval-default",
    ):
        self.provider = provider
        self.model_spec = model_spec
        self.config = config or NoisyPolicyConfig()
        self.candidate_budget = int(candidate_budget)
        self.recipe_id = recipe_id
        self.rng = np.random.default_rng(self.config.seed)
        self.telemetry: list[EvalPolicyTrace] = []

    def __call__(
        self,
        move_history: list[tuple[int, int, int]],
        time_ms_override: int,
        player: int,
    ) -> tuple[int | None, int | None]:
        del time_ms_override, player
        context = self._context(move_history)
        if context.legal_table.rows.shape[0] == 0:
            return None, None
        started = time.monotonic()
        evaluation = self.provider.evaluate_root(context)
        elapsed = (time.monotonic() - started) * 1000.0
        self.telemetry.append(_trace_from_evaluation(context, evaluation, elapsed_ms=elapsed))
        row = _sample_row(
            evaluation.row_priors,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            rng=self.rng,
        )
        q, r = context.legal_table.rows[int(row)]
        return int(q), int(r)

    def _context(self, moves: list[tuple[int, int, int]]) -> SearchContext:
        history = encode_move_history(moves)
        return build_search_context(
            history,
            model_spec=self.model_spec,
            recipe_id=self.recipe_id,
            candidate_budget=self.candidate_budget,
            near_radius=self.config.near_radius,
            constrain_threats=self.config.constrain_threats,
            inference_protocol="local_model_eval_v1",
        )


class NoisyModelPlayer(PolicyPlayer):
    """Compatibility constructor that still evaluates through PolicyProvider."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | None = None,
        config: NoisyPolicyConfig | None = None,
        cfg: Config | None = None,
        model_spec: ModelSpec | None = None,
    ):
        cfg = cfg or Config()
        spec = model_spec or model_spec_from_config(cfg)
        manifest = protocol_manifest_from_contract(inference_contract(cfg), timeout_ms=float(getattr(cfg.inference, "timeout_ms", 1000.0)))
        client = LocalEvaluator(model, manifest=manifest, device=device)
        super().__init__(
            create_policy_provider(model_spec=spec, client=client),
            model_spec=spec,
            config=config or NoisyPolicyConfig(),
            candidate_budget=int(getattr(cfg.model, "candidate_budget", 256)),
            recipe_id=f"eval:{spec.kind}",
        )


def policy_player_from_model(
    model: torch.nn.Module,
    *,
    model_spec: ModelSpec | None = None,
    cfg: Config | None = None,
    device: torch.device | None = None,
    temperature: float = 0.35,
    top_p: float = 0.98,
    near_radius: int = 8,
    constrain_threats: bool = True,
    seed: int = 0,
) -> PolicyPlayer:
    cfg = cfg or Config()
    spec = model_spec or model_spec_from_config(cfg)
    return NoisyModelPlayer(
        model,
        device=device,
        cfg=cfg,
        model_spec=spec,
        config=NoisyPolicyConfig(
            temperature=temperature,
            top_p=top_p,
            near_radius=near_radius,
            constrain_threats=constrain_threats,
            seed=seed,
        ),
    )


def noisy_model_player(
    model: torch.nn.Module,
    *,
    device: torch.device | None = None,
    temperature: float = 0.35,
    top_p: float = 0.98,
    near_radius: int = 8,
    constrain_threats: bool = True,
    seed: int = 0,
    cfg: Config | None = None,
    model_spec: ModelSpec | None = None,
) -> PlayerFn:
    return policy_player_from_model(
        model,
        device=device,
        temperature=temperature,
        top_p=top_p,
        near_radius=near_radius,
        constrain_threats=constrain_threats,
        seed=seed,
        cfg=cfg,
        model_spec=model_spec,
    )


def greedy_model_player(
    model: torch.nn.Module,
    *,
    device: torch.device | None = None,
    cfg: Config | None = None,
    model_spec: ModelSpec | None = None,
) -> PlayerFn:
    return noisy_model_player(
        model,
        device=device,
        temperature=1e-4,
        top_p=1.0,
        cfg=cfg,
        model_spec=model_spec,
    )


def _sample_row(
    priors: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    probs = np.asarray(priors, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        raise ValueError("cannot sample from an empty provider evaluation")
    if temperature <= 1e-4:
        return int(np.argmax(probs))
    scaled = np.power(np.maximum(probs, 1e-12), 1.0 / max(float(temperature), 1e-4))
    scaled /= max(float(scaled.sum()), 1e-12)
    keep = _top_p_indices(scaled, top_p)
    kept = scaled[keep]
    kept /= max(float(kept.sum()), 1e-12)
    return int(rng.choice(keep, p=kept))


def _top_p_indices(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return np.arange(len(probs))
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, max(float(top_p), 1e-6), side="left") + 1
    return order[: max(1, cutoff)]


def _trace_from_evaluation(
    context: SearchContext,
    evaluation: SearchEvaluation,
    *,
    elapsed_ms: float,
) -> EvalPolicyTrace:
    input_contract = "global_graph_v1" if context.graph_batch is not None else "crop_tensor_v1"
    output_contract = (
        "global_place_value_v1"
        if evaluation.policy_provider == "GlobalGraphPolicyProvider"
        else "row_mapped_policy_value_v1"
    )
    return EvalPolicyTrace(
        trace_id=context.trace_id,
        model_family=evaluation.model_family,
        provider_type=evaluation.policy_provider,
        input_contract=input_contract,
        output_contract=output_contract,
        legal_row_count=int(context.legal_table.rows.shape[0]),
        pair_rows_scored=0,
        inference_protocol=evaluation.inference_protocol,
        warnings=tuple(evaluation.warnings),
        timings_ms={"provider_call_ms": float(elapsed_ms), **dict(evaluation.timings)},
    )

