"""Reusable arena/eval players backed by V2 policy providers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from hexorl.action_contract.tactical_oracle import scan_tactical_oracle_from_history
from hexorl.config import Config
from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.history import encode_move_history
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.engine.encoding import encode_board_and_legal
from hexorl.engine.rust import engine_available
from hexorl.graph.tensorize import build_graph_batch_from_history
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


class LocalModelInferenceClient:
    """Small in-process client used by eval to feed the public PolicyProvider."""

    def __init__(self, model: torch.nn.Module, *, device: torch.device | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.manifest = _EvalManifest("local_model_eval_v1")
        self.model.eval()

    def evaluate_dense(self, tensor: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(tensor[:count], dtype=np.float32)).to(
            device=self.device,
            dtype=model_input_dtype(self.model),
        )
        with torch.no_grad():
            out = self.model(x)
        policy = out["policy"].detach().float().cpu().numpy()
        value_head = out.get("value")
        if value_head is None:
            value = np.zeros((count,), dtype=np.float32)
        else:
            value = _value_from_output(self.model, value_head).detach().float().cpu().numpy()
        return policy, value

    def evaluate_sparse(
        self,
        tensor: np.ndarray,
        count: int,
        candidate_indices: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.from_numpy(np.asarray(tensor[:count], dtype=np.float32)).to(
            device=self.device,
            dtype=model_input_dtype(self.model),
        )
        indices = torch.from_numpy(np.asarray(candidate_indices[:count], dtype=np.int64)).to(self.device)
        features = torch.from_numpy(np.asarray(candidate_features[:count], dtype=np.float32)).to(self.device)
        mask = torch.from_numpy(np.asarray(candidate_mask[:count], dtype=np.bool_)).to(self.device)
        with torch.no_grad():
            out = self.model(
                x,
                candidate_indices=indices,
                candidate_features=features,
                candidate_mask=mask,
            )
        dense = out["policy"].detach().float().cpu().numpy()
        if "sparse_policy" in out:
            sparse = out["sparse_policy"].detach().float().cpu().numpy()
        else:
            rows = np.asarray(candidate_indices[:count], dtype=np.int64)
            sparse = np.take_along_axis(dense, rows.clip(0, dense.shape[1] - 1), axis=1)
            sparse = np.where(np.asarray(candidate_mask[:count], dtype=np.bool_), sparse, -80.0)
        value_head = out.get("value")
        if value_head is None:
            value = np.zeros((count,), dtype=np.float32)
        else:
            value = _value_from_output(self.model, value_head).detach().float().cpu().numpy()
        return dense, value, sparse

    def evaluate_global_graph(self, graph_batch) -> dict[str, np.ndarray | dict[str, object]]:
        kwargs = {
            "token_features": _batched_tensor(graph_batch.token_features, self.device, torch.float32),
            "token_type": _batched_tensor(graph_batch.token_type, self.device, torch.long),
            "token_qr": _batched_tensor(graph_batch.token_qr, self.device, torch.float32),
            "token_mask": _batched_tensor(graph_batch.token_mask, self.device, torch.bool),
            "legal_token_indices": _batched_tensor(graph_batch.legal_token_indices, self.device, torch.long),
            "legal_mask": _batched_tensor(graph_batch.legal_mask, self.device, torch.bool),
            "opp_legal_qr": _batched_tensor(graph_batch.opp_legal_qr, self.device, torch.float32),
            "opp_legal_mask": _batched_tensor(graph_batch.opp_legal_mask, self.device, torch.bool),
            "pair_first_indices": _batched_tensor(graph_batch.pair_first_indices, self.device, torch.long),
            "pair_second_indices": _batched_tensor(graph_batch.pair_second_indices, self.device, torch.long),
            "pair_token_indices": _batched_tensor(graph_batch.pair_token_indices, self.device, torch.long),
            "relation_type": _batched_tensor(graph_batch.relation_type, self.device, torch.long),
            "relation_bias": torch.from_numpy(np.asarray(graph_batch.relation_bias, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device),
        }
        with torch.no_grad():
            out = self.model(**kwargs)
        result: dict[str, np.ndarray | dict[str, object]] = {
            "metadata": {"legal_qr": np.asarray(graph_batch.legal_qr, dtype=np.int32)}
        }
        for key, value in out.items():
            if key == "value":
                result[key] = _value_from_output(self.model, value).detach().float().cpu().numpy()
            else:
                result[key] = value.detach().float().cpu().numpy().reshape(-1)
        return result


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
        tensor, offset_q, offset_r, legal_rows, _legal_bytes = encode_board_and_legal(
            history,
            self.config.near_radius,
            self.config.constrain_threats,
        )
        legal_table = LegalActionTable.from_rows(
            [(int(q), int(r)) for q, r in legal_rows.tolist()],
            source="rust:legal",
            history_hash=stable_digest(("eval-history", history)),
            current_player=len(moves) % 2,
            placements_remaining=1,
        )
        candidate_table = None
        graph_batch = None
        if self.model_spec.is_global_graph:
            graph_batch = build_graph_batch_from_history(
                history,
                radius=self.config.near_radius,
                max_pair_rows=0,
                include_pair_rows=False,
            )
        elif self.model_spec.kind == "graph_hybrid":
            oracle = scan_tactical_oracle_from_history(
                history,
                [(int(q), int(r)) for q, r in legal_rows.tolist()],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
            )
            candidate_table = CandidateContractBuilder().build(
                [(int(q), int(r)) for q, r in legal_rows.tolist()],
                [],
                offset_q=int(offset_q),
                offset_r=int(offset_r),
                budget=self.candidate_budget,
                storage_width=self.candidate_budget,
                winning_moves=oracle.win_now_cells,
                forced_block_moves=oracle.forced_block_cells,
                cover_cells=oracle.cover_cells,
                open_four_cells=oracle.open_four_cells,
                open_five_cells=oracle.open_five_cells,
            )
        return SearchContext.create(
            phase="root",
            legal_table=legal_table,
            model_family=self.model_spec.kind,
            model_spec_version=str(self.model_spec.version),
            recipe_id=self.recipe_id,
            search_id="arena",
            pair_strategy_id="none",
            tensor=tensor.reshape(1, 13, 33, 33).astype(np.float32, copy=False),
            history_bytes=history,
            candidate_table=candidate_table,
            graph_batch=graph_batch,
            inference_protocol="local_model_eval_v1",
            extra={"offset_q": int(offset_q), "offset_r": int(offset_r)},
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
        client = LocalModelInferenceClient(model, device=device)
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


def model_input_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return torch.float32
    return dtype if dtype.is_floating_point else torch.float32


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


def _value_from_output(model: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
    if value.ndim > 1 and value.shape[-1] > 1 and hasattr(model, "bins_to_value"):
        return model.bins_to_value(value)
    return value.reshape(value.shape[0], -1)[:, 0]


def _batched_tensor(value: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).unsqueeze(0).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class _EvalManifest:
    transport: str
