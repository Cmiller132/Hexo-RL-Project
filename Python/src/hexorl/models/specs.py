"""Model identity specs and typed per-family parameter schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

MODEL_SPEC_VERSION = 1
REQUIRED_MODEL_KINDS = ("dense_cnn", "restnet", "graph_hybrid", "global_xattn", "global_line_window", "global_relation_graph")
GLOBAL_MODEL_KINDS = frozenset({"global_xattn", "global_line_window", "global_relation_graph"})
CROP_MODEL_KINDS = frozenset({"dense_cnn", "restnet", "graph_hybrid"})


class ModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    channels: int = 128
    heads: tuple[str, ...] = ("policy", "value")
    dropout: float = 0.0
    n_bins: int = 65


class DenseCnnParams(ModelParams):
    blocks: int = 16


class RestNetParams(DenseCnnParams):
    attention_positions: tuple[int, ...] = ()
    attention_heads: int = 8
    attention_mlp_ratio: float = 2.0
    attention_dropout: float = 0.0
    relative_bias: bool = False


class GraphHybridParams(RestNetParams):
    graph_token_set: str = "graph512_turn_pair_prior"
    graph_token_budget: int = 512
    graph_layers: int = 3
    sparse_policy: bool = False
    candidate_budget: int = 256


class GlobalGraphParams(ModelParams):
    graph_token_budget: int = 512
    graph_layers: int = 3
    attention_heads: int = 8


class ModelSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    kind: str
    version: int = MODEL_SPEC_VERSION
    params: Any = ModelParams()
    source_name: str | None = None

    @property
    def is_global_graph(self) -> bool:
        return self.kind in GLOBAL_MODEL_KINDS

    @property
    def is_crop(self) -> bool:
        return self.kind in CROP_MODEL_KINDS

    def manifest(self) -> dict[str, Any]:
        params = self.params.model_dump(mode="json") if hasattr(self.params, "model_dump") else dict(self.params)
        return {"kind": self.kind, "version": self.version, "params": params, "source_name": self.source_name}


def model_spec_from_config(cfg: Any, *, registry: Any | None = None) -> ModelSpec:
    if registry is None:
        from hexorl.models.factory import REGISTRY

        registry = REGISTRY
    source_name = str(getattr(cfg.model, "architecture", "cnn")).lower()
    descriptor = registry.resolve(source_name)
    params = _params_from_config(cfg, descriptor.params_schema, descriptor.components.heads)
    return ModelSpec(kind=descriptor.name, params=params, source_name=source_name)


def _params_from_config(cfg: Any, params_schema: type[ModelParams], heads: tuple[str, ...]) -> ModelParams:
    model_cfg = cfg.model
    provided = set(getattr(model_cfg, "model_fields_set", set()))
    allowed = set(params_schema.model_fields)
    defaults = type(model_cfg)()
    nonparam_runtime_fields = {
        "attention_heads",
        "graph_layers",
        "pair_prior_mix",
        "pair_strategy",
        "pair_strategy_max_pairs",
        "sparse_prior_mix",
        "sparse_prior_stage",
    }
    disallowed_explicit = sorted(
        name
        for name in (provided - allowed) - {"architecture", "blocks"} - nonparam_runtime_fields
        if getattr(model_cfg, name, None) != getattr(defaults, name, None)
    )
    if disallowed_explicit:
        raise ValueError(f"{params_schema.__name__} does not accept model fields: {disallowed_explicit}")
    payload = {name: getattr(model_cfg, name) for name in allowed if hasattr(model_cfg, name) and name not in {"heads", "n_bins"}}
    for name, value in list(payload.items()):
        if isinstance(value, list):
            payload[name] = tuple(value)
    payload["heads"] = tuple(heads)
    payload["n_bins"] = 65
    return params_schema.model_validate(payload)


__all__ = [
    "CROP_MODEL_KINDS", "DenseCnnParams", "GLOBAL_MODEL_KINDS", "GlobalGraphParams",
    "GraphHybridParams", "MODEL_SPEC_VERSION", "ModelParams", "ModelSpec", "REQUIRED_MODEL_KINDS",
    "RestNetParams", "model_spec_from_config",
]
