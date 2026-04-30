"""Discriminated model specs and legacy config-name normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


MODEL_SPEC_VERSION = 1

REQUIRED_MODEL_KINDS = (
    "dense_cnn",
    "restnet",
    "graph_hybrid",
    "global_xattn",
    "global_line_window",
    "global_relation_graph",
)

MODEL_KIND_ALIASES: dict[str, str] = {
    "cnn": "dense_cnn",
    "restnet": "restnet",
    "graph": "graph_hybrid",
    "graph_hybrid_0": "graph_hybrid",
    "global_xattn_0": "global_xattn",
    "global_line_window_0": "global_line_window",
    "global_graph_option1": "global_relation_graph",
    "global_pair_twostage_0": "global_relation_graph",
    "global_graph_full_0": "global_relation_graph",
    "global_hybrid_action_0": "global_relation_graph",
    "global_graph768_champion": "global_relation_graph",
}

GLOBAL_MODEL_KINDS = frozenset({"global_xattn", "global_line_window", "global_relation_graph"})
CROP_MODEL_KINDS = frozenset({"dense_cnn", "restnet", "graph_hybrid"})


@dataclass(frozen=True)
class ModelSpec:
    """Runtime model identity selected by kind, with config values as parameters."""

    kind: str
    version: int = MODEL_SPEC_VERSION
    params: Mapping[str, Any] = field(default_factory=dict)
    source_name: str | None = None

    @property
    def is_global_graph(self) -> bool:
        return self.kind in GLOBAL_MODEL_KINDS

    @property
    def is_crop(self) -> bool:
        return self.kind in CROP_MODEL_KINDS

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "params": dict(self.params),
            "source_name": self.source_name,
        }


def normalize_model_kind(name: str) -> str:
    key = str(name).lower()
    try:
        return MODEL_KIND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown model spec kind or migration alias {name!r}; "
            f"expected one of {sorted(set(REQUIRED_MODEL_KINDS) | set(MODEL_KIND_ALIASES))}"
        ) from exc


def model_spec_from_config(cfg: Any) -> ModelSpec:
    model_cfg = cfg.model
    source_name = str(getattr(model_cfg, "architecture", "cnn")).lower()
    kind = normalize_model_kind(source_name)
    params = {
        "channels": int(getattr(model_cfg, "channels", 128)),
        "blocks": int(getattr(model_cfg, "blocks", 16)),
        "heads": list(getattr(model_cfg, "heads", ["policy", "value"])),
        "attention_positions": list(getattr(model_cfg, "attention_positions", [])),
        "attention_heads": int(getattr(model_cfg, "attention_heads", 8)),
        "attention_mlp_ratio": float(getattr(model_cfg, "attention_mlp_ratio", 2.0)),
        "attention_dropout": float(getattr(model_cfg, "attention_dropout", 0.0)),
        "dropout": float(getattr(model_cfg, "dropout", 0.0)),
        "relative_bias": bool(getattr(model_cfg, "relative_bias", False)),
        "graph_token_set": str(getattr(model_cfg, "graph_token_set", "graph512_turn_pair_prior")),
        "graph_token_budget": int(getattr(model_cfg, "graph_token_budget", 512)),
        "graph_layers": int(getattr(model_cfg, "graph_layers", 3)),
        "sparse_policy": bool(getattr(model_cfg, "sparse_policy", False)),
        "candidate_budget": int(getattr(model_cfg, "candidate_budget", 256)),
        "n_bins": 65,
    }
    return ModelSpec(kind=kind, params=params, source_name=source_name)
