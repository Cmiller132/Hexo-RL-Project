"""Registered model architecture authority."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from hexorl.models.specs import (
    AliasDecision,
    ArchitectureSpec,
    ResolvedArchitectureSpec,
    dense_spec,
    global_graph_spec,
    resolve_spec,
)


_SPECS: dict[str, ArchitectureSpec] = {
    "cnn": dense_spec(
        "cnn",
        family_id="cnn",
        recipe_id="dense_cnn",
        description="KataGo-style dense crop CNN.",
    ),
    "restnet": dense_spec(
        "restnet",
        family_id="restnet",
        recipe_id="restnet",
        description="Paper-style dense ResTNet trunk with fixed RRRTRRTRRT sequence.",
        sparse_policy_capable=False,
        pair_capabilities=(),
        crop_pair_capable=False,
        default_outputs=("policy", "value", "opp_policy"),
        requires_attention_head_divisibility=True,
        supports_attention_positions=True,
    ),
    "restnet_crop_scout": dense_spec(
        "restnet_crop_scout",
        family_id="restnet_crop_scout",
        recipe_id="restnet_crop_scout",
        description="Legacy dense crop CNN with configured spatial transformer blocks.",
        requires_attention_head_divisibility=True,
        supports_attention_positions=True,
    ),
    "graph_hybrid_0": dense_spec(
        "graph_hybrid_0",
        family_id="crop_sparse_graph_hybrid",
        recipe_id="graph_hybrid_0",
        description="Crop-compatible sparse graph hybrid scout.",
        replay_sparse_diagnostics=True,
        requires_attention_head_divisibility=True,
    ),
    "global_graph_option1": global_graph_spec(
        "global_graph_option1",
        family_id="relation_graph",
        recipe_id="global_graph",
        description="Relation-biased global graph over legal action rows.",
        relation_required=True,
    ),
    "global_xattn_0": global_graph_spec(
        "global_xattn_0",
        family_id="context_cross_attention",
        recipe_id="global_graph",
        description="Global graph with legal-to-context cross attention.",
        relation_required=False,
    ),
    "global_line_window_0": global_graph_spec(
        "global_line_window_0",
        family_id="line_window_cover",
        recipe_id="global_graph",
        description="Global graph with line/window tactical gating.",
        relation_required=True,
    ),
    "global_pair_twostage_0": global_graph_spec(
        "global_pair_twostage_0",
        family_id="pair_two_stage",
        recipe_id="global_graph",
        description="Global graph with pair-specific refinement heads.",
        relation_required=False,
    ),
    "global_graph_full_0": global_graph_spec(
        "global_graph_full_0",
        family_id="full_relation_graph",
        recipe_id="global_graph",
        description="Full relation global graph variant.",
        relation_required=True,
    ),
    "global_hybrid_action_0": global_graph_spec(
        "global_hybrid_action_0",
        family_id="crop_diagnostic_global_action",
        recipe_id="global_graph",
        description="Global graph with optional crop action context.",
        relation_required=False,
    ),
    "global_graph768_champion": global_graph_spec(
        "global_graph768_champion",
        family_id="scaled_relation_graph",
        recipe_id="global_graph",
        description="Scaled global graph champion recipe.",
        relation_required=True,
    ),
    "global_graph768_devwin_0": global_graph_spec(
        "global_graph768_devwin_0",
        family_id="scaled_development_window_graph",
        recipe_id="global_graph",
        description="Scaled global graph with development-window Window6 encoding.",
        relation_required=True,
    ),
}

_ALIASES: dict[str, AliasDecision] = {
    "graph": AliasDecision(
        alias="graph",
        target=None,
        decision="deleted deprecated crop-compatible alias; use 'graph_hybrid_0'",
        runtime_supported=False,
    )
}


def architecture_ids() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))


def global_graph_architecture_ids() -> tuple[str, ...]:
    return tuple(sorted(arch for arch, spec in _SPECS.items() if spec.global_graph))


def relation_required_architecture_ids() -> tuple[str, ...]:
    return tuple(sorted(arch for arch, spec in _SPECS.items() if spec.relation_required))


def deprecated_aliases() -> Mapping[str, AliasDecision]:
    return dict(_ALIASES)


def normalize_architecture_id(architecture: object) -> str:
    arch = str(architecture).lower()
    if arch in _SPECS:
        return arch
    if arch in _ALIASES:
        decision = _ALIASES[arch]
        raise ValueError(
            f"architecture alias {arch!r} is not a runtime architecture; "
            f"decision: {decision.decision}"
        )
    raise ValueError(f"unknown model architecture {architecture!r}")


def architecture_spec(architecture: object) -> ArchitectureSpec:
    return _SPECS[normalize_architecture_id(architecture)]


def is_global_graph_architecture(architecture: object) -> bool:
    try:
        return architecture_spec(architecture).global_graph
    except ValueError:
        return False


def is_graph_architecture(architecture: object) -> bool:
    try:
        return architecture_spec(architecture).graph
    except ValueError:
        return False


def global_graph_family(architecture: object) -> str:
    spec = architecture_spec(architecture)
    if not spec.global_graph:
        raise ValueError(f"{architecture!r} is not a global graph architecture")
    return spec.family_id


def replay_uses_sparse_diagnostics(
    heads: Iterable[str],
    *,
    architecture: object = "cnn",
    sparse_policy: bool = False,
    graph: bool = False,
) -> bool:
    head_set = {str(head) for head in heads}
    spec = architecture_spec(architecture)
    return bool(
        sparse_policy
        or "sparse_policy" in head_set
        or "pair_policy" in head_set
        or bool(graph)
        or spec.replay_sparse_diagnostics
    )


def resolve_model_spec(
    cfg_or_model,
    *,
    heads: Sequence[str] | None = None,
) -> ResolvedArchitectureSpec:
    model_cfg = getattr(cfg_or_model, "model", cfg_or_model)
    buffer_cfg = getattr(cfg_or_model, "buffer", None)
    spec = architecture_spec(getattr(model_cfg, "architecture", "cnn"))
    requested_heads = heads if heads is not None else getattr(model_cfg, "heads", None)
    return resolve_spec(
        spec,
        requested_heads,
        lookahead_horizons=getattr(buffer_cfg, "lookahead_horizons", ()),
        sparse_policy=bool(getattr(model_cfg, "sparse_policy", False)),
        sparse_prior_stage=int(getattr(model_cfg, "sparse_prior_stage", 0)),
    )


def architecture_display_summary(
    model: Mapping[str, object],
    family: Mapping[str, object] | None = None,
) -> str:
    family = family or {}
    arch = str(model.get("architecture") or family.get("architecture") or "cnn").lower()
    try:
        spec = architecture_spec(arch)
    except ValueError:
        spec = architecture_spec("cnn")
    channels = model.get("channels") or family.get("channels") or "?"
    blocks = model.get("blocks") or family.get("blocks") or "?"
    heads = model.get("heads") or []
    if spec.global_graph:
        return (
            f"{spec.family_id} global graph, {channels} channels, "
            f"{model.get('graph_token_budget', '?')} tokens, "
            f"{model.get('graph_layers', '?')} graph layers, heads: {len(heads)}."
        )
    if spec.graph and not spec.global_graph:
        return (
            f"{spec.family_id}, {channels} channels, {blocks} residual blocks, "
            f"{model.get('graph_token_budget', '?')} {model.get('graph_token_set', 'tokens')}, "
            f"{model.get('graph_layers', '?')} graph layers, heads: {len(heads)}."
        )
    if spec.supports_attention_positions:
        return (
            f"{spec.family_id} hybrid trunk, {channels} channels, {blocks} blocks, "
            f"attention at {model.get('attention_positions') or []}, heads: {len(heads)}."
        )
    return f"CNN residual trunk, {channels} channels, {blocks} blocks, heads: {len(heads)}."


def trial_model_summary(family: Mapping[str, object], static: Mapping[str, object]) -> str:
    arch = str(family.get("architecture") or "cnn")
    try:
        spec = architecture_spec(arch)
    except ValueError:
        return f"{arch or 'model'} {family.get('channels', '?')}x{family.get('blocks', '?')}"
    if spec.global_graph:
        return f"{spec.architecture_id} {static.get('graph_token_budget', '?')} tokens x {static.get('graph_layers', '?')} layers"
    if spec.graph and not spec.global_graph:
        return f"{spec.architecture_id} {static.get('graph_token_budget', '?')} tokens x {static.get('graph_layers', '?')} layers"
    return f"{spec.architecture_id} {family.get('channels', '?')}x{family.get('blocks', '?')}"
