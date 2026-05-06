"""Architecture specs and resolution rules."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

from hexorl.contracts import (
    OutputContract,
    ROW_TABLE_DEFINITIONS,
    RowTableDefinition,
    ValueDecoderContract,
    row_table_definitions_for,
)


LOOKAHEAD_FAMILY = "lookahead_*"


@dataclass(frozen=True)
class AliasDecision:
    alias: str
    target: str | None
    decision: str
    runtime_supported: bool


@dataclass(frozen=True)
class ArchitectureSpec:
    architecture_id: str
    family_id: str
    recipe_id: str
    description: str
    default_outputs: tuple[str, ...]
    supported_optional_outputs: tuple[str, ...] = ()
    forbidden_outputs: tuple[str, ...] = ()
    dynamic_output_families: tuple[str, ...] = (LOOKAHEAD_FAMILY,)
    selfplay_required_outputs: tuple[str, ...] = ("policy", "value")
    training_adapter_id: str = "crop:v1"
    inference_adapter_id: str = "crop:v1"
    policy_provider_id: str = "dense_board:v1"
    value_provider_id: str = "binned_value_65:v1"
    pair_capabilities: tuple[str, ...] = ()
    input_contract_id: str = "crop_13x33x33:v1"
    row_families: tuple[str, ...] = ("dense_board",)
    relation_required: bool = False
    graph: bool = False
    global_graph: bool = False
    sparse_policy_capable: bool = False
    crop_pair_capable: bool = False
    graph_pair_capable: bool = False
    replay_sparse_diagnostics: bool = False
    requires_attention_head_divisibility: bool = False
    supports_attention_positions: bool = False
    display_name: str = "model"
    default_loss_weights: Mapping[str, float] = field(default_factory=dict)
    output_contracts: Mapping[str, OutputContract] = field(default_factory=dict)
    value_decoder: ValueDecoderContract = field(default_factory=ValueDecoderContract)

    @property
    def all_static_outputs(self) -> frozenset[str]:
        return frozenset(self.default_outputs) | frozenset(self.supported_optional_outputs)

    def supports_output(self, output: str) -> bool:
        if output in self.forbidden_outputs:
            return False
        if output in self.all_static_outputs:
            return True
        return output.startswith("lookahead_") and LOOKAHEAD_FAMILY in self.dynamic_output_families

    def row_table_definitions(self) -> Mapping[str, RowTableDefinition]:
        return row_table_definitions_for(self.row_families)


@dataclass(frozen=True)
class ResolvedArchitectureSpec:
    spec: ArchitectureSpec
    architecture_id: str
    outputs: tuple[str, ...]
    output_contracts: Mapping[str, OutputContract]
    row_table_definitions: Mapping[str, RowTableDefinition]
    value_decoder: ValueDecoderContract
    default_loss_weights: Mapping[str, float]
    config_head_aliases: Mapping[str, str] = field(default_factory=dict)

    def has_output(self, name: str) -> bool:
        return name in self.outputs

    @property
    def family_id(self) -> str:
        return self.spec.family_id

    @property
    def recipe_id(self) -> str:
        return self.spec.recipe_id

    @property
    def selfplay_required_outputs(self) -> tuple[str, ...]:
        return self.spec.selfplay_required_outputs

    @property
    def global_graph(self) -> bool:
        return self.spec.global_graph

    @property
    def graph(self) -> bool:
        return self.spec.graph

    @property
    def pair_capabilities(self) -> tuple[str, ...]:
        return self.spec.pair_capabilities


def merge_resolved_loss_weights(
    resolved: ResolvedArchitectureSpec,
    configured_loss_weights: Mapping[str, float],
) -> dict[str, float]:
    """Return effective loss weights without mutating config objects."""

    merged = dict(configured_loss_weights)
    for name, weight in resolved.default_loss_weights.items():
        if name.startswith("lookahead_"):
            if not resolved.global_graph:
                continue
            weight = float(merged.get("value", weight)) * 0.1
        merged.setdefault(name, weight)
    return merged


def output_contracts_for(names: Sequence[str]) -> dict[str, OutputContract]:
    contracts: dict[str, OutputContract] = {}
    for name in names:
        if name == "policy":
            contracts[name] = OutputContract(
                name=name,
                kind="policy",
                prediction_key="policy",
                row_family="dense_board",
                runtime_consumed=True,
                required_for_selfplay=True,
            )
        elif name == "sparse_policy":
            contracts[name] = OutputContract(
                name=name,
                kind="policy",
                prediction_key=name,
                row_family="candidate",
                runtime_consumed=True,
                optional=True,
            )
        elif name == "pair_policy":
            contracts[name] = OutputContract(
                name=name,
                kind="pair_policy",
                prediction_key=name,
                row_family="pair_joint",
                runtime_consumed=False,
                optional=True,
            )
        elif name == "policy_place":
            contracts[name] = OutputContract(
                name=name,
                kind="policy",
                prediction_key=name,
                row_family="legal",
                runtime_consumed=True,
                required_for_selfplay=True,
            )
        elif name == "policy_pair_first":
            contracts[name] = OutputContract(
                name=name,
                kind="pair_policy",
                prediction_key=name,
                row_family="legal",
                runtime_consumed=True,
                optional=True,
            )
        elif name == "policy_pair_joint":
            contracts[name] = OutputContract(
                name=name,
                kind="pair_policy",
                prediction_key=name,
                row_family="pair_joint",
                runtime_consumed=True,
                optional=True,
            )
        elif name == "policy_pair_second":
            contracts[name] = OutputContract(
                name=name,
                kind="pair_policy",
                prediction_key=name,
                row_family="known_first_pair",
                runtime_consumed=True,
                optional=True,
            )
        elif name == "opp_policy":
            contracts[name] = OutputContract(
                name=name,
                kind="policy",
                prediction_key=name,
                row_family="opponent_legal",
                optional=True,
            )
        elif name == "value" or name.startswith("lookahead_") or name == "regret_value":
            contracts[name] = OutputContract(
                name=name,
                kind="value",
                prediction_key=name,
                state_row="state",
                runtime_consumed=name == "value",
                required_for_selfplay=name == "value",
                optional=name != "value",
            )
        elif name in {"axis", "axis_delta_norm", "regret_rank", "moves_left", "tactical", "legal_token_quality"}:
            contracts[name] = OutputContract(
                name=name,
                kind="auxiliary",
                prediction_key=name,
                row_family="legal" if name == "legal_token_quality" else None,
                state_row=None if name == "legal_token_quality" else "state",
                optional=True,
            )
    return contracts


DENSE_OPTIONAL_OUTPUTS = (
    "sparse_policy",
    "pair_policy",
    "opp_policy",
    "axis",
    "axis_delta_norm",
    "regret_rank",
    "regret_value",
    "moves_left",
)

GLOBAL_OPTIONAL_OUTPUTS = (
    "opp_policy",
    "policy_pair_first",
    "policy_pair_joint",
    "policy_pair_second",
    "regret_rank",
    "regret_value",
    "moves_left",
    "tactical",
    "axis",
    "axis_delta_norm",
    "legal_token_quality",
)

DENSE_LOSS_DEFAULTS = {
    "policy": 1.0,
    "value": 1.5,
    "sparse_policy": 0.25,
    "pair_policy": 0.05,
}

GLOBAL_LOSS_DEFAULTS = {
    "policy_place": 1.0,
    "policy_pair_first": 0.05,
    "policy_pair_second": 0.05,
    "policy_pair_joint": 0.05,
    "opp_policy": 0.15,
    "value": 1.0,
    "regret_rank": 0.1,
    "regret_value": 0.1,
    "moves_left": 0.05,
    "tactical": 0.05,
    "legal_token_quality": 0.05,
}


def dense_spec(
    architecture_id: str,
    *,
    family_id: str,
    recipe_id: str,
    description: str,
    sparse_policy_capable: bool = True,
    replay_sparse_diagnostics: bool = False,
    requires_attention_head_divisibility: bool = False,
    supports_attention_positions: bool = False,
) -> ArchitectureSpec:
    contracts = output_contracts_for(("policy", "value", *DENSE_OPTIONAL_OUTPUTS))
    contracts["opp_policy"] = OutputContract(
        name="opp_policy",
        kind="policy",
        prediction_key="opp_policy",
        row_family="dense_board",
        optional=True,
    )
    return ArchitectureSpec(
        architecture_id=architecture_id,
        family_id=family_id,
        recipe_id=recipe_id,
        description=description,
        default_outputs=("policy", "value"),
        supported_optional_outputs=DENSE_OPTIONAL_OUTPUTS,
        selfplay_required_outputs=("policy", "value"),
        training_adapter_id="crop:v1",
        inference_adapter_id="crop:v1",
        policy_provider_id="dense_board:v1",
        pair_capabilities=("crop_pair_policy",),
        row_families=("dense_board", "candidate", "pair_joint"),
        sparse_policy_capable=sparse_policy_capable,
        crop_pair_capable=True,
        replay_sparse_diagnostics=replay_sparse_diagnostics,
        requires_attention_head_divisibility=requires_attention_head_divisibility,
        supports_attention_positions=supports_attention_positions,
        display_name=family_id,
        default_loss_weights=DENSE_LOSS_DEFAULTS,
        output_contracts=contracts,
    )


def global_graph_spec(
    architecture_id: str,
    *,
    family_id: str,
    recipe_id: str,
    description: str,
    relation_required: bool,
) -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id=architecture_id,
        family_id=family_id,
        recipe_id=recipe_id,
        description=description,
        default_outputs=("policy_place", "value"),
        supported_optional_outputs=GLOBAL_OPTIONAL_OUTPUTS,
        selfplay_required_outputs=("policy_place", "value"),
        training_adapter_id="global_graph:v1",
        inference_adapter_id="global_graph:v1",
        policy_provider_id="global_legal:v1",
        pair_capabilities=("graph_pair_first", "graph_pair_joint", "graph_pair_second"),
        input_contract_id="global_graph_tokens:v1",
        row_families=(
            "legal",
            "opponent_legal",
            "pair_joint",
            "known_first_pair",
            "graph_token",
        ),
        relation_required=relation_required,
        graph=True,
        global_graph=True,
        graph_pair_capable=True,
        replay_sparse_diagnostics=True,
        requires_attention_head_divisibility=True,
        display_name=family_id,
        default_loss_weights=GLOBAL_LOSS_DEFAULTS,
        output_contracts=output_contracts_for(("policy_place", "value", *GLOBAL_OPTIONAL_OUTPUTS)),
    )


def resolve_outputs(
    spec: ArchitectureSpec,
    requested_heads: Sequence[str] | None,
    *,
    lookahead_horizons: Sequence[int] = (),
    sparse_policy: bool = False,
    sparse_prior_stage: int = 0,
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    requested = list(requested_heads or spec.default_outputs)
    aliases: dict[str, str] = {}
    if spec.global_graph and requested == ["policy", "value"]:
        requested = ["policy_place", "value", LOOKAHEAD_FAMILY]
    if spec.global_graph:
        translated = []
        for head in requested:
            if head == "policy":
                aliases["policy"] = "policy_place"
                translated.append("policy_place")
            else:
                translated.append(head)
        requested = translated

    if not requested:
        requested = list(spec.default_outputs)

    outputs: list[str] = []
    for head in requested:
        if head == LOOKAHEAD_FAMILY:
            for horizon in lookahead_horizons:
                outputs.append(f"lookahead_{int(horizon)}")
            continue
        outputs.append(str(head))

    if sparse_policy or int(sparse_prior_stage) > 0:
        if spec.sparse_policy_capable and "sparse_policy" not in outputs:
            outputs.append("sparse_policy")

    for required in spec.selfplay_required_outputs:
        if required not in outputs:
            raise ValueError(
                f"model architecture {spec.architecture_id} requires self-play output "
                f"{required!r}; requested heads={requested!r}"
            )

    unknown = [head for head in outputs if not spec.supports_output(head)]
    if unknown:
        raise ValueError(
            f"model architecture {spec.architecture_id} does not support outputs {unknown}"
        )

    ordered = tuple(dict.fromkeys(outputs))
    return ordered, aliases


def resolve_spec(
    spec: ArchitectureSpec,
    requested_heads: Sequence[str] | None,
    *,
    lookahead_horizons: Sequence[int] = (),
    sparse_policy: bool = False,
    sparse_prior_stage: int = 0,
) -> ResolvedArchitectureSpec:
    outputs, aliases = resolve_outputs(
        spec,
        requested_heads,
        lookahead_horizons=lookahead_horizons,
        sparse_policy=sparse_policy,
        sparse_prior_stage=sparse_prior_stage,
    )
    contracts = dict(spec.output_contracts)
    contracts.update(output_contracts_for([name for name in outputs if name.startswith("lookahead_")]))
    active_contracts = {name: contracts[name] for name in outputs if name in contracts}
    loss_defaults = dict(spec.default_loss_weights)
    for name in outputs:
        if name.startswith("lookahead_"):
            loss_defaults.setdefault(name, 0.1)
    return ResolvedArchitectureSpec(
        spec=spec,
        architecture_id=spec.architecture_id,
        outputs=outputs,
        output_contracts=active_contracts,
        row_table_definitions=spec.row_table_definitions(),
        value_decoder=spec.value_decoder,
        default_loss_weights=loss_defaults,
        config_head_aliases=aliases,
    )


def replace_loss_defaults(spec: ArchitectureSpec, defaults: Mapping[str, float]) -> ArchitectureSpec:
    return replace(spec, default_loss_weights=dict(defaults))
