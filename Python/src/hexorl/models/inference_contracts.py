"""Model-owned inference operation and transport contract declarations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from hexorl.contracts.candidates import CANDIDATE_FEATURES
from hexorl.graph.semantic_builder import GRAPH_FEATURE_DIM, GRAPH_SCHEMA_VERSION, RELATION_SCHEMA_VERSION
from hexorl.models.specs import MODEL_SPEC_VERSION


CONTRACT_VERSION = 3
BOARD_SIZE = 33
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
MAX_CANDIDATES = 512
MAX_PAIR_CANDIDATES = 512
MAX_GRAPH_TOKENS = 1024
MAX_GRAPH_ACTIONS = 1024
MAX_GRAPH_PAIRS = 4096

DTYPE_FLOAT32 = "float32"
DTYPE_INT16 = "int16"
DTYPE_INT32 = "int32"
DTYPE_INT64 = "int64"
DTYPE_UINT8 = "uint8"

DIM_BATCH = "B"
DIM_CANDIDATE = "K"
DIM_PAIR = "P"
DIM_TOKEN = "T"
DIM_LEGAL = "L"
DIM_OPP_LEGAL = "O"
DIM_GRAPH_PAIR = "G"

DECODER_POLICY_LOGITS = "policy_logits"
DECODER_VALUE_BINS = "value_bins_to_scalar"
DECODER_REGRET_BINS = "regret_bins_to_scalar"
DECODER_SCALAR = "scalar"

OP_PLACE_VALUE = "place_value"
OP_SPARSE_PLACE_VALUE = "sparse_place_value"
OP_PAIR_POLICY = "pair_policy"
OP_REGRET = "regret"
OP_GRAPH_PLACE_VALUE = "graph_place_value"
OP_GRAPH_PAIR_POLICY = "graph_pair_policy"

BatchingMode = Literal["stack_over_b", "pad_and_stack", "singleton"]
ShapeDim = int | str


@dataclass(frozen=True)
class CapacitySpec:
    max_batch_size: int
    max_candidate_rows: int = 0
    max_pair_rows: int = 0
    max_graph_tokens: int = 0
    max_graph_actions: int = 0
    max_graph_pairs: int = 0

    def dim_capacity(self, dim: str) -> int:
        return {
            DIM_BATCH: self.max_batch_size,
            DIM_CANDIDATE: self.max_candidate_rows,
            DIM_PAIR: self.max_pair_rows,
            DIM_TOKEN: self.max_graph_tokens,
            DIM_LEGAL: self.max_graph_actions,
            DIM_OPP_LEGAL: self.max_graph_actions,
            DIM_GRAPH_PAIR: self.max_graph_pairs,
        }[str(dim)]

    def bounded_for_runtime(self, *, max_batch_size: int) -> "CapacitySpec":
        return CapacitySpec(
            max_batch_size=int(max_batch_size),
            max_candidate_rows=int(self.max_candidate_rows),
            max_pair_rows=int(self.max_pair_rows),
            max_graph_tokens=int(self.max_graph_tokens),
            max_graph_actions=int(self.max_graph_actions),
            max_graph_pairs=int(self.max_graph_pairs),
        )


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[ShapeDim, ...]
    semantic: str
    batching: BatchingMode

    def dynamic_dims(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(dim) for dim in self.shape if isinstance(dim, str)))


@dataclass(frozen=True)
class HeadDecoderSpec:
    name: str
    kind: str
    min_value: float | None = None
    max_value: float | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None


@dataclass(frozen=True)
class OutputHeadSpec:
    name: str
    tensor: TensorSpec
    decoder: HeadDecoderSpec
    row_mapping: str
    required: bool = False


@dataclass(frozen=True)
class TransportLayoutSpec:
    name: str
    input_tensors: tuple[TensorSpec, ...]
    output_tensors: tuple[TensorSpec, ...]

    def hash(self) -> str:
        return stable_contract_hash(asdict(self))


@dataclass(frozen=True)
class InferenceOperationSpec:
    name: str
    capability: str
    required_inputs: tuple[str, ...]
    output_heads: tuple[str, ...]
    required_heads: tuple[str, ...]
    layout: TransportLayoutSpec

    @property
    def layout_hash(self) -> str:
        return self.layout.hash()

    @property
    def code(self) -> int:
        digest = hashlib.sha256(self.name.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") or 1


@dataclass(frozen=True)
class ModelInferenceContract:
    model_family: str
    model_spec_version: int
    contract_version: int
    operations: tuple[InferenceOperationSpec, ...]
    heads: tuple[OutputHeadSpec, ...]
    capacities: CapacitySpec
    input_contract: str
    action_contract: str
    graph_schema_version: int | None = None
    relation_schema_version: int | None = None
    candidate_contract_version: int = 1
    pair_action_contract_version: int = 1
    ffi_protocol_version: int = 1
    legal_row_encoding: str = "rust-legal-row-u64-v1"
    history_row_encoding: str = "rust-compact-history-row-v1"
    pair_row_encoding: str = "candidate-index-pair-v1"

    def operation(self, name: str) -> InferenceOperationSpec:
        return _by_name(self.operations, name, "operation")

    def head(self, name: str) -> OutputHeadSpec:
        return _by_name(self.heads, name, "head")

    def operation_codes(self) -> dict[str, int]:
        return {op.name: op.code for op in self.operations}

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return stable_contract_hash(self.canonical_dict())


def _by_name(items: tuple[Any, ...], name: str, kind: str) -> Any:
    for item in items:
        if item.name == name:
            return item
    raise KeyError(f"model inference {kind} is not declared: {name}")


def stable_contract_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _tensor(name: str, dtype: str, shape: tuple[ShapeDim, ...], semantic: str, batching: BatchingMode) -> TensorSpec:
    return TensorSpec(name, dtype, shape, semantic, batching)


def crop_input_tensors(*, candidates: bool, pairs: bool) -> tuple[TensorSpec, ...]:
    tensors = [_tensor("tensor", DTYPE_FLOAT32, (DIM_BATCH, 13, BOARD_SIZE, BOARD_SIZE), "crop_input", "stack_over_b")]
    if candidates:
        tensors += [
            _tensor("candidate_indices", DTYPE_INT64, (DIM_BATCH, DIM_CANDIDATE), "candidate_rows", "pad_and_stack"),
            _tensor("candidate_features", DTYPE_FLOAT32, (DIM_BATCH, DIM_CANDIDATE, CANDIDATE_FEATURES), "candidate_rows", "pad_and_stack"),
            _tensor("candidate_mask", DTYPE_UINT8, (DIM_BATCH, DIM_CANDIDATE), "candidate_rows", "pad_and_stack"),
        ]
    if pairs:
        tensors += [
            _tensor("pair_candidate_indices", DTYPE_INT64, (DIM_BATCH, DIM_PAIR, 2), "pair_rows", "pad_and_stack"),
            _tensor("pair_candidate_mask", DTYPE_UINT8, (DIM_BATCH, DIM_PAIR), "pair_rows", "pad_and_stack"),
        ]
    return tuple(tensors)


def graph_input_tensors() -> tuple[TensorSpec, ...]:
    return (
        _tensor("token_features", DTYPE_FLOAT32, (DIM_TOKEN, GRAPH_FEATURE_DIM), "graph_tokens", "pad_and_stack"),
        _tensor("token_type", DTYPE_INT64, (DIM_TOKEN,), "graph_tokens", "pad_and_stack"),
        _tensor("token_qr", DTYPE_INT32, (DIM_TOKEN, 2), "graph_tokens", "pad_and_stack"),
        _tensor("token_mask", DTYPE_UINT8, (DIM_TOKEN,), "graph_tokens", "pad_and_stack"),
        _tensor("legal_token_indices", DTYPE_INT64, (DIM_LEGAL,), "legal_rows", "pad_and_stack"),
        _tensor("legal_mask", DTYPE_UINT8, (DIM_LEGAL,), "legal_rows", "pad_and_stack"),
        _tensor("opp_legal_qr", DTYPE_INT32, (DIM_OPP_LEGAL, 2), "opp_legal_rows", "pad_and_stack"),
        _tensor("opp_legal_mask", DTYPE_UINT8, (DIM_OPP_LEGAL,), "opp_legal_rows", "pad_and_stack"),
        _tensor("pair_token_indices", DTYPE_INT64, (DIM_GRAPH_PAIR,), "graph_pair_rows", "pad_and_stack"),
        _tensor("pair_first_indices", DTYPE_INT64, (DIM_GRAPH_PAIR,), "graph_pair_rows", "pad_and_stack"),
        _tensor("pair_second_indices", DTYPE_INT64, (DIM_GRAPH_PAIR,), "graph_pair_rows", "pad_and_stack"),
        _tensor("relation_type", DTYPE_INT64, (DIM_TOKEN, DIM_TOKEN), "graph_relations", "pad_and_stack"),
        _tensor("relation_bias", DTYPE_FLOAT32, (1, DIM_TOKEN, DIM_TOKEN), "graph_relations", "pad_and_stack"),
    )


def output_head_specs(head_names: tuple[str, ...], *, required_heads: tuple[str, ...]) -> tuple[OutputHeadSpec, ...]:
    from hexorl.models.heads import HEAD_REGISTRY

    required = set(required_heads)
    return tuple(
        OutputHeadSpec(name, HEAD_REGISTRY[name].output_tensor, HEAD_REGISTRY[name].decoder, HEAD_REGISTRY[name].row_mapping, name in required)
        for name in head_names
    )


def _head(name: str, shape: tuple[ShapeDim, ...], rows: str, decoder: str, **kwargs: Any) -> OutputHeadSpec:
    tensor = _tensor(name, DTYPE_FLOAT32, shape, rows, "pad_and_stack")
    required = bool(kwargs.pop("required", False))
    if decoder == DECODER_VALUE_BINS:
        kwargs.setdefault("clamp_min", -1.0)
        kwargs.setdefault("clamp_max", 1.0)
    elif decoder == DECODER_REGRET_BINS:
        kwargs.setdefault("clamp_min", 0.0)
        kwargs.setdefault("clamp_max", 4.0)
    elif decoder == DECODER_POLICY_LOGITS:
        kwargs.setdefault("clamp_min", -80.0)
        kwargs.setdefault("clamp_max", 80.0)
    return OutputHeadSpec(name, tensor, HeadDecoderSpec(name, decoder, **kwargs), rows, required)


def make_crop_contract(*, family_name: str, capabilities: tuple[str, ...], cfg: Any, required_heads: tuple[str, ...], output_heads: tuple[str, ...], graph_hybrid: bool) -> ModelInferenceContract:
    del capabilities
    capacity = CapacitySpec(int(getattr(cfg.inference, "max_batch_size", 1)), max(int(getattr(cfg.model, "candidate_budget", MAX_CANDIDATES)), 1), max(int(getattr(cfg.model, "pair_strategy_max_pairs", MAX_PAIR_CANDIDATES)), 1))
    heads = output_head_specs(output_heads, required_heads=required_heads)
    names = tuple(head.name for head in heads)
    ops = [_operation(OP_PLACE_VALUE, "DENSE_PLACE_POLICY", crop_input_tensors(candidates=False, pairs=False), heads, ("policy", "value"), required_heads)]
    if graph_hybrid and "sparse_policy" in names:
        ops.append(_operation(OP_SPARSE_PLACE_VALUE, "SPARSE_PLACE_POLICY", crop_input_tensors(candidates=True, pairs=False), heads, ("policy", "value", "sparse_policy"), ("policy", "value", "sparse_policy")))
    if graph_hybrid and "pair_policy" in names:
        ops.append(_operation(OP_PAIR_POLICY, "JOINT_PAIR_POLICY", crop_input_tensors(candidates=True, pairs=True), heads, ("policy", "value", "sparse_policy", "pair_policy"), ("policy", "value", "pair_policy")))
    if {"regret_rank", "regret_value"}.issubset(set(names)):
        ops.append(_operation(OP_REGRET, "REGRET_HEAD", crop_input_tensors(candidates=False, pairs=False), heads, ("regret_rank", "regret_value"), ("regret_rank", "regret_value")))
    return ModelInferenceContract(family_name, MODEL_SPEC_VERSION, CONTRACT_VERSION, tuple(ops), heads, capacity, "crop_tensor_v3", "legal_action_table_v1")


def make_graph_contract(*, family_name: str, cfg: Any, required_heads: tuple[str, ...], output_heads: tuple[str, ...]) -> ModelInferenceContract:
    capacity = CapacitySpec(1, max_graph_tokens=int(getattr(cfg.model, "graph_token_budget", MAX_GRAPH_TOKENS)), max_graph_actions=MAX_GRAPH_ACTIONS, max_graph_pairs=MAX_GRAPH_PAIRS, max_pair_rows=MAX_GRAPH_PAIRS)
    heads = output_head_specs(output_heads, required_heads=required_heads)
    names = tuple(head.name for head in heads)
    base = tuple(name for name in ("policy_place", "value", "opp_policy", "regret_rank", "regret_value") if name in names)
    pair = tuple(name for name in ("policy_place", "value", "opp_policy", "policy_pair_first", "policy_pair_joint", "policy_pair_second", "regret_rank", "regret_value") if name in names)
    ops = [_operation(OP_GRAPH_PLACE_VALUE, "GLOBAL_PLACE_POLICY", graph_input_tensors(), heads, base, required_heads)]
    if {"policy_pair_joint", "policy_pair_second"}.issubset(set(names)):
        ops.append(_operation(OP_GRAPH_PAIR_POLICY, "JOINT_PAIR_POLICY", graph_input_tensors(), heads, pair, ("policy_place", "value", "policy_pair_joint", "policy_pair_second")))
    return ModelInferenceContract(family_name, MODEL_SPEC_VERSION, CONTRACT_VERSION, tuple(ops), heads, capacity, "global_graph_v3", "legal_action_table_v1", GRAPH_SCHEMA_VERSION, RELATION_SCHEMA_VERSION)


def _operation(name: str, capability: str, inputs: tuple[TensorSpec, ...], heads: tuple[OutputHeadSpec, ...], outputs: tuple[str, ...], required: tuple[str, ...]) -> InferenceOperationSpec:
    output_tensors = tuple(head.tensor for head in heads if head.name in outputs)
    return InferenceOperationSpec(name, capability, tuple(t.name for t in inputs), outputs, required, TransportLayoutSpec(f"{name}_layout", inputs, output_tensors))


__all__ = [
    "BOARD_AREA",
    "BOARD_SIZE",
    "CONTRACT_VERSION",
    "DECODER_POLICY_LOGITS",
    "DECODER_REGRET_BINS",
    "DECODER_SCALAR",
    "DECODER_VALUE_BINS",
    "DIM_BATCH",
    "DIM_CANDIDATE",
    "DIM_GRAPH_PAIR",
    "DIM_LEGAL",
    "DIM_OPP_LEGAL",
    "DIM_PAIR",
    "DIM_TOKEN",
    "CapacitySpec",
    "HeadDecoderSpec",
    "InferenceOperationSpec",
    "ModelInferenceContract",
    "OP_GRAPH_PAIR_POLICY",
    "OP_GRAPH_PLACE_VALUE",
    "OP_PAIR_POLICY",
    "OP_PLACE_VALUE",
    "OP_REGRET",
    "OP_SPARSE_PLACE_VALUE",
    "OutputHeadSpec",
    "TensorSpec",
    "TransportLayoutSpec",
    "make_crop_contract",
    "make_graph_contract",
]
