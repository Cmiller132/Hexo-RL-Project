"""Game record format — compact serialization for the ring buffer.

Each record represents one position from a self-play game:
  - move_history: compact bytes (i32 LE triples: player, q, r) 
  - policy_target: sparse dict {action_idx: probability} 
  - value_target: f32 (outcome or EMA lookahead)
  - game_id: u32 (for recency weighting)
  - player: u8 (which player generated this record — used for perspective flip)
"""

import json
import pickle
import struct
import zlib
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field

from hexorl.v1_pair_contract import V1TerminalTacticalPayload


# Constants matching Rust encoder (must stay in sync)
NUM_CHANNELS = 13
BOARD_SIZE = 33
BOARD_AREA = 33 * 33  # 1089
COMPACT_MAGIC_V2 = b"HXG2"
COMPACT_VERSION_V2 = 11
COMPACT_VERSION_MIN = 2
PolicyTargetV2 = List[Tuple[int, int, float]]

V1_PAIR_SEARCH_SCHEMA_VERSION = 2
V1_PAIR_SEARCH_COMPACT_MAGIC = b"HXVM"
V1_PAIR_SEARCH_COMPACT_VERSION = 3
V1_PAIR_SEARCH_COMPRESSION_ZLIB = 1
V1_PAIR_SUPPORT_TYPES = frozenset(
    {
        "exhaustive_legal_pair_table",
        "admitted_candidate_set_without_explicit_negatives",
        "admitted_candidate_set_with_explicit_negatives",
        "completed_q_candidate_posterior",
    }
)
V1_PAIR_TARGET_SUPPORT_FLAGS = frozenset(
    {
        "admitted",
        "explicit_negative",
        "forced",
        "sampled_negative",
        "terminal_exact",
        "terminal_equivalent",
        "terminal_cover",
        "covers_all_opponent_win_requirements",
        "impossible_to_cover",
        "unsampled",
    }
)
V1_BASE_SUPPORT_FLAGS = frozenset({"admitted", "explicit_negative", "unsampled"})
V1_INCLUSION_KINDS = frozenset(
    {
        "stochastic_sample",
        "deterministic_top_k",
        "tactical_protected",
        "structured_quota",
        "diagnostic_canary",
        "unknown",
    }
)
V1_CORRECTION_MODES = frozenset(
    {
        "exact_importance",
        "clipped_propensity",
        "uncorrected_logged",
        "training_forbidden",
    }
)

PairCoord = Tuple[int, int]
PairKey = Tuple[PairCoord, PairCoord]
_V1_SUPPORT_FLAG_ORDER = (
    "admitted",
    "explicit_negative",
    "forced",
    "sampled_negative",
    "terminal_exact",
    "terminal_equivalent",
    "terminal_cover",
    "covers_all_opponent_win_requirements",
    "impossible_to_cover",
    "unsampled",
)


def _pair_coord(value: Any) -> PairCoord:
    return (int(value[0]), int(value[1]))


def _canonical_pair_key(first: Any, second: Any) -> PairKey:
    a = _pair_coord(first)
    b = _pair_coord(second)
    if a == b:
        raise ValueError(f"duplicate coordinates are illegal for V1 pair metadata: {a}")
    return (a, b) if a <= b else (b, a)


def _finite_optional(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{field_name} must be finite when provided")
    return out


def _string_tuple(values: Tuple[str, ...] | List[str]) -> Tuple[str, ...]:
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class V1CandidateSourceContribution:
    """Auditable source contribution for one V1 pair candidate."""

    source_type: str
    source_rank: Optional[int] = None
    source_weight: Optional[float] = None
    local_probability_or_score: Optional[float] = None
    quota_id: Optional[str] = None
    inclusion_kind: str = "unknown"
    exact_inclusion_probability: Optional[float] = None
    heuristic_propensity: Optional[float] = None
    correction_mode: str = "training_forbidden"

    def __post_init__(self) -> None:
        if not str(self.source_type):
            raise ValueError("V1 source_contribution.source_type is required")
        if self.inclusion_kind not in V1_INCLUSION_KINDS:
            raise ValueError(f"Unsupported V1 inclusion_kind: {self.inclusion_kind!r}")
        if self.correction_mode not in V1_CORRECTION_MODES:
            raise ValueError(f"Unsupported V1 correction_mode: {self.correction_mode!r}")
        if self.source_rank is not None and int(self.source_rank) < 0:
            raise ValueError("V1 source_rank cannot be negative")
        object.__setattr__(self, "source_rank", None if self.source_rank is None else int(self.source_rank))
        object.__setattr__(self, "source_weight", _finite_optional(self.source_weight, "source_weight"))
        object.__setattr__(
            self,
            "local_probability_or_score",
            _finite_optional(self.local_probability_or_score, "local_probability_or_score"),
        )
        object.__setattr__(
            self,
            "exact_inclusion_probability",
            _finite_optional(self.exact_inclusion_probability, "exact_inclusion_probability"),
        )
        object.__setattr__(
            self,
            "heuristic_propensity",
            _finite_optional(self.heuristic_propensity, "heuristic_propensity"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_rank": self.source_rank,
            "source_weight": self.source_weight,
            "local_probability_or_score": self.local_probability_or_score,
            "quota_id": self.quota_id,
            "inclusion_kind": self.inclusion_kind,
            "exact_inclusion_probability": self.exact_inclusion_probability,
            "heuristic_propensity": self.heuristic_propensity,
            "correction_mode": self.correction_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1CandidateSourceContribution":
        return cls(
            source_type=str(data.get("source_type", "")),
            source_rank=data.get("source_rank"),
            source_weight=data.get("source_weight"),
            local_probability_or_score=data.get("local_probability_or_score"),
            quota_id=data.get("quota_id"),
            inclusion_kind=str(data.get("inclusion_kind", "unknown")),
            exact_inclusion_probability=data.get("exact_inclusion_probability"),
            heuristic_propensity=data.get("heuristic_propensity"),
            correction_mode=str(data.get("correction_mode", "training_forbidden")),
        )


@dataclass(frozen=True)
class V1ProposalPropensityMetadata:
    """Typed proposal logging used for sampled pair correction."""

    proposal_policy: str
    correction_mode: str
    total_proposal_probability: Optional[float] = None
    log_proposal_probability: Optional[float] = None
    sampling_without_replacement: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.proposal_policy):
            raise ValueError("V1 proposal_propensity_metadata.proposal_policy is required")
        if self.correction_mode not in V1_CORRECTION_MODES:
            raise ValueError(f"Unsupported V1 proposal correction_mode: {self.correction_mode!r}")
        prob = _finite_optional(self.total_proposal_probability, "total_proposal_probability")
        if prob is not None and prob < 0.0:
            raise ValueError("total_proposal_probability cannot be negative")
        object.__setattr__(self, "total_proposal_probability", prob)
        object.__setattr__(
            self,
            "log_proposal_probability",
            _finite_optional(self.log_proposal_probability, "log_proposal_probability"),
        )
        object.__setattr__(self, "sampling_without_replacement", bool(self.sampling_without_replacement))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_policy": self.proposal_policy,
            "correction_mode": self.correction_mode,
            "total_proposal_probability": self.total_proposal_probability,
            "log_proposal_probability": self.log_proposal_probability,
            "sampling_without_replacement": self.sampling_without_replacement,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1ProposalPropensityMetadata":
        return cls(
            proposal_policy=str(data.get("proposal_policy", "")),
            correction_mode=str(data.get("correction_mode", "training_forbidden")),
            total_proposal_probability=data.get("total_proposal_probability"),
            log_proposal_probability=data.get("log_proposal_probability"),
            sampling_without_replacement=bool(data.get("sampling_without_replacement", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class V1ProposalCorrectionParameters:
    """Root/search correction parameters logged with V1 pair metadata."""

    correction_mode: str
    min_log: float
    max_log: float
    prior_temperature: float

    def __post_init__(self) -> None:
        if self.correction_mode not in V1_CORRECTION_MODES:
            raise ValueError(f"Unsupported V1 correction_mode: {self.correction_mode!r}")
        min_log = float(self.min_log)
        max_log = float(self.max_log)
        temp = float(self.prior_temperature)
        if not np.isfinite(min_log) or not np.isfinite(max_log):
            raise ValueError("V1 proposal correction log bounds must be finite")
        if min_log > max_log:
            raise ValueError("V1 proposal correction min_log cannot exceed max_log")
        if not np.isfinite(temp) or temp <= 0.0:
            raise ValueError("V1 proposal correction prior_temperature must be positive")
        object.__setattr__(self, "min_log", min_log)
        object.__setattr__(self, "max_log", max_log)
        object.__setattr__(self, "prior_temperature", temp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correction_mode": self.correction_mode,
            "min_log": self.min_log,
            "max_log": self.max_log,
            "prior_temperature": self.prior_temperature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1ProposalCorrectionParameters":
        return cls(
            correction_mode=str(data.get("correction_mode", "training_forbidden")),
            min_log=float(data.get("min_log", 0.0)),
            max_log=float(data.get("max_log", 0.0)),
            prior_temperature=float(data.get("prior_temperature", 1.0)),
        )


@dataclass(frozen=True)
class V1ReservoirRefillEvent:
    """Telemetry for an explicitly allowed V1 candidate reservoir refill."""

    node_id: str
    reason: str
    generation: int
    requested_count: int
    added_count: int

    def __post_init__(self) -> None:
        if not str(self.node_id):
            raise ValueError("V1 reservoir_refill_events.node_id is required")
        if not str(self.reason):
            raise ValueError("V1 reservoir_refill_events.reason is required")
        for field_name in ("generation", "requested_count", "added_count"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"V1 reservoir refill {field_name} cannot be negative")
            object.__setattr__(self, field_name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "reason": self.reason,
            "generation": self.generation,
            "requested_count": self.requested_count,
            "added_count": self.added_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1ReservoirRefillEvent":
        return cls(
            node_id=str(data.get("node_id", "")),
            reason=str(data.get("reason", "")),
            generation=int(data.get("generation", 0)),
            requested_count=int(data.get("requested_count", 0)),
            added_count=int(data.get("added_count", 0)),
        )


@dataclass(frozen=True)
class V1CandidatePair:
    """One candidate/support row in the V1 sampled pair-action replay schema."""

    candidate_id: str
    pair_key: PairKey
    first_legal_row_id: int
    second_legal_row_id: int
    row_table_schema_version: int
    source_contributions: Tuple[V1CandidateSourceContribution, ...]
    proposal_propensity_metadata: V1ProposalPropensityMetadata
    forced_exploration_flag: bool
    terminal_exact_flag: bool
    terminal_equivalence_flag: bool
    target_support_flags: Tuple[str, ...]
    admission_generation: int
    root_or_interior: str
    candidate_selection_reason: str = ""

    def __post_init__(self) -> None:
        first, second = self.pair_key
        object.__setattr__(self, "pair_key", _canonical_pair_key(first, second))
        if not str(self.candidate_id):
            raise ValueError("V1 candidate_pair.candidate_id is required")
        if int(self.first_legal_row_id) < 0 or int(self.second_legal_row_id) < 0:
            raise ValueError("V1 candidate legal row ids cannot be negative")
        if int(self.first_legal_row_id) == int(self.second_legal_row_id):
            raise ValueError("V1 candidate legal row ids must refer to distinct rows")
        if int(self.row_table_schema_version) <= 0:
            raise ValueError("V1 candidate row_table_schema_version must be positive")
        if int(self.admission_generation) < 0:
            raise ValueError("V1 candidate admission_generation cannot be negative")
        if self.root_or_interior not in {"root", "interior"}:
            raise ValueError("V1 candidate root_or_interior must be 'root' or 'interior'")
        sources = tuple(
            source
            if isinstance(source, V1CandidateSourceContribution)
            else V1CandidateSourceContribution.from_dict(source)
            for source in self.source_contributions
        )
        if not sources:
            raise ValueError("V1 candidate source_contributions are required")
        proposal = (
            self.proposal_propensity_metadata
            if isinstance(self.proposal_propensity_metadata, V1ProposalPropensityMetadata)
            else V1ProposalPropensityMetadata.from_dict(self.proposal_propensity_metadata)
        )
        flags = _string_tuple(self.target_support_flags)
        unknown = set(flags) - V1_PAIR_TARGET_SUPPORT_FLAGS
        if unknown:
            raise ValueError(f"Unsupported V1 target_support_flags: {sorted(unknown)!r}")
        base_flags = set(flags) & V1_BASE_SUPPORT_FLAGS
        if len(base_flags) != 1:
            raise ValueError(
                "V1 target_support_flags must contain exactly one of "
                "'admitted', 'explicit_negative', or 'unsampled'"
            )
        if "unsampled" in flags and "sampled_negative" in flags:
            raise ValueError("V1 unsampled legal pairs cannot be marked sampled_negative")
        if bool(self.forced_exploration_flag) != ("forced" in flags):
            raise ValueError("V1 forced_exploration_flag must match the 'forced' target support flag")
        if bool(self.terminal_exact_flag) != ("terminal_exact" in flags):
            raise ValueError("V1 terminal_exact_flag must match the 'terminal_exact' target support flag")
        if bool(self.terminal_equivalence_flag) != ("terminal_equivalent" in flags):
            raise ValueError(
                "V1 terminal_equivalence_flag must match the 'terminal_equivalent' target support flag"
            )
        object.__setattr__(self, "first_legal_row_id", int(self.first_legal_row_id))
        object.__setattr__(self, "second_legal_row_id", int(self.second_legal_row_id))
        object.__setattr__(self, "row_table_schema_version", int(self.row_table_schema_version))
        object.__setattr__(self, "source_contributions", sources)
        object.__setattr__(self, "proposal_propensity_metadata", proposal)
        object.__setattr__(self, "forced_exploration_flag", bool(self.forced_exploration_flag))
        object.__setattr__(self, "terminal_exact_flag", bool(self.terminal_exact_flag))
        object.__setattr__(self, "terminal_equivalence_flag", bool(self.terminal_equivalence_flag))
        object.__setattr__(self, "target_support_flags", flags)
        object.__setattr__(self, "admission_generation", int(self.admission_generation))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "pair_key": [list(self.pair_key[0]), list(self.pair_key[1])],
            "first_legal_row_id": self.first_legal_row_id,
            "second_legal_row_id": self.second_legal_row_id,
            "row_table_schema_version": self.row_table_schema_version,
            "source_contributions": [source.to_dict() for source in self.source_contributions],
            "proposal_propensity_metadata": self.proposal_propensity_metadata.to_dict(),
            "forced_exploration_flag": self.forced_exploration_flag,
            "terminal_exact_flag": self.terminal_exact_flag,
            "terminal_equivalence_flag": self.terminal_equivalence_flag,
            "target_support_flags": list(self.target_support_flags),
            "admission_generation": self.admission_generation,
            "root_or_interior": self.root_or_interior,
            "candidate_selection_reason": self.candidate_selection_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1CandidatePair":
        pair_key = data.get("pair_key", [])
        if len(pair_key) != 2:
            raise ValueError("V1 candidate pair_key must contain two coordinates")
        return cls(
            candidate_id=str(data.get("candidate_id", "")),
            pair_key=(_pair_coord(pair_key[0]), _pair_coord(pair_key[1])),
            first_legal_row_id=int(data.get("first_legal_row_id", -1)),
            second_legal_row_id=int(data.get("second_legal_row_id", -1)),
            row_table_schema_version=int(data.get("row_table_schema_version", 0)),
            source_contributions=tuple(
                V1CandidateSourceContribution.from_dict(item)
                for item in data.get("source_contributions", [])
            ),
            proposal_propensity_metadata=V1ProposalPropensityMetadata.from_dict(
                data.get("proposal_propensity_metadata", {})
            ),
            forced_exploration_flag=bool(data.get("forced_exploration_flag", False)),
            terminal_exact_flag=bool(data.get("terminal_exact_flag", False)),
            terminal_equivalence_flag=bool(data.get("terminal_equivalence_flag", False)),
            target_support_flags=tuple(data.get("target_support_flags", ())),
            admission_generation=int(data.get("admission_generation", 0)),
            root_or_interior=str(data.get("root_or_interior", "")),
            candidate_selection_reason=str(data.get("candidate_selection_reason", "")),
        )


@dataclass(frozen=True)
class V1SearchPairMetadata:
    """Versioned V1 candidate-aware pair-search replay metadata.

    This metadata is intentionally separate from legacy ``pair_policy_target_v2``.
    Until V1 training consumers exist, records carrying this container must not
    be treated as complete legacy pair-policy targets.
    """

    candidate_selector_version: str
    support_type: str
    legal_pair_count: int
    legal_row_schema_version: int
    pair_row_schema_version: int
    candidate_pairs: Tuple[V1CandidatePair, ...]
    proposal_correction_parameters: V1ProposalCorrectionParameters
    root_gumbel_values: Tuple[float, ...]
    root_admission_order: Tuple[int, ...]
    root_simulation_allocation: Tuple[int, ...]
    visit_counts: Tuple[int, ...]
    q_values: Tuple[float, ...]
    completed_q_values: Tuple[float, ...]
    selected_pair: Optional[PairKey]
    target_support_flags: Tuple[Tuple[str, ...], ...]
    terminal_equivalence_flags: Tuple[bool, ...]
    terminal_tactical_payload: V1TerminalTacticalPayload | Mapping[str, Any] | None = None
    search_surprise_metrics: Dict[str, float] = field(default_factory=dict)
    neural_calls_per_expanded_full_turn_node: Optional[float] = None
    reservoir_refill_events: Tuple[V1ReservoirRefillEvent, ...] = ()
    schema_version: int = V1_PAIR_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != V1_PAIR_SEARCH_SCHEMA_VERSION:
            raise ValueError(f"Unsupported V1 pair search schema_version {self.schema_version}")
        if not str(self.candidate_selector_version):
            raise ValueError("V1 candidate_selector_version is required")
        if self.support_type not in V1_PAIR_SUPPORT_TYPES:
            raise ValueError(f"Unsupported V1 support_type: {self.support_type!r}")
        if int(self.legal_pair_count) < 0:
            raise ValueError("V1 legal_pair_count cannot be negative")
        if int(self.legal_row_schema_version) <= 0 or int(self.pair_row_schema_version) <= 0:
            raise ValueError("V1 legal/pair schema versions must be positive")
        candidates = tuple(
            candidate
            if isinstance(candidate, V1CandidatePair)
            else V1CandidatePair.from_dict(candidate)
            for candidate in self.candidate_pairs
        )
        if len(candidates) > int(self.legal_pair_count):
            raise ValueError("V1 candidate_pairs cannot exceed legal_pair_count")
        pair_keys = [candidate.pair_key for candidate in candidates]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("V1 candidate_pairs contain duplicate pair_key values")
        selected = None if self.selected_pair is None else _canonical_pair_key(*self.selected_pair)
        if selected is not None and selected not in set(pair_keys):
            raise ValueError("V1 selected_pair must be present in candidate_pairs")
        correction = (
            self.proposal_correction_parameters
            if isinstance(self.proposal_correction_parameters, V1ProposalCorrectionParameters)
            else V1ProposalCorrectionParameters.from_dict(self.proposal_correction_parameters)
        )
        target_flags = tuple(_string_tuple(flags) for flags in self.target_support_flags)
        terminal_flags = tuple(bool(flag) for flag in self.terminal_equivalence_flags)
        tactical_payload = (
            self.terminal_tactical_payload
            if isinstance(self.terminal_tactical_payload, V1TerminalTacticalPayload)
            else V1TerminalTacticalPayload.from_mapping(self.terminal_tactical_payload)
        )
        arrays = {
            "root_gumbel_values": tuple(float(v) for v in self.root_gumbel_values),
            "root_admission_order": tuple(int(v) for v in self.root_admission_order),
            "root_simulation_allocation": tuple(int(v) for v in self.root_simulation_allocation),
            "visit_counts": tuple(int(v) for v in self.visit_counts),
            "q_values": tuple(float(v) for v in self.q_values),
            "completed_q_values": tuple(float(v) for v in self.completed_q_values),
            "target_support_flags": target_flags,
            "terminal_equivalence_flags": terminal_flags,
        }
        for name, values in arrays.items():
            if len(values) != len(candidates):
                raise ValueError(
                    f"V1 {name} length {len(values)} must match candidate_pairs length {len(candidates)}"
                )
        for name in ("root_gumbel_values", "q_values", "completed_q_values"):
            if any(not np.isfinite(value) for value in arrays[name]):
                raise ValueError(f"V1 {name} must contain finite values")
        for name in ("root_admission_order", "root_simulation_allocation", "visit_counts"):
            if any(int(value) < 0 for value in arrays[name]):
                raise ValueError(f"V1 {name} cannot contain negative values")
        for idx, candidate in enumerate(candidates):
            flags = target_flags[idx]
            if flags != candidate.target_support_flags:
                raise ValueError("V1 target_support_flags must mirror candidate target_support_flags")
            if terminal_flags[idx] != candidate.terminal_equivalence_flag:
                raise ValueError(
                    "V1 terminal_equivalence_flags must mirror candidate terminal_equivalence_flag"
                )
            if (
                "explicit_negative" in flags
                and self.support_type != "admitted_candidate_set_with_explicit_negatives"
            ):
                raise ValueError("V1 explicit negatives require support_type with explicit negatives")
            if "unsampled" in flags and "explicit_negative" in flags:
                raise ValueError("V1 unsampled legal pairs cannot be explicit negatives")
        surprise = {str(key): float(value) for key, value in self.search_surprise_metrics.items()}
        if any(not np.isfinite(value) for value in surprise.values()):
            raise ValueError("V1 search_surprise_metrics must be finite")
        calls = _finite_optional(
            self.neural_calls_per_expanded_full_turn_node,
            "neural_calls_per_expanded_full_turn_node",
        )
        if calls is not None and calls < 0.0:
            raise ValueError("V1 neural_calls_per_expanded_full_turn_node cannot be negative")
        refills = tuple(
            event
            if isinstance(event, V1ReservoirRefillEvent)
            else V1ReservoirRefillEvent.from_dict(event)
            for event in self.reservoir_refill_events
        )
        object.__setattr__(self, "legal_pair_count", int(self.legal_pair_count))
        object.__setattr__(self, "legal_row_schema_version", int(self.legal_row_schema_version))
        object.__setattr__(self, "pair_row_schema_version", int(self.pair_row_schema_version))
        object.__setattr__(self, "candidate_pairs", candidates)
        object.__setattr__(self, "proposal_correction_parameters", correction)
        object.__setattr__(self, "root_gumbel_values", arrays["root_gumbel_values"])
        object.__setattr__(self, "root_admission_order", arrays["root_admission_order"])
        object.__setattr__(self, "root_simulation_allocation", arrays["root_simulation_allocation"])
        object.__setattr__(self, "visit_counts", arrays["visit_counts"])
        object.__setattr__(self, "q_values", arrays["q_values"])
        object.__setattr__(self, "completed_q_values", arrays["completed_q_values"])
        object.__setattr__(self, "selected_pair", selected)
        object.__setattr__(self, "target_support_flags", target_flags)
        object.__setattr__(self, "terminal_equivalence_flags", terminal_flags)
        object.__setattr__(self, "terminal_tactical_payload", tactical_payload)
        object.__setattr__(self, "search_surprise_metrics", surprise)
        object.__setattr__(self, "neural_calls_per_expanded_full_turn_node", calls)
        object.__setattr__(self, "reservoir_refill_events", refills)
        object.__setattr__(self, "schema_version", V1_PAIR_SEARCH_SCHEMA_VERSION)

    def target_support_for_pair(self, pair: PairKey) -> Tuple[str, ...]:
        key = _canonical_pair_key(*pair)
        for idx, candidate in enumerate(self.candidate_pairs):
            if candidate.pair_key == key:
                return self.target_support_flags[idx]
        return ()

    def is_pair_explicit_negative(self, pair: PairKey) -> bool:
        return "explicit_negative" in self.target_support_for_pair(pair)

    def explicit_negative_pairs(self) -> Tuple[PairKey, ...]:
        return tuple(
            candidate.pair_key
            for idx, candidate in enumerate(self.candidate_pairs)
            if "explicit_negative" in self.target_support_flags[idx]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_selector_version": self.candidate_selector_version,
            "support_type": self.support_type,
            "legal_pair_count": self.legal_pair_count,
            "legal_row_schema_version": self.legal_row_schema_version,
            "pair_row_schema_version": self.pair_row_schema_version,
            "candidate_pairs": [candidate.to_dict() for candidate in self.candidate_pairs],
            "proposal_correction_parameters": self.proposal_correction_parameters.to_dict(),
            "root_gumbel_values": list(self.root_gumbel_values),
            "root_admission_order": list(self.root_admission_order),
            "root_simulation_allocation": list(self.root_simulation_allocation),
            "visit_counts": list(self.visit_counts),
            "q_values": list(self.q_values),
            "completed_q_values": list(self.completed_q_values),
            "selected_pair": None
            if self.selected_pair is None
            else [list(self.selected_pair[0]), list(self.selected_pair[1])],
            "target_support_flags": [list(flags) for flags in self.target_support_flags],
            "terminal_equivalence_flags": list(self.terminal_equivalence_flags),
            "terminal_tactical_payload": self.terminal_tactical_payload.to_dict(),
            "search_surprise_metrics": dict(self.search_surprise_metrics),
            "neural_calls_per_expanded_full_turn_node": self.neural_calls_per_expanded_full_turn_node,
            "reservoir_refill_events": [event.to_dict() for event in self.reservoir_refill_events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "V1SearchPairMetadata":
        schema_version = int(data.get("schema_version", 0))
        if schema_version >= V1_PAIR_SEARCH_SCHEMA_VERSION and "terminal_tactical_payload" not in data:
            raise ValueError("V1 schema 2 metadata requires terminal_tactical_payload")
        selected_raw = data.get("selected_pair")
        selected_pair = None
        if selected_raw is not None:
            if len(selected_raw) != 2:
                raise ValueError("V1 selected_pair must contain two coordinates")
            selected_pair = (_pair_coord(selected_raw[0]), _pair_coord(selected_raw[1]))
        return cls(
            schema_version=schema_version,
            candidate_selector_version=str(data.get("candidate_selector_version", "")),
            support_type=str(data.get("support_type", "")),
            legal_pair_count=int(data.get("legal_pair_count", -1)),
            legal_row_schema_version=int(data.get("legal_row_schema_version", 0)),
            pair_row_schema_version=int(data.get("pair_row_schema_version", 0)),
            candidate_pairs=tuple(
                V1CandidatePair.from_dict(item)
                for item in data.get("candidate_pairs", [])
            ),
            proposal_correction_parameters=V1ProposalCorrectionParameters.from_dict(
                data.get("proposal_correction_parameters", {})
            ),
            root_gumbel_values=tuple(data.get("root_gumbel_values", ())),
            root_admission_order=tuple(data.get("root_admission_order", ())),
            root_simulation_allocation=tuple(data.get("root_simulation_allocation", ())),
            visit_counts=tuple(data.get("visit_counts", ())),
            q_values=tuple(data.get("q_values", ())),
            completed_q_values=tuple(data.get("completed_q_values", ())),
            selected_pair=selected_pair,
            target_support_flags=tuple(tuple(flags) for flags in data.get("target_support_flags", ())),
            terminal_equivalence_flags=tuple(data.get("terminal_equivalence_flags", ())),
            terminal_tactical_payload=V1TerminalTacticalPayload.from_mapping(
                data.get("terminal_tactical_payload", {})
            ),
            search_surprise_metrics=dict(data.get("search_surprise_metrics", {})),
            neural_calls_per_expanded_full_turn_node=data.get("neural_calls_per_expanded_full_turn_node"),
            reservoir_refill_events=tuple(
                V1ReservoirRefillEvent.from_dict(item)
                for item in data.get("reservoir_refill_events", ())
            ),
        )


def _v1_flag_mask(flags: Tuple[str, ...]) -> int:
    mask = 0
    for bit, name in enumerate(_V1_SUPPORT_FLAG_ORDER):
        if name in flags:
            mask |= 1 << bit
    return mask


def _v1_flags_from_mask(mask: int) -> Tuple[str, ...]:
    return tuple(name for bit, name in enumerate(_V1_SUPPORT_FLAG_ORDER) if int(mask) & (1 << bit))


def _v1_pack_array(values: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    arr = np.asarray(values)
    return (arr.dtype.str, tuple(int(dim) for dim in arr.shape), arr.tobytes(order="C"))


def _v1_unpack_array(payload: tuple[str, tuple[int, ...], bytes]) -> np.ndarray:
    dtype, shape, blob = payload
    return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(tuple(shape))


def _v1_encode_compact_payload(metadata: V1SearchPairMetadata) -> tuple[Any, ...]:
    strings: list[str] = [""]
    string_ids: dict[str, int] = {"": 0}

    def sid(value: Any) -> int:
        text = "" if value is None else str(value)
        idx = string_ids.get(text)
        if idx is None:
            idx = len(strings)
            strings.append(text)
            string_ids[text] = idx
        return idx

    n = len(metadata.candidate_pairs)
    pair_qr = np.zeros((n, 4), dtype=np.int16)
    legal_row_ids = np.zeros((n, 2), dtype=np.int32)
    row_schema = np.zeros(n, dtype=np.uint16)
    bool_flags = np.zeros(n, dtype=np.uint8)
    target_masks = np.zeros(n, dtype=np.uint16)
    admissions = np.zeros(n, dtype=np.uint16)
    roots = np.zeros(n, dtype=np.uint8)
    candidate_ids = np.zeros(n, dtype=np.uint16)
    selection_reasons = np.zeros(n, dtype=np.uint16)
    source_payloads: list[tuple[tuple[Any, ...], ...]] = []
    proposal_payloads: list[tuple[Any, ...]] = []
    for idx, candidate in enumerate(metadata.candidate_pairs):
        pair_qr[idx] = (
            int(candidate.pair_key[0][0]),
            int(candidate.pair_key[0][1]),
            int(candidate.pair_key[1][0]),
            int(candidate.pair_key[1][1]),
        )
        legal_row_ids[idx] = (int(candidate.first_legal_row_id), int(candidate.second_legal_row_id))
        row_schema[idx] = int(candidate.row_table_schema_version)
        flags = 0
        if candidate.forced_exploration_flag:
            flags |= 1
        if candidate.terminal_exact_flag:
            flags |= 2
        if candidate.terminal_equivalence_flag:
            flags |= 4
        bool_flags[idx] = flags
        target_masks[idx] = _v1_flag_mask(candidate.target_support_flags)
        admissions[idx] = int(candidate.admission_generation)
        roots[idx] = 1 if candidate.root_or_interior == "root" else 0
        candidate_ids[idx] = sid(candidate.candidate_id)
        selection_reasons[idx] = sid(candidate.candidate_selection_reason)
        source_payloads.append(
            tuple(
                (
                    sid(source.source_type),
                    -1 if source.source_rank is None else int(source.source_rank),
                    np.nan if source.source_weight is None else float(source.source_weight),
                    np.nan
                    if source.local_probability_or_score is None
                    else float(source.local_probability_or_score),
                    sid(source.quota_id),
                    sid(source.inclusion_kind),
                    np.nan
                    if source.exact_inclusion_probability is None
                    else float(source.exact_inclusion_probability),
                    np.nan if source.heuristic_propensity is None else float(source.heuristic_propensity),
                    sid(source.correction_mode),
                )
                for source in candidate.source_contributions
            )
        )
        proposal = candidate.proposal_propensity_metadata
        proposal_payloads.append(
            (
                sid(proposal.proposal_policy),
                sid(proposal.correction_mode),
                np.nan
                if proposal.total_proposal_probability is None
                else float(proposal.total_proposal_probability),
                np.nan
                if proposal.log_proposal_probability is None
                else float(proposal.log_proposal_probability),
                1 if proposal.sampling_without_replacement else 0,
                sid(proposal.notes),
            )
        )

    selected = None
    if metadata.selected_pair is not None:
        selected = (
            int(metadata.selected_pair[0][0]),
            int(metadata.selected_pair[0][1]),
            int(metadata.selected_pair[1][0]),
            int(metadata.selected_pair[1][1]),
        )
    correction = metadata.proposal_correction_parameters
    scalar_payload = (
        sid(metadata.candidate_selector_version),
        sid(metadata.support_type),
        int(metadata.legal_pair_count),
        int(metadata.legal_row_schema_version),
        int(metadata.pair_row_schema_version),
        int(metadata.schema_version),
    )
    correction_payload = (
        sid(correction.correction_mode),
        float(correction.min_log),
        float(correction.max_log),
        float(correction.prior_temperature),
    )
    search_array_payload = (
        _v1_pack_array(np.asarray(metadata.root_gumbel_values, dtype=np.float32)),
        _v1_pack_array(np.asarray(metadata.root_admission_order, dtype=np.uint16)),
        _v1_pack_array(np.asarray(metadata.root_simulation_allocation, dtype=np.uint32)),
        _v1_pack_array(np.asarray(metadata.visit_counts, dtype=np.uint32)),
        _v1_pack_array(np.asarray(metadata.q_values, dtype=np.float32)),
        _v1_pack_array(np.asarray(metadata.completed_q_values, dtype=np.float32)),
        _v1_pack_array(np.asarray([_v1_flag_mask(flags) for flags in metadata.target_support_flags], dtype=np.uint16)),
        _v1_pack_array(np.asarray(metadata.terminal_equivalence_flags, dtype=np.bool_)),
    )
    surprise_payload = tuple(
        (sid(key), float(value)) for key, value in sorted(metadata.search_surprise_metrics.items())
    )
    neural_payload = (
        np.nan
        if metadata.neural_calls_per_expanded_full_turn_node is None
        else float(metadata.neural_calls_per_expanded_full_turn_node)
    )
    refill_payload = tuple(
        (
            sid(event.node_id),
            sid(event.reason),
            int(event.generation),
            int(event.requested_count),
            int(event.added_count),
        )
        for event in metadata.reservoir_refill_events
    )
    tactical_payload = json.dumps(
        metadata.terminal_tactical_payload.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    tactical_payload_id = sid(tactical_payload)
    return (
        V1_PAIR_SEARCH_COMPACT_VERSION,
        tuple(strings),
        scalar_payload,
        (
            _v1_pack_array(pair_qr),
            _v1_pack_array(legal_row_ids),
            _v1_pack_array(row_schema),
            _v1_pack_array(bool_flags),
            _v1_pack_array(target_masks),
            _v1_pack_array(admissions),
            _v1_pack_array(roots),
            _v1_pack_array(candidate_ids),
            _v1_pack_array(selection_reasons),
        ),
        tuple(source_payloads),
        tuple(proposal_payloads),
        correction_payload,
        search_array_payload,
        selected,
        surprise_payload,
        neural_payload,
        refill_payload,
        tactical_payload_id,
    )


def _v1_decode_compact_payload(payload: tuple[Any, ...]) -> V1SearchPairMetadata:
    (
        compact_version,
        strings,
        scalars,
        candidate_arrays,
        source_payloads,
        proposal_payloads,
        correction_payload,
        search_arrays,
        selected,
        surprise_payload,
        neural_calls,
        refill_payloads,
        tactical_payload_sid,
    ) = payload
    if int(compact_version) != V1_PAIR_SEARCH_COMPACT_VERSION:
        raise ValueError(f"Unsupported V1 compact metadata version {compact_version}")

    def s(idx: int) -> str:
        return str(strings[int(idx)])

    (
        pair_qr,
        legal_row_ids,
        row_schema,
        bool_flags,
        target_masks,
        admissions,
        roots,
        candidate_ids,
        selection_reasons,
    ) = (_v1_unpack_array(item) for item in candidate_arrays)
    candidates = []
    for idx in range(int(pair_qr.shape[0])):
        source_contributions = tuple(
            V1CandidateSourceContribution(
                source_type=s(source[0]),
                source_rank=None if int(source[1]) < 0 else int(source[1]),
                source_weight=None if np.isnan(float(source[2])) else float(source[2]),
                local_probability_or_score=None if np.isnan(float(source[3])) else float(source[3]),
                quota_id=None if int(source[4]) == 0 else s(source[4]),
                inclusion_kind=s(source[5]),
                exact_inclusion_probability=None if np.isnan(float(source[6])) else float(source[6]),
                heuristic_propensity=None if np.isnan(float(source[7])) else float(source[7]),
                correction_mode=s(source[8]),
            )
            for source in source_payloads[idx]
        )
        proposal = proposal_payloads[idx]
        proposal_metadata = V1ProposalPropensityMetadata(
            proposal_policy=s(proposal[0]),
            correction_mode=s(proposal[1]),
            total_proposal_probability=None if np.isnan(float(proposal[2])) else float(proposal[2]),
            log_proposal_probability=None if np.isnan(float(proposal[3])) else float(proposal[3]),
            sampling_without_replacement=bool(proposal[4]),
            notes=s(proposal[5]),
        )
        flags = int(bool_flags[idx])
        candidates.append(
            V1CandidatePair(
                candidate_id=s(candidate_ids[idx]),
                pair_key=(
                    (int(pair_qr[idx, 0]), int(pair_qr[idx, 1])),
                    (int(pair_qr[idx, 2]), int(pair_qr[idx, 3])),
                ),
                first_legal_row_id=int(legal_row_ids[idx, 0]),
                second_legal_row_id=int(legal_row_ids[idx, 1]),
                row_table_schema_version=int(row_schema[idx]),
                source_contributions=source_contributions,
                proposal_propensity_metadata=proposal_metadata,
                forced_exploration_flag=bool(flags & 1),
                terminal_exact_flag=bool(flags & 2),
                terminal_equivalence_flag=bool(flags & 4),
                target_support_flags=_v1_flags_from_mask(int(target_masks[idx])),
                admission_generation=int(admissions[idx]),
                root_or_interior="root" if int(roots[idx]) == 1 else "interior",
                candidate_selection_reason=s(selection_reasons[idx]),
            )
        )

    selected_pair = None
    if selected is not None:
        selected_pair = ((int(selected[0]), int(selected[1])), (int(selected[2]), int(selected[3])))
    correction = V1ProposalCorrectionParameters(
        correction_mode=s(correction_payload[0]),
        min_log=float(correction_payload[1]),
        max_log=float(correction_payload[2]),
        prior_temperature=float(correction_payload[3]),
    )
    (
        root_gumbels,
        admission_order,
        simulation_alloc,
        visit_counts,
        q_values,
        completed_q,
        target_masks_arr,
        terminal_flags,
    ) = (_v1_unpack_array(item) for item in search_arrays)
    selector_id, support_id, legal_pair_count, legal_schema, pair_schema, schema_version = scalars
    return V1SearchPairMetadata(
        schema_version=int(schema_version),
        candidate_selector_version=s(selector_id),
        support_type=s(support_id),
        legal_pair_count=int(legal_pair_count),
        legal_row_schema_version=int(legal_schema),
        pair_row_schema_version=int(pair_schema),
        candidate_pairs=tuple(candidates),
        proposal_correction_parameters=correction,
        root_gumbel_values=tuple(float(value) for value in root_gumbels.tolist()),
        root_admission_order=tuple(int(value) for value in admission_order.tolist()),
        root_simulation_allocation=tuple(int(value) for value in simulation_alloc.tolist()),
        visit_counts=tuple(int(value) for value in visit_counts.tolist()),
        q_values=tuple(float(value) for value in q_values.tolist()),
        completed_q_values=tuple(float(value) for value in completed_q.tolist()),
        selected_pair=selected_pair,
        target_support_flags=tuple(_v1_flags_from_mask(int(mask)) for mask in target_masks_arr.tolist()),
        terminal_equivalence_flags=tuple(bool(value) for value in terminal_flags.tolist()),
        terminal_tactical_payload=V1TerminalTacticalPayload.from_mapping(
            json.loads(s(tactical_payload_sid))
        ),
        search_surprise_metrics={s(key): float(value) for key, value in surprise_payload},
        neural_calls_per_expanded_full_turn_node=None
        if np.isnan(float(neural_calls))
        else float(neural_calls),
        reservoir_refill_events=tuple(
            V1ReservoirRefillEvent(
                node_id=s(event[0]),
                reason=s(event[1]),
                generation=int(event[2]),
                requested_count=int(event[3]),
                added_count=int(event[4]),
            )
            for event in refill_payloads
        ),
    )


def v1_search_metadata_to_compact_bytes(
    metadata: Optional[V1SearchPairMetadata],
    *,
    compression: str | bool = "zlib",
) -> bytes:
    if metadata is None:
        return b""
    payload = pickle.dumps(_v1_encode_compact_payload(metadata), protocol=5)
    compression_name = "zlib" if compression is True else str(compression or "none").lower()
    if compression_name in {"1", "true", "enabled"}:
        compression_name = "zlib"
    if compression_name == "zlib":
        payload = zlib.compress(payload, level=6)
        return V1_PAIR_SEARCH_COMPACT_MAGIC + struct.pack("<BB", V1_PAIR_SEARCH_COMPACT_VERSION, V1_PAIR_SEARCH_COMPRESSION_ZLIB) + payload
    if compression_name in {"none", "false", "0", "disabled"}:
        return V1_PAIR_SEARCH_COMPACT_MAGIC + struct.pack("<BB", V1_PAIR_SEARCH_COMPACT_VERSION, 0) + payload
    raise ValueError(f"Unsupported V1 metadata compression mode {compression!r}")


def v1_search_metadata_from_compact_bytes(blob: bytes | None) -> Optional[V1SearchPairMetadata]:
    if not blob:
        return None
    if not blob.startswith(V1_PAIR_SEARCH_COMPACT_MAGIC):
        raise ValueError("Unsupported legacy V1 metadata payload: compact schema v2 is required")
    if len(blob) < len(V1_PAIR_SEARCH_COMPACT_MAGIC) + 2:
        raise ValueError("Truncated V1 compact metadata header")
    version, compression_id = struct.unpack_from("<BB", blob, len(V1_PAIR_SEARCH_COMPACT_MAGIC))
    if int(version) != V1_PAIR_SEARCH_COMPACT_VERSION:
        raise ValueError(f"Unsupported V1 compact metadata version {version}")
    payload = blob[len(V1_PAIR_SEARCH_COMPACT_MAGIC) + 2 :]
    if int(compression_id) == V1_PAIR_SEARCH_COMPRESSION_ZLIB:
        payload = zlib.decompress(payload)
    elif int(compression_id) != 0:
        raise ValueError(f"Unsupported V1 compact metadata compression id {compression_id}")
    return _v1_decode_compact_payload(pickle.loads(payload))


@dataclass
class PositionRecord:
    """One position from a self-play game — data needed for training."""

    # Compact move history: flat bytes of (player:i32, q:i32, r:i32) LE triples.
    # Encodes all moves played so far (from initial empty board).
    # Rust's encode_compact_record replays this into (13,33,33) tensors on demand.
    move_history: bytes

    # Sparse policy target: maps action index (flat BOARD_AREA index: q*33 + r + offset)
    # to visit probability. Top-K only to save space and prune low-visit noise.
    policy_target: Dict[int, float]

    # Root Q-value from MCTS (from current player's perspective).
    root_value: float

    # Which player generated this record (0 or 1).
    player: int

    # Game outcome from P0's perspective. None until game ends.
    # 1.0 = P0 wins, -1.0 = P1 wins.
    outcome: Optional[float] = None

    # Unique game identifier for recency-weighted sampling.
    game_id: int = 0

    # Whether this position was generated with full MCTS sims (True) or
    # low-sim PCR (Playout Cap Randomization). PCR samples get lower weight.
    is_full_search: bool = True

    # Turn index within the game (0-based). Used for temperature schedule lookup.
    turn_index: int = 0

    # MCTS value of the selected action from the acting player's perspective.
    # Used by RGSC Eq. 2; missing values are not valid regret targets.
    selected_action_value: Optional[float] = None

    # Lookahead value targets at multiple horizons (KataGo-style).
    lookahead_values: List[float] = field(default_factory=list)
    opp_policy_target: Dict[int, float] = field(default_factory=dict)
    opp_policy_weight: float = 0.0
    policy_target_v2: PolicyTargetV2 = field(default_factory=list)
    opp_policy_target_v2: PolicyTargetV2 = field(default_factory=list)
    opp_policy_legal_v2: List[Tuple[int, int]] = field(default_factory=list)
    pair_policy_target_v2: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = field(default_factory=list)
    pair_policy_complete: bool = False
    v1_search_metadata: Optional[V1SearchPairMetadata] = None
    target_policy_mass_outside_window: float = 0.0
    missing_target_policy_mass: float = 0.0
    candidate_recall_mcts_top1: float = 1.0
    candidate_recall_mcts_top4: float = 1.0
    candidate_recall_mcts_top8: float = 1.0
    candidate_recall_winning_move: float = 1.0
    candidate_recall_forced_block: float = 1.0
    candidate_recall_two_placement_cover: float = 1.0
    candidate_discovery_top1: float = 1.0
    candidate_discovery_top4: float = 1.0
    candidate_discovery_top8: float = 1.0
    candidate_discovery_winning_move: float = 1.0
    candidate_discovery_forced_block: float = 1.0
    candidate_discovery_two_placement_cover: float = 1.0
    candidate_discovery_open_four: float = 1.0
    candidate_discovery_open_five: float = 1.0
    candidate_critical_count: int = 0
    candidate_critical_overflow_count: int = 0
    candidate_critical_overflow_examples: Tuple[Tuple[int, int], ...] = ()
    sparse_prior_stage: int = 0
    sparse_prior_root_candidate_count: int = 0
    sparse_prior_leaf_candidate_count: float = 0.0
    sparse_prior_root_hit_frac: float = 0.0
    sparse_prior_leaf_hit_frac: float = 0.0
    fallback_prior_use: float = 0.0
    fallback_prior_use_on_mcts_top1: float = 0.0
    fallback_prior_use_on_mcts_top4: float = 0.0
    fallback_prior_use_on_mcts_top8: float = 0.0
    sparse_vs_dense_disagreement: float = 0.0
    sparse_prior_forward_ms: float = 0.0
    sparse_prior_candidate_build_ms: float = 0.0
    pair_prior_candidate_count: int = 0
    pair_prior_hit_frac: float = 0.0
    pair_fallback_prior_use: float = 0.0
    pair_fallback_prior_use_on_mcts_top1: float = 0.0
    pair_fallback_prior_use_on_mcts_top4: float = 0.0
    pair_fallback_prior_use_on_mcts_top8: float = 0.0
    regret_rank: float = 0.0
    regret_value: float = 0.0
    regret_weight: float = 0.0
    axis_label: int = -1
    moves_left: float = 0.0
    value_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.v1_search_metadata is None:
            return
        metadata = (
            self.v1_search_metadata
            if isinstance(self.v1_search_metadata, V1SearchPairMetadata)
            else V1SearchPairMetadata.from_dict(self.v1_search_metadata)
        )
        if self.pair_policy_target_v2:
            raise ValueError(
                "V1 pair search metadata must not be mixed with legacy pair_policy_target_v2 "
                "until V1 pair training consumers are implemented"
            )
        self.v1_search_metadata = metadata
        self.pair_policy_complete = False

    def to_value_target(self) -> float:
        """Compute the training value target for this position.

        From the current player's perspective:
          - If current player == P0: target = outcome
          - If current player == P1: target = -outcome
        """
        if self.outcome is None:
            return 0.0
        return self.outcome if self.player == 0 else -self.outcome

    def to_dense_policy(self) -> np.ndarray:
        """Convert sparse policy target to dense (BOARD_AREA,) float32 array."""
        dense = np.zeros(BOARD_AREA, dtype=np.float32)
        for idx, prob in self.policy_target.items():
            if 0 <= idx < BOARD_AREA:
                dense[idx] = prob
        return dense

    def to_dense_opp_policy(self) -> np.ndarray:
        """Convert sparse opponent-policy target to dense (BOARD_AREA,) float32 array."""
        dense = np.zeros(BOARD_AREA, dtype=np.float32)
        for idx, prob in self.opp_policy_target.items():
            if 0 <= idx < BOARD_AREA:
                dense[idx] = prob
        return dense


@dataclass
class GameRecord:
    """Complete record of one self-play game.

    Contains all positions played, plus the final outcome.
    """

    # All positions in this game (one per move, except terminal state).
    positions: List[PositionRecord] = field(default_factory=list)

    # Final game outcome from P0's perspective.
    outcome: float = 0.0

    # Unique game ID (monotonic counter).
    game_id: int = 0

    # Total number of placements in the game.
    game_length: int = 0

    # Complete placement history including the terminal move. Position histories
    # remain prefixes before each decision.
    final_move_history: bytes = b""

    # True when the game stopped because the move cap or another non-terminal
    # guard fired before either player won.
    truncated: bool = False
    terminal_reason: str = "unknown"
    rgsc_restart_attempted: bool = False
    rgsc_restart_used: bool = False
    rgsc_restart_reason: str = "disabled"
    rgsc_restart_entry_index: Optional[int] = None
    rgsc_restart_entry_id: Optional[int] = None
    rgsc_restart_move_count: int = 0
    rgsc_prb_inserted: bool = False
    rgsc_metrics: Dict[str, float] = field(default_factory=dict)

    def assign_outcomes(self):
        """Assign the game outcome to all positions."""
        for pos in self.positions:
            pos.outcome = self.outcome

    def to_compact_bytes(self) -> bytes:
        """Serialize the game record into compact bytes for buffer storage.

        V2 records start with a magic/version prefix. from_compact_bytes still
        accepts legacy records that started directly with game_id/outcome.
        """
        parts = bytearray()

        # Header
        parts.extend(COMPACT_MAGIC_V2)
        parts.extend(struct.pack("<HIfI", COMPACT_VERSION_V2, self.game_id, self.outcome, len(self.positions)))

        for pos in self.positions:
            # Move history
            parts.extend(struct.pack("<I", len(pos.move_history)))
            parts.extend(pos.move_history)

            # Flags
            parts.extend(struct.pack("<BB", pos.player, int(pos.is_full_search)))

            # Root value
            parts.extend(struct.pack("<f", pos.root_value))
            parts.extend(struct.pack(
                "<f",
                float("nan") if pos.selected_action_value is None else float(pos.selected_action_value),
            ))

            # Policy target (legacy dense-crop sparse)
            entries = list(pos.policy_target.items())
            parts.extend(struct.pack("<H", len(entries)))
            for idx, prob in entries:
                parts.extend(struct.pack("<Hf", idx, prob))

            # Turn index
            parts.extend(struct.pack("<I", pos.turn_index))

            # Auxiliary targets
            opp_entries = list(pos.opp_policy_target.items())
            parts.extend(struct.pack("<H", len(opp_entries)))
            for idx, prob in opp_entries:
                parts.extend(struct.pack("<Hf", idx, prob))
            v2_entries = list(pos.policy_target_v2)
            parts.extend(struct.pack("<H", len(v2_entries)))
            for q, r, prob in v2_entries:
                parts.extend(struct.pack("<iif", int(q), int(r), float(prob)))
            opp_v2_entries = list(pos.opp_policy_target_v2)
            parts.extend(struct.pack("<H", len(opp_v2_entries)))
            for q, r, prob in opp_v2_entries:
                parts.extend(struct.pack("<iif", int(q), int(r), float(prob)))
            opp_legal_v2_entries = list(pos.opp_policy_legal_v2)
            parts.extend(struct.pack("<H", len(opp_legal_v2_entries)))
            for q, r in opp_legal_v2_entries:
                parts.extend(struct.pack("<ii", int(q), int(r)))
            pair_v2_entries = [] if pos.v1_search_metadata is not None else list(pos.pair_policy_target_v2)
            parts.extend(struct.pack("<H", len(pair_v2_entries)))
            for first, second, prob in pair_v2_entries:
                q1, r1 = first
                q2, r2 = second
                parts.extend(struct.pack("<iiiif", int(q1), int(r1), int(q2), int(r2), float(prob)))
            parts.extend(struct.pack("<B", int(bool(pos.pair_policy_complete and pos.v1_search_metadata is None))))
            parts.extend(struct.pack(
                "<ffffffff",
                float(pos.target_policy_mass_outside_window),
                float(pos.missing_target_policy_mass),
                float(pos.candidate_recall_mcts_top1),
                float(pos.candidate_recall_mcts_top4),
                float(pos.candidate_recall_mcts_top8),
                float(pos.candidate_recall_winning_move),
                float(pos.candidate_recall_forced_block),
                float(pos.candidate_recall_two_placement_cover),
            ))
            parts.extend(struct.pack(
                "<ffhf",
                pos.regret_rank,
                pos.regret_value,
                pos.axis_label,
                pos.moves_left,
            ))
            parts.extend(struct.pack("<f", float(pos.opp_policy_weight)))
            parts.extend(struct.pack("<f", float(pos.value_weight)))
            parts.extend(struct.pack("<f", float(pos.regret_weight)))
            parts.extend(struct.pack(
                "<18f",
                float(pos.sparse_prior_stage),
                float(pos.sparse_prior_root_candidate_count),
                float(pos.sparse_prior_leaf_candidate_count),
                float(pos.sparse_prior_root_hit_frac),
                float(pos.sparse_prior_leaf_hit_frac),
                float(pos.fallback_prior_use),
                float(pos.fallback_prior_use_on_mcts_top1),
                float(pos.fallback_prior_use_on_mcts_top4),
                float(pos.fallback_prior_use_on_mcts_top8),
                float(pos.sparse_vs_dense_disagreement),
                float(pos.sparse_prior_forward_ms),
                float(pos.sparse_prior_candidate_build_ms),
                float(pos.pair_prior_candidate_count),
                float(pos.pair_prior_hit_frac),
                float(pos.pair_fallback_prior_use),
                float(pos.pair_fallback_prior_use_on_mcts_top1),
                float(pos.pair_fallback_prior_use_on_mcts_top4),
                float(pos.pair_fallback_prior_use_on_mcts_top8),
            ))
            discovery_examples = list(pos.candidate_critical_overflow_examples[:8])
            parts.extend(struct.pack(
                "<8fHHH",
                float(pos.candidate_discovery_top1),
                float(pos.candidate_discovery_top4),
                float(pos.candidate_discovery_top8),
                float(pos.candidate_discovery_winning_move),
                float(pos.candidate_discovery_forced_block),
                float(pos.candidate_discovery_two_placement_cover),
                float(pos.candidate_discovery_open_four),
                float(pos.candidate_discovery_open_five),
                int(pos.candidate_critical_count),
                int(pos.candidate_critical_overflow_count),
                len(discovery_examples),
            ))
            for q, r in discovery_examples:
                parts.extend(struct.pack("<ii", int(q), int(r)))

            v1_metadata_blob = v1_search_metadata_to_compact_bytes(pos.v1_search_metadata)
            parts.extend(struct.pack("<I", len(v1_metadata_blob)))
            parts.extend(v1_metadata_blob)

        return bytes(parts)

    @staticmethod
    def from_compact_bytes(data: bytes) -> "GameRecord":
        """Deserialize a game record from compact bytes."""
        offset = 0

        is_v2 = data[:4] == COMPACT_MAGIC_V2
        if is_v2:
            offset += 4
            version, game_id, outcome, num_pos = struct.unpack_from("<HIfI", data, offset)
            offset += struct.calcsize("<HIfI")
            if not (COMPACT_VERSION_MIN <= version <= COMPACT_VERSION_V2):
                raise ValueError(f"Unsupported compact GameRecord version {version}")
        else:
            game_id = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            outcome = struct.unpack_from("<f", data, offset)[0]
            offset += 4
            num_pos = struct.unpack_from("<I", data, offset)[0]
            offset += 4

        positions = []
        for _ in range(num_pos):
            # Move history
            mh_len = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            move_history = data[offset:offset + mh_len]
            offset += mh_len

            # Flags
            player = data[offset]
            offset += 1
            is_full = bool(data[offset])
            offset += 1

            # Root value
            root_value = struct.unpack_from("<f", data, offset)[0]
            offset += 4
            selected_action_value: Optional[float] = None
            if is_v2 and version >= 4:
                selected_action_value_raw = struct.unpack_from("<f", data, offset)[0]
                offset += 4
                if np.isfinite(selected_action_value_raw):
                    selected_action_value = float(selected_action_value_raw)

            # Policy target
            num_entries = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            policy = {}
            for _ in range(num_entries):
                idx = struct.unpack_from("<H", data, offset)[0]
                offset += 2
                prob = struct.unpack_from("<f", data, offset)[0]
                offset += 4
                policy[idx] = prob

            # Turn index
            turn_idx = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            opp_policy = {}
            policy_v2: PolicyTargetV2 = []
            opp_policy_v2: PolicyTargetV2 = []
            opp_policy_legal_v2: List[Tuple[int, int]] = []
            pair_policy_v2: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = []
            pair_policy_complete = False
            target_policy_mass_outside_window = 0.0
            missing_target_policy_mass = 0.0
            candidate_recall_mcts_top1 = 1.0
            candidate_recall_mcts_top4 = 1.0
            candidate_recall_mcts_top8 = 1.0
            candidate_recall_winning_move = 1.0
            candidate_recall_forced_block = 1.0
            candidate_recall_two_placement_cover = 1.0
            candidate_discovery_top1 = 1.0
            candidate_discovery_top4 = 1.0
            candidate_discovery_top8 = 1.0
            candidate_discovery_winning_move = 1.0
            candidate_discovery_forced_block = 1.0
            candidate_discovery_two_placement_cover = 1.0
            candidate_discovery_open_four = 1.0
            candidate_discovery_open_five = 1.0
            candidate_critical_count = 0
            candidate_critical_overflow_count = 0
            candidate_critical_overflow_examples: Tuple[Tuple[int, int], ...] = ()
            sparse_prior_stage = 0
            sparse_prior_root_candidate_count = 0
            sparse_prior_leaf_candidate_count = 0.0
            sparse_prior_root_hit_frac = 0.0
            sparse_prior_leaf_hit_frac = 0.0
            fallback_prior_use = 0.0
            fallback_prior_use_on_mcts_top1 = 0.0
            fallback_prior_use_on_mcts_top4 = 0.0
            fallback_prior_use_on_mcts_top8 = 0.0
            sparse_vs_dense_disagreement = 0.0
            sparse_prior_forward_ms = 0.0
            sparse_prior_candidate_build_ms = 0.0
            pair_prior_candidate_count = 0
            pair_prior_hit_frac = 0.0
            pair_fallback_prior_use = 0.0
            pair_fallback_prior_use_on_mcts_top1 = 0.0
            pair_fallback_prior_use_on_mcts_top4 = 0.0
            pair_fallback_prior_use_on_mcts_top8 = 0.0
            regret_rank = 0.0
            regret_value = 0.0
            regret_weight = 0.0
            axis_label = -1
            moves_left = 0.0
            opp_policy_weight = 0.0
            value_weight = 1.0
            v1_search_metadata: Optional[V1SearchPairMetadata] = None
            if offset < len(data):
                num_opp_entries = struct.unpack_from("<H", data, offset)[0]
                offset += 2
                for _ in range(num_opp_entries):
                    idx = struct.unpack_from("<H", data, offset)[0]
                    offset += 2
                    prob = struct.unpack_from("<f", data, offset)[0]
                    offset += 4
                    opp_policy[idx] = prob
                if is_v2:
                    num_v2_entries = struct.unpack_from("<H", data, offset)[0]
                    offset += 2
                    for _ in range(num_v2_entries):
                        q, r, prob = struct.unpack_from("<iif", data, offset)
                        offset += struct.calcsize("<iif")
                        policy_v2.append((int(q), int(r), float(prob)))
                    num_opp_v2_entries = struct.unpack_from("<H", data, offset)[0]
                    offset += 2
                    for _ in range(num_opp_v2_entries):
                        q, r, prob = struct.unpack_from("<iif", data, offset)
                        offset += struct.calcsize("<iif")
                        opp_policy_v2.append((int(q), int(r), float(prob)))
                    if version >= 8:
                        num_opp_legal_v2_entries = struct.unpack_from("<H", data, offset)[0]
                        offset += 2
                        for _ in range(num_opp_legal_v2_entries):
                            q, r = struct.unpack_from("<ii", data, offset)
                            offset += struct.calcsize("<ii")
                            opp_policy_legal_v2.append((int(q), int(r)))
                    if version >= 3:
                        num_pair_v2_entries = struct.unpack_from("<H", data, offset)[0]
                        offset += 2
                        for _ in range(num_pair_v2_entries):
                            q1, r1, q2, r2, prob = struct.unpack_from("<iiiif", data, offset)
                            offset += struct.calcsize("<iiiif")
                            pair_policy_v2.append(((int(q1), int(r1)), (int(q2), int(r2)), float(prob)))
                        if version >= 9:
                            pair_policy_complete = bool(struct.unpack_from("<B", data, offset)[0])
                            offset += 1
                        else:
                            pair_policy_complete = bool(pair_policy_v2)
                        (
                            target_policy_mass_outside_window,
                            missing_target_policy_mass,
                            candidate_recall_mcts_top1,
                            candidate_recall_mcts_top4,
                            candidate_recall_mcts_top8,
                            candidate_recall_winning_move,
                            candidate_recall_forced_block,
                            candidate_recall_two_placement_cover,
                        ) = struct.unpack_from("<ffffffff", data, offset)
                        offset += struct.calcsize("<ffffffff")
                    else:
                        (
                            target_policy_mass_outside_window,
                            missing_target_policy_mass,
                            candidate_recall_mcts_top1,
                            candidate_recall_mcts_top4,
                            candidate_recall_mcts_top8,
                        ) = struct.unpack_from("<fffff", data, offset)
                        offset += struct.calcsize("<fffff")
                regret_rank, regret_value, axis_label, moves_left = struct.unpack_from(
                    "<ffhf", data, offset
                )
                offset += struct.calcsize("<ffhf")
                if is_v2 and version >= 4:
                    opp_policy_weight = struct.unpack_from("<f", data, offset)[0]
                    offset += 4
                elif opp_policy or opp_policy_v2:
                    opp_policy_weight = 1.0
                if is_v2 and version >= 5:
                    value_weight = struct.unpack_from("<f", data, offset)[0]
                    offset += 4
                if is_v2 and version >= 6:
                    regret_weight = struct.unpack_from("<f", data, offset)[0]
                    offset += 4
                if is_v2 and version >= 7:
                    (
                        sparse_prior_stage_f,
                        sparse_prior_root_candidate_count_f,
                        sparse_prior_leaf_candidate_count,
                        sparse_prior_root_hit_frac,
                        sparse_prior_leaf_hit_frac,
                        fallback_prior_use,
                        fallback_prior_use_on_mcts_top1,
                        fallback_prior_use_on_mcts_top4,
                        fallback_prior_use_on_mcts_top8,
                        sparse_vs_dense_disagreement,
                        sparse_prior_forward_ms,
                        sparse_prior_candidate_build_ms,
                        pair_prior_candidate_count_f,
                        pair_prior_hit_frac,
                        pair_fallback_prior_use,
                        pair_fallback_prior_use_on_mcts_top1,
                        pair_fallback_prior_use_on_mcts_top4,
                        pair_fallback_prior_use_on_mcts_top8,
                    ) = struct.unpack_from("<18f", data, offset)
                    offset += struct.calcsize("<18f")
                    sparse_prior_stage = int(sparse_prior_stage_f)
                    sparse_prior_root_candidate_count = int(sparse_prior_root_candidate_count_f)
                    pair_prior_candidate_count = int(pair_prior_candidate_count_f)
                if is_v2 and version >= 8:
                    (
                        candidate_discovery_top1,
                        candidate_discovery_top4,
                        candidate_discovery_top8,
                        candidate_discovery_winning_move,
                        candidate_discovery_forced_block,
                        candidate_discovery_two_placement_cover,
                        candidate_discovery_open_four,
                        candidate_discovery_open_five,
                        candidate_critical_count,
                        candidate_critical_overflow_count,
                        overflow_example_count,
                    ) = struct.unpack_from("<8fHHH", data, offset)
                    offset += struct.calcsize("<8fHHH")
                    examples = []
                    for _ in range(overflow_example_count):
                        q, r = struct.unpack_from("<ii", data, offset)
                        offset += struct.calcsize("<ii")
                        examples.append((int(q), int(r)))
                    candidate_critical_overflow_examples = tuple(examples)
                if is_v2 and version >= 10:
                    metadata_len = struct.unpack_from("<I", data, offset)[0]
                    offset += 4
                    if metadata_len:
                        metadata_blob = data[offset:offset + metadata_len]
                        offset += metadata_len
                        v1_search_metadata = v1_search_metadata_from_compact_bytes(metadata_blob)

            positions.append(PositionRecord(
                move_history=move_history,
                policy_target=policy,
                root_value=root_value,
                selected_action_value=selected_action_value,
                player=player,
                outcome=outcome,
                game_id=game_id,
                is_full_search=is_full,
                turn_index=turn_idx,
                opp_policy_target=opp_policy,
                opp_policy_weight=opp_policy_weight,
                value_weight=value_weight,
                policy_target_v2=policy_v2,
                opp_policy_target_v2=opp_policy_v2,
                opp_policy_legal_v2=opp_policy_legal_v2,
                pair_policy_target_v2=pair_policy_v2,
                pair_policy_complete=pair_policy_complete,
                v1_search_metadata=v1_search_metadata,
                target_policy_mass_outside_window=target_policy_mass_outside_window,
                missing_target_policy_mass=missing_target_policy_mass,
                candidate_recall_mcts_top1=candidate_recall_mcts_top1,
                candidate_recall_mcts_top4=candidate_recall_mcts_top4,
                candidate_recall_mcts_top8=candidate_recall_mcts_top8,
                candidate_recall_winning_move=candidate_recall_winning_move,
                candidate_recall_forced_block=candidate_recall_forced_block,
                candidate_recall_two_placement_cover=candidate_recall_two_placement_cover,
                candidate_discovery_top1=candidate_discovery_top1,
                candidate_discovery_top4=candidate_discovery_top4,
                candidate_discovery_top8=candidate_discovery_top8,
                candidate_discovery_winning_move=candidate_discovery_winning_move,
                candidate_discovery_forced_block=candidate_discovery_forced_block,
                candidate_discovery_two_placement_cover=candidate_discovery_two_placement_cover,
                candidate_discovery_open_four=candidate_discovery_open_four,
                candidate_discovery_open_five=candidate_discovery_open_five,
                candidate_critical_count=int(candidate_critical_count),
                candidate_critical_overflow_count=int(candidate_critical_overflow_count),
                candidate_critical_overflow_examples=candidate_critical_overflow_examples,
                sparse_prior_stage=sparse_prior_stage,
                sparse_prior_root_candidate_count=sparse_prior_root_candidate_count,
                sparse_prior_leaf_candidate_count=sparse_prior_leaf_candidate_count,
                sparse_prior_root_hit_frac=sparse_prior_root_hit_frac,
                sparse_prior_leaf_hit_frac=sparse_prior_leaf_hit_frac,
                fallback_prior_use=fallback_prior_use,
                fallback_prior_use_on_mcts_top1=fallback_prior_use_on_mcts_top1,
                fallback_prior_use_on_mcts_top4=fallback_prior_use_on_mcts_top4,
                fallback_prior_use_on_mcts_top8=fallback_prior_use_on_mcts_top8,
                sparse_vs_dense_disagreement=sparse_vs_dense_disagreement,
                sparse_prior_forward_ms=sparse_prior_forward_ms,
                sparse_prior_candidate_build_ms=sparse_prior_candidate_build_ms,
                pair_prior_candidate_count=pair_prior_candidate_count,
                pair_prior_hit_frac=pair_prior_hit_frac,
                pair_fallback_prior_use=pair_fallback_prior_use,
                pair_fallback_prior_use_on_mcts_top1=pair_fallback_prior_use_on_mcts_top1,
                pair_fallback_prior_use_on_mcts_top4=pair_fallback_prior_use_on_mcts_top4,
                pair_fallback_prior_use_on_mcts_top8=pair_fallback_prior_use_on_mcts_top8,
                regret_rank=regret_rank,
                regret_value=regret_value,
                regret_weight=regret_weight,
                axis_label=axis_label,
                moves_left=moves_left,
            ))

        return GameRecord(
            positions=positions,
            outcome=outcome,
            game_id=game_id,
            game_length=num_pos,
            final_move_history=positions[-1].move_history if positions else b"",
        )

    @staticmethod
    def from_game_data(
        move_history_bytes: bytes,
        policy_targets: List[Dict[int, float]],
        root_values: List[float],
        players: List[int],
        outcome: float,
        game_id: int,
        is_full_search: bool = True,
        policy_targets_v2: Optional[List[PolicyTargetV2]] = None,
        pair_policy_targets_v2: Optional[List[List[Tuple[Tuple[int, int], Tuple[int, int], float]]]] = None,
    ) -> "GameRecord":
        """Construct a GameRecord from raw game data.

        Args:
            move_history_bytes: For each position, the compact move history
                up to (but not including) that position's action.
            policy_targets: For each position, sparse policy dict.
            root_values: For each position, root Q-value.
            players: For each position, the player who made the action.
            outcome: Final game outcome from P0's perspective.
            game_id: Monotonic game counter.
            is_full_search: Whether full MCTS sim count was used.
        """
        positions = []
        # Each position encodes the board state BEFORE the move.
        # move_history_bytes is a list of bytes, one per position.
        if isinstance(move_history_bytes, bytes):
            # Single contiguous byte buffer — split by position.
            # Each position's history is the prefix up to that move.
            num_moves = len(policy_targets)
            pos_histories = _split_history_bytes(move_history_bytes, num_moves)
        else:
            pos_histories = move_history_bytes

        policy_targets_v2 = policy_targets_v2 or [[] for _ in policy_targets]
        pair_policy_targets_v2 = pair_policy_targets_v2 or [[] for _ in policy_targets]

        for i, (history, policy, rv, player) in enumerate(
            zip(pos_histories, policy_targets, root_values, players)
        ):
            positions.append(PositionRecord(
                move_history=history,
                policy_target=policy,
                policy_target_v2=policy_targets_v2[i] if i < len(policy_targets_v2) else [],
                pair_policy_target_v2=pair_policy_targets_v2[i] if i < len(pair_policy_targets_v2) else [],
                pair_policy_complete=bool(pair_policy_targets_v2[i]) if i < len(pair_policy_targets_v2) else False,
                root_value=rv,
                selected_action_value=rv,
                player=player,
                outcome=outcome,
                game_id=game_id,
                is_full_search=is_full_search,
                turn_index=i,
            ))

        return GameRecord(
            positions=positions,
            outcome=outcome,
            game_id=game_id,
            game_length=len(positions),
            final_move_history=move_history_bytes if isinstance(move_history_bytes, bytes) else (
                pos_histories[-1] if pos_histories else b""
            ),
        )


def _split_history_bytes(full_history: bytes, num_positions: int) -> List[bytes]:
    """Split a contiguous move history into per-position prefixes.

    Each triple is 12 bytes (i32 LE × 3). Position i gets the first i triples.
    """
    result = []
    stride = 12  # player:i32, q:i32, r:i32
    for i in range(num_positions):
        result.append(full_history[:i * stride])
    return result


def sparsify_policy(
    dense_policy: np.ndarray,
    top_k: int = 20,
) -> Dict[int, float]:
    """Convert dense policy array to sparse top-K dict.

    Args:
        dense_policy: (BOARD_AREA,) float32 array of visit probabilities.
        top_k: Number of top entries to keep.

    Returns:
        Dict mapping action indices to probabilities, renormalized over top-K.
    """
    if len(dense_policy) == 0:
        return {}
    indices = np.argpartition(-dense_policy, min(top_k, len(dense_policy) - 1))[:top_k]
    values = dense_policy[indices]
    total = values.sum()
    if total > 0:
        values = values / total
    else:
        values = np.full_like(values, 1.0 / max(len(values), 1), dtype=np.float32)
    return {int(idx): float(val) for idx, val in zip(indices, values)}


def policy_v2_from_visits(
    moves_q: List[int],
    moves_r: List[int],
    visits: List[int],
    top_k: Optional[int] = None,
) -> PolicyTargetV2:
    """Build normalized global action targets directly from MCTS root visits."""
    entries = [
        (int(q), int(r), float(v))
        for q, r, v in zip(moves_q, moves_r, visits)
        if float(v) > 0.0
    ]
    if not entries:
        return []
    entries.sort(key=lambda item: (-item[2], item[0], item[1]))
    total_all = sum(v for _, _, v in entries)
    if top_k is not None and int(top_k) > 0:
        entries = entries[:max(1, int(top_k))]
    total = total_all
    if total <= 0.0:
        p = 1.0 / len(entries)
        return [(q, r, p) for q, r, _ in entries]
    return [(q, r, v / total) for q, r, v in entries]


def pair_policy_v2_from_place_target(
    policy_v2: PolicyTargetV2,
    *,
    top_k: Optional[int] = None,
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """Diagnostic-only synthetic pair target from place-action probabilities.

    Production training must use search-observed pair targets assigned in
    ``hexorl.buffer.targets``. This helper is retained for legacy tests and
    diagnostics that explicitly want the synthetic product baseline.
    """
    merged: dict[tuple[int, int], float] = {}
    for q, r, prob in policy_v2:
        if prob > 0.0:
            key = (int(q), int(r))
            merged[key] = merged.get(key, 0.0) + float(prob)
    entries = [(q, r, prob) for (q, r), prob in merged.items()]
    entries.sort(key=lambda item: (-item[2], item[0], item[1]))
    if len(entries) < 2:
        return []
    pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    limit = len(entries) if top_k is None else min(len(entries), max(2, int(top_k)))
    for i in range(limit):
        q1, r1, p1 = entries[i]
        for j in range(i + 1, limit):
            q2, r2, p2 = entries[j]
            pairs.append(((q1, r1), (q2, r2), p1 * p2))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    if top_k is not None:
        pairs = pairs[: max(1, int(top_k))]
    total = sum(prob for _a, _b, prob in pairs)
    if total <= 0.0:
        return []
    return [(a, b, float(prob / total)) for a, b, prob in pairs]


def dense_policy_from_v2(
    policy_v2: PolicyTargetV2,
    offset_q: int,
    offset_r: int,
    top_k: int = 64,
) -> tuple[Dict[int, float], float]:
    """Project global V2 targets into the legacy crop-local sparse target."""
    dense = np.zeros(BOARD_AREA, dtype=np.float32)
    outside_mass = 0.0
    for q, r, prob in policy_v2:
        idx = action_to_board_index(q, r, offset_q, offset_r)
        if idx >= 0:
            dense[idx] += float(prob)
        else:
            outside_mass += float(prob)
    policy = sparsify_policy(dense, top_k=top_k) if dense.sum() > 0.0 else {}
    return policy, outside_mass


def action_to_board_index(q: int, r: int, offset_q: int = -16, offset_r: int = -16) -> int:
    """Convert axial hex coordinates (q, r) to flat BOARD_AREA index.

    The board tensor uses a 33×33 window centered at the board centroid.
    offset_q and offset_r are board coordinates for tensor index (0, 0).
    Returns -1 when the action is outside the encoded policy window.
    """
    gi = q - offset_q
    gj = r - offset_r
    if not (0 <= gi < BOARD_SIZE and 0 <= gj < BOARD_SIZE):
        return -1
    return gi * BOARD_SIZE + gj
