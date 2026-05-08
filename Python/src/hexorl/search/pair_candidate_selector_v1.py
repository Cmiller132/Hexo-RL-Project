"""V1 pair-candidate selector and admission contract.

The selector owns source quotas, canonicalization, metadata, tactical
protection, and deterministic admission ordering.  It does not choose final
actions and does not project pair scores into single-cell logits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from hexorl.search.pair_scorer_v1 import (
    DirectPairRetrievalCandidateV1,
    LegalRowIdentityV1,
    PairCoord,
    PairKey,
    PairRowIdentityV1,
    canonical_pair_key,
    direct_pair_retrieval_v1,
    hex_distance,
    pair_identity_from_legal_row_ids,
    parse_legal_rows_v1,
    parse_pair_rows_v1,
    same_hex_line,
    same_origin_axis,
)
from hexorl.selfplay.records import (
    V1CandidatePair,
    V1CandidateSourceContribution,
    V1ProposalPropensityMetadata,
)


PAIR_CANDIDATE_SELECTOR_VERSION_V1 = "pair_candidate_selector_v1"

SOURCE_TERMINAL_EXACT = "terminal_exact_v1"
SOURCE_DIRECT_PAIR_RETRIEVAL = "direct_pair_retrieval"
SOURCE_ANCHOR_CONDITIONED_COMPLETION = "anchor_conditioned_completion"
SOURCE_CELL_MARGINAL_CROSS = "cell_marginal_cross"
SOURCE_STRUCTURED_DIVERSITY = "structured_diversity"
SOURCE_BLIND_CANARY = "blind_canary"

DEFAULT_SOURCE_PRIORITY = (
    SOURCE_DIRECT_PAIR_RETRIEVAL,
    SOURCE_ANCHOR_CONDITIONED_COMPLETION,
    SOURCE_CELL_MARGINAL_CROSS,
    SOURCE_STRUCTURED_DIVERSITY,
    SOURCE_BLIND_CANARY,
)


def _default_source_quotas() -> dict[str, int]:
    return {
        SOURCE_DIRECT_PAIR_RETRIEVAL: 64,
        SOURCE_ANCHOR_CONDITIONED_COMPLETION: 32,
        SOURCE_CELL_MARGINAL_CROSS: 32,
        SOURCE_STRUCTURED_DIVERSITY: 16,
        SOURCE_BLIND_CANARY: 2,
    }


@dataclass(frozen=True)
class PairCandidateSelectorV1Config:
    candidate_budget: int = 128
    source_quotas: Mapping[str, int] = field(default_factory=_default_source_quotas)
    source_priority: tuple[str, ...] = DEFAULT_SOURCE_PRIORITY
    direct_retrieval_top_k: int | None = None
    direct_retrieval_block_size: int = 512
    cell_marginal_top_cells: int = 16
    structured_diversity_pool_rows: int = 256
    blind_canary_seed: int = 0xC0DEC0DE
    candidate_id_prefix: str = PAIR_CANDIDATE_SELECTOR_VERSION_V1

    def __post_init__(self) -> None:
        budget = int(self.candidate_budget)
        if budget < 0:
            raise ValueError("V1 candidate_budget cannot be negative")
        quotas = {str(source): int(quota) for source, quota in self.source_quotas.items()}
        if any(quota < 0 for quota in quotas.values()):
            raise ValueError("V1 source quotas cannot be negative")
        priority = tuple(str(source) for source in self.source_priority)
        direct_top_k = None if self.direct_retrieval_top_k is None else int(self.direct_retrieval_top_k)
        if direct_top_k is not None and direct_top_k < 0:
            raise ValueError("V1 direct_retrieval_top_k cannot be negative")
        block_size = int(self.direct_retrieval_block_size)
        if block_size <= 0:
            raise ValueError("V1 direct_retrieval_block_size must be positive")
        top_cells = int(self.cell_marginal_top_cells)
        if top_cells < 0:
            raise ValueError("V1 cell_marginal_top_cells cannot be negative")
        diversity_pool = int(self.structured_diversity_pool_rows)
        if diversity_pool < 0:
            raise ValueError("V1 structured_diversity_pool_rows cannot be negative")
        if not str(self.candidate_id_prefix):
            raise ValueError("V1 candidate_id_prefix is required")
        object.__setattr__(self, "candidate_budget", budget)
        object.__setattr__(self, "source_quotas", quotas)
        object.__setattr__(self, "source_priority", priority)
        object.__setattr__(self, "direct_retrieval_top_k", direct_top_k)
        object.__setattr__(self, "direct_retrieval_block_size", block_size)
        object.__setattr__(self, "cell_marginal_top_cells", top_cells)
        object.__setattr__(self, "structured_diversity_pool_rows", diversity_pool)
        object.__setattr__(self, "blind_canary_seed", int(self.blind_canary_seed))
        object.__setattr__(self, "candidate_id_prefix", str(self.candidate_id_prefix))

    def quota_for(self, source_type: str) -> int:
        return int(self.source_quotas.get(source_type, 0))


@dataclass(frozen=True)
class PairCandidateV1:
    candidate_id: str
    pair_key: PairKey
    first_legal_row_id: int
    second_legal_row_id: int
    row_table_schema_version: int
    source_contributions: tuple[V1CandidateSourceContribution, ...]
    proposal_propensity_metadata: V1ProposalPropensityMetadata
    forced_exploration_flag: bool
    tactical_protected_flag: bool
    terminal_exact_flag: bool
    terminal_equivalence_flag: bool
    target_support_flags: tuple[str, ...]
    admission_generation: int
    root_or_interior: Literal["root", "interior"]
    pair_row_key: int | None = None
    selector_score: float = 0.0
    rich_rerank_score: float | None = None
    source_scores: Mapping[str, float] = field(default_factory=dict)
    candidate_selection_reason: str = ""

    def __post_init__(self) -> None:
        pair_key = canonical_pair_key(*self.pair_key)
        first_id = int(self.first_legal_row_id)
        second_id = int(self.second_legal_row_id)
        if first_id < 0 or second_id < 0 or first_id == second_id:
            raise ValueError("V1 admitted candidates require distinct nonnegative legal-row IDs")
        if first_id > second_id:
            first_id, second_id = second_id, first_id
        sources = tuple(
            source
            if isinstance(source, V1CandidateSourceContribution)
            else V1CandidateSourceContribution.from_dict(source)
            for source in self.source_contributions
        )
        if not sources:
            raise ValueError("V1 admitted candidate is missing source metadata")
        proposal = self.proposal_propensity_metadata
        if proposal is None:
            raise ValueError("V1 admitted candidate is missing proposal metadata")
        if not isinstance(proposal, V1ProposalPropensityMetadata):
            proposal = V1ProposalPropensityMetadata.from_dict(proposal)
        flags = tuple(str(flag) for flag in self.target_support_flags)
        if not flags or "admitted" not in flags:
            raise ValueError("V1 admitted candidate is missing target-support metadata")
        tactical = bool(self.tactical_protected_flag)
        if tactical and not any(source.inclusion_kind == "tactical_protected" for source in sources):
            raise ValueError("V1 tactical-protected candidate requires a tactical source contribution")
        selector_score = float(self.selector_score)
        if not np.isfinite(selector_score):
            raise ValueError("V1 selector_score must be finite")
        rich_score = None if self.rich_rerank_score is None else float(self.rich_rerank_score)
        if rich_score is not None and not np.isfinite(rich_score):
            raise ValueError("V1 rich_rerank_score must be finite when provided")
        scores = {str(source): float(score) for source, score in self.source_scores.items()}
        if any(not np.isfinite(score) for score in scores.values()):
            raise ValueError("V1 source_scores must be finite")
        replay = V1CandidatePair(
            candidate_id=str(self.candidate_id),
            pair_key=pair_key,
            first_legal_row_id=first_id,
            second_legal_row_id=second_id,
            row_table_schema_version=int(self.row_table_schema_version),
            source_contributions=sources,
            proposal_propensity_metadata=proposal,
            forced_exploration_flag=bool(self.forced_exploration_flag),
            terminal_exact_flag=bool(self.terminal_exact_flag),
            terminal_equivalence_flag=bool(self.terminal_equivalence_flag),
            target_support_flags=flags,
            admission_generation=int(self.admission_generation),
            root_or_interior=str(self.root_or_interior),
            candidate_selection_reason=str(self.candidate_selection_reason),
        )
        object.__setattr__(self, "candidate_id", replay.candidate_id)
        object.__setattr__(self, "pair_key", replay.pair_key)
        object.__setattr__(self, "first_legal_row_id", replay.first_legal_row_id)
        object.__setattr__(self, "second_legal_row_id", replay.second_legal_row_id)
        object.__setattr__(self, "row_table_schema_version", replay.row_table_schema_version)
        object.__setattr__(self, "source_contributions", replay.source_contributions)
        object.__setattr__(self, "proposal_propensity_metadata", replay.proposal_propensity_metadata)
        object.__setattr__(self, "forced_exploration_flag", replay.forced_exploration_flag)
        object.__setattr__(self, "tactical_protected_flag", tactical)
        object.__setattr__(self, "terminal_exact_flag", replay.terminal_exact_flag)
        object.__setattr__(self, "terminal_equivalence_flag", replay.terminal_equivalence_flag)
        object.__setattr__(self, "target_support_flags", replay.target_support_flags)
        object.__setattr__(self, "admission_generation", replay.admission_generation)
        object.__setattr__(self, "root_or_interior", replay.root_or_interior)
        object.__setattr__(self, "pair_row_key", None if self.pair_row_key is None else int(self.pair_row_key))
        object.__setattr__(self, "selector_score", selector_score)
        object.__setattr__(self, "rich_rerank_score", rich_score)
        object.__setattr__(self, "source_scores", scores)
        object.__setattr__(self, "candidate_selection_reason", replay.candidate_selection_reason)

    @property
    def row_id_pair(self) -> tuple[int, int]:
        return (self.first_legal_row_id, self.second_legal_row_id)

    def to_replay_candidate_pair(self) -> V1CandidatePair:
        return V1CandidatePair(
            candidate_id=self.candidate_id,
            pair_key=self.pair_key,
            first_legal_row_id=self.first_legal_row_id,
            second_legal_row_id=self.second_legal_row_id,
            row_table_schema_version=self.row_table_schema_version,
            source_contributions=self.source_contributions,
            proposal_propensity_metadata=self.proposal_propensity_metadata,
            forced_exploration_flag=self.forced_exploration_flag,
            terminal_exact_flag=self.terminal_exact_flag,
            terminal_equivalence_flag=self.terminal_equivalence_flag,
            target_support_flags=self.target_support_flags,
            admission_generation=self.admission_generation,
            root_or_interior=self.root_or_interior,
            candidate_selection_reason=self.candidate_selection_reason,
        )


@dataclass(frozen=True)
class PairCandidateSelectorTelemetryV1:
    legal_row_count: int
    legal_pair_count: int
    proposed_by_source: Mapping[str, int]
    pre_budget_admitted_by_source: Mapping[str, int]
    admitted_by_source: Mapping[str, int]
    evicted_by_source: Mapping[str, int]
    duplicate_proposals: int
    quota_evictions: int
    budget_evictions: int
    protected_count: int
    canary_count: int
    canary_loss_count: int
    direct_retrieval_scored_pair_count: int
    direct_retrieval_block_size: int


@dataclass(frozen=True)
class PairCandidateSelectorResultV1:
    candidates: tuple[PairCandidateV1, ...]
    telemetry: PairCandidateSelectorTelemetryV1
    candidate_selector_version: str = PAIR_CANDIDATE_SELECTOR_VERSION_V1

    @property
    def replay_candidate_pairs(self) -> tuple[V1CandidatePair, ...]:
        return tuple(candidate.to_replay_candidate_pair() for candidate in self.candidates)


@dataclass(frozen=True)
class _SourceProposal:
    identity: PairRowIdentityV1
    source_type: str
    score: float
    source_rank: int
    quota_id: str
    inclusion_kind: str
    correction_mode: str
    protected: bool = False
    forced: bool = False
    terminal_exact: bool = False
    terminal_equivalence: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        score = float(self.score)
        if not np.isfinite(score):
            raise ValueError("V1 source proposal score must be finite")
        rank = int(self.source_rank)
        if rank < 0:
            raise ValueError("V1 source proposal rank cannot be negative")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "source_rank", rank)
        object.__setattr__(self, "source_type", str(self.source_type))
        object.__setattr__(self, "quota_id", str(self.quota_id))
        object.__setattr__(self, "inclusion_kind", str(self.inclusion_kind))
        object.__setattr__(self, "correction_mode", str(self.correction_mode))

    @property
    def row_id_pair(self) -> tuple[int, int]:
        return self.identity.row_id_pair


@dataclass
class _CandidateState:
    identity: PairRowIdentityV1
    source_contributions: list[V1CandidateSourceContribution] = field(default_factory=list)
    source_scores: dict[str, float] = field(default_factory=dict)
    protected: bool = False
    forced: bool = False
    terminal_exact: bool = False
    terminal_equivalence: bool = False
    selector_score: float = 0.0
    reason_parts: list[str] = field(default_factory=list)


def select_pair_candidates_v1(
    legal_rows: Sequence[Any] | Mapping[str, Any],
    *,
    pair_rows: Sequence[Any] | Mapping[str, Any] | None = None,
    tactical_payload: Mapping[str, Any] | None = None,
    legal_cell_embeddings: np.ndarray | Sequence[Sequence[float]] | None = None,
    direct_retrieval_rows: Sequence[Any] | None = None,
    pair_completion_logits: np.ndarray | Sequence[Sequence[float]] | None = None,
    anchor_completion_scores: Mapping[Any, float] | Sequence[Any] | None = None,
    cell_marginal_logits: np.ndarray | Sequence[float] | None = None,
    rich_pair_rerank: Callable[[Sequence[PairCandidateV1]], Mapping[Any, float] | Sequence[float]] | None = None,
    config: PairCandidateSelectorV1Config | None = None,
    admission_generation: int = 0,
    root_or_interior: Literal["root", "interior"] = "root",
    legal_row_schema_version: int | None = None,
    pair_row_schema_version: int | None = None,
) -> PairCandidateSelectorResultV1:
    cfg = config or PairCandidateSelectorV1Config()
    rows_in_input_order = parse_legal_rows_v1(legal_rows)
    legal_by_id = {row.row_id: row for row in rows_in_input_order}
    sorted_rows = tuple(sorted(rows_in_input_order, key=lambda row: (row.row_id, row.cell)))
    pair_row_identities = parse_pair_rows_v1(pair_rows)
    pair_rows_by_id_pair = {row.row_id_pair: row for row in pair_row_identities}
    legal_schema = int(legal_row_schema_version or _schema_version_from(legal_rows) or 1)
    pair_schema = int(
        pair_row_schema_version
        or _schema_version_from(pair_rows)
        or (int(tactical_payload["pair_row_schema_version"]) if tactical_payload and "pair_row_schema_version" in tactical_payload else 1)
    )
    if legal_schema <= 0 or pair_schema <= 0:
        raise ValueError("V1 legal and pair schema versions must be positive")
    legal_pair_count = len(sorted_rows) * max(0, len(sorted_rows) - 1) // 2

    proposed_by_source: dict[str, int] = {}
    admitted_by_source: dict[str, int] = {}
    duplicate_proposals = 0
    quota_evictions = 0
    direct_scored_pair_count = 0
    direct_block_size = cfg.direct_retrieval_block_size
    states: dict[tuple[int, int], _CandidateState] = {}

    tactical = _tactical_proposals(tactical_payload, legal_by_id, pair_rows_by_id_pair)
    proposed_by_source[SOURCE_TERMINAL_EXACT] = len(tactical)
    for proposal in tactical:
        duplicate_proposals += _admit_proposal(states, proposal)
        admitted_by_source[SOURCE_TERMINAL_EXACT] = admitted_by_source.get(SOURCE_TERMINAL_EXACT, 0) + 1

    source_proposals: dict[str, list[_SourceProposal]] = {
        SOURCE_DIRECT_PAIR_RETRIEVAL: [],
        SOURCE_ANCHOR_CONDITIONED_COMPLETION: [],
        SOURCE_CELL_MARGINAL_CROSS: [],
        SOURCE_STRUCTURED_DIVERSITY: [],
        SOURCE_BLIND_CANARY: [],
    }

    direct_rows, direct_scored_pair_count, direct_block_size = _direct_retrieval_proposals(
        rows_in_input_order,
        legal_by_id,
        pair_rows_by_id_pair,
        legal_cell_embeddings=legal_cell_embeddings,
        direct_retrieval_rows=direct_retrieval_rows,
        config=cfg,
    )
    source_proposals[SOURCE_DIRECT_PAIR_RETRIEVAL] = direct_rows
    source_proposals[SOURCE_ANCHOR_CONDITIONED_COMPLETION] = _anchor_completion_proposals(
        rows_in_input_order,
        legal_by_id,
        pair_rows_by_id_pair,
        pair_completion_logits=pair_completion_logits,
        anchor_completion_scores=anchor_completion_scores,
        limit=max(cfg.quota_for(SOURCE_ANCHOR_CONDITIONED_COMPLETION), cfg.candidate_budget, 1),
    )
    source_proposals[SOURCE_CELL_MARGINAL_CROSS] = _cell_marginal_cross_proposals(
        rows_in_input_order,
        legal_by_id,
        pair_rows_by_id_pair,
        cell_marginal_logits=cell_marginal_logits,
        top_cells=cfg.cell_marginal_top_cells,
        limit=max(cfg.quota_for(SOURCE_CELL_MARGINAL_CROSS), cfg.candidate_budget, 1),
    )
    diversity_rows = _deterministic_row_pool(
        sorted_rows,
        seed=cfg.blind_canary_seed ^ 0x5155_7101,
        limit=cfg.structured_diversity_pool_rows,
    )
    canary_rows = _deterministic_row_pool(
        sorted_rows,
        seed=cfg.blind_canary_seed,
        limit=cfg.structured_diversity_pool_rows,
    )
    source_proposals[SOURCE_STRUCTURED_DIVERSITY] = _structured_diversity_proposals(
        diversity_rows,
        legal_by_id,
        pair_rows_by_id_pair,
        seed=cfg.blind_canary_seed ^ 0x5155_7101,
        limit=max(cfg.quota_for(SOURCE_STRUCTURED_DIVERSITY), 0),
    )
    source_proposals[SOURCE_BLIND_CANARY] = _blind_canary_proposals(
        canary_rows,
        legal_by_id,
        pair_rows_by_id_pair,
        seed=cfg.blind_canary_seed,
        limit=cfg.quota_for(SOURCE_BLIND_CANARY),
        legal_pair_count=legal_pair_count,
    )

    for source_type, proposals in source_proposals.items():
        deduped, duplicates = _dedup_source_proposals(proposals)
        source_proposals[source_type] = deduped
        duplicate_proposals += duplicates
        proposed_by_source[source_type] = len(deduped)

    for source_type in cfg.source_priority:
        quota = cfg.quota_for(source_type)
        if quota <= 0:
            quota_evictions += len([p for p in source_proposals.get(source_type, ()) if p.row_id_pair not in states])
            continue
        admitted_new = 0
        for proposal in source_proposals.get(source_type, ()):
            if proposal.row_id_pair in states:
                duplicate_proposals += _admit_proposal(states, proposal)
                continue
            if admitted_new >= quota:
                quota_evictions += 1
                continue
            duplicate_proposals += _admit_proposal(states, proposal)
            admitted_new += 1
            admitted_by_source[source_type] = admitted_by_source.get(source_type, 0) + 1

    preliminary = tuple(
        _candidate_from_state(
            state,
            config=cfg,
            row_table_schema_version=pair_schema,
            admission_generation=admission_generation,
            root_or_interior=root_or_interior,
        )
        for state in states.values()
    )
    preliminary = _apply_rich_rerank(preliminary, rich_pair_rerank)
    protected = sorted(
        (candidate for candidate in preliminary if candidate.tactical_protected_flag),
        key=_candidate_order_key,
    )
    nonprotected = sorted(
        (candidate for candidate in preliminary if not candidate.tactical_protected_flag),
        key=_candidate_order_key,
    )
    selected_protected = protected
    remaining = max(0, cfg.candidate_budget - len(selected_protected))
    selected_nonprotected = nonprotected[:remaining]
    budget_evictions = (
        max(0, len(nonprotected) - len(selected_nonprotected))
    )
    selected = tuple(selected_protected + selected_nonprotected)
    _validate_admitted_candidates(selected)
    final_by_source: dict[str, int] = {}
    for candidate in selected:
        for source in candidate.source_contributions:
            final_by_source[source.source_type] = final_by_source.get(source.source_type, 0) + 1
    selected_row_ids = {candidate.row_id_pair for candidate in selected}
    evicted_by_source: dict[str, int] = {}
    for candidate in preliminary:
        if candidate.row_id_pair in selected_row_ids:
            continue
        for source in candidate.source_contributions:
            evicted_by_source[source.source_type] = evicted_by_source.get(source.source_type, 0) + 1
    final_canaries = sum(1 for candidate in selected if SOURCE_BLIND_CANARY in candidate.source_scores)
    proposed_canaries = proposed_by_source.get(SOURCE_BLIND_CANARY, 0)
    telemetry = PairCandidateSelectorTelemetryV1(
        legal_row_count=len(sorted_rows),
        legal_pair_count=legal_pair_count,
        proposed_by_source=dict(sorted(proposed_by_source.items())),
        pre_budget_admitted_by_source=dict(sorted(admitted_by_source.items())),
        admitted_by_source=dict(sorted(final_by_source.items())),
        evicted_by_source=dict(sorted(evicted_by_source.items())),
        duplicate_proposals=duplicate_proposals,
        quota_evictions=quota_evictions,
        budget_evictions=budget_evictions,
        protected_count=len(selected_protected),
        canary_count=final_canaries,
        canary_loss_count=max(0, proposed_canaries - final_canaries),
        direct_retrieval_scored_pair_count=direct_scored_pair_count,
        direct_retrieval_block_size=direct_block_size,
    )
    return PairCandidateSelectorResultV1(candidates=selected, telemetry=telemetry)


def _schema_version_from(value: Any) -> int | None:
    if isinstance(value, Mapping) and "schema_version" in value:
        return int(value["schema_version"])
    return None


def _tactical_proposals(
    tactical_payload: Mapping[str, Any] | None,
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
) -> list[_SourceProposal]:
    if not tactical_payload:
        return []
    impossible_to_cover = bool(tactical_payload.get("impossible_to_cover", False)) or str(
        tactical_payload.get("status", "")
    ) == "hot_cover_impossible"
    specs = (
        ("hot_completion_pairs", "own_hot_completion", True, False),
        ("hot_cover_pairs", "opponent_hot_cover", False, False),
        ("terminal_equivalent_pairs", "terminal_equivalent", False, True),
    )
    proposals: list[_SourceProposal] = []
    for field_name, reason, terminal_exact, terminal_equivalent in specs:
        rows = parse_pair_rows_v1(tactical_payload.get(field_name, ()))
        for rank, row in enumerate(sorted(rows, key=lambda item: item.row_id_pair)):
            identity = pair_identity_from_legal_row_ids(
                row.first_legal_row_id,
                row.second_legal_row_id,
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
            proposals.append(
                _SourceProposal(
                    identity=identity,
                    source_type=SOURCE_TERMINAL_EXACT,
                    score=1_000_000.0 - rank,
                    source_rank=rank,
                    quota_id=field_name,
                    inclusion_kind="tactical_protected",
                    correction_mode="uncorrected_logged",
                    protected=True,
                    terminal_exact=terminal_exact,
                    terminal_equivalence=terminal_equivalent,
                    reason=(
                        f"{reason};impossible_to_cover_flag"
                        if impossible_to_cover and field_name == "hot_cover_pairs"
                        else reason
                    ),
                )
            )
    return proposals


def _direct_retrieval_proposals(
    rows_in_input_order: Sequence[LegalRowIdentityV1],
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
    *,
    legal_cell_embeddings: np.ndarray | Sequence[Sequence[float]] | None,
    direct_retrieval_rows: Sequence[Any] | None,
    config: PairCandidateSelectorV1Config,
) -> tuple[list[_SourceProposal], int, int]:
    quota = config.quota_for(SOURCE_DIRECT_PAIR_RETRIEVAL)
    top_k = config.direct_retrieval_top_k
    if top_k is None:
        top_k = max(quota, config.candidate_budget, 1)
    candidates: tuple[DirectPairRetrievalCandidateV1, ...] = ()
    scored_pair_count = 0
    block_size = config.direct_retrieval_block_size
    if direct_retrieval_rows is not None:
        candidates = tuple(_parse_direct_candidate(item, legal_by_id, pair_rows_by_id_pair) for item in direct_retrieval_rows)
    elif legal_cell_embeddings is not None and top_k > 0:
        result = direct_pair_retrieval_v1(
            rows_in_input_order,
            legal_cell_embeddings,
            top_k=top_k,
            block_size=config.direct_retrieval_block_size,
        )
        candidates = result.candidates
        scored_pair_count = result.scored_pair_count
        block_size = result.block_size
    proposals: list[_SourceProposal] = []
    for rank, candidate in enumerate(sorted(candidates, key=lambda item: (-item.score, item.first_legal_row_id, item.second_legal_row_id))):
        identity = pair_identity_from_legal_row_ids(
            candidate.first_legal_row_id,
            candidate.second_legal_row_id,
            legal_by_id,
            pair_rows_by_id_pair=pair_rows_by_id_pair,
        )
        proposals.append(
            _SourceProposal(
                identity=identity,
                source_type=SOURCE_DIRECT_PAIR_RETRIEVAL,
                score=float(candidate.score),
                source_rank=rank,
                quota_id=SOURCE_DIRECT_PAIR_RETRIEVAL,
                inclusion_kind="deterministic_top_k",
                correction_mode="uncorrected_logged",
                reason=SOURCE_DIRECT_PAIR_RETRIEVAL,
            )
        )
    return proposals, scored_pair_count, block_size


def _parse_direct_candidate(
    item: Any,
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
) -> DirectPairRetrievalCandidateV1:
    if isinstance(item, DirectPairRetrievalCandidateV1):
        return item
    if isinstance(item, PairRowIdentityV1):
        return DirectPairRetrievalCandidateV1(identity=item, score=0.0, source_rank=0)
    if isinstance(item, Mapping):
        if "identity" in item:
            identity = _identity_from_pair_like(item["identity"], legal_by_id, pair_rows_by_id_pair)
        else:
            identity = _identity_from_pair_like(item, legal_by_id, pair_rows_by_id_pair)
        return DirectPairRetrievalCandidateV1(
            identity=identity,
            score=float(item.get("score", 0.0)),
            source_rank=int(item.get("source_rank", 0)),
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) == 2:
            identity = _identity_from_pair_like(item[0], legal_by_id, pair_rows_by_id_pair)
            return DirectPairRetrievalCandidateV1(identity=identity, score=float(item[1]), source_rank=0)
        if len(item) >= 3:
            identity = _identity_from_pair_like((item[0], item[1]), legal_by_id, pair_rows_by_id_pair)
            return DirectPairRetrievalCandidateV1(identity=identity, score=float(item[2]), source_rank=0)
    raise ValueError(f"unsupported direct retrieval candidate row: {item!r}")


def _anchor_completion_proposals(
    rows_in_input_order: Sequence[LegalRowIdentityV1],
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
    *,
    pair_completion_logits: np.ndarray | Sequence[Sequence[float]] | None,
    anchor_completion_scores: Mapping[Any, float] | Sequence[Any] | None,
    limit: int,
) -> list[_SourceProposal]:
    scored: list[tuple[PairRowIdentityV1, float]] = []
    if pair_completion_logits is not None:
        logits = np.asarray(pair_completion_logits, dtype=np.float64)
        n = len(rows_in_input_order)
        if logits.shape != (n, n):
            raise ValueError(f"pair_completion_logits shape {logits.shape} does not match legal rows {(n, n)}")
        for first_idx in range(n):
            for second_idx in range(first_idx + 1, n):
                forward = float(logits[first_idx, second_idx])
                backward = float(logits[second_idx, first_idx])
                if not np.isfinite(forward) and not np.isfinite(backward):
                    continue
                if np.isfinite(forward) and np.isfinite(backward):
                    score = 0.5 * (forward + backward)
                else:
                    score = forward if np.isfinite(forward) else backward
                identity = pair_identity_from_legal_row_ids(
                    rows_in_input_order[first_idx].row_id,
                    rows_in_input_order[second_idx].row_id,
                    legal_by_id,
                    pair_rows_by_id_pair=pair_rows_by_id_pair,
                )
                scored.append((identity, score))
    if anchor_completion_scores is not None and not isinstance(anchor_completion_scores, Mapping):
        score_array = np.asarray(anchor_completion_scores, dtype=np.float64)
        if score_array.shape == (len(rows_in_input_order), len(rows_in_input_order)):
            for first_idx in range(len(rows_in_input_order)):
                for second_idx in range(first_idx + 1, len(rows_in_input_order)):
                    score = float(score_array[first_idx, second_idx])
                    if not np.isfinite(score):
                        continue
                    identity = pair_identity_from_legal_row_ids(
                        rows_in_input_order[first_idx].row_id,
                        rows_in_input_order[second_idx].row_id,
                        legal_by_id,
                        pair_rows_by_id_pair=pair_rows_by_id_pair,
                    )
                    scored.append((identity, score))
        else:
            for pair_like, score in _iter_pair_scores(anchor_completion_scores):
                identity = _identity_from_pair_like(pair_like, legal_by_id, pair_rows_by_id_pair)
                scored.append((identity, float(score)))
    elif anchor_completion_scores is not None:
        for pair_like, score in _iter_pair_scores(anchor_completion_scores):
            identity = _identity_from_pair_like(pair_like, legal_by_id, pair_rows_by_id_pair)
            scored.append((identity, float(score)))
    ranked = _rank_scored_identities(scored)[: max(0, int(limit))]
    return [
        _SourceProposal(
            identity=identity,
            source_type=SOURCE_ANCHOR_CONDITIONED_COMPLETION,
            score=score,
            source_rank=rank,
            quota_id=SOURCE_ANCHOR_CONDITIONED_COMPLETION,
            inclusion_kind="deterministic_top_k",
            correction_mode="uncorrected_logged",
            reason=SOURCE_ANCHOR_CONDITIONED_COMPLETION,
        )
        for rank, (identity, score) in enumerate(ranked)
    ]


def _cell_marginal_cross_proposals(
    rows_in_input_order: Sequence[LegalRowIdentityV1],
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
    *,
    cell_marginal_logits: np.ndarray | Sequence[float] | None,
    top_cells: int,
    limit: int,
) -> list[_SourceProposal]:
    if cell_marginal_logits is None or top_cells <= 0 or limit <= 0:
        return []
    logits = np.asarray(cell_marginal_logits, dtype=np.float64).reshape(-1)
    if logits.shape[0] != len(rows_in_input_order):
        raise ValueError(
            f"cell_marginal_logits length {logits.shape[0]} does not match legal rows {len(rows_in_input_order)}"
        )
    ranked_cells = sorted(
        (
            (rows_in_input_order[idx], float(logit))
            for idx, logit in enumerate(logits.tolist())
            if np.isfinite(float(logit))
        ),
        key=lambda item: (-item[1], item[0].row_id, item[0].cell),
    )[: min(int(top_cells), len(rows_in_input_order))]
    scored: list[tuple[PairRowIdentityV1, float]] = []
    for first_idx in range(len(ranked_cells)):
        for second_idx in range(first_idx + 1, len(ranked_cells)):
            first, first_score = ranked_cells[first_idx]
            second, second_score = ranked_cells[second_idx]
            identity = pair_identity_from_legal_row_ids(
                first.row_id,
                second.row_id,
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
            scored.append((identity, first_score + second_score))
    ranked = _rank_scored_identities(scored)[: max(0, int(limit))]
    return [
        _SourceProposal(
            identity=identity,
            source_type=SOURCE_CELL_MARGINAL_CROSS,
            score=score,
            source_rank=rank,
            quota_id=SOURCE_CELL_MARGINAL_CROSS,
            inclusion_kind="deterministic_top_k",
            correction_mode="uncorrected_logged",
            reason=SOURCE_CELL_MARGINAL_CROSS,
        )
        for rank, (identity, score) in enumerate(ranked)
    ]


def _structured_diversity_proposals(
    sorted_rows: Sequence[LegalRowIdentityV1],
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
    *,
    seed: int,
    limit: int,
) -> list[_SourceProposal]:
    if limit <= 0 or len(sorted_rows) < 2:
        return []
    per_bucket: dict[tuple[int, int, bool, bool], tuple[int, PairRowIdentityV1, float]] = {}
    extras: list[tuple[int, PairRowIdentityV1, float]] = []
    for first_idx in range(len(sorted_rows)):
        first = sorted_rows[first_idx]
        for second in sorted_rows[first_idx + 1 :]:
            identity = pair_identity_from_legal_row_ids(
                first.row_id,
                second.row_id,
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
            dist = hex_distance(first.cell, second.cell)
            bucket = 0 if dist <= 2 else 1 if dist <= 5 else 2
            axis = _pair_axis_code(first.cell, second.cell)
            line = same_hex_line(first.cell, second.cell)
            origin_axis = same_origin_axis(first.cell, second.cell)
            hash_value = _stable_u64(seed, first.row_id, second.row_id, bucket, axis)
            score = 1.0 - (hash_value / float(2**64 - 1))
            bucket_key = (bucket, axis, line, origin_axis)
            current = per_bucket.get(bucket_key)
            if current is None or hash_value < current[0]:
                per_bucket[bucket_key] = (hash_value, identity, score)
            extras.append((hash_value, identity, score))
    protected_bucket_rows = sorted(per_bucket.values(), key=lambda item: (item[0], item[1].row_id_pair))
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[PairRowIdentityV1, float]] = []
    for _hash, identity, score in protected_bucket_rows:
        if identity.row_id_pair not in seen:
            seen.add(identity.row_id_pair)
            ordered.append((identity, score))
    for _hash, identity, score in sorted(extras, key=lambda item: (item[0], item[1].row_id_pair)):
        if len(ordered) >= limit:
            break
        if identity.row_id_pair in seen:
            continue
        seen.add(identity.row_id_pair)
        ordered.append((identity, score))
    return [
        _SourceProposal(
            identity=identity,
            source_type=SOURCE_STRUCTURED_DIVERSITY,
            score=score,
            source_rank=rank,
            quota_id=SOURCE_STRUCTURED_DIVERSITY,
            inclusion_kind="structured_quota",
            correction_mode="uncorrected_logged",
            reason=SOURCE_STRUCTURED_DIVERSITY,
        )
        for rank, (identity, score) in enumerate(ordered[:limit])
    ]


def _deterministic_row_pool(
    sorted_rows: Sequence[LegalRowIdentityV1],
    *,
    seed: int,
    limit: int,
) -> tuple[LegalRowIdentityV1, ...]:
    """Return a row-order-independent bounded pool for non-exhaustive sources.

    Direct pair retrieval remains blockwise-exact over the full legal table.
    Structured diversity and blind canaries are proposal-support sources; on
    late 500-move games the full Rust legal table can contain thousands of
    rows, so enumerating every legal pair in Python makes V1 unable to produce
    game records.  A deterministic hash pool preserves row identity and avoids
    legal-order bias while keeping these auxiliary sources bounded.
    """

    row_limit = int(limit)
    if row_limit <= 0 or len(sorted_rows) <= row_limit:
        return tuple(sorted_rows)
    ranked = sorted(
        sorted_rows,
        key=lambda row: (_stable_u64(seed, row.row_id, row.q, row.r), row.row_id, row.cell),
    )
    return tuple(sorted(ranked[:row_limit], key=lambda row: (row.row_id, row.cell)))


def _blind_canary_proposals(
    sorted_rows: Sequence[LegalRowIdentityV1],
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
    *,
    seed: int,
    limit: int,
    legal_pair_count: int,
) -> list[_SourceProposal]:
    if limit <= 0 or len(sorted_rows) < 2:
        return []
    selected: list[tuple[int, PairRowIdentityV1]] = []
    for first_idx in range(len(sorted_rows)):
        first = sorted_rows[first_idx]
        for second in sorted_rows[first_idx + 1 :]:
            identity = pair_identity_from_legal_row_ids(
                first.row_id,
                second.row_id,
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
            selected.append((_stable_u64(seed, first.row_id, second.row_id, legal_pair_count), identity))
    selected.sort(key=lambda item: (item[0], item[1].row_id_pair))
    proposals = []
    for rank, (hash_value, identity) in enumerate(selected[:limit]):
        score = 1.0 - (hash_value / float(2**64 - 1))
        proposals.append(
            _SourceProposal(
                identity=identity,
                source_type=SOURCE_BLIND_CANARY,
                score=score,
                source_rank=rank,
                quota_id=SOURCE_BLIND_CANARY,
                inclusion_kind="diagnostic_canary",
                correction_mode="training_forbidden",
                forced=True,
                reason=SOURCE_BLIND_CANARY,
            )
        )
    return proposals


def _iter_pair_scores(source: Mapping[Any, float] | Sequence[Any]) -> list[tuple[Any, float]]:
    if isinstance(source, Mapping):
        return [(key, float(value)) for key, value in source.items()]
    out: list[tuple[Any, float]] = []
    for item in source:
        if isinstance(item, Mapping):
            pair = item.get("pair", item.get("pair_key", item))
            out.append((pair, float(item.get("score", item.get("logit", 0.0)))))
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) == 2:
                out.append((item[0], float(item[1])))
            elif len(item) >= 3:
                out.append(((item[0], item[1]), float(item[2])))
            else:
                raise ValueError(f"unsupported pair score row: {item!r}")
            continue
        raise ValueError(f"unsupported pair score row: {item!r}")
    return out


def _identity_from_pair_like(
    pair_like: Any,
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
) -> PairRowIdentityV1:
    if isinstance(pair_like, DirectPairRetrievalCandidateV1):
        pair_like = pair_like.identity
    if isinstance(pair_like, PairRowIdentityV1):
        return pair_identity_from_legal_row_ids(
            pair_like.first_legal_row_id,
            pair_like.second_legal_row_id,
            legal_by_id,
            pair_rows_by_id_pair=pair_rows_by_id_pair,
        )
    if isinstance(pair_like, Mapping):
        if "first_legal_row_id" in pair_like and "second_legal_row_id" in pair_like:
            return pair_identity_from_legal_row_ids(
                int(pair_like["first_legal_row_id"]),
                int(pair_like["second_legal_row_id"]),
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
        if "first" in pair_like and "second" in pair_like:
            return _identity_from_cells(pair_like["first"], pair_like["second"], legal_by_id, pair_rows_by_id_pair)
        if "pair_key" in pair_like:
            return _identity_from_pair_like(pair_like["pair_key"], legal_by_id, pair_rows_by_id_pair)
    if isinstance(pair_like, Sequence) and not isinstance(pair_like, (str, bytes)):
        if len(pair_like) == 2 and all(isinstance(value, (int, np.integer)) for value in pair_like):
            return pair_identity_from_legal_row_ids(
                int(pair_like[0]),
                int(pair_like[1]),
                legal_by_id,
                pair_rows_by_id_pair=pair_rows_by_id_pair,
            )
        if len(pair_like) == 2:
            return _identity_from_cells(pair_like[0], pair_like[1], legal_by_id, pair_rows_by_id_pair)
        if len(pair_like) >= 4 and all(isinstance(value, (int, np.integer)) for value in pair_like[:4]):
            return _identity_from_cells(
                (int(pair_like[0]), int(pair_like[1])),
                (int(pair_like[2]), int(pair_like[3])),
                legal_by_id,
                pair_rows_by_id_pair,
            )
    raise ValueError(f"cannot resolve V1 pair identity from {pair_like!r}")


def _identity_from_cells(
    first: Any,
    second: Any,
    legal_by_id: Mapping[int, LegalRowIdentityV1],
    pair_rows_by_id_pair: Mapping[tuple[int, int], PairRowIdentityV1],
) -> PairRowIdentityV1:
    first_cell, second_cell = canonical_pair_key(first, second)
    id_by_cell = {row.cell: row.row_id for row in legal_by_id.values()}
    if first_cell not in id_by_cell or second_cell not in id_by_cell:
        raise ValueError(f"V1 pair references cells outside legal rows: {(first_cell, second_cell)}")
    return pair_identity_from_legal_row_ids(
        id_by_cell[first_cell],
        id_by_cell[second_cell],
        legal_by_id,
        pair_rows_by_id_pair=pair_rows_by_id_pair,
    )


def _rank_scored_identities(scored: Sequence[tuple[PairRowIdentityV1, float]]) -> list[tuple[PairRowIdentityV1, float]]:
    best: dict[tuple[int, int], tuple[PairRowIdentityV1, float]] = {}
    for identity, score_raw in scored:
        score = float(score_raw)
        if not np.isfinite(score):
            continue
        key = identity.row_id_pair
        current = best.get(key)
        if current is None or (-score, identity.row_id_pair, identity.pair_key) < (
            -current[1],
            current[0].row_id_pair,
            current[0].pair_key,
        ):
            best[key] = (identity, score)
    return sorted(best.values(), key=lambda item: (-item[1], item[0].row_id_pair, item[0].pair_key))


def _dedup_source_proposals(proposals: Sequence[_SourceProposal]) -> tuple[list[_SourceProposal], int]:
    best: dict[tuple[int, int], _SourceProposal] = {}
    duplicates = 0
    for proposal in proposals:
        current = best.get(proposal.row_id_pair)
        if current is None:
            best[proposal.row_id_pair] = proposal
            continue
        duplicates += 1
        if _proposal_sort_key(proposal) < _proposal_sort_key(current):
            best[proposal.row_id_pair] = proposal
    return sorted(best.values(), key=_proposal_sort_key), duplicates


def _admit_proposal(states: dict[tuple[int, int], _CandidateState], proposal: _SourceProposal) -> int:
    duplicate = 1 if proposal.row_id_pair in states else 0
    state = states.get(proposal.row_id_pair)
    if state is None:
        state = _CandidateState(identity=proposal.identity, selector_score=proposal.score)
        states[proposal.row_id_pair] = state
    state.protected = state.protected or proposal.protected
    state.forced = state.forced or proposal.forced
    state.terminal_exact = state.terminal_exact or proposal.terminal_exact
    state.terminal_equivalence = state.terminal_equivalence or proposal.terminal_equivalence
    state.selector_score = max(float(state.selector_score), float(proposal.score))
    state.source_scores[proposal.source_type] = max(
        float(state.source_scores.get(proposal.source_type, -np.inf)),
        float(proposal.score),
    )
    state.source_contributions.append(_source_contribution(proposal))
    if proposal.reason:
        state.reason_parts.append(proposal.reason)
    return duplicate


def _source_contribution(proposal: _SourceProposal) -> V1CandidateSourceContribution:
    heuristic = 1.0 / float(proposal.source_rank + 1)
    if proposal.inclusion_kind == "tactical_protected":
        heuristic = 1.0
    return V1CandidateSourceContribution(
        source_type=proposal.source_type,
        source_rank=proposal.source_rank,
        source_weight=1.0,
        local_probability_or_score=proposal.score,
        quota_id=proposal.quota_id,
        inclusion_kind=proposal.inclusion_kind,
        exact_inclusion_probability=None,
        heuristic_propensity=heuristic,
        correction_mode=proposal.correction_mode,
    )


def _candidate_from_state(
    state: _CandidateState,
    *,
    config: PairCandidateSelectorV1Config,
    row_table_schema_version: int,
    admission_generation: int,
    root_or_interior: str,
) -> PairCandidateV1:
    contributions = tuple(sorted(state.source_contributions, key=_source_contribution_sort_key))
    flags = ["admitted"]
    if state.forced:
        flags.append("forced")
    if state.terminal_exact:
        flags.append("terminal_exact")
    if state.terminal_equivalence:
        flags.append("terminal_equivalent")
    reason_text = ";".join(state.reason_parts)
    if "opponent_hot_cover" in reason_text:
        flags.append("terminal_cover")
        flags.append("covers_all_opponent_win_requirements")
    if "impossible_to_cover_flag" in reason_text:
        flags.append("impossible_to_cover")
    proposal = V1ProposalPropensityMetadata(
        proposal_policy=PAIR_CANDIDATE_SELECTOR_VERSION_V1,
        correction_mode=_proposal_correction_mode(contributions),
        total_proposal_probability=None,
        log_proposal_probability=None,
        sampling_without_replacement=False,
        notes=",".join(sorted({source.source_type for source in contributions})),
    )
    identity = state.identity
    return PairCandidateV1(
        candidate_id=_candidate_id(config, identity, row_table_schema_version),
        pair_key=identity.pair_key,
        first_legal_row_id=identity.first_legal_row_id,
        second_legal_row_id=identity.second_legal_row_id,
        row_table_schema_version=row_table_schema_version,
        source_contributions=contributions,
        proposal_propensity_metadata=proposal,
        forced_exploration_flag=state.forced,
        tactical_protected_flag=state.protected,
        terminal_exact_flag=state.terminal_exact,
        terminal_equivalence_flag=state.terminal_equivalence,
        target_support_flags=tuple(flags),
        admission_generation=int(admission_generation),
        root_or_interior=root_or_interior,
        pair_row_key=identity.pair_row_key,
        selector_score=state.selector_score,
        source_scores=dict(state.source_scores),
        candidate_selection_reason=",".join(sorted(set(state.reason_parts))),
    )


def _candidate_id(
    config: PairCandidateSelectorV1Config,
    identity: PairRowIdentityV1,
    row_table_schema_version: int,
) -> str:
    row_key = "none" if identity.pair_row_key is None else f"{identity.pair_row_key:016x}"
    return (
        f"{config.candidate_id_prefix}:schema{int(row_table_schema_version)}:"
        f"{identity.first_legal_row_id}:{identity.second_legal_row_id}:{row_key}"
    )


def _proposal_correction_mode(contributions: Sequence[V1CandidateSourceContribution]) -> str:
    modes = {source.correction_mode for source in contributions}
    if modes == {"training_forbidden"}:
        return "training_forbidden"
    return "uncorrected_logged"


def _apply_rich_rerank(
    candidates: Sequence[PairCandidateV1],
    callback: Callable[[Sequence[PairCandidateV1]], Mapping[Any, float] | Sequence[float]] | None,
) -> tuple[PairCandidateV1, ...]:
    if callback is None or not candidates:
        return tuple(candidates)
    scores_raw = callback(tuple(candidates))
    if isinstance(scores_raw, Mapping):
        scores = [_rich_score_from_mapping(candidate, scores_raw) for candidate in candidates]
    else:
        scores = [float(score) for score in scores_raw]
        if len(scores) != len(candidates):
            raise ValueError("rich_pair_rerank returned a score sequence with the wrong length")
    return tuple(_replace_rich_score(candidate, score) for candidate, score in zip(candidates, scores))


def _rich_score_from_mapping(candidate: PairCandidateV1, scores: Mapping[Any, float]) -> float:
    for key in (
        candidate.candidate_id,
        candidate.row_id_pair,
        candidate.pair_key,
        tuple(candidate.pair_key),
    ):
        if key in scores:
            return float(scores[key])
    return 0.0


def _replace_rich_score(candidate: PairCandidateV1, score: float) -> PairCandidateV1:
    return PairCandidateV1(
        candidate_id=candidate.candidate_id,
        pair_key=candidate.pair_key,
        first_legal_row_id=candidate.first_legal_row_id,
        second_legal_row_id=candidate.second_legal_row_id,
        row_table_schema_version=candidate.row_table_schema_version,
        source_contributions=candidate.source_contributions,
        proposal_propensity_metadata=candidate.proposal_propensity_metadata,
        forced_exploration_flag=candidate.forced_exploration_flag,
        tactical_protected_flag=candidate.tactical_protected_flag,
        terminal_exact_flag=candidate.terminal_exact_flag,
        terminal_equivalence_flag=candidate.terminal_equivalence_flag,
        target_support_flags=candidate.target_support_flags,
        admission_generation=candidate.admission_generation,
        root_or_interior=candidate.root_or_interior,
        pair_row_key=candidate.pair_row_key,
        selector_score=candidate.selector_score,
        rich_rerank_score=float(score),
        source_scores=candidate.source_scores,
        candidate_selection_reason=candidate.candidate_selection_reason,
    )


def _validate_admitted_candidates(candidates: Sequence[PairCandidateV1]) -> None:
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        if not candidate.source_contributions:
            raise ValueError("V1 admission produced a candidate without source metadata")
        if candidate.proposal_propensity_metadata is None:
            raise ValueError("V1 admission produced a candidate without proposal metadata")
        if not candidate.target_support_flags:
            raise ValueError("V1 admission produced a candidate without target-support metadata")
        if candidate.row_id_pair in seen:
            raise ValueError(f"V1 admission produced duplicate pair row {candidate.row_id_pair}")
        seen.add(candidate.row_id_pair)


def _candidate_order_key(candidate: PairCandidateV1) -> tuple[int, float, float, int, int, PairKey]:
    rich = 0.0 if candidate.rich_rerank_score is None else float(candidate.rich_rerank_score)
    return (
        0 if candidate.tactical_protected_flag else 1,
        -rich,
        -float(candidate.selector_score),
        candidate.first_legal_row_id,
        candidate.second_legal_row_id,
        candidate.pair_key,
    )


def _proposal_sort_key(proposal: _SourceProposal) -> tuple[float, int, int, PairKey, str]:
    return (
        -float(proposal.score),
        proposal.identity.first_legal_row_id,
        proposal.identity.second_legal_row_id,
        proposal.identity.pair_key,
        proposal.quota_id,
    )


def _source_contribution_sort_key(source: V1CandidateSourceContribution) -> tuple[int, int, str, str]:
    priority = {source_type: idx for idx, source_type in enumerate((SOURCE_TERMINAL_EXACT,) + DEFAULT_SOURCE_PRIORITY)}
    return (
        priority.get(source.source_type, 999),
        999_999 if source.source_rank is None else int(source.source_rank),
        source.source_type,
        "" if source.quota_id is None else str(source.quota_id),
    )


def _pair_axis_code(first: PairCoord, second: PairCoord) -> int:
    if int(first[0]) == int(second[0]):
        return 0
    if int(first[1]) == int(second[1]):
        return 1
    if int(first[0]) + int(first[1]) == int(second[0]) + int(second[1]):
        return 2
    return 3


def _stable_u64(*values: int) -> int:
    mask = (1 << 64) - 1
    value = 0xCBF29CE484222325
    for raw in values:
        number = int(raw) & mask
        for byte in number.to_bytes(8, "little", signed=False):
            value ^= byte
            value = (value * 0x100000001B3) & mask
    return value


__all__ = [
    "PAIR_CANDIDATE_SELECTOR_VERSION_V1",
    "SOURCE_ANCHOR_CONDITIONED_COMPLETION",
    "SOURCE_BLIND_CANARY",
    "SOURCE_CELL_MARGINAL_CROSS",
    "SOURCE_DIRECT_PAIR_RETRIEVAL",
    "SOURCE_STRUCTURED_DIVERSITY",
    "SOURCE_TERMINAL_EXACT",
    "PairCandidateSelectorResultV1",
    "PairCandidateSelectorTelemetryV1",
    "PairCandidateSelectorV1Config",
    "PairCandidateV1",
    "select_pair_candidates_v1",
]
