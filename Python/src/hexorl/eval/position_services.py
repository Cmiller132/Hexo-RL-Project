"""Read-only position contract services shared by eval and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hexorl.contracts.candidates import CandidateContractBuilder, CandidateTable
from hexorl.contracts.identity import stable_digest
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.pairs import PairActionTable, PairActionTableBuilder, PairStrategy
from hexorl.contracts.symmetry import (
    transform_history,
    transform_pair_policy_target,
    transform_policy_target,
)
from hexorl.engine.encoding import encode_board_and_legal
from hexorl.engine.tactical import scan_tactical_oracle_from_history
from hexorl.graph.tensorize import GraphBatch, build_graph_batch_from_history
from hexorl.models.specs import ModelSpec
from hexorl.search.context import SearchContext


PolicyTarget = Sequence[tuple[int, int, float]]
PairPolicyTarget = Sequence[tuple[tuple[int, int], tuple[int, int], float]]


@dataclass(frozen=True)
class PositionContracts:
    tensor: np.ndarray
    offset_q: int
    offset_r: int
    legal_table: LegalActionTable
    candidate_table: CandidateTable | None
    pair_table: PairActionTable | None


def compact_history_hash(history: bytes) -> str:
    return stable_digest(("history", bytes(history)))


def build_position_contracts(
    history: bytes,
    *,
    policy_target: PolicyTarget = (),
    pair_policy_target: PairPolicyTarget = (),
    include_candidates: bool = True,
    include_pair_rows: bool = False,
    candidate_budget: int = 512,
    near_radius: int = 8,
    constrain_threats: bool = True,
) -> PositionContracts:
    """Build canonical read-only contracts for a compact-history position."""

    tensor, offset_q, offset_r, legal_rows, _legal_bytes = encode_board_and_legal(
        history,
        near_radius,
        constrain_threats,
    )
    rows = [(int(q), int(r)) for q, r in legal_rows.tolist()]
    legal_table = LegalActionTable.from_rows(
        rows,
        source="rust:legal",
        history_hash=compact_history_hash(history),
        current_player=(len(history) // 12) % 2,
        placements_remaining=1,
    )
    candidate_table = None
    pair_table = None
    if include_candidates:
        width = min(max(int(candidate_budget), 1), 512)
        oracle = scan_tactical_oracle_from_history(
            history,
            rows,
            offset_q=int(offset_q),
            offset_r=int(offset_r),
        )
        candidate_table = CandidateContractBuilder().build(
            rows,
            list(policy_target),
            offset_q=int(offset_q),
            offset_r=int(offset_r),
            budget=min(max(len(rows), 1), width),
            storage_width=width,
            winning_moves=oracle.win_now_cells,
            forced_block_moves=oracle.forced_block_cells,
            cover_cells=oracle.cover_cells,
            open_four_cells=oracle.open_four_cells,
            open_five_cells=oracle.open_five_cells,
        )
        if include_pair_rows and len(rows) >= 2:
            pair_table = PairActionTableBuilder().build(
                candidate_table,
                list(pair_policy_target),
                strategy=PairStrategy(
                    generation_mode="capped_fill",
                    max_pairs=min(512, max(1, len(rows) * (len(rows) - 1) // 2)),
                ),
                legal_moves=rows,
            )
    return PositionContracts(
        tensor=np.asarray(tensor, dtype=np.float32).reshape(1, 13, 33, 33),
        offset_q=int(offset_q),
        offset_r=int(offset_r),
        legal_table=legal_table,
        candidate_table=candidate_table,
        pair_table=pair_table,
    )


def build_graph_contract(
    history: bytes,
    *,
    policy_target: PolicyTarget = (),
    pair_policy_target: PairPolicyTarget = (),
    include_pair_rows: bool = False,
    near_radius: int = 8,
    max_pair_rows: int = 4096,
) -> GraphBatch:
    return build_graph_batch_from_history(
        history,
        policy_target=policy_target,
        pair_policy_target=pair_policy_target,
        radius=near_radius,
        max_pair_rows=max_pair_rows,
        include_pair_rows=include_pair_rows,
    )


def build_search_context(
    history: bytes,
    *,
    model_spec: ModelSpec,
    recipe_id: str,
    policy_target: PolicyTarget = (),
    pair_policy_target: PairPolicyTarget = (),
    candidate_budget: int = 256,
    near_radius: int = 8,
    constrain_threats: bool = True,
    inference_protocol: str = "local_model_eval_v1",
) -> SearchContext:
    include_candidates = model_spec.kind == "graph_hybrid"
    contracts = build_position_contracts(
        history,
        policy_target=policy_target,
        pair_policy_target=pair_policy_target,
        include_candidates=include_candidates,
        include_pair_rows=False,
        candidate_budget=candidate_budget,
        near_radius=near_radius,
        constrain_threats=constrain_threats,
    )
    graph_batch = None
    if model_spec.is_global_graph:
        graph_batch = build_graph_contract(
            history,
            policy_target=policy_target,
            pair_policy_target=pair_policy_target,
            include_pair_rows=False,
            near_radius=near_radius,
            max_pair_rows=0,
        )
    return SearchContext.create(
        phase="root",
        legal_table=contracts.legal_table,
        model_family=model_spec.kind,
        model_spec_version=str(model_spec.version),
        recipe_id=recipe_id,
        search_id="arena",
        pair_strategy_id="none",
        tensor=contracts.tensor,
        history_bytes=history,
        candidate_table=contracts.candidate_table,
        graph_batch=graph_batch,
        inference_protocol=inference_protocol,
        extra={"offset_q": contracts.offset_q, "offset_r": contracts.offset_r},
    )


def position_contract_payload(contracts: PositionContracts) -> dict[str, Any]:
    return {
        "legal_table_hash": contracts.legal_table.table_hash,
        "legal_table_source": contracts.legal_table.source,
        "candidate_contract_hash": "" if contracts.candidate_table is None else contracts.candidate_table.table_hash,
        "candidate_contract_source": "" if contracts.candidate_table is None else contracts.candidate_table.source,
        "pair_table_hash": "" if contracts.pair_table is None else contracts.pair_table.table_hash,
        "pair_table_source": "" if contracts.pair_table is None else contracts.pair_table.source,
    }


def transform_position_inputs(
    history: bytes,
    *,
    policy_target: PolicyTarget = (),
    pair_policy_target: PairPolicyTarget = (),
    symmetry_index: int,
) -> tuple[bytes, tuple[tuple[int, int, float], ...], tuple[tuple[tuple[int, int], tuple[int, int], float], ...]]:
    transformed_history = transform_history(history, symmetry_index)
    transformed_policy = tuple(transform_policy_target(list(policy_target), symmetry_index))
    transformed_pair = tuple(transform_pair_policy_target(list(pair_policy_target), symmetry_index))
    return transformed_history, transformed_policy, transformed_pair
