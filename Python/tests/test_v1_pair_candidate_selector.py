import pytest

from hexorl.search.pair_candidate_selector_v1 import (
    SOURCE_ANCHOR_CONDITIONED_COMPLETION,
    SOURCE_BLIND_CANARY,
    SOURCE_DIRECT_PAIR_RETRIEVAL,
    SOURCE_STRUCTURED_DIVERSITY,
    SOURCE_TERMINAL_EXACT,
    PairCandidateSelectorV1Config,
    PairCandidateV1,
    select_pair_candidates_v1,
)
from hexorl.search.pair_scorer_v1 import direct_pair_retrieval_v1
from hexorl.selfplay.records import V1CandidateSourceContribution, V1ProposalPropensityMetadata


LEGAL_ROWS = (
    (10, -1, 1),
    (2, 0, 0),
    (7, 1, 0),
    (4, 0, 1),
    (9, 2, -1),
    (12, -2, 2),
)


def _cell(row_id: int) -> tuple[int, int]:
    for legal_id, q, r in LEGAL_ROWS:
        if legal_id == row_id:
            return (q, r)
    raise AssertionError(f"unknown row id {row_id}")


def _pair_row(first_id: int, second_id: int, row_id: int = 0) -> tuple[int, int, int, int, int, int, int, int]:
    first_id, second_id = sorted((int(first_id), int(second_id)))
    first = _cell(first_id)
    second = _cell(second_id)
    return (
        int(row_id),
        first_id,
        second_id,
        first[0],
        first[1],
        second[0],
        second[1],
        (first_id << 32) | second_id,
    )


def _source() -> V1CandidateSourceContribution:
    return V1CandidateSourceContribution(
        source_type=SOURCE_DIRECT_PAIR_RETRIEVAL,
        source_rank=0,
        source_weight=1.0,
        local_probability_or_score=1.0,
        quota_id=SOURCE_DIRECT_PAIR_RETRIEVAL,
        inclusion_kind="deterministic_top_k",
        heuristic_propensity=1.0,
        correction_mode="uncorrected_logged",
    )


def _proposal() -> V1ProposalPropensityMetadata:
    return V1ProposalPropensityMetadata(
        proposal_policy="pair_candidate_selector_v1",
        correction_mode="uncorrected_logged",
        total_proposal_probability=1.0,
    )


def test_v1_selector_enforces_source_quotas_and_metadata():
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=8,
        source_quotas={
            SOURCE_DIRECT_PAIR_RETRIEVAL: 2,
            SOURCE_ANCHOR_CONDITIONED_COMPLETION: 1,
            SOURCE_BLIND_CANARY: 0,
        },
        source_priority=(SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_ANCHOR_CONDITIONED_COMPLETION, SOURCE_BLIND_CANARY),
    )

    result = select_pair_candidates_v1(
        LEGAL_ROWS,
        direct_retrieval_rows=[
            ((2, 4), 10.0),
            ((2, 7), 9.0),
            ((2, 9), 8.0),
        ],
        anchor_completion_scores={
            (_cell(4), _cell(9)): 7.0,
            (_cell(7), _cell(9)): 6.0,
        },
        config=cfg,
    )

    assert [candidate.row_id_pair for candidate in result.candidates] == [(2, 4), (2, 7), (4, 9)]
    assert result.telemetry.admitted_by_source[SOURCE_DIRECT_PAIR_RETRIEVAL] == 2
    assert result.telemetry.admitted_by_source[SOURCE_ANCHOR_CONDITIONED_COMPLETION] == 1
    assert result.telemetry.quota_evictions == 2
    for candidate in result.candidates:
        assert candidate.source_contributions
        assert candidate.proposal_propensity_metadata.proposal_policy == "pair_candidate_selector_v1"
        assert "admitted" in candidate.target_support_flags
    assert [pair.pair_key for pair in result.replay_candidate_pairs] == [
        candidate.pair_key for candidate in result.candidates
    ]


def test_v1_candidate_rejects_missing_source_proposal_or_support_metadata():
    with pytest.raises(ValueError, match="source metadata"):
        PairCandidateV1(
            candidate_id="bad-source",
            pair_key=(_cell(2), _cell(4)),
            first_legal_row_id=2,
            second_legal_row_id=4,
            row_table_schema_version=1,
            source_contributions=(),
            proposal_propensity_metadata=_proposal(),
            forced_exploration_flag=False,
            tactical_protected_flag=False,
            terminal_exact_flag=False,
            terminal_equivalence_flag=False,
            target_support_flags=("admitted",),
            admission_generation=0,
            root_or_interior="root",
        )

    with pytest.raises(ValueError, match="proposal metadata"):
        PairCandidateV1(
            candidate_id="bad-proposal",
            pair_key=(_cell(2), _cell(4)),
            first_legal_row_id=2,
            second_legal_row_id=4,
            row_table_schema_version=1,
            source_contributions=(_source(),),
            proposal_propensity_metadata=None,  # type: ignore[arg-type]
            forced_exploration_flag=False,
            tactical_protected_flag=False,
            terminal_exact_flag=False,
            terminal_equivalence_flag=False,
            target_support_flags=("admitted",),
            admission_generation=0,
            root_or_interior="root",
        )

    with pytest.raises(ValueError, match="target-support metadata"):
        PairCandidateV1(
            candidate_id="bad-support",
            pair_key=(_cell(2), _cell(4)),
            first_legal_row_id=2,
            second_legal_row_id=4,
            row_table_schema_version=1,
            source_contributions=(_source(),),
            proposal_propensity_metadata=_proposal(),
            forced_exploration_flag=False,
            tactical_protected_flag=False,
            terminal_exact_flag=False,
            terminal_equivalence_flag=False,
            target_support_flags=(),
            admission_generation=0,
            root_or_interior="root",
        )


def test_v1_selector_canonicalizes_and_deduplicates_across_sources():
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=5,
        source_quotas={
            SOURCE_DIRECT_PAIR_RETRIEVAL: 2,
            SOURCE_ANCHOR_CONDITIONED_COMPLETION: 2,
            SOURCE_BLIND_CANARY: 0,
        },
        source_priority=(SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_ANCHOR_CONDITIONED_COMPLETION, SOURCE_BLIND_CANARY),
    )

    result = select_pair_candidates_v1(
        LEGAL_ROWS,
        direct_retrieval_rows=[((4, 2), 5.0), ((2, 4), 4.0)],
        anchor_completion_scores={(_cell(4), _cell(2)): 3.0},
        config=cfg,
    )

    assert [candidate.row_id_pair for candidate in result.candidates] == [(2, 4)]
    candidate = result.candidates[0]
    assert candidate.pair_key == tuple(sorted((_cell(2), _cell(4))))
    assert {
        source.source_type for source in candidate.source_contributions
    } == {SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_ANCHOR_CONDITIONED_COMPLETION}
    assert result.telemetry.duplicate_proposals >= 2


def test_v1_tactical_protected_candidates_survive_quota_budget_and_rerank():
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=2,
        source_quotas={SOURCE_DIRECT_PAIR_RETRIEVAL: 1, SOURCE_BLIND_CANARY: 0},
        source_priority=(SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_BLIND_CANARY),
    )
    tactical_payload = {
        "pair_row_schema_version": 1,
        "hot_completion_pairs": [_pair_row(2, 4, row_id=0), _pair_row(7, 9, row_id=1)],
        "hot_cover_pairs": [],
        "terminal_equivalent_pairs": [],
    }

    result = select_pair_candidates_v1(
        LEGAL_ROWS,
        tactical_payload=tactical_payload,
        direct_retrieval_rows=[((10, 12), 99.0)],
        rich_pair_rerank=lambda candidates: {candidate.row_id_pair: -1000.0 for candidate in candidates},
        config=cfg,
    )

    assert [candidate.row_id_pair for candidate in result.candidates] == [(2, 4), (7, 9)]
    assert all(candidate.tactical_protected_flag for candidate in result.candidates)
    assert all(candidate.terminal_exact_flag for candidate in result.candidates)
    assert all(SOURCE_TERMINAL_EXACT in candidate.source_scores for candidate in result.candidates)
    assert result.telemetry.protected_count == 2
    assert result.telemetry.budget_evictions == 1


def test_v1_tactical_protected_candidates_expand_beyond_candidate_budget():
    tactical_payload = {
        "pair_row_schema_version": 1,
        "hot_completion_pairs": [
            _pair_row(2, 4, row_id=0),
            _pair_row(2, 7, row_id=1),
            _pair_row(4, 9, row_id=2),
        ],
        "hot_cover_pairs": [],
        "terminal_equivalent_pairs": [],
    }
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=2,
        source_quotas={SOURCE_DIRECT_PAIR_RETRIEVAL: 0, SOURCE_BLIND_CANARY: 0},
        source_priority=(SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_BLIND_CANARY),
    )

    result = select_pair_candidates_v1(
        LEGAL_ROWS,
        tactical_payload=tactical_payload,
        config=cfg,
    )

    assert len(result.candidates) == 3
    assert result.telemetry.protected_count == 3
    assert result.telemetry.budget_evictions == 0
    assert all(candidate.tactical_protected_flag for candidate in result.candidates)


def test_v1_impossible_cover_payload_marks_hot_cover_candidate_support():
    tactical_payload = {
        "status": "hot_cover_impossible",
        "pair_row_schema_version": 1,
        "hot_completion_pairs": [],
        "hot_cover_pairs": [_pair_row(2, 4, row_id=0)],
        "terminal_equivalent_pairs": [],
        "impossible_to_cover": True,
    }
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=1,
        source_quotas={SOURCE_BLIND_CANARY: 0},
        source_priority=(SOURCE_BLIND_CANARY,),
    )

    result = select_pair_candidates_v1(
        LEGAL_ROWS,
        tactical_payload=tactical_payload,
        config=cfg,
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.tactical_protected_flag is True
    assert "terminal_cover" in candidate.target_support_flags
    assert "covers_all_opponent_win_requirements" in candidate.target_support_flags
    assert "impossible_to_cover" in candidate.target_support_flags


def test_v1_blind_canary_is_deterministic_and_marked_training_forbidden():
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=4,
        source_quotas={SOURCE_BLIND_CANARY: 2},
        source_priority=(SOURCE_BLIND_CANARY,),
        blind_canary_seed=12345,
    )

    first = select_pair_candidates_v1(LEGAL_ROWS, config=cfg)
    second = select_pair_candidates_v1(tuple(reversed(LEGAL_ROWS)), config=cfg)

    assert [candidate.row_id_pair for candidate in first.candidates] == [
        candidate.row_id_pair for candidate in second.candidates
    ]
    assert len(first.candidates) == 2
    for candidate in first.candidates:
        assert candidate.forced_exploration_flag
        assert "forced" in candidate.target_support_flags
        assert candidate.proposal_propensity_metadata.correction_mode == "training_forbidden"
        assert {
            source.inclusion_kind for source in candidate.source_contributions
        } == {"diagnostic_canary"}


def test_v1_direct_pair_retrieval_is_exact_and_chunking_stable():
    embeddings = [
        [1.0, 0.0, 0.5],
        [0.0, 2.0, 1.0],
        [3.0, 1.0, -0.5],
        [-1.0, 2.0, 0.25],
        [0.5, -0.25, 2.0],
        [2.0, -1.0, 0.75],
    ]

    block_one = direct_pair_retrieval_v1(LEGAL_ROWS, embeddings, top_k=6, block_size=1)
    block_three = direct_pair_retrieval_v1(LEGAL_ROWS, embeddings, top_k=6, block_size=3)

    assert [candidate.identity.row_id_pair for candidate in block_one.candidates] == [
        candidate.identity.row_id_pair for candidate in block_three.candidates
    ]
    assert [candidate.score for candidate in block_one.candidates] == pytest.approx(
        [candidate.score for candidate in block_three.candidates]
    )
    assert all(
        candidate.first_legal_row_id != candidate.second_legal_row_id
        for candidate in block_one.candidates
    )
    assert block_one.scored_pair_count == len(LEGAL_ROWS) * (len(LEGAL_ROWS) - 1) // 2


def test_v1_auxiliary_sources_use_bounded_deterministic_row_pool_for_large_tables():
    legal_rows = tuple((idx, idx % 97, idx // 97) for idx in range(600))
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=12,
        source_quotas={
            SOURCE_DIRECT_PAIR_RETRIEVAL: 0,
            SOURCE_ANCHOR_CONDITIONED_COMPLETION: 0,
            SOURCE_STRUCTURED_DIVERSITY: 6,
            SOURCE_BLIND_CANARY: 2,
        },
        source_priority=(SOURCE_STRUCTURED_DIVERSITY, SOURCE_BLIND_CANARY),
        structured_diversity_pool_rows=32,
        blind_canary_seed=777,
    )

    first = select_pair_candidates_v1(legal_rows, config=cfg)
    second = select_pair_candidates_v1(tuple(reversed(legal_rows)), config=cfg)

    assert first.telemetry.legal_row_count == 600
    assert first.telemetry.legal_pair_count == 600 * 599 // 2
    assert first.telemetry.proposed_by_source[SOURCE_STRUCTURED_DIVERSITY] <= 6
    assert first.telemetry.proposed_by_source[SOURCE_BLIND_CANARY] <= 2
    assert [candidate.row_id_pair for candidate in first.candidates] == [
        candidate.row_id_pair for candidate in second.candidates
    ]
    assert all(
        SOURCE_STRUCTURED_DIVERSITY in candidate.source_scores
        or SOURCE_BLIND_CANARY in candidate.source_scores
        for candidate in first.candidates
    )


def test_v1_selector_final_admission_is_stable_under_legal_row_order_changes():
    embeddings_by_id = {
        10: [1.0, 0.0, 0.5],
        2: [0.0, 2.0, 1.0],
        7: [3.0, 1.0, -0.5],
        4: [-1.0, 2.0, 0.25],
        9: [0.5, -0.25, 2.0],
        12: [2.0, -1.0, 0.75],
    }
    cfg = PairCandidateSelectorV1Config(
        candidate_budget=4,
        source_quotas={SOURCE_DIRECT_PAIR_RETRIEVAL: 4, SOURCE_BLIND_CANARY: 0},
        source_priority=(SOURCE_DIRECT_PAIR_RETRIEVAL, SOURCE_BLIND_CANARY),
        direct_retrieval_top_k=4,
        direct_retrieval_block_size=2,
    )

    forward_embeddings = [embeddings_by_id[row_id] for row_id, _q, _r in LEGAL_ROWS]
    reversed_rows = tuple(reversed(LEGAL_ROWS))
    reversed_embeddings = [embeddings_by_id[row_id] for row_id, _q, _r in reversed_rows]

    forward = select_pair_candidates_v1(LEGAL_ROWS, legal_cell_embeddings=forward_embeddings, config=cfg)
    reversed_result = select_pair_candidates_v1(
        reversed_rows,
        legal_cell_embeddings=reversed_embeddings,
        config=cfg,
    )

    assert [candidate.row_id_pair for candidate in forward.candidates] == [
        candidate.row_id_pair for candidate in reversed_result.candidates
    ]
    assert [candidate.pair_key for candidate in forward.candidates] == [
        candidate.pair_key for candidate in reversed_result.candidates
    ]
