import numpy as np

from hexorl.graph.batch import GraphBatch, graph_batch_with_admitted_pair_rows
from hexorl.selfplay.records import (
    V1CandidatePair,
    V1CandidateSourceContribution,
    V1ProposalPropensityMetadata,
)
from hexorl.v1_pair_contract import (
    V1_PAIR_FEATURE_DIM,
    V1_PAIR_FEATURE_NAMES,
    V1_PAIR_FEATURE_SCHEMA_VERSION,
    V1TerminalTacticalPayload,
    terminal_tactical_target_vector,
    v1_pair_features_for_candidates,
)


def _candidate(pair_key=((0, 0), (2, 0)), *, support=("admitted",)):
    return V1CandidatePair(
        candidate_id="candidate",
        pair_key=pair_key,
        first_legal_row_id=0,
        second_legal_row_id=1,
        row_table_schema_version=7,
        source_contributions=(
            V1CandidateSourceContribution(
                source_type="direct_pair_retrieval",
                source_rank=0,
                source_weight=1.0,
                local_probability_or_score=1.0,
                inclusion_kind="deterministic_top_k",
                correction_mode="exact_importance",
            ),
        ),
        proposal_propensity_metadata=V1ProposalPropensityMetadata(
            proposal_policy="pair_candidate_selector_v1",
            correction_mode="exact_importance",
            total_proposal_probability=1.0,
        ),
        forced_exploration_flag=False,
        terminal_exact_flag=False,
        terminal_equivalence_flag=False,
        target_support_flags=support,
        admission_generation=0,
        root_or_interior="root",
    )


def test_canonical_v1_pair_feature_schema_is_versioned_and_ordered():
    assert V1_PAIR_FEATURE_SCHEMA_VERSION == 2
    assert V1_PAIR_FEATURE_NAMES == (
        "axial_distance_norm",
        "same_axis",
        "same_line",
        "same_window",
        "terminal_exact_win",
        "terminal_equivalent_win",
        "terminal_exact_cover",
        "covers_all_opponent_win_requirements",
        "impossible_to_cover",
        "phase_full_turn",
        "phase_known_first",
        "phase_both_legal",
    )
    assert V1_PAIR_FEATURE_DIM == 12


def test_train_and_infer_v1_pair_features_match_for_same_candidate_row():
    candidate = _candidate()
    features = v1_pair_features_for_candidates([candidate])
    base_graph = GraphBatch(
        token_features=np.zeros((2, 4), dtype=np.float32),
        token_type=np.zeros(2, dtype=np.int32),
        token_qr=np.array([[0, 0], [2, 0]], dtype=np.int32),
        token_mask=np.ones(2, dtype=bool),
        legal_token_indices=np.array([0, 1], dtype=np.int64),
        legal_qr=np.array([[0, 0], [2, 0]], dtype=np.int32),
        legal_mask=np.ones(2, dtype=bool),
        pair_token_indices=np.zeros(0, dtype=np.int64),
        pair_first_indices=np.zeros(0, dtype=np.int64),
        pair_second_indices=np.zeros(0, dtype=np.int64),
        relation_bias=np.zeros((0, 0), dtype=np.float32),
        relation_type=np.zeros((0, 0), dtype=np.int32),
        policy_target=np.zeros(2, dtype=np.float32),
        opp_legal_qr=np.zeros((0, 2), dtype=np.int32),
        opp_legal_mask=np.zeros(0, dtype=bool),
        opp_policy_target=np.zeros(0, dtype=np.float32),
        pair_first_policy_target=np.zeros(2, dtype=np.float32),
        pair_policy_target=np.zeros(0, dtype=np.float32),
        pair_second_policy_target=np.zeros(2, dtype=np.float32),
        tactical_target=np.zeros(0, dtype=np.float32),
        placements_remaining=2,
        current_player=0,
    )
    graph = graph_batch_with_admitted_pair_rows(
        base_graph,
        [candidate.pair_key],
        pair_features=features,
    )

    np.testing.assert_allclose(graph.pair_features, features)
    assert graph.pair_features.shape == (1, V1_PAIR_FEATURE_DIM)


def test_terminal_tactical_targets_come_from_payload_without_candidate_masks():
    payload = V1TerminalTacticalPayload(
        status="hot_cover_impossible",
        opponent_win_requirements=((1, 0), (2, 0)),
        hot_cover_pairs=(((1, 0), (2, 0)),),
        impossible_to_cover=True,
    )

    target = terminal_tactical_target_vector(payload)

    assert target.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]


def test_terminal_tactical_payload_accepts_rust_pair_row_tuples():
    rust_pair_row = (42, 3, 9, 1, 0, 2, 0, "pair-key")
    payload = V1TerminalTacticalPayload.from_mapping(
        {
            "status": "hot_completion_available",
            "hot_completion_pairs": [rust_pair_row],
            "pair_row_schema_version": 11,
        }
    )

    assert payload.hot_completion_pairs == (((1, 0), (2, 0)),)
