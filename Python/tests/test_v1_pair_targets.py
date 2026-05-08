import numpy as np
import pytest

from hexorl.selfplay.records import (
    V1CandidatePair,
    V1CandidateSourceContribution,
    V1ProposalCorrectionParameters,
    V1ProposalPropensityMetadata,
    V1SearchPairMetadata,
)
from hexorl.train.v1_pair_targets import (
    build_v1_pair_training_targets,
    collate_v1_pair_training_targets,
)


def _source(rank: int, source_type: str = "direct_pair_retrieval"):
    return V1CandidateSourceContribution(
        source_type=source_type,
        source_rank=rank,
        source_weight=1.0,
        local_probability_or_score=1.0 / float(rank + 1),
        quota_id=source_type,
        inclusion_kind="deterministic_top_k",
        exact_inclusion_probability=1.0,
        correction_mode="exact_importance",
    )


def _proposal(prob: float = 1.0):
    return V1ProposalPropensityMetadata(
        proposal_policy="pair_candidate_selector_v1",
        correction_mode="exact_importance",
        total_proposal_probability=prob,
        log_proposal_probability=float(np.log(prob)),
        sampling_without_replacement=True,
    )


def _candidate(
    idx: int,
    first,
    second,
    flags=("admitted",),
    *,
    source_type: str = "direct_pair_retrieval",
):
    flag_set = set(flags)
    return V1CandidatePair(
        candidate_id=f"cand-{idx}",
        pair_key=(first, second),
        first_legal_row_id=idx,
        second_legal_row_id=idx + 1,
        row_table_schema_version=1,
        source_contributions=(_source(idx, source_type),),
        proposal_propensity_metadata=_proposal(0.5),
        forced_exploration_flag="forced" in flag_set,
        terminal_exact_flag="terminal_exact" in flag_set,
        terminal_equivalence_flag="terminal_equivalent" in flag_set,
        target_support_flags=tuple(flags),
        admission_generation=0,
        root_or_interior="root",
    )


def _metadata(
    candidates,
    *,
    visits=None,
    q_values=None,
    completed_q=None,
    selected_pair=None,
    support_type="admitted_candidate_set_with_explicit_negatives",
):
    n = len(candidates)
    visits = tuple(visits if visits is not None else [0] * n)
    q_values = tuple(q_values if q_values is not None else [0.0] * n)
    completed_q = tuple(completed_q if completed_q is not None else [0.0] * n)
    return V1SearchPairMetadata(
        candidate_selector_version="pair_candidate_selector_v1",
        support_type=support_type,
        legal_pair_count=max(n, 6),
        legal_row_schema_version=1,
        pair_row_schema_version=1,
        candidate_pairs=tuple(candidates),
        proposal_correction_parameters=V1ProposalCorrectionParameters(
            correction_mode="exact_importance",
            min_log=-4.0,
            max_log=4.0,
            prior_temperature=1.0,
        ),
        root_gumbel_values=tuple(float(i) for i in range(n)),
        root_admission_order=tuple(range(n)),
        root_simulation_allocation=tuple([1] * n),
        visit_counts=visits,
        q_values=q_values,
        completed_q_values=completed_q,
        selected_pair=selected_pair,
        target_support_flags=tuple(c.target_support_flags for c in candidates),
        terminal_equivalence_flags=tuple(c.terminal_equivalence_flag for c in candidates),
        neural_calls_per_expanded_full_turn_node=1.0,
    )


def test_v1_targets_train_only_logged_support_and_do_not_mark_unsampled_negative():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted",)),
        _candidate(1, (0, 0), (0, 1), ("explicit_negative", "sampled_negative")),
        _candidate(2, (1, 0), (0, 1), ("unsampled",)),
    ]
    metadata = _metadata(
        candidates,
        visits=[7, 0, 99],
        q_values=[0.2, -0.4, -0.9],
        completed_q=[0.3, -0.3, -1.0],
        selected_pair=candidates[0].pair_key,
    )

    targets = build_v1_pair_training_targets(metadata)

    assert targets.pair_joint_mask.tolist() == [True, False, False]
    assert targets.pair_joint_target.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert targets.explicit_negative_mask.tolist() == [False, True, False]
    assert targets.sampled_negative_mask.tolist() == [False, True, False]
    assert targets.unsampled_mask.tolist() == [False, False, True]
    assert targets.ranking_negative_indices.tolist() == [1]
    assert targets.selected_pair_index == 0


def test_v1_targets_use_completed_q_soft_posterior_when_visits_are_zero():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted",)),
        _candidate(1, (0, 0), (0, 1), ("admitted", "forced")),
    ]
    metadata = _metadata(
        candidates,
        visits=[0, 0],
        completed_q=[0.0, 2.0],
        support_type="completed_q_candidate_posterior",
    )

    targets = build_v1_pair_training_targets(metadata, posterior_temperature=1.0)

    assert targets.forced_mask.tolist() == [False, True]
    assert targets.pair_joint_target[1] > targets.pair_joint_target[0]
    assert targets.pair_joint_target.sum() == pytest.approx(1.0)


def test_v1_terminal_equivalent_mass_collapses_without_filler_pair_training():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted", "terminal_equivalent")),
        _candidate(1, (0, 0), (0, 1), ("admitted", "terminal_equivalent")),
        _candidate(2, (1, 0), (0, 1), ("admitted",)),
    ]
    metadata = _metadata(candidates, visits=[5, 0, 5])

    targets = build_v1_pair_training_targets(metadata)

    assert targets.terminal_equivalent_mask.tolist() == [True, True, False]
    assert targets.pair_completion_target.tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert targets.pair_joint_target[0] == pytest.approx(targets.pair_joint_target[1])
    assert targets.pair_joint_target.sum() == pytest.approx(1.0)


def test_v1_terminal_equivalent_policy_can_be_omitted_explicitly():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted", "terminal_exact")),
        _candidate(1, (0, 0), (0, 1), ("admitted",)),
    ]
    metadata = _metadata(candidates, visits=[9, 1])

    targets = build_v1_pair_training_targets(
        metadata,
        terminal_equivalent_mass="omit_policy",
    )

    assert targets.terminal_exact_mask.tolist() == [True, False]
    assert targets.pair_joint_mask.tolist() == [False, True]
    assert targets.pair_joint_target.tolist() == pytest.approx([0.0, 1.0])


def test_v1_targets_export_contract_dictionary():
    candidates = [_candidate(0, (0, 0), (1, 0), ("admitted",))]
    metadata = _metadata(candidates, visits=[1])

    payload = build_v1_pair_training_targets(metadata).as_dict()

    assert payload["v1_pair_target_schema_version"] == 2
    assert payload["v1_pair_schema_version"] == 2
    assert payload["v1_candidate_pair_qr"].shape == (1, 4)
    assert payload["v1_pair_legal_row_ids"].tolist() == [[0, 1]]


def test_v1_targets_emit_unordered_safe_conditionals_and_cell_marginals():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted",)),
        _candidate(1, (0, 0), (0, 1), ("admitted",)),
    ]
    metadata = _metadata(candidates, visits=[3, 1])

    targets = build_v1_pair_training_targets(metadata, legal_row_count=3, softening_alpha=0.0)

    assert targets.cell_marginal_target.tolist() == pytest.approx([0.375, 0.5, 0.125])
    assert targets.conditional_pair_indices.tolist() == [0, 0, 1, 1]
    assert targets.conditional_first_legal_row_ids.tolist() == [0, 1, 1, 2]
    assert targets.conditional_second_legal_row_ids.tolist() == [1, 0, 2, 1]
    assert targets.conditional_mask.tolist() == [True, True, True, True]
    assert targets.conditional_target.tolist() == pytest.approx([1.0, 0.75, 0.25, 1.0])


def test_v1_target_collation_keeps_support_masks_and_unsampled_out_of_training():
    candidates = [
        _candidate(0, (0, 0), (1, 0), ("admitted",)),
        _candidate(1, (0, 0), (0, 1), ("explicit_negative", "sampled_negative")),
        _candidate(2, (1, 0), (0, 1), ("unsampled",)),
    ]
    target = build_v1_pair_training_targets(_metadata(candidates, visits=[5, 0, 0]))

    batch = collate_v1_pair_training_targets([target], legal_width=4, pair_width=3)

    assert batch["v1_pair_schema_version"].tolist() == [2]
    assert batch["v1_support_type_id"].tolist() == [3]
    assert batch["v1_pair_joint_mask"].tolist() == [[True, False, False]]
    assert batch["v1_pair_ranking_mask"].tolist() == [[True, True, False]]
    assert batch["v1_unsampled_pair_mask"].tolist() == [[False, False, True]]
