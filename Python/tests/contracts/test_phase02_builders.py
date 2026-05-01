import struct

import numpy as np
import pytest

from hexorl.contracts.candidates import CANDIDATE_FEATURES, CandidateContractBuilder
from hexorl.contracts.pairs import PairActionTable, PairActionTableBuilder, PairStrategy
from hexorl.contracts.symmetry import compose_symmetries, inverse_symmetry, transform_pair_policy_target, transform_policy_target
from hexorl.graph.collate import collate_graph_batches
from hexorl.graph.semantic_builder import GraphSemanticBuilder, GraphSemanticContract, GraphTokenType
from hexorl.graph.tensorize import GraphTensorizer, graph_batch_with_pair_table


def test_candidate_contract_builder_owns_rows_features_targets_and_hash():
    table = CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [(1, 0, 0.75), (2, 0, 0.25)],
        offset_q=-16,
        offset_r=-16,
        budget=2,
        storage_width=3,
        winning_moves=[(0, 1)],
    )

    assert table.rows.shape == (3, 2)
    assert table.features.shape == (3, CANDIDATE_FEATURES)
    assert table.mask.tolist() == [True, True, True]
    assert table.target.sum() == pytest.approx(1.0)
    assert table.missing_mass == pytest.approx(0.0)
    assert table.recall_winning_move == pytest.approx(1.0)
    assert table.table_hash == CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [(1, 0, 0.75), (2, 0, 0.25)],
        offset_q=-16,
        offset_r=-16,
        budget=2,
        storage_width=3,
        winning_moves=[(0, 1)],
    ).table_hash
    with pytest.raises(ValueError, match="read-only"):
        table.rows[0, 0] = 9


def test_pair_action_table_builder_is_phase_aware_cap_aware_and_projection_only():
    candidates = CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [],
        offset_q=-16,
        offset_r=-16,
        budget=3,
        storage_width=3,
    )
    table = PairActionTableBuilder().build(
        candidates,
        [((1, 0), (0, 0), 1.0)],
        strategy=PairStrategy(generation_mode="full_capped", max_pairs=3, allow_full=True),
        legal_moves=[(0, 0), (1, 0), (0, 1)],
    )

    assert table.phase == "first_placement"
    assert table.possible_pair_count == 3
    assert table.selected_pair_count == 3
    assert table.first_policy_target[candidates.rows.tolist().index([1, 0])] == pytest.approx(1.0)
    assert table.pair_indices.shape == (3, 2)
    with pytest.raises(ValueError, match="full pair strategy cap is smaller"):
        PairActionTableBuilder().build(
            candidates,
            [],
            strategy=PairStrategy(generation_mode="full_capped", max_pairs=1, allow_full=True),
            legal_moves=[(0, 0), (1, 0), (0, 1)],
        )


def test_second_placement_pair_table_validates_known_first():
    candidates = CandidateContractBuilder().build(
        [(9, 9), (0, 0), (1, 0)],
        [],
        offset_q=-16,
        offset_r=-16,
        budget=3,
        storage_width=3,
        critical_actions=[(9, 9), (0, 0), (1, 0)],
        source="rust:synthetic",
    )
    table = PairActionTableBuilder().build(
        candidates,
        [((9, 9), (1, 0), 1.0)],
        strategy=PairStrategy(generation_mode="full_capped", max_pairs=2, allow_full=True),
        legal_moves=[(0, 0), (1, 0)],
        known_first=(9, 9),
        source="rust:synthetic",
    )

    assert table.phase == "second_placement_known_first"
    assert table.known_first == (9, 9)
    assert table.target.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="does not match known_first"):
        PairActionTableBuilder().build(
            candidates,
            [((8, 8), (1, 0), 1.0)],
            strategy=PairStrategy(generation_mode="selected", max_pairs=1),
            legal_moves=[(0, 0), (1, 0)],
            known_first=(9, 9),
            source="rust:synthetic",
        )


def test_selected_pair_generation_mode_materializes_only_target_pairs():
    candidates = CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [],
        offset_q=-16,
        offset_r=-16,
        budget=3,
        storage_width=3,
    )
    table = PairActionTableBuilder().build(
        candidates,
        [((0, 0), (1, 0), 1.0)],
        strategy=PairStrategy(generation_mode="selected", max_pairs=1),
        legal_moves=[(0, 0), (1, 0), (0, 1)],
    )

    assert table.generation_mode == "selected"
    assert table.possible_pair_count == 3
    assert table.selected_pair_count == 1
    assert table.mask.tolist() == [True]
    assert table.rows.tolist() == [[0, 0, 1, 0]]


def test_graph_semantic_builder_tensorizer_and_collator_are_split_and_mutation_safe():
    semantic = GraphSemanticBuilder().build(b"", include_pair_rows=False)
    assert isinstance(semantic, GraphSemanticContract)
    with pytest.raises(ValueError, match="read-only"):
        semantic.token_type[0] = 99
    batch = GraphTensorizer().tensorize(semantic)
    batch.token_type[0] = 99
    assert int(semantic.token_type[0]) == int(GraphTokenType.STATE)
    collated = collate_graph_batches([GraphTensorizer().tensorize(semantic)])
    assert collated.token_features.shape[0] == 1


def test_graph_pair_projection_derives_from_pair_action_table():
    history = struct.pack("<iii", 0, 0, 0)
    semantic = GraphSemanticBuilder().build(history, include_pair_rows=False)
    graph = GraphTensorizer().tensorize(semantic)
    legal_rows = [(int(q), int(r)) for q, r in graph.legal_qr[:3].tolist()]
    candidates = CandidateContractBuilder().build(
        legal_rows,
        [],
        offset_q=0,
        offset_r=0,
        budget=3,
        storage_width=3,
        critical_actions=legal_rows,
    )
    pair_table = PairActionTableBuilder().build(
        candidates,
        [(legal_rows[1], legal_rows[0], 1.0)],
        strategy=PairStrategy(generation_mode="full_capped", max_pairs=3, allow_full=True),
        legal_moves=legal_rows,
    )
    projected = graph_batch_with_pair_table(graph, pair_table)

    assert projected.pair_first_indices.shape[0] == pair_table.selected_pair_count
    assert projected.pair_policy_target.sum() == pytest.approx(1.0)
    assert projected.pair_first_policy_target[1] == pytest.approx(1.0)


def test_graph_pair_projection_rejects_stale_pair_table_references():
    history = struct.pack("<iii", 0, 0, 0)
    graph = GraphTensorizer().tensorize(GraphSemanticBuilder().build(history, include_pair_rows=False))
    legal_rows = [(int(q), int(r)) for q, r in graph.legal_qr[:3].tolist()]
    candidates = CandidateContractBuilder().build(
        legal_rows,
        [],
        offset_q=0,
        offset_r=0,
        budget=3,
        storage_width=3,
        critical_actions=legal_rows,
    )
    good = PairActionTableBuilder().build(
        candidates,
        [(legal_rows[0], legal_rows[1], 1.0)],
        strategy=PairStrategy(generation_mode="full_capped", max_pairs=3, allow_full=True),
        legal_moves=legal_rows,
    )
    bad_refs = np.asarray(good.first_candidate_rows, dtype=np.int64).copy()
    bad_refs[0] = 2
    stale = PairActionTable(
        rows=good.rows,
        first_candidate_rows=bad_refs,
        second_candidate_rows=good.second_candidate_rows,
        mask=good.mask,
        target=good.target,
        first_policy_target=good.first_policy_target,
        phase=good.phase,
        source=good.source,
        known_first=good.known_first,
        generation_mode=good.generation_mode,
        possible_pair_count=good.possible_pair_count,
        selected_pair_count=good.selected_pair_count,
        missing_mass=good.missing_mass,
        candidate_table_hash=good.candidate_table_hash,
    )

    with pytest.raises(ValueError, match="references do not match PairActionTable rows"):
        graph_batch_with_pair_table(graph, stale)


def test_candidate_pair_d6_inverse_composition_preserves_target_mass():
    target = [(0, 0, 0.4), (1, 0, 0.6)]
    pair_target = [((0, 0), (1, 0), 1.0)]
    sym = 5
    inv = inverse_symmetry(sym)
    assert compose_symmetries(sym, inv) == 0
    transformed = transform_policy_target(target, sym)
    restored = transform_policy_target(transformed, inv)
    transformed_pair = transform_pair_policy_target(pair_target, sym)
    restored_pair = transform_pair_policy_target(transformed_pair, inv)
    assert sum(prob for *_qr, prob in restored) == pytest.approx(1.0)
    assert sum(prob for *_qr, prob in restored_pair) == pytest.approx(1.0)


def test_graph_tensorizer_rejects_corrupted_semantic_shape():
    semantic = GraphSemanticBuilder().build(b"", include_pair_rows=False)
    corrupted = GraphSemanticContract(
        token_features=np.zeros((1, 1), dtype=np.float32),
        token_type=semantic.token_type,
        token_qr=semantic.token_qr,
        token_mask=semantic.token_mask,
        legal_token_indices=semantic.legal_token_indices,
        legal_qr=semantic.legal_qr,
        legal_mask=semantic.legal_mask,
        pair_token_indices=semantic.pair_token_indices,
        pair_first_indices=semantic.pair_first_indices,
        pair_second_indices=semantic.pair_second_indices,
        relation_bias=semantic.relation_bias,
        relation_type=semantic.relation_type,
        policy_target=semantic.policy_target,
        opp_legal_qr=semantic.opp_legal_qr,
        opp_legal_mask=semantic.opp_legal_mask,
        opp_policy_target=semantic.opp_policy_target,
        pair_first_policy_target=semantic.pair_first_policy_target,
        pair_policy_target=semantic.pair_policy_target,
        tactical_target=semantic.tactical_target,
        placements_remaining=semantic.placements_remaining,
        current_player=semantic.current_player,
    )
    with pytest.raises(ValueError, match="token feature shape"):
        GraphTensorizer().tensorize(corrupted)
