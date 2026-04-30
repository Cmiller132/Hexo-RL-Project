import struct

import numpy as np
import pytest

from hexorl.contracts.history import MoveHistory, encode_move_history
from hexorl.contracts.legal import LegalActionTable
from hexorl.contracts.symmetry import (
    apply_tensor_symmetry,
    compose_symmetries,
    inverse_symmetry,
    transform_dense_policy,
    transform_history,
    transform_legal_table,
    transform_pair_policy_target,
    transform_policy_target,
    transform_qr,
)
from hexorl.contracts.validation import ContractValidationError


def _pack(rows):
    return b"".join(struct.pack("<iii", int(player), int(q), int(r)) for player, q, r in rows)


def test_move_history_validation_hash_source_and_debug_payload():
    history = MoveHistory.decode(encode_move_history([(0, 0, 0), (1, 1, 0)]), source="rust")

    assert history.rows == ((0, 0, 0), (1, 1, 0))
    assert history.current_player == 1
    assert history.placements_remaining == 1
    assert history.identity.schema_version == history.schema_version
    assert history.history_hash == MoveHistory.decode(history.encode(), source="rust").history_hash
    assert history.debug_payload()["history_hash"] == history.history_hash


@pytest.mark.parametrize(
    "payload, match",
    [
        (b"\x00", "multiple of 12"),
        (_pack([(0, 0, 0), (0, 1, 0)]), "invalid player order"),
        (_pack([(0, 0, 0), (1, 0, 0)]), "duplicate occupied"),
    ],
)
def test_move_history_rejects_malformed_and_semantically_invalid_rows(payload, match):
    with pytest.raises(ContractValidationError, match=match):
        MoveHistory.decode(payload, source="rust")


def test_move_history_rejects_fallback_source_and_requires_fixture_opt_in():
    with pytest.raises(ContractValidationError, match="fallback"):
        MoveHistory.decode(b"", source="fallback")
    with pytest.raises(ContractValidationError, match="fixture"):
        MoveHistory.decode(b"", source="fixture")

    fixture = MoveHistory.decode(encode_move_history([(0, 4, 4)]), source="fixture", allow_fixture=True)
    assert fixture.source == "fixture"


def test_legal_action_table_identity_mutation_and_negative_validation():
    table = LegalActionTable.from_rows(
        [(0, 0), (1, 0)],
        source="rust:legal",
        occupied_count=1,
        current_player=1,
        placements_remaining=2,
        history_hash="history-a",
    )

    assert table.rows.flags.writeable is False
    assert table.dense_indices.flags.writeable is False
    assert table.table_hash == LegalActionTable.from_rows(
        [(0, 0), (1, 0)],
        source="rust:legal",
        occupied_count=1,
        current_player=1,
        placements_remaining=2,
        history_hash="history-a",
    ).table_hash
    with pytest.raises(ValueError):
        table.rows[0, 0] = 99
    with pytest.raises(ContractValidationError, match="duplicate"):
        LegalActionTable.from_rows([(0, 0), (0, 0)], source="rust:legal")
    with pytest.raises(ContractValidationError, match="fallback"):
        LegalActionTable.from_rows([(0, 0)], source="fallback")
    with pytest.raises(ContractValidationError, match="occupied"):
        table.assert_semantic_consistency(occupied={(1, 0)})


def test_d6_composition_inverse_mass_and_non_mutation():
    history = encode_move_history([(0, 0, 0), (1, 1, 0)])
    table = LegalActionTable.from_rows([(1, 0), (0, 1)], source="rust:legal")
    policy = [(1, 0, 0.25), (0, 1, 0.75)]
    pairs = [((1, 0), (0, 1), 0.6), ((2, 0), (0, 2), 0.4)]
    dense = np.zeros(33 * 33, dtype=np.float32)
    dense[17 * 33 + 16] = 1.0
    tensor = np.zeros((1, 33, 33), dtype=np.float32)
    tensor[0, 17, 16] = 1.0

    for first in range(12):
        for second in range(12):
            composed = compose_symmetries(first, second)
            assert transform_qr(transform_qr((2, -1), first), second) == transform_qr((2, -1), composed)
        restored = transform_history(transform_history(history, first), inverse_symmetry(first))
        assert restored == history

    transformed_table = transform_legal_table(table, 1)
    assert table.rows.tolist() == [[1, 0], [0, 1]]
    assert transformed_table.table_hash != table.table_hash
    assert sum(prob for *_qr, prob in transform_policy_target(policy, 2)) == pytest.approx(1.0)
    assert sum(prob for *_qr, prob in transform_pair_policy_target(pairs, 3)) == pytest.approx(1.0)
    assert float(transform_dense_policy(dense, 4).sum()) == pytest.approx(1.0)
    assert apply_tensor_symmetry(tensor, 5).sum() == pytest.approx(tensor.sum())
