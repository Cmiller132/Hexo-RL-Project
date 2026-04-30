import numpy as np
import pytest

from hexorl.contracts.validation import ContractValidationError
from hexorl.search.priors import PRIOR_SOURCE_DENSE, SearchEvaluation
from hexorl.selfplay.game_runner import GameRunRequest


def test_golden_runner_aligns_engine_rows_policy_mcts_and_replay(runner_factory):
    runner, _telemetry, writer = runner_factory()
    result = runner.run_game(GameRunRequest(run_id="r", game_id=4, game_index=0, seed=4))

    record = writer.records[0]
    position = record.positions[0]
    assert result.ok is True
    assert position.policy_target_v2 == ((0, 0, 1.0),)
    assert position.policy_target_dense
    assert record.final_move_history.endswith((0).to_bytes(4, "little", signed=True))


def test_bad_masks_and_stale_legal_rows_fail_before_search_consumption(legal_context):
    with pytest.raises(ContractValidationError, match="prior length"):
        SearchEvaluation(
            context=legal_context,
            value=0.0,
            legal_row_ids=np.asarray([0], dtype=np.int64),
            legal_dense_indices=legal_context.legal_table.dense_indices[:1],
            row_priors=np.ones(1, dtype=np.float32),
            prior_source=np.asarray([PRIOR_SOURCE_DENSE], dtype=np.uint8),
            policy_provider="fake",
            model_family="dense_cnn",
            model_spec_version="v2",
            inference_protocol="fake",
        )

    with pytest.raises(ContractValidationError, match="non-finite"):
        SearchEvaluation(
            context=legal_context,
            value=float("nan"),
            legal_row_ids=np.arange(2, dtype=np.int64),
            legal_dense_indices=legal_context.legal_table.dense_indices,
            row_priors=np.ones(2, dtype=np.float32),
            prior_source=np.full(2, PRIOR_SOURCE_DENSE, dtype=np.uint8),
            policy_provider="fake",
            model_family="dense_cnn",
            model_spec_version="v2",
            inference_protocol="fake",
        )
