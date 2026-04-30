import numpy as np
import pytest

from hexorl.contracts.validation import ContractValidationError
from hexorl.engine.legal import decode_legal_bytes
from hexorl.search.context import SearchContext
from hexorl.search.engine_adapter import EngineAdapter, EngineAdapterError, create_engine_adapter
from hexorl.search.pair_strategy import PairEvaluation
from hexorl.search.priors import SearchEvaluation


def _eval(engine, legal_table):
    init = engine.init_root()
    assert init is not None
    tensor, oq, or_, legal_bytes, root_generation = init
    rows = decode_legal_bytes(legal_bytes)
    legal = legal_table.from_rows(
        [(int(q), int(r)) for q, r in rows.tolist()],
        source="rust:legal",
        allow_fixture=True,
        history_hash="fixture",
    )
    ctx = SearchContext.create(
        phase="root",
        legal_table=legal,
        model_family="dense_cnn",
        tensor=tensor.reshape(1, 13, 33, 33),
        root_generation=root_generation,
        extra={"offset_q": oq, "offset_r": or_, "legal_bytes": legal_bytes},
    )
    return SearchEvaluation(
        context=ctx,
        value=0.0,
        legal_row_ids=np.arange(legal.rows.shape[0]),
        legal_dense_indices=legal.dense_indices,
        row_priors=np.ones(legal.rows.shape[0], dtype=np.float32),
        prior_source=np.ones(legal.rows.shape[0], dtype=np.uint8),
        policy_provider="test",
        model_family="dense_cnn",
        model_spec_version="1",
        inference_protocol="test",
    )


def test_engine_adapter_is_only_rust_mcts_caller():
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    assert isinstance(engine, EngineAdapter)


def test_engine_adapter_rejects_raw_logits():
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    with pytest.raises(TypeError):
        engine.expand_root(np.ones(1089, dtype=np.float32))


def test_engine_adapter_validates_legal_row_identity(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    bad_ctx = SearchContext.create(
        phase="root",
        legal_table=ev.context.legal_table,
        model_family="dense_cnn",
        root_generation=ev.context.root_generation,
        extra={"offset_q": 99, "offset_r": 99, "legal_bytes": ev.context.extra["legal_bytes"]},
    )
    bad = SearchEvaluation(
        context=bad_ctx,
        value=0.0,
        legal_row_ids=ev.legal_row_ids,
        legal_dense_indices=ev.legal_dense_indices,
        row_priors=ev.row_priors,
        prior_source=ev.prior_source,
        policy_provider="test",
        model_family="dense_cnn",
        model_spec_version="1",
        inference_protocol="test",
    )
    with pytest.raises(ContractValidationError):
        engine.expand_root(bad)


def test_engine_adapter_rejects_stale_root_token(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    stale_ctx = SearchContext.create(
        phase="root",
        legal_table=ev.context.legal_table,
        model_family="dense_cnn",
        root_generation=int(ev.context.root_generation or 0) + 1,
        extra=ev.context.extra,
    )
    stale = SearchEvaluation(
        context=stale_ctx,
        value=0.0,
        legal_row_ids=ev.legal_row_ids,
        legal_dense_indices=ev.legal_dense_indices,
        row_priors=ev.row_priors,
        prior_source=ev.prior_source,
        policy_provider="test",
        model_family="dense_cnn",
        model_spec_version="1",
        inference_protocol="test",
    )
    with pytest.raises(ContractValidationError, match="stale root token"):
        engine.expand_root(stale)


def test_engine_adapter_empty_pair_eval_is_noop(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    pair = PairEvaluation.empty(strategy_name="none", context=ev.context)
    engine.apply_root_pair_priors(pair)
    assert engine.pair_influence == "none"


def test_engine_adapter_validates_pair_row_identity(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    pair = PairEvaluation(
        strategy_name="diagnostic_full_root",
        phase="root",
        root_scope=True,
        pair_table_identity="fixture-pair-table",
        pair_rows=np.asarray([[(0, 0, 1, 0)]], dtype=np.int32),
        pair_priors=np.asarray([1.0], dtype=np.float32),
        pair_prior_source=np.asarray([1], dtype=np.uint8),
        known_first=None,
        total_possible_pairs=1,
        selected_pair_rows=1,
        scored_pair_rows=1,
        influence="pair_joint",
    )
    engine.apply_root_pair_priors(pair)
    assert engine.pair_influence == "pair_joint"
    with pytest.raises(ValueError):
        pair.pair_rows[0, 0] = 9


def test_engine_adapter_rejects_stale_batch_token(legal_table):
    engine = create_engine_adapter(num_simulations=2, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    _tensor, _count, batch_generation = engine.select_leaves(1)
    ctx = SearchContext.create(phase="leaf", legal_table=legal_table, model_family="dense_cnn", batch_generation=batch_generation + 1)
    ev = SearchEvaluation(
        context=ctx,
        value=0.0,
        legal_row_ids=np.arange(legal_table.rows.shape[0]),
        legal_dense_indices=legal_table.dense_indices,
        row_priors=np.ones(legal_table.rows.shape[0], dtype=np.float32),
        prior_source=np.ones(legal_table.rows.shape[0], dtype=np.uint8),
        policy_provider="test",
        model_family="dense_cnn",
        model_spec_version="1",
        inference_protocol="test",
    )
    with pytest.raises(ContractValidationError, match="stale batch token"):
        engine.expand_and_backprop([ev])


def test_engine_adapter_maps_mcts_error_to_structured_python_error(legal_table):
    class FailingBackend:
        def init_root(self):
            return None

        def select_leaves(self, _batch_size):
            raise ValueError("backend failed")

    engine = EngineAdapter(FailingBackend())
    with pytest.raises(EngineAdapterError) as err:
        engine.select_leaves(1)
    assert err.value.trace.operation == "select_leaves"


def test_engine_adapter_rejects_mutated_search_evaluation(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    with pytest.raises(ValueError):
        ev.row_priors[0] = 2.0


def test_engine_adapter_uses_no_legacy_panic_or_tokenless_mcts_api():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    worker_source = (root / "src" / "hexorl" / "selfplay" / "worker.py").read_text(encoding="utf-8")
    adapter_source = (root / "src" / "hexorl" / "search" / "engine_adapter.py").read_text(encoding="utf-8")
    assert "RealMCTSEngine" not in worker_source
    assert "MockMCTSEngine" not in worker_source
    assert ".init_root(" not in worker_source
    assert ".select_leaves(" not in worker_source
    assert "args[:5]" not in adapter_source
    assert "args[:2]" not in adapter_source
    assert "must return tensor, offsets, legal bytes, and root token" in adapter_source
    assert "must return tensor batch, count, and batch token" in adapter_source


def test_policy_search_debug_bundle_localizes_mapping_or_mcts_failure(legal_table):
    engine = create_engine_adapter(num_simulations=1, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    payload = {
        "context": ev.context.identity_payload(),
        "evaluation_hash": ev.evaluation_hash,
        "policy_provider": ev.policy_provider,
        "prior_source": ev.prior_source.tolist(),
        "legal_rows": ev.context.legal_table.rows.tolist(),
    }
    assert payload["context"]["legal_table_hash"] == ev.context.legal_table.table_hash
    assert payload["policy_provider"] == "test"


def test_mcts_integration_consumes_policy_provider_outputs(legal_table):
    engine = create_engine_adapter(num_simulations=2, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    engine.expand_root(ev)
    assert engine.last_trace is not None
    assert engine.last_trace.legal_table_hash == ev.context.legal_table.table_hash


def test_mcts_integration_consumes_pair_strategy_outputs_when_enabled(legal_table):
    engine = create_engine_adapter(num_simulations=2, c_puct=1.5, near_radius=2, seed=1, force_mock=True)
    ev = _eval(engine, legal_table)
    pair = PairEvaluation(
        strategy_name="two_stage_root_only",
        phase="root",
        root_scope=True,
        pair_table_identity="fixture-pair-table",
        pair_rows=np.asarray([[(0, 0, 1, 0)]], dtype=np.int32),
        pair_priors=np.asarray([1.0], dtype=np.float32),
        pair_prior_source=np.asarray([1], dtype=np.uint8),
        known_first=None,
        total_possible_pairs=1,
        selected_pair_rows=1,
        scored_pair_rows=1,
        influence="pair_joint",
    )
    engine.expand_root(ev)
    engine.apply_root_pair_priors(pair)
    assert engine.prior_source_summary()["pair_influence"] == "pair_joint"


def test_engine_adapter_rejects_stale_hashes_duplicate_rows_and_nonfinite_priors(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn")
    with pytest.raises(ContractValidationError):
        SearchEvaluation(
            context=ctx,
            value=0.0,
            legal_row_ids=np.arange(legal_table.rows.shape[0]),
            legal_dense_indices=legal_table.dense_indices,
            row_priors=np.asarray([1.0, float("nan"), 1.0], dtype=np.float32),
            prior_source=np.ones(3, dtype=np.uint8),
            policy_provider="test",
            model_family="dense_cnn",
            model_spec_version="1",
            inference_protocol="test",
        )
