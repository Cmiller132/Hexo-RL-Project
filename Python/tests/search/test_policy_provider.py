import numpy as np
import pytest

from hexorl.contracts.candidates import CandidateContractBuilder
from hexorl.contracts.validation import ContractValidationError
from hexorl.graph.batch import build_graph_batch_from_history
from hexorl.models.specs import ModelSpec
from hexorl.search.context import SearchContext
from hexorl.search.policy_provider import (
    DensePolicyProvider,
    GlobalGraphPolicyProvider,
    GraphHybridPolicyProvider,
    RestNetPolicyProvider,
    create_policy_provider,
)
from hexorl.search.priors import SearchEvaluation


def _tensor():
    return np.zeros((1, 13, 33, 33), dtype=np.float32)


def test_dense_policy_provider_returns_row_mapped_priors(legal_table, dense_spec, fake_client):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn", tensor=_tensor())
    ev = DensePolicyProvider(client=fake_client, model_spec=dense_spec).evaluate_root(ctx)
    assert isinstance(ev, SearchEvaluation)
    assert ev.row_priors.shape[0] == legal_table.rows.shape[0]
    assert np.array_equal(ev.legal_dense_indices, legal_table.dense_indices)
    assert fake_client.dense_calls == 1


def test_restnet_policy_provider_returns_row_mapped_priors(legal_table, fake_client):
    spec = ModelSpec(kind="restnet", source_name="fixture")
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="restnet", tensor=_tensor())
    ev = RestNetPolicyProvider(client=fake_client, model_spec=spec).evaluate_root(ctx)
    assert ev.model_family == "restnet"
    assert ev.policy_provider == "RestNetPolicyProvider"


def test_graph_hybrid_policy_provider_uses_candidate_legal_rows(legal_table, fake_client):
    spec = ModelSpec(kind="graph_hybrid", source_name="fixture")
    cand = CandidateContractBuilder().build(
        [(0, 0), (1, 0), (0, 1)],
        [],
        offset_q=0,
        offset_r=0,
        budget=3,
        storage_width=3,
        source="fixture",
        allow_fixture=True,
    )
    ctx = SearchContext.create(
        phase="root",
        legal_table=legal_table,
        model_family="graph_hybrid",
        tensor=_tensor(),
        candidate_table=cand,
    )
    ev = GraphHybridPolicyProvider(client=fake_client, model_spec=spec).evaluate_root(ctx)
    assert ev.row_priors.shape[0] == legal_table.rows.shape[0]
    assert ev.raw_metadata["candidate_table_hash"] == cand.table_hash


def test_global_graph_policy_provider_maps_legal_logits_to_legal_rows(legal_table, fake_client):
    spec = ModelSpec(kind="global_xattn", source_name="fixture")
    graph = build_graph_batch_from_history(b"", include_pair_rows=False)
    graph = graph.__class__(
        **{**graph.__dict__, "legal_qr": legal_table.rows, "legal_mask": np.ones(3, dtype=np.bool_)}
    )
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="global_xattn", graph_batch=graph)
    ev = GlobalGraphPolicyProvider(client=fake_client, model_spec=spec).evaluate_root(ctx)
    assert ev.value == pytest.approx(0.75)
    assert ev.row_priors.shape[0] == 3
    assert fake_client.graph_calls == 1


def test_search_evaluation_rejects_prior_length_mismatch(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn")
    with pytest.raises(ContractValidationError):
        SearchEvaluation(
            context=ctx,
            value=0.0,
            legal_row_ids=np.arange(2),
            legal_dense_indices=legal_table.dense_indices[:2],
            row_priors=np.ones(2, dtype=np.float32),
            prior_source=np.ones(2, dtype=np.uint8),
            policy_provider="test",
            model_family="dense_cnn",
            model_spec_version="1",
            inference_protocol="test",
        )


def test_search_evaluation_rejects_unmapped_model_rows(legal_table):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn")
    with pytest.raises(ContractValidationError):
        SearchEvaluation(
            context=ctx,
            value=0.0,
            legal_row_ids=np.asarray([0, 2, 3]),
            legal_dense_indices=legal_table.dense_indices,
            row_priors=np.ones(3, dtype=np.float32),
            prior_source=np.ones(3, dtype=np.uint8),
            policy_provider="test",
            model_family="dense_cnn",
            model_spec_version="1",
            inference_protocol="test",
        )


def test_policy_source_traceability_records_provider_family_protocol(legal_table, dense_spec, fake_client):
    ctx = SearchContext.create(phase="root", legal_table=legal_table, model_family="dense_cnn", tensor=_tensor())
    ev = create_policy_provider(model_spec=dense_spec, client=fake_client).evaluate_root(ctx)
    assert ev.policy_provider == "DensePolicyProvider"
    assert ev.model_family == "dense_cnn"
    assert ev.inference_protocol == "test-protocol"
