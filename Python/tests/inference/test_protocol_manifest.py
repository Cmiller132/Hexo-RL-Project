from hexorl.inference.protocol import (
    InferenceRequestKind,
    default_protocol_manifest,
)


def test_protocol_manifest_contains_phase04_required_fields():
    manifest = default_protocol_manifest(max_batch_size=8, timeout_ms=250.0)
    payload = manifest.canonical_dict()
    required = {
        "protocol_version",
        "request_kind",
        "request_schema_version",
        "response_schema_version",
        "model_family",
        "model_spec_version",
        "input_contract",
        "output_contract",
        "action_contract",
        "graph_schema_version",
        "relation_schema_version",
        "candidate_contract_version",
        "pair_action_contract_version",
        "ffi_protocol_version",
        "legal_row_encoding",
        "history_row_encoding",
        "pair_row_encoding",
        "heads",
        "adapter_name",
        "adapter_version",
        "transport",
        "max_batch_size",
        "max_legal_rows",
        "max_candidate_rows",
        "max_pair_rows",
        "max_graph_tokens",
        "max_graph_relations",
        "timeout_ms",
        "heartbeat_interval_ms",
        "created_by_git_sha",
        "config_hash",
    }
    assert required <= set(payload)
    assert all(kind.value in manifest.request_kind for kind in InferenceRequestKind)
    assert len(manifest.hash()) == 64
