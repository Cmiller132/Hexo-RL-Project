from hexorl.graph.batch import build_graph_batch_from_history
from hexorl.inference.adapters.global_graph import GlobalGraphPolicyValueAdapter
from hexorl.inference.protocol import InferenceRequestKind, default_protocol_manifest, make_request


def test_global_graph_adapter_accepts_graph_contract_request():
    manifest = default_protocol_manifest(max_batch_size=1, timeout_ms=100.0)
    graph = build_graph_batch_from_history(b"", include_pair_rows=False)
    request = make_request(
        kind=InferenceRequestKind.GLOBAL_GRAPH_POLICY_VALUE,
        manifest=manifest,
        payload={"graph_batch": graph},
        deadline_monotonic_s=999.0,
        slot_generation=0,
    )
    GlobalGraphPolicyValueAdapter(manifest).validate_request(request)
