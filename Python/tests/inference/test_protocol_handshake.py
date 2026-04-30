from hexorl.inference.protocol import (
    InferenceRequestKind,
    default_protocol_manifest,
    load_server_manifest,
    negotiate_protocol,
    publish_server_manifest,
    remove_server_manifest,
)


def test_protocol_handshake_accepts_matching_manifest_and_kind():
    manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    handshake = negotiate_protocol(
        client_manifest=manifest,
        server_manifest=manifest,
        request_kind=InferenceRequestKind.SPARSE_POLICY_VALUE,
    )
    assert handshake.accepted
    assert handshake.client_manifest_hash == manifest.hash()
    assert handshake.server_manifest_hash == manifest.hash()
    assert handshake.selected_request_kind == InferenceRequestKind.SPARSE_POLICY_VALUE.value
    assert handshake.selected_capacity == manifest.max_batch_size


def test_client_loads_published_server_manifest_for_handshake():
    manifest = default_protocol_manifest(max_batch_size=3, timeout_ms=100.0, heads=("policy", "value"))
    publish_server_manifest(manifest, num_workers=7, max_batch_size=3)
    try:
        loaded = load_server_manifest(num_workers=7, max_batch_size=3)
    finally:
        remove_server_manifest(num_workers=7, max_batch_size=3)
    handshake = negotiate_protocol(
        client_manifest=loaded,
        server_manifest=manifest,
        request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        required_heads=("policy", "value"),
    )
    assert handshake.server_manifest_hash == manifest.hash()
