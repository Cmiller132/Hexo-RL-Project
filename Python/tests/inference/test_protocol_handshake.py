from hexorl.inference.protocol import (
    InferenceRequestKind,
    default_protocol_manifest,
    negotiate_protocol,
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
