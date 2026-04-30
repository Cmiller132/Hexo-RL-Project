import pytest

from hexorl.inference.protocol import (
    InferenceProtocolMismatch,
    InferenceRequestKind,
    default_protocol_manifest,
    negotiate_protocol,
)


def test_protocol_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    server_manifest = default_protocol_manifest(max_batch_size=5, timeout_ms=100.0)
    with pytest.raises(InferenceProtocolMismatch):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        )
