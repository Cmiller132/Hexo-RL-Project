from dataclasses import replace

import pytest

from hexorl.inference.protocol import (
    InferenceProtocolMismatch,
    InferenceRequestKind,
    default_protocol_manifest,
    negotiate_protocol,
)


def test_protocol_contract_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    server_manifest = replace(client_manifest, output_contract="different")
    with pytest.raises(InferenceProtocolMismatch):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        )


def test_protocol_request_kind_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    server_manifest = replace(client_manifest, request_kind=(InferenceRequestKind.SPARSE_POLICY_VALUE.value,))
    with pytest.raises(InferenceProtocolMismatch, match="unsupported"):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        )


def test_protocol_schema_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    server_manifest = replace(client_manifest, request_schema_version=client_manifest.request_schema_version + 1)
    with pytest.raises(InferenceProtocolMismatch, match="request schema"):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
        )


def test_protocol_head_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0, heads=("policy", "value"))
    server_manifest = replace(client_manifest, heads=("value",))
    with pytest.raises(InferenceProtocolMismatch, match="required heads"):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.DENSE_POLICY_VALUE,
            required_heads=("policy", "value"),
        )


def test_protocol_capacity_mismatch_raises_before_enqueue():
    client_manifest = default_protocol_manifest(max_batch_size=4, timeout_ms=100.0)
    server_manifest = replace(client_manifest, max_pair_rows=1)
    with pytest.raises(InferenceProtocolMismatch, match="pair capacity"):
        negotiate_protocol(
            client_manifest=client_manifest,
            server_manifest=server_manifest,
            request_kind=InferenceRequestKind.PAIR_SCORING,
        )
