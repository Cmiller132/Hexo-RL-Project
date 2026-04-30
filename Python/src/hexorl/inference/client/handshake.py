"""Client-side inference protocol negotiation."""

from __future__ import annotations

from hexorl.inference.protocol import (
    InferenceProtocolManifest,
    InferenceRequestKind,
    load_server_manifest,
    negotiate_protocol,
)


def load_declared_server_manifest(
    *,
    num_workers: int,
    max_batch_size: int,
    explicit_manifest: InferenceProtocolManifest | None = None,
) -> InferenceProtocolManifest:
    if explicit_manifest is not None:
        return explicit_manifest
    return load_server_manifest(num_workers=num_workers, max_batch_size=max_batch_size)


def select_request_kind(
    *,
    client_manifest: InferenceProtocolManifest,
    server_manifest: InferenceProtocolManifest,
) -> InferenceRequestKind:
    client_kinds = [InferenceRequestKind(kind) for kind in client_manifest.request_kind]
    server_kinds = set(server_manifest.request_kind)
    for kind in client_kinds:
        if kind.value in server_kinds:
            return kind
    raise RuntimeError(
        "no compatible inference request kind between client and server: "
        f"client={list(client_manifest.request_kind)} server={list(server_manifest.request_kind)}"
    )


def negotiate_client_handshake(
    *,
    client_manifest: InferenceProtocolManifest,
    server_manifest: InferenceProtocolManifest,
):
    selected_kind = select_request_kind(
        client_manifest=client_manifest,
        server_manifest=server_manifest,
    )
    return negotiate_protocol(
        client_manifest=client_manifest,
        server_manifest=server_manifest,
        request_kind=selected_kind,
        required_heads=tuple(client_manifest.heads),
    )


__all__ = [
    "load_declared_server_manifest",
    "negotiate_client_handshake",
    "select_request_kind",
]
