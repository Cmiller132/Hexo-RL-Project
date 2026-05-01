"""Client-side inference protocol negotiation."""

from __future__ import annotations

from hexorl.inference.protocol import InferenceProtocolManifest, load_server_manifest, negotiate_protocol


def load_declared_server_manifest(
    *,
    num_workers: int,
    max_batch_size: int,
    explicit_manifest: InferenceProtocolManifest | None = None,
) -> InferenceProtocolManifest:
    if explicit_manifest is not None:
        return explicit_manifest
    return load_server_manifest(num_workers=num_workers, max_batch_size=max_batch_size)


def negotiate_client_handshake(
    *,
    client_manifest: InferenceProtocolManifest,
    server_manifest: InferenceProtocolManifest,
    operation_name: str | None = None,
):
    operation = operation_name or next(iter(server_manifest.operations), "")
    return negotiate_protocol(
        client_manifest=client_manifest,
        server_manifest=server_manifest,
        operation_name=operation,
        required_heads=tuple(server_manifest.model_contract.operation(operation).required_heads),
    )


__all__ = ["load_declared_server_manifest", "negotiate_client_handshake"]
