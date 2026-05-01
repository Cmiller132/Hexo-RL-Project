"""Worker-side generic inference client."""

from __future__ import annotations

import time
from typing import Optional

from hexorl.inference.client.handshake import load_declared_server_manifest, negotiate_client_handshake
from hexorl.inference.client.transport import ShmTransport, TransportState
from hexorl.inference.protocol import InferenceProtocolManifest, make_request
from hexorl.inference.arena import InferenceQueue, connect_inference_queue


class RemoteEvaluator:
    """Generic shared-memory evaluator.

    Model families own semantic operations. This client owns only handshake,
    dynamic shared-memory transport, timing, and response validation.
    """

    def __init__(
        self,
        worker_id: int,
        num_workers: int,
        max_batch_size: int,
        timeout_ms: float = 1000.0,
        manifest: InferenceProtocolManifest | None = None,
        server_manifest: InferenceProtocolManifest | None = None,
    ):
        self.worker_id = int(worker_id)
        self.num_workers = int(num_workers)
        self.max_batch = int(max_batch_size)
        self.timeout_ms = float(timeout_ms)
        self._manifest_explicit = manifest is not None
        self.manifest = manifest
        self.server_manifest = server_manifest
        self._queue: Optional[InferenceQueue] = None
        self._slot = None
        self._transport: ShmTransport | None = None
        self._connected = False
        self.n_submits = 0
        self.total_wait_ms = 0.0
        self.handshake = None

    def connect(self):
        if self._connected:
            return
        server_manifest = load_declared_server_manifest(
            num_workers=self.num_workers,
            max_batch_size=self.max_batch,
            explicit_manifest=self.server_manifest,
        )
        if self.manifest is None:
            self.manifest = server_manifest
        self._queue = connect_inference_queue(self.num_workers, self.max_batch, self.manifest)
        self._slot = self._queue.get_slot(self.worker_id)
        self._transport = ShmTransport(
            worker_id=self.worker_id,
            slot=self._slot,
            timeout_ms=self.timeout_ms,
            manifest=self.manifest,
        )
        self._transport.state = TransportState.HANDSHAKING
        self.handshake = negotiate_client_handshake(
            client_manifest=self.manifest,
            server_manifest=server_manifest,
        )
        self._transport.mark_ready()
        self._connected = True

    def disconnect(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        if self._queue is not None:
            self._queue.close()
            self._queue = None
        self._slot = None
        self._connected = False

    def close(self) -> None:
        self.disconnect()

    def evaluate(self, operation_name: str, payload: dict[str, object]):
        if not self._connected or self._transport is None or self.manifest is None:
            raise RuntimeError("RemoteEvaluator not connected. Call connect() first.")
        request = make_request(
            operation_name=operation_name,
            manifest=self.manifest,
            payload=payload,
            deadline_monotonic_s=time.monotonic() + self.timeout_ms / 1000.0,
            slot_generation=self._transport.generation,
        )
        response = self._transport.round_trip(request)
        self.total_wait_ms += float(response.telemetry.get("wait_ms", 0.0))
        self.n_submits += 1
        return response

    @property
    def avg_wait_ms(self) -> float:
        if self.n_submits == 0:
            return 0.0
        return self.total_wait_ms / self.n_submits

    def __del__(self):
        self.disconnect()


InferenceClient = RemoteEvaluator
