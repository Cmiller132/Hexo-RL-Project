"""Server-side model forward execution."""

from __future__ import annotations

import time

import torch

from hexorl.inference.server.metrics import ServerMetrics
from hexorl.inference.server.outputs import (
    DenseForwardOutputs,
    GraphForwardOutputs,
    decode_dense_outputs,
    decode_graph_outputs,
)


class ServerExecutor:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        forward_stream,
        fp16: bool,
        metrics: ServerMetrics,
    ) -> None:
        self.model = model
        self.device = device
        self.forward_stream = forward_stream
        self.fp16 = bool(fp16)
        self.metrics = metrics

    def forward_dense(
        self,
        batch_tensor: torch.Tensor,
        sparse_inputs: dict[str, torch.Tensor] | None = None,
    ) -> DenseForwardOutputs:
        t0 = time.monotonic()
        model_t0 = time.monotonic()
        with torch.inference_mode():
            out = self._call_model(batch_tensor, sparse_inputs)
        model_ms = (time.monotonic() - model_t0) * 1000.0

        post_t0 = time.monotonic()
        decoded = decode_dense_outputs(out, sparse_inputs=sparse_inputs)
        post_download_ms = (time.monotonic() - post_t0) * 1000.0

        self.metrics.total_forward_ms += (time.monotonic() - t0) * 1000.0
        self.metrics.total_model_ms += model_ms
        self.metrics.total_postprocess_ms += post_download_ms
        self.metrics.total_download_ms += 0.0
        return decoded

    def forward_graph(self, graph_inputs: dict[str, torch.Tensor]) -> GraphForwardOutputs:
        t0 = time.monotonic()
        model_t0 = time.monotonic()
        with torch.inference_mode():
            out = self._call_graph_model(graph_inputs)
        model_ms = (time.monotonic() - model_t0) * 1000.0

        post_t0 = time.monotonic()
        decoded = decode_graph_outputs(out)
        post_download_ms = (time.monotonic() - post_t0) * 1000.0

        self.metrics.total_forward_ms += (time.monotonic() - t0) * 1000.0
        self.metrics.total_model_ms += model_ms
        self.metrics.total_postprocess_ms += post_download_ms
        self.metrics.total_download_ms += 0.0
        return decoded

    def _call_model(
        self,
        batch_tensor: torch.Tensor,
        sparse_inputs: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if self.device.type == "cuda" and self.forward_stream is not None:
            with torch.cuda.stream(self.forward_stream):
                out = self._call_model_with_autocast(batch_tensor, sparse_inputs)
            self.forward_stream.synchronize()
            return out
        return self._call_model_with_autocast(batch_tensor, sparse_inputs)

    def _call_model_with_autocast(
        self,
        batch_tensor: torch.Tensor,
        sparse_inputs: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if self.fp16 and self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self.model(batch_tensor, **sparse_inputs) if sparse_inputs else self.model(batch_tensor)
        return self.model(batch_tensor, **sparse_inputs) if sparse_inputs else self.model(batch_tensor)

    def _call_graph_model(self, graph_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.device.type == "cuda" and self.forward_stream is not None:
            with torch.cuda.stream(self.forward_stream):
                out = self._call_graph_model_with_autocast(graph_inputs)
            self.forward_stream.synchronize()
            return out
        return self._call_graph_model_with_autocast(graph_inputs)

    def _call_graph_model_with_autocast(self, graph_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.fp16 and self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self.model(**graph_inputs)
        return self.model(**graph_inputs)


__all__ = ["ServerExecutor"]
