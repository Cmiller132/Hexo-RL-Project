"""Server-side model forward execution."""

from __future__ import annotations

import time

import torch

from hexorl.inference.server.metrics import ServerMetrics
from hexorl.inference.server.outputs import DecodedOutputs, decode_outputs
from hexorl.models.inputs import CropInputs, GraphInputs


class ServerExecutor:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        forward_stream,
        fp16: bool,
        metrics: ServerMetrics,
        manifest,
    ) -> None:
        self.model = model
        self.device = device
        self.forward_stream = forward_stream
        self.fp16 = bool(fp16)
        self.metrics = metrics
        self.manifest = manifest

    def forward(self, collated) -> DecodedOutputs:
        t0 = time.monotonic()
        model_t0 = time.monotonic()
        with torch.inference_mode():
            raw = self._call_model(collated)
        model_ms = (time.monotonic() - model_t0) * 1000.0
        post_t0 = time.monotonic()
        decoded = decode_outputs(raw, operation=collated.operation, contract=self.manifest.model_contract)
        post_ms = (time.monotonic() - post_t0) * 1000.0
        self.metrics.total_forward_ms += (time.monotonic() - t0) * 1000.0
        self.metrics.total_model_ms += model_ms
        self.metrics.total_postprocess_ms += post_ms
        return decoded

    def _call_model(self, collated) -> dict[str, torch.Tensor]:
        if self.device.type == "cuda" and self.forward_stream is not None:
            with torch.cuda.stream(self.forward_stream):
                out = self._call_model_with_autocast(collated)
            self.forward_stream.synchronize()
            return out
        return self._call_model_with_autocast(collated)

    def _call_model_with_autocast(self, collated) -> dict[str, torch.Tensor]:
        if self.fp16 and self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                return self._invoke(collated)
        return self._invoke(collated)

    def _invoke(self, collated) -> dict[str, torch.Tensor]:
        if "token_features" in collated.model_inputs:
            return self.model(GraphInputs(**collated.model_inputs))
        payload = dict(collated.model_inputs)
        tensor = payload.pop("tensor")
        return self.model(CropInputs(tensor=tensor, **payload))


__all__ = ["ServerExecutor"]
