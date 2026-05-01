"""Contract-driven model output validation and decoding."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hexorl.inference.protocol import InferenceOutputValidationError
from hexorl.models.heads.value import bins_to_scalar, bins_to_value
from hexorl.models.inference_contracts import (
    DECODER_POLICY_LOGITS,
    DECODER_REGRET_BINS,
    DECODER_SCALAR,
    DECODER_VALUE_BINS,
)


DecodedOutputs = dict[str, np.ndarray]


def assert_finite_tensor(tensor: torch.Tensor, *, head_name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise InferenceOutputValidationError(f"inference model output contains non-finite values: {head_name}")


def bounded_policy_logits(logits: torch.Tensor, *, head_name: str) -> torch.Tensor:
    out = logits.float()
    assert_finite_tensor(out, head_name=head_name)
    return out.clone().clamp(-80.0, 80.0)


def bounded_value_logits(logits: torch.Tensor, *, head_name: str) -> torch.Tensor:
    out = logits.float()
    assert_finite_tensor(out, head_name=head_name)
    return out.clone().clamp(-80.0, 80.0)


def decode_outputs(raw_outputs: dict[str, torch.Tensor], *, operation: Any, contract: Any) -> DecodedOutputs:
    decoded: DecodedOutputs = {}
    for head_name in operation.output_heads:
        if head_name not in raw_outputs:
            if head_name in operation.required_heads:
                raise InferenceOutputValidationError(
                    f"inference model omitted required head {head_name} for operation {operation.name}"
                )
            continue
        head = contract.head(head_name)
        tensor = raw_outputs[head_name]
        kind = head.decoder.kind
        if kind == DECODER_POLICY_LOGITS:
            out = bounded_policy_logits(tensor, head_name=head_name)
        elif kind == DECODER_VALUE_BINS:
            out = bins_to_value(bounded_value_logits(tensor, head_name=head_name)).float()
            if head.decoder.clamp_min is not None and head.decoder.clamp_max is not None:
                out = out.clamp(float(head.decoder.clamp_min), float(head.decoder.clamp_max))
            assert_finite_tensor(out, head_name=f"{head_name}_scalar")
        elif kind == DECODER_REGRET_BINS:
            logits = bounded_value_logits(tensor, head_name=head_name)
            out = bins_to_scalar(
                logits,
                min_value=float(head.decoder.min_value if head.decoder.min_value is not None else 0.0),
                max_value=float(head.decoder.max_value if head.decoder.max_value is not None else 4.0),
            ).float()
            if head.decoder.clamp_min is not None and head.decoder.clamp_max is not None:
                out = out.clamp(float(head.decoder.clamp_min), float(head.decoder.clamp_max))
            assert_finite_tensor(out, head_name=f"{head_name}_scalar")
        elif kind == DECODER_SCALAR:
            out = tensor.detach().float().reshape(-1)
            assert_finite_tensor(out, head_name=head_name)
        else:
            raise InferenceOutputValidationError(f"unknown inference decoder {kind!r} for head {head_name}")
        decoded[head_name] = out.detach().float().cpu().numpy()
    return decoded


__all__ = [
    "DecodedOutputs",
    "assert_finite_tensor",
    "bounded_policy_logits",
    "bounded_value_logits",
    "decode_outputs",
]
