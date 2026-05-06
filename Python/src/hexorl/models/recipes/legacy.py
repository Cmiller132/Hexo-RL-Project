"""Stage 2 recipe wrapper around retained legacy PyTorch implementations."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hexorl.models.specs import ResolvedArchitectureSpec
from hexorl.models.families.global_graph import GlobalHexGraphNet
from hexorl.models.families.network import HexNet
from hexorl.models.families.network import restore_family_state as _restore_family_state


def _select_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _attach_resolved_metadata(model: nn.Module, resolved: ResolvedArchitectureSpec) -> nn.Module:
    model.hexorl_architecture = resolved.architecture_id
    model.hexorl_family = resolved.family_id
    model.hexorl_recipe = resolved.recipe_id
    model.hexorl_outputs = tuple(resolved.outputs)
    model.hexorl_output_contracts = dict(resolved.output_contracts)
    model.hexorl_row_table_definitions = dict(resolved.row_table_definitions)
    model.hexorl_value_decoder = resolved.value_decoder
    model.hexorl_pair_capabilities = tuple(resolved.pair_capabilities)
    return model


def build_legacy_model(
    cfg,
    resolved: ResolvedArchitectureSpec,
    *,
    device: Optional[torch.device] = None,
    inference: bool = False,
) -> nn.Module:
    model_cfg = cfg.model
    inference_cfg = cfg.inference
    selected_device = _select_device(device)
    if resolved.global_graph:
        model = GlobalHexGraphNet(
            channels=model_cfg.channels,
            layers=getattr(model_cfg, "graph_layers", 3),
            heads=getattr(model_cfg, "attention_heads", 8),
            architecture=resolved.architecture_id,
            dropout=getattr(model_cfg, "dropout", 0.0),
            output_heads=list(resolved.outputs),
        )
        model.graph_context_tokens = int(getattr(model_cfg, "graph_token_budget", 256))
        model.graph_legal_rows = int(getattr(model_cfg, "candidate_budget", 256))
    else:
        model = HexNet(
            channels=model_cfg.channels,
            blocks=model_cfg.blocks,
            heads=list(resolved.outputs),
            architecture=resolved.architecture_id,
            attention_positions=list(getattr(model_cfg, "attention_positions", [])),
            attention_heads=getattr(model_cfg, "attention_heads", 8),
            attention_mlp_ratio=getattr(model_cfg, "attention_mlp_ratio", 2.0),
            attention_dropout=getattr(model_cfg, "attention_dropout", 0.0),
            dropout=getattr(model_cfg, "dropout", 0.0),
            relative_bias=getattr(model_cfg, "relative_bias", False),
            graph_token_set=getattr(model_cfg, "graph_token_set", "graph512_turn_pair_prior"),
            graph_token_budget=getattr(model_cfg, "graph_token_budget", 512),
            graph_layers=getattr(model_cfg, "graph_layers", 3),
            sparse_policy=getattr(model_cfg, "sparse_policy", False),
        )
    model = _attach_resolved_metadata(model, resolved)
    model = model.to(selected_device)
    if inference and inference_cfg.fp16 and selected_device.type == "cuda":
        model = model.half()
    if inference:
        model.eval()
    return model


def bins_to_value(logits: torch.Tensor) -> torch.Tensor:
    return HexNet.bins_to_value(logits)


def load_model_state(model: nn.Module, state_dict: dict, *, allow_partial: bool = False):
    return _restore_family_state(model, state_dict, allow_partial=allow_partial)
