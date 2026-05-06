import numpy as np
import torch
import pytest

from hexorl.config import Config
from hexorl.models.assembly import build_model_from_config
from hexorl.models.contracts import RowTableInstance
from hexorl.models.registry import (
    architecture_ids,
    architecture_spec,
    deprecated_aliases,
    global_graph_architecture_ids,
    is_global_graph_architecture,
    relation_required_architecture_ids,
    resolve_model_spec,
)


def test_stage2_registry_resolves_all_current_architectures_and_alias_decision():
    ids = set(architecture_ids())
    assert {
        "cnn",
        "restnet",
        "graph_hybrid_0",
        "global_graph_option1",
        "global_xattn_0",
        "global_line_window_0",
        "global_pair_twostage_0",
        "global_graph_full_0",
        "global_hybrid_action_0",
        "global_graph768_champion",
    } <= ids
    assert set(global_graph_architecture_ids()) == {
        "global_graph_option1",
        "global_xattn_0",
        "global_line_window_0",
        "global_pair_twostage_0",
        "global_graph_full_0",
        "global_hybrid_action_0",
        "global_graph768_champion",
    }
    assert set(relation_required_architecture_ids()) == {
        "global_graph_option1",
        "global_line_window_0",
        "global_graph_full_0",
        "global_graph768_champion",
    }
    assert deprecated_aliases()["graph"].target == "graph_hybrid_0"
    assert deprecated_aliases()["graph"].runtime_supported is False


def test_stage2_every_architecture_exposes_search_policy_and_value_capabilities():
    for architecture in architecture_ids():
        spec = architecture_spec(architecture)

        assert spec.policy_provider_id
        assert spec.value_provider_id
        assert set(spec.selfplay_required_outputs) <= set(spec.default_outputs)


def test_stage2_global_default_heads_are_spec_resolved_not_policy_alias():
    cfg = Config.model_validate(
        {
            "model": {
                "architecture": "global_xattn_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_layers": 1,
            },
            "inference": {"fp16": False},
        }
    )
    resolved = resolve_model_spec(cfg)

    assert cfg.model.heads[:2] == ["policy_place", "value"]
    assert "policy" not in resolved.outputs
    assert {"lookahead_4", "lookahead_12", "lookahead_36"} <= set(resolved.outputs)
    assert resolved.output_contracts["policy_place"].row_family == "legal"
    assert resolved.value_decoder.name == "binned_expected_value_65"
    assert "legal" in resolved.row_table_definitions


def test_stage2_row_table_instance_identity_hash_is_available():
    definition = architecture_spec("global_xattn_0").row_table_definitions()["legal"]
    rows = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    mask = np.asarray([True, True, True])

    instance = RowTableInstance(
        definition=definition,
        rows=rows,
        mask=mask,
        phase="search_any",
        source="rust_mcts_root",
        feature_schema_version=1,
        relation_schema_version=2,
    )
    different_phase = RowTableInstance(
        definition=definition,
        rows=rows,
        mask=mask,
        phase="known_first",
        source="rust_mcts_root",
        feature_schema_version=1,
        relation_schema_version=2,
    )

    assert instance.definition.id == "legal:v1"
    assert instance.identity_hash.startswith("sha256:")
    assert instance.identity_hash != different_phase.identity_hash


def test_stage2_selfplay_required_outputs_cannot_be_disabled():
    with pytest.raises(ValueError, match="requires self-play output 'value'"):
        Config.model_validate({"model": {"heads": ["policy"]}})

    with pytest.raises(ValueError, match="requires self-play output 'policy_place'"):
        Config.model_validate(
            {
                "model": {
                    "architecture": "global_xattn_0",
                    "channels": 16,
                    "attention_heads": 4,
                    "graph_layers": 1,
                    "heads": ["value"],
                }
            }
        )


def test_stage2_dynamic_lookahead_family_expands_from_buffer_horizons():
    cfg = Config.model_validate(
        {
            "model": {"heads": ["policy", "value", "lookahead_*"]},
            "buffer": {
                "lookahead_horizons": [2, 5],
                "lookahead_lambdas": [0.8, 0.9],
            },
            "train": {
                "loss_weights": {
                    "policy": 1.0,
                    "value": 1.0,
                    "lookahead_2": 0.1,
                    "lookahead_5": 0.1,
                }
            },
        }
    )

    assert resolve_model_spec(cfg).outputs == ("policy", "value", "lookahead_2", "lookahead_5")


@pytest.mark.parametrize(
    "architecture",
    ["cnn", "restnet", "graph_hybrid_0", "global_xattn_0"],
)
def test_stage2_assembly_attaches_resolved_metadata_for_current_bundles(architecture):
    model_cfg = {
        "architecture": architecture,
        "channels": 8,
        "blocks": 1,
        "heads": ["policy", "value"],
    }
    if architecture in {"restnet", "graph_hybrid_0"} or is_global_graph_architecture(architecture):
        model_cfg["attention_heads"] = 4
    if architecture == "restnet":
        model_cfg["attention_positions"] = [1]
    if architecture == "graph_hybrid_0":
        model_cfg.update(
            {
                "graph_token_set": "graph256_cells",
                "graph_token_budget": 64,
                "graph_layers": 1,
            }
        )
    if is_global_graph_architecture(architecture):
        model_cfg.update({"graph_layers": 1, "heads": ["policy_place", "value"]})
    cfg = Config.model_validate({"model": model_cfg, "inference": {"fp16": False}})

    model = build_model_from_config(cfg, device=torch.device("cpu"), inference=False)

    assert getattr(model, "hexorl_architecture") == architecture
    assert tuple(getattr(model, "hexorl_outputs"))
    assert getattr(model, "hexorl_output_contracts")
    assert getattr(model, "hexorl_row_table_definitions")
    assert getattr(model, "hexorl_value_decoder").name == "binned_expected_value_65"


def test_stage2_specs_own_display_and_adapter_metadata():
    spec = architecture_spec("global_pair_twostage_0")

    assert spec.family_id == "pair_two_stage"
    assert spec.training_adapter_id == "global_graph:v1"
    assert spec.inference_adapter_id == "global_graph:v1"
    assert spec.policy_provider_id == "global_legal:v1"
    assert "graph_pair_second" in spec.pair_capabilities
