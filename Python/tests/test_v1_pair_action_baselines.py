import pytest

from hexorl.autotune import candidate_recipes_from_config
from hexorl.config import Config
from hexorl.config.schema import AUTOTUNE_PAIR_STRATEGY_MODES
from hexorl.models.registry import architecture_ids


BASELINE_XATTN_ENTRY = "global_xattn_0:none"
BASELINE_CHAMPION_ENTRY = "global_graph768_champion:none"
V1_ARCHITECTURE_ID = "global_pair_biaffine_0"
V1_PAIR_STRATEGY_ID = "sampled_joint_pair_v1"
V1_PLAN_ENTRY = f"{V1_ARCHITECTURE_ID}:{V1_PAIR_STRATEGY_ID}"
V1_CANDIDATE_ID = "global_pair_biaffine_0__sampled_joint_pair_v1__v1"


def _v1_config_identifiers_registered() -> bool:
    return (
        V1_ARCHITECTURE_ID in set(architecture_ids())
        and V1_PAIR_STRATEGY_ID in set(AUTOTUNE_PAIR_STRATEGY_MODES)
    )


def test_v1_side_by_side_baselines_materialize_unchanged():
    cfg = Config.model_validate(
        {
            "autotune": {
                "scout": {
                    "candidate_plan": [
                        BASELINE_XATTN_ENTRY,
                        BASELINE_CHAMPION_ENTRY,
                    ]
                }
            }
        }
    )

    recipes = {recipe.candidate_id: recipe for recipe in candidate_recipes_from_config(cfg)}

    assert set(recipes) == {
        "global_xattn_0__none__v1",
        "global_graph768_champion__none__v1",
    }

    xattn_cfg = recipes["global_xattn_0__none__v1"].materialize_config(Config())
    assert xattn_cfg.model.architecture == "global_xattn_0"
    assert xattn_cfg.model.heads == ["policy_place", "value"]
    assert xattn_cfg.model.pair_strategy == "none"
    assert xattn_cfg.model.pair_strategy_max_pairs == 0
    assert xattn_cfg.model.pair_prior_mix == pytest.approx(0.0)
    assert xattn_cfg.model.graph_token_budget == 256
    assert xattn_cfg.model.graph_layers == 1

    champion_cfg = recipes["global_graph768_champion__none__v1"].materialize_config(Config())
    assert champion_cfg.model.architecture == "global_graph768_champion"
    assert champion_cfg.model.heads == ["policy_place", "value"]
    assert champion_cfg.model.pair_strategy == "none"
    assert champion_cfg.model.pair_strategy_max_pairs == 0
    assert champion_cfg.model.pair_prior_mix == pytest.approx(0.0)
    assert champion_cfg.model.graph_token_set == "graph768_champion"
    assert champion_cfg.model.graph_token_budget == 768
    assert champion_cfg.model.graph_layers == 6
    assert champion_cfg.inference.fp16 is False
    assert champion_cfg.train.graph_microbatch_size == 1
    assert champion_cfg.train.graph_microbatch_autotune_max_size == 4
    assert champion_cfg.train.graph_microbatch_memory_headroom == pytest.approx(0.60)


@pytest.mark.xfail(
    condition=not _v1_config_identifiers_registered(),
    reason=(
        "V1 runtime identifiers are not registered yet: expected "
        "global_pair_biaffine_0:sampled_joint_pair_v1"
    ),
    strict=True,
)
def test_v1_pair_action_candidate_identifier_materializes_once_registered():
    cfg = Config.model_validate(
        {
            "autotune": {
                "scout": {
                    "candidate_plan": [
                        BASELINE_XATTN_ENTRY,
                        BASELINE_CHAMPION_ENTRY,
                        V1_PLAN_ENTRY,
                    ]
                }
            }
        }
    )

    recipes = {recipe.candidate_id: recipe for recipe in candidate_recipes_from_config(cfg)}
    assert V1_CANDIDATE_ID in recipes

    v1_cfg = recipes[V1_CANDIDATE_ID].materialize_config(Config())
    assert v1_cfg.model.architecture == V1_ARCHITECTURE_ID
    assert v1_cfg.model.pair_strategy == V1_PAIR_STRATEGY_ID
    assert v1_cfg.model.pair_strategy_max_pairs > 0
