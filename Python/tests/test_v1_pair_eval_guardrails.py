from hexorl.autotune import candidate_recipes_from_config
from hexorl.config import Config
from hexorl.eval.v1_pair_scorecard import (
    V1_PAIR_REQUIRED_BASELINES,
    V1_PAIR_REQUIRED_METRICS,
    validate_v1_pair_scorecard_payload,
    v1_pair_scorecard_payload_template,
)


def test_v1_pair_scorecard_schema_gate_requires_all_metrics_and_blocker():
    payload = v1_pair_scorecard_payload_template(
        schema_only_blocker=(
            "equal-wall-clock arena is not claimed until current runtime exposes "
            "deterministic fair sequential DAG neural baseline games"
        )
    )

    result = validate_v1_pair_scorecard_payload(payload)

    assert result.hard_pass is True
    assert result.schema_only is True
    assert result.strength_claimed is False
    assert set(payload["side_by_side_baselines"]) == set(V1_PAIR_REQUIRED_BASELINES)
    assert set(V1_PAIR_REQUIRED_METRICS) <= set(payload["metrics"])


def test_v1_pair_scorecard_rejects_missing_metric_and_unbacked_strength_claim():
    payload = v1_pair_scorecard_payload_template(schema_only_blocker="schema only")
    payload["metrics"].pop("pair_scores_per_second")
    payload["equal_wall_clock"]["strength_claimed"] = True
    payload["equal_wall_clock"]["schema_only_blocker"] = ""

    result = validate_v1_pair_scorecard_payload(payload)

    assert result.hard_pass is False
    assert "missing_metric:pair_scores_per_second" in result.failures
    assert "equal_wall_clock_evidence_artifact_paths" in result.failures
    assert "missing_equal_wall_clock_comparison:global_xattn_0:none" in result.failures


def test_v1_autotune_recipe_declares_scorecard_baselines_and_metrics():
    cfg = Config.model_validate(
        {
            "autotune": {
                "scout": {
                    "candidate_plan": [
                        "global_xattn_0:none",
                        "global_graph768_champion:none",
                        "global_pair_biaffine_0:sampled_joint_pair_v1",
                    ]
                }
            }
        }
    )

    recipes = {recipe.candidate_id: recipe for recipe in candidate_recipes_from_config(cfg)}
    v1 = recipes["global_pair_biaffine_0__sampled_joint_pair_v1__v1"]

    assert v1.metadata["v1_pair_scorecard_schema"] == "v1_pair_scorecard:v1"
    assert set(v1.metadata["required_side_by_side_baselines"]) == set(V1_PAIR_REQUIRED_BASELINES)
    assert set(v1.metadata["required_v1_pair_metrics"]) == set(V1_PAIR_REQUIRED_METRICS)
