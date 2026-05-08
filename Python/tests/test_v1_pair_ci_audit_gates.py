from pathlib import Path

import pytest

from hexorl.autotune import candidate_recipes_from_config
from hexorl.config import Config
from hexorl.selfplay.records import (
    V1CandidatePair,
    V1ProposalPropensityMetadata,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v1_strategy_branch_does_not_use_pair_to_single_projection():
    text = _read("Python/src/hexorl/search/pair_strategy.py")
    branch = text.split("if normalized == PAIR_STRATEGY_SAMPLED_JOINT_PAIR_V1:", 1)[1]
    branch = branch.split("raise ValueError", 1)[0]

    assert "pair_logits_to_action_logits" not in branch
    assert "pair_joint_logits" in branch
    assert 'PAIR_STRATEGY_SAMPLED_JOINT_PAIR_V1 = "sampled_joint_pair_v1"' in text


def test_v1_training_eval_modules_do_not_call_legacy_pair_projection():
    paths = (
        "Python/src/hexorl/train/v1_pair_targets.py",
        "Python/src/hexorl/train/loss_plan.py",
        "Python/src/hexorl/train/losses.py",
        "Python/src/hexorl/replay/training_batch.py",
        "Python/src/hexorl/eval/v1_pair_scorecard.py",
        "Python/src/hexorl/autotune/recipes.py",
    )

    offenders = [path for path in paths if "pair_logits_to_action_logits" in _read(path)]

    assert offenders == []


def test_v1_recipe_has_no_threat_filtered_legal_and_baselines_stay_none():
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

    v1 = recipes["global_pair_biaffine_0__sampled_joint_pair_v1__v1"].materialize_config(Config())
    xattn = recipes["global_xattn_0__none__v1"].materialize_config(Config())
    champion = recipes["global_graph768_champion__none__v1"].materialize_config(Config())

    assert v1.selfplay.legal_row_mode == "full_rust_legal"
    assert v1.selfplay.tactical_mode == "proposal_and_label"
    assert v1.selfplay.constrain_threats is False
    assert xattn.model.pair_strategy == "none"
    assert champion.model.pair_strategy == "none"


def test_v1_candidates_require_source_and_proposal_metadata():
    with pytest.raises(ValueError, match="source_contributions"):
        V1CandidatePair(
            candidate_id="missing-source",
            pair_key=((0, 0), (1, 0)),
            first_legal_row_id=0,
            second_legal_row_id=1,
            row_table_schema_version=1,
            source_contributions=(),
            proposal_propensity_metadata=V1ProposalPropensityMetadata(
                proposal_policy="pair_candidate_selector_v1",
                correction_mode="exact_importance",
                total_proposal_probability=1.0,
            ),
            forced_exploration_flag=False,
            terminal_exact_flag=False,
            terminal_equivalence_flag=False,
            target_support_flags=("admitted",),
            admission_generation=0,
            root_or_interior="root",
        )

    with pytest.raises(ValueError, match="proposal_policy"):
        V1ProposalPropensityMetadata(
            proposal_policy="",
            correction_mode="exact_importance",
            total_proposal_probability=1.0,
        )
