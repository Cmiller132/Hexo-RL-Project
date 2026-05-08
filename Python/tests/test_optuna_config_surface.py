import hashlib
import json

import pytest

from hexorl.autotune import (
    CandidateRecipe,
    ModelRecipe,
    PairStrategySpec,
    candidate_recipes_from_config,
    config_hash,
    write_candidate_artifacts,
)
from hexorl.config import Config


def test_autotune_config_defaults_match_scout_plan():
    cfg = Config()

    assert cfg.autotune.scout.enabled is True
    assert cfg.autotune.scout.max_candidates == 8
    assert cfg.autotune.scout.min_epochs == 12
    assert cfg.autotune.scout.min_generated_selfplay_positions_per_epoch == 3000
    assert cfg.autotune.scout.schedule_quantum_epochs == 2
    assert cfg.autotune.scout.include_dense_control is False
    assert cfg.autotune.scout.candidate_plan == [
        "global_xattn_0:none",
        "global_line_window_0:none",
        "global_pair_twostage_0:none",
        "global_graph_full_0:none",
        "global_graph768_champion:none",
        "global_pair_twostage_0:root_pair_mcts",
        "global_pair_twostage_0:full_pair_mcts",
        "global_graph_full_0:root_pair_mcts",
    ]
    assert cfg.autotune.optuna.storage == "sqlite:///runs/<run_id>/optuna.sqlite3"
    assert cfg.autotune.optuna.phase1_pruner == "nop"
    assert cfg.autotune.runtime_probe.speed_quarantine_positions_per_sec == pytest.approx(2.0)
    assert cfg.autotune.pair_strategy.modes == [
        "none",
        "root_pair_mcts",
        "full_pair_mcts",
        "sampled_joint_pair_v1",
    ]
    assert cfg.autotune.scoring.target_scalar == "classical_survival_lcb"
    assert cfg.autotune.final_eval.classical_arena_games == 400


def test_autotune_config_is_strict_and_validates_initial_scout_plan():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Config.model_validate({"autotune": {"scout": {"max_candidatez": 8}}})

    with pytest.raises(ValueError, match="duplicate"):
        Config.model_validate(
            {
                "autotune": {
                    "scout": {
                        "candidate_plan": [
                            "global_graph768_champion:none",
                            "global_graph768_champion:none",
                        ]
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="global-graph-only"):
        Config.model_validate(
            {
                "autotune": {
                    "scout": {
                        "candidate_plan": [
                            "global_graph768_champion:none",
                            "cnn:none",
                        ]
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="global_graph768_champion:none"):
        Config.model_validate(
            {
                "autotune": {
                    "scout": {
                        "candidate_plan": ["global_xattn_0:none"],
                    }
                }
            }
        )


def test_initial_scout_recipes_have_deterministic_candidate_ids():
    recipes = candidate_recipes_from_config(Config())
    ids = [recipe.candidate_id for recipe in recipes]

    assert len(recipes) == 8
    assert len(set(ids)) == 8
    assert "global_xattn_0__none__v1" in ids
    assert "global_pair_twostage_0__root_pair_mcts__v1" in ids
    assert "global_pair_twostage_0__full_pair_mcts__v1" in ids
    assert "global_graph768_champion__none__v1" in ids
    champion = next(recipe for recipe in recipes if recipe.model.architecture_id == "global_graph768_champion")
    assert champion.model.graph_token_budget == 768
    assert champion.pair_strategy.mode == "none"
    assert champion.runtime.inference_fp16 is False
    assert champion.runtime.graph_microbatch_size == 1
    assert champion.runtime.graph_microbatch_autotune_max_size == 4
    assert champion.runtime.graph_microbatch_memory_headroom == pytest.approx(0.60)


def test_candidate_recipe_materializes_valid_config_without_mutating_base():
    base = Config()
    base.run.seed = 777
    recipe = CandidateRecipe(
        model=ModelRecipe(
            architecture_id="global_xattn_0",
            graph_token_budget=256,
            graph_layers=1,
            output_heads=["policy_place", "value"],
        )
    )

    cfg = recipe.materialize_config(base)

    assert base.model.architecture == "cnn"
    assert base.model.pair_strategy == "none"
    assert cfg is not base
    assert cfg.run.seed == 777
    assert cfg.model.architecture == "global_xattn_0"
    assert cfg.model.heads == ["policy_place", "value"]
    assert cfg.model.pair_strategy == "none"
    assert cfg.model.pair_strategy_max_pairs == 0
    assert cfg.selfplay.num_workers == base.selfplay.num_workers
    assert cfg.selfplay.batch_size_per_worker == base.selfplay.batch_size_per_worker
    assert cfg.inference.max_batch_size == base.inference.max_batch_size
    assert cfg.inference.max_wait_us == base.inference.max_wait_us
    assert Config.model_validate(cfg.model_dump(mode="json")).model.architecture == "global_xattn_0"


def test_graph768_champion_materializes_conservative_training_runtime():
    recipe = next(
        recipe
        for recipe in candidate_recipes_from_config(Config())
        if recipe.model.architecture_id == "global_graph768_champion"
    )

    cfg = recipe.materialize_config(Config())

    assert cfg.model.architecture == "global_graph768_champion"
    assert cfg.model.graph_token_budget == 768
    assert cfg.model.graph_layers == 6
    assert cfg.inference.fp16 is False
    assert cfg.train.graph_microbatch_size == 1
    assert cfg.train.graph_microbatch_autotune_max_size == 4
    assert cfg.train.graph_microbatch_memory_headroom == pytest.approx(0.60)


def test_v1_pair_candidate_materializes_conservative_runtime_without_architecture_change():
    config = Config.model_validate(
        {
            "autotune": {
                "scout": {
                    "candidate_plan": [
                        "global_xattn_0:none",
                        "global_graph768_champion:none",
                        "global_pair_biaffine_0:sampled_joint_pair_v1",
                    ],
                }
            }
        }
    )
    recipe = next(
        recipe
        for recipe in candidate_recipes_from_config(config)
        if recipe.model.architecture_id == "global_pair_biaffine_0"
    )

    cfg = recipe.materialize_config(config)

    assert cfg.model.architecture == "global_pair_biaffine_0"
    assert cfg.model.graph_token_set == "graph512_turn_pair_prior"
    assert cfg.model.graph_token_budget == 512
    assert cfg.model.graph_layers == 3
    assert cfg.model.pair_strategy == "sampled_joint_pair_v1"
    assert cfg.model.pair_strategy_max_pairs == 256
    assert cfg.model.heads == [
        "cell_marginal_logits",
        "pair_completion_logits",
        "pair_proposal_score",
        "pair_joint_logits",
        "value",
        "terminal_tactical_v1",
    ]
    assert cfg.selfplay.legal_row_mode == "full_rust_legal"
    assert cfg.selfplay.tactical_mode == "proposal_and_label"
    assert cfg.selfplay.constrain_threats is False
    assert cfg.buffer.capacity == 120_000
    assert cfg.selfplay.num_workers == 2
    assert cfg.selfplay.batch_size_per_worker == 4
    assert cfg.inference.max_batch_size == 48
    assert cfg.inference.max_wait_us == 500
    assert cfg.inference.fp16 is True
    assert cfg.train.graph_microbatch_size == 1
    assert cfg.train.graph_microbatch_autotune_max_size == 4
    assert cfg.train.graph_microbatch_memory_headroom == pytest.approx(0.60)


@pytest.mark.parametrize("mode", ["root_pair_mcts", "full_pair_mcts"])
def test_pair_mcts_modes_materialize_as_explicit_runtime_modes(mode):
    recipe = CandidateRecipe(
        model=ModelRecipe(
            architecture_id="global_pair_twostage_0",
            output_heads=["policy_place", "value", "policy_pair_first", "policy_pair_joint", "policy_pair_second"],
        ),
        pair_strategy=PairStrategySpec(mode=mode, pair_row_budget=256),
    )

    cfg = recipe.materialize_config(Config())

    assert recipe.candidate_id == f"global_pair_twostage_0__{mode}__v1"
    assert cfg.model.pair_strategy == mode
    assert cfg.model.pair_strategy_max_pairs == 256
    assert cfg.model.pair_prior_mix == pytest.approx(0.35)

def test_config_hash_is_canonical_sorted_json_sha256():
    cfg = Config()
    expected = hashlib.sha256(
        json.dumps(
            cfg.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert config_hash(cfg) == expected
    assert len(config_hash(cfg)) == 64


def test_candidate_artifact_writer_creates_candidate_first_layout(tmp_path):
    recipe = CandidateRecipe(model=ModelRecipe(architecture_id="global_xattn_0"))
    cfg = recipe.materialize_config(Config())

    paths = write_candidate_artifacts(
        tmp_path / "runs",
        "scout_run_001",
        recipe,
        cfg,
        optuna_trial={"number": 3, "state": "WAITING"},
        git_sha="abc123",
        host_profile={"gpu": "test"},
        study_name="study_architecture_scout_v1",
        trial_number=3,
    )

    assert paths.candidate_dir == (
        tmp_path / "runs" / "scout_run_001" / "candidates" / "global_xattn_0__none__v1"
    )
    for path in [
        paths.candidate_manifest,
        paths.recipe_json,
        paths.full_config_toml,
        paths.full_config_json,
        paths.optuna_trial_json,
        paths.events_jsonl,
        paths.scorecards_jsonl,
    ]:
        assert path.exists()
    assert paths.checkpoints_dir.is_dir()
    assert paths.debug_bundles_dir.is_dir()

    manifest = json.loads(paths.candidate_manifest.read_text(encoding="utf-8"))
    assert manifest["candidate_id"] == "global_xattn_0__none__v1"
    assert manifest["architecture_id"] == "global_xattn_0"
    assert manifest["pair_strategy_mode"] == "none"
    assert manifest["recipe_schema_version"] == 1
    assert manifest["config_hash"] == config_hash(cfg)

    full_config = json.loads(paths.full_config_json.read_text(encoding="utf-8"))
    assert full_config["model"]["architecture"] == "global_xattn_0"
    assert "[model]" in paths.full_config_toml.read_text(encoding="utf-8")
