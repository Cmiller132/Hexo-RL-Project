import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from hexorl.config import Config
from hexorl.tuning.asha import ASHARungTable, DEFAULT_ASHA_RESOURCES, TrialObservation
from hexorl.tuning.bohb import BOHBSampler, SearchSpace
from hexorl.tuning.pb2 import PB2Observation, PB2Scheduler


def test_asha_default_resources_start_at_epoch_8():
    assert DEFAULT_ASHA_RESOURCES == (8, 12, 14)
    assert ASHARungTable.default().resources[0] == 8


def test_asha_rungs_are_persisted_and_replayable(tmp_path):
    table = ASHARungTable.default()
    table.record(_obs("a", 8, 0.4))
    table.record(_obs("b", 8, 0.9))
    table.record(_obs("c", 8, 0.2, hard_failure=True))
    decision = table.decision_for(8)
    assert decision["promoted"] == ["b"]
    assert decision["quarantined"] == ["c"]

    path = tmp_path / "asha.json"
    table.save(path)
    loaded = ASHARungTable.load(path)
    assert loaded.replay_decisions() == [
        {
            "resource": 8,
            "promoted": ["b"],
            "pruned": ["a"],
            "quarantined": ["c"],
            "mode": "asha_same_resource",
            "promotion_semantics": "same_resource_rung_successive_halving",
        }
    ]


def test_asha_promotion_uses_same_resource_level_only():
    table = ASHARungTable(resources=(8, 12), promotion_fraction=0.5)
    table.record(_obs("low_at_8", 8, 0.1))
    table.record(_obs("high_at_8", 8, 0.9))
    table.record(_obs("different_resource", 12, 99.0))
    assert table.promoted_trials(8) == ["high_at_8"]


def test_asha_hard_failures_are_quarantined_not_ranked():
    table = ASHARungTable.default()
    table.record(_obs("failed_high_score", 8, 99.0, hard_failure=True))
    table.record(_obs("healthy", 8, 0.1))
    decision = table.decision_for(8)
    assert decision["promoted"] == ["healthy"]
    assert decision["quarantined"] == ["failed_high_score"]


def test_bohb_creates_hyperband_brackets():
    sampler = BOHBSampler(_space(), min_resource=4, max_resource=16, eta=2)
    brackets = sampler.create_brackets()
    assert [bracket.budgets for bracket in brackets] == [(4, 8, 16), (8, 16), (16,)]
    assert all(bracket.initial_configs >= 1 for bracket in brackets)


def test_bohb_samples_from_good_bad_density_models(tmp_path):
    sampler = BOHBSampler(_space(), warmup_points=3, random_fraction=0.0)
    for idx, family in enumerate(["graph", "graph", "cnn", "cnn"]):
        config = {
            "model_family": family,
            "batch_size": 64,
            "graph_token_budget": 512 if family == "graph" else None,
        }
        if family != "graph":
            config.pop("graph_token_budget")
        sampler.observe(config, score=10.0 - idx)

    record = sampler.sample(seed=7)
    assert record["source"] == "good_bad_density"
    assert record["density_model"]["valid_points"] == 4
    assert "good" in record["density_model"]
    assert record["candidate_scores"][0]["l_over_g"] >= record["candidate_scores"][-1]["l_over_g"]
    assert isinstance(record["config"]["batch_size"], int)

    path = tmp_path / "bohb.json"
    sampler.save(path)
    loaded = BOHBSampler.load(path)
    assert loaded.replay_sample(record)["source"] == "good_bad_density"


def test_bohb_handles_conditional_graph_space():
    config = _space().sample_random(__import__("random").Random(2), {"model_family": "graph"})
    assert "graph_token_budget" in config
    cnn_config = _space().sample_random(__import__("random").Random(2), {"model_family": "cnn"})
    assert "graph_token_budget" not in cnn_config


def test_pb2_fits_continuous_response_model():
    scheduler = _pb2_with_observations()
    model = scheduler.fit_response_model("graph")
    assert model["param_names"] == ["c_puct", "learning_rate"]
    assert model["kernel"] == "rbf_time_parameter"
    assert len(model["weights"]) == len(model["rows"])
    assert model["residual_std"] >= 0.0


def test_pb2_proposals_are_uncertainty_aware_and_clamped(tmp_path):
    scheduler = _pb2_with_observations()
    event = scheduler.propose(
        {"learning_rate": 0.03, "c_puct": 1.5},
        seed=4,
        compatible_group="graph",
        candidates=16,
    )
    assert event["source_method"] == "pb2"
    assert event["accepted_mutation"]["uncertainty"] > 0.0
    assert 1e-4 <= event["final_values"]["learning_rate"] <= 0.1
    assert 0.5 <= event["final_values"]["c_puct"] <= 3.0
    assert PB2Scheduler.replay_proposal(event) == event["final_values"]

    path = tmp_path / "pb2.json"
    scheduler.save(path)
    loaded = PB2Scheduler.load(path)
    assert loaded.events[0]["source_method"] == "pb2"


def test_pb2_scheduler_replay_reproduces_mutations():
    scheduler = _pb2_with_observations()
    event = scheduler.propose(
        {"learning_rate": 0.02, "c_puct": 1.2},
        seed=11,
        compatible_group="graph",
        candidates=8,
    )
    assert PB2Scheduler.replay_proposal(event) == event["final_values"]


def test_pb2_respects_conditional_parameters():
    scheduler = PB2Scheduler(
        {
            "learning_rate": (1e-4, 0.1),
            "pair_policy_loss": (0.01, 0.5),
        },
        parameter_conditions={
            "pair_policy_loss": {"key": "pair_policy", "values": [True]},
        },
    )
    for idx in range(3):
        scheduler.observe(
            PB2Observation(
                trial_id=f"t{idx}",
                epoch=8 + idx,
                params={"learning_rate": 0.01 * (idx + 1), "pair_policy_loss": 0.1},
                score=float(idx),
                compatible_group="dense",
            )
        )
    event = scheduler.propose(
        {"learning_rate": 0.02, "pair_policy_loss": 0.1},
        seed=3,
        compatible_group="dense",
        context={"pair_policy": False},
        candidates=4,
    )
    assert event["fit_inputs"]["param_names"] == ["learning_rate"]
    assert event["final_values"]["pair_policy_loss"] == 0.1


def test_phase3_sparse_decisive_candidate_gate_uses_discovery_metrics():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(candidate_recall_gate=0.95, target_epoch_seconds=100.0)
    services = module.EvaluationServices.__new__(module.EvaluationServices)
    services.args = supervisor.args
    trial = SimpleNamespace(family=SimpleNamespace(sparse_policy=True), last_score=1.0, score_history=[])
    buffer = {
        "size": 10,
        "avg_candidate_recall_mcts_top8": 1.0,
        "avg_candidate_recall_winning_move": 1.0,
        "avg_candidate_recall_forced_block": 1.0,
        "avg_candidate_recall_two_placement_cover": 1.0,
        "candidate_discovery_top1": 0.2,
        "candidate_discovery_top4": 0.3,
        "candidate_discovery_top8": 0.4,
        "candidate_discovery_winning_move": 0.98,
        "candidate_discovery_forced_block": 1.0,
        "candidate_discovery_two_placement_cover": 1.0,
    }

    candidate = module.EvaluationServices.candidate_recall(services, trial, buffer)
    reason = module.Phase3Supervisor._hard_prune_reason(
        supervisor,
        trial,
        {"buffer": buffer, "selfplay": {"positions_done": 10}},
    )

    assert candidate["candidate_discovery_top8"] == 0.4
    assert "candidate_recall_mcts_top8" not in candidate
    assert candidate["gate_pass"] is False
    assert reason == "decisive_candidate_discovery_below_gate:0.9800"


def test_phase3_sparse_calibration_top8_candidate_gate_is_score_only():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(candidate_recall_gate=0.95, target_epoch_seconds=100.0)
    trial = SimpleNamespace(family=SimpleNamespace(sparse_policy=True), last_score=1.0, score_history=[])
    buffer = {
        "size": 10,
        "candidate_discovery_top8": 0.4,
        "candidate_discovery_winning_move": 1.0,
        "candidate_discovery_forced_block": 1.0,
        "candidate_discovery_two_placement_cover": 1.0,
    }

    reason = module.Phase3Supervisor._hard_prune_reason(
        supervisor,
        trial,
        {"stage": "3A_calibration", "buffer": buffer, "selfplay": {"positions_done": 10}},
    )

    assert reason == ""


def test_phase3_sparse_candidate_gate_failure_penalizes_scheduler_score(monkeypatch):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(
        candidate_recall_gate=0.95,
        classical_score_min_epochs=12,
        eval_games=4,
        eval_time_ms=25,
        seed=9300,
        target_epoch_seconds=100.0,
    )
    services = module.EvaluationServices.__new__(module.EvaluationServices)
    services.s = supervisor
    services.args = supervisor.args
    trial = SimpleNamespace(
        checkpoint_path="checkpoint.pt",
        cfg=SimpleNamespace(),
        epoch=1,
        family=SimpleNamespace(sparse_policy=True),
        metrics_history=[
            {
                "epoch_elapsed_s": 10.0,
                "train": {
                    "loss_value": 1.0,
                    "policy_full_search_frac": 1.0,
                },
                "buffer": {
                    "size": 10,
                    "candidate_discovery_top8": 0.40,
                    "candidate_discovery_winning_move": 1.0,
                    "candidate_discovery_forced_block": 1.0,
                    "candidate_discovery_two_placement_cover": 1.0,
                    "avg_missing_target_policy_mass": 0.0,
                },
                "selfplay": {"positions_per_min": 1.0},
            }
        ],
        trial_id="graph_low_recall",
    )

    monkeypatch.setattr(
        supervisor,
        "_arena_checkpoint_vs_classical",
        lambda *args, **kwargs: {
            "model_win_rate": 0.0,
            "winrate_std": 0.0,
            "classical_survival_score": 0.0,
            "illegal_or_crash_rate": 0.0,
        },
    )
    monkeypatch.setattr(
        module.EvaluationServices,
        "tactical_suite",
        lambda *args, **kwargs: {"tactical_suite_score": 1.0},
    )

    row = module.EvaluationServices.evaluate_trial(services, trial, stage="3A_calibration")

    assert row["candidate_recall"]["gate_pass"] is False
    assert row["candidate_recall_penalty"] == 0.09
    assert row["scheduler_score"] == row["strength_score"] - 0.01 - 0.09


def test_phase3_throughput_memory_uses_selfplay_truncation_rate():
    module = _load_phase3_autotune_module()
    services = module.EvaluationServices.__new__(module.EvaluationServices)

    direct = module.EvaluationServices.throughput_memory(
        services,
        {
            "epoch_elapsed_s": 10.0,
            "buffer": {"size": 20},
            "train": {"batches_per_sec": 3.0},
            "selfplay": {"positions_per_min": 120.0, "truncation_rate": 0.75},
        },
    )
    derived = module.EvaluationServices.throughput_memory(
        services,
        {
            "epoch_elapsed_s": 10.0,
            "buffer": {"size": 20},
            "train": {"batches_per_sec": 3.0},
            "selfplay": {"positions_per_min": 120.0, "games_done": 8, "truncated_games": 2},
        },
    )

    assert direct["truncation_rate"] == 0.75
    assert derived["truncation_rate"] == 0.25


def test_transient_train_exception_does_not_quarantine_family():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.blocked_families = {}
    supervisor.trials = []
    supervisor.log = _CaptureLog()
    family = module.FamilySpec(
        name="graph_hybrid_0",
        description="graph",
        architecture="graph_hybrid_0",
        graph=True,
        sparse_policy=True,
        available=True,
    )

    module.Phase3Supervisor._quarantine_family(
        supervisor,
        family,
        "train_exception:RuntimeError:Inference server failed to start within 30s",
        stage="3A_calibration",
    )

    assert family.available is True
    assert "graph_hybrid_0" not in supervisor.blocked_families
    assert supervisor.log.events[-1][0] == "family_quarantine_skipped"


def test_selfplay_no_positions_does_not_quarantine_family():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.blocked_families = {}
    supervisor.trials = []
    supervisor.log = _CaptureLog()
    family = module.FamilySpec(
        name="best_restnet_33",
        description="restnet",
        architecture="restnet",
        available=True,
    )

    module.Phase3Supervisor._quarantine_family(
        supervisor,
        family,
        "selfplay_generated_no_positions",
        stage="3A_calibration",
    )

    assert family.available is True
    assert "best_restnet_33" not in supervisor.blocked_families
    assert supervisor.log.events[-1][0] == "family_quarantine_skipped"


def test_graph_low_memory_runtime_sweep_retests_one_and_two_workers():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(runtime_sweep_workers="2,3", runtime_sweep_max_candidates=2)
    supervisor.host = SimpleNamespace(logical_cpus=32, cuda_available=True, cuda_memory_gb=12.0)
    supervisor._low_memory_cuda_host = lambda: True
    cfg = SimpleNamespace(
        selfplay=SimpleNamespace(num_workers=2, batch_size_per_worker=8),
        inference=SimpleNamespace(max_wait_us=500),
        runtime=SimpleNamespace(selfplay_cpu_reserve=2),
    )
    trial = SimpleNamespace(
        cfg=cfg,
        family=SimpleNamespace(graph=True),
        static=SimpleNamespace(full_sims=512),
    )

    candidates = module.Phase3Supervisor._runtime_sweep_candidates(supervisor, trial)

    assert [candidate["workers"] for candidate in candidates] == [1, 2]
    assert [candidate["max_wait_us"] for candidate in candidates] == [500, 500]


def test_high_search_low_memory_runtime_sweep_retests_safe_workers():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(runtime_sweep_workers="2,3", runtime_sweep_max_candidates=2)
    supervisor.host = SimpleNamespace(logical_cpus=32, cuda_available=True, cuda_memory_gb=12.0)
    supervisor._low_memory_cuda_host = lambda: True
    cfg = SimpleNamespace(
        selfplay=SimpleNamespace(num_workers=2, batch_size_per_worker=16),
        inference=SimpleNamespace(max_wait_us=500),
        runtime=SimpleNamespace(selfplay_cpu_reserve=2),
    )
    trial = SimpleNamespace(
        cfg=cfg,
        family=SimpleNamespace(graph=False),
        static=SimpleNamespace(full_sims=800),
    )

    candidates = module.Phase3Supervisor._runtime_sweep_candidates(supervisor, trial)

    assert [candidate["workers"] for candidate in candidates] == [1, 2]
    assert [candidate["batch_size_per_worker"] for candidate in candidates] == [8, 8]
    assert [candidate["max_wait_us"] for candidate in candidates] == [500, 500]


def test_runtime_sweep_memory_summary_rejects_marginal_wsl_headroom():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)

    summary = module.Phase3Supervisor._summarize_runtime_memory(
        supervisor,
        {"total_gb": 23.5, "available_gb": 8.0, "used_gb": 15.5, "swap_used_gb": 0.0},
        {"total_gb": 23.5, "available_gb": 5.5, "used_gb": 18.0, "swap_used_gb": 0.2},
        [{"total_gb": 23.5, "available_gb": 3.5, "used_gb": 20.0, "swap_used_gb": 0.2}],
    )

    assert summary["unsafe"] is True


def test_low_memory_restnet_recommended_recipe_caps_train_batch():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.host = SimpleNamespace(
        cuda_available=True,
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
        physical_cpus=16,
    )
    family = module.FamilySpec(
        name="best_restnet_33",
        description="restnet",
        architecture="restnet_crop_scout",
        attention_positions=(5, 10, 14),
        available=True,
    )

    recipe = module.Phase3Supervisor._recommended_recipe(supervisor, family)

    assert recipe.train_batch_size == 128
    assert recipe.full_sims == 512
    assert recipe.pcr_low_sims == 128


def test_low_memory_static_recipe_caps_memory_hungry_bohb_batch():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.host = SimpleNamespace(
        cuda_available=True,
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
        physical_cpus=16,
    )
    restnet = module.FamilySpec(
        name="best_restnet_33",
        description="restnet",
        architecture="restnet_crop_scout",
        attention_positions=(5, 10, 14),
        available=True,
    )
    graph = module.FamilySpec(
        name="graph_hybrid_0",
        description="graph",
        architecture="graph_hybrid_0",
        graph=True,
        sparse_policy=True,
        available=True,
    )
    dense = module.FamilySpec(
        name="best_current_33",
        description="cnn",
        architecture="cnn",
        available=True,
    )

    assert module.Phase3Supervisor._static_recipe_from_bohb_config(
        supervisor, restnet, {"full_sims": 800, "train_batch_size": 384}
    ).train_batch_size == 128
    assert module.Phase3Supervisor._static_recipe_from_bohb_config(
        supervisor, restnet, {"full_sims": 800, "train_batch_size": 384}
    ).full_sims == 512
    assert module.Phase3Supervisor._static_recipe_from_bohb_config(
        supervisor, graph, {"full_sims": 512, "train_batch_size": 384}
    ).train_batch_size == 128
    assert module.Phase3Supervisor._static_recipe_from_bohb_config(
        supervisor, dense, {"full_sims": 800, "train_batch_size": 384}
    ).train_batch_size == 256
    assert module.Phase3Supervisor._static_recipe_from_bohb_config(
        supervisor, dense, {"full_sims": 800, "train_batch_size": 384}
    ).full_sims == 800


def test_finalist_pool_includes_staged_global_graph_scouts():
    module = _load_phase3_autotune_module()
    from hexorl.models.registry import global_graph_architecture_ids

    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)

    families = module.Phase3Supervisor._finalist_pool(supervisor)
    by_name = {family.name: family for family in families}

    assert set(module.GLOBAL_GRAPH_SCOUT_FAMILIES) <= set(global_graph_architecture_ids())
    assert set(module.GLOBAL_GRAPH_SCOUT_FAMILIES) <= set(by_name)
    assert "global_graph768_champion" not in by_name
    for name in module.GLOBAL_GRAPH_SCOUT_FAMILIES:
        assert by_name[name].graph is True
        assert by_name[name].global_graph is True
        assert by_name[name].sparse_policy is False


def test_global_graph_config_uses_graph_native_heads_and_compact_replay_width(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.base_cfg = module.Config()
    supervisor.host = SimpleNamespace(
        logical_cpus=32,
        physical_cpus=16,
        system="linux",
        cuda_available=True,
        cuda_name="test-gpu",
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
    )
    supervisor.args = SimpleNamespace(seed=9300, train_batches=4, max_game_moves=64)
    family = module.FamilySpec(
        name="global_xattn_0",
        description="global",
        architecture="global_xattn_0",
        graph=True,
        global_graph=True,
        available=True,
    )
    recipe = module.Phase3Supervisor._recommended_recipe(supervisor, family)

    cfg = module.Phase3Supervisor._make_config(
        supervisor,
        family,
        recipe,
        module.DynamicParams(),
        tmp_path,
        "3A_calibration",
    )
    replay = module.Phase3Supervisor._make_replay_buffer(supervisor, cfg, family)

    assert cfg.model.architecture == "global_xattn_0"
    assert cfg.model.sparse_policy is False
    assert "policy_place" in cfg.model.heads
    assert "policy" not in cfg.model.heads
    assert cfg.model.pair_strategy == "none"
    assert cfg.model.pair_strategy_max_pairs == 0
    assert replay.max_policy_v2_entries <= module.REPLAY_POLICY_WIDTH_CAP
    assert replay.max_policy_v2_entries < module.FULL_GLOBAL_POLICY_ROWS
    assert replay.memory_estimate()["feature_groups"]["opp_policy"] is False


def test_crop_replay_width_is_capped_below_full_global_rows(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.base_cfg = module.Config()
    supervisor.host = SimpleNamespace(
        logical_cpus=32,
        physical_cpus=16,
        system="linux",
        cuda_available=True,
        cuda_name="test-gpu",
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
    )
    supervisor.args = SimpleNamespace(seed=9300, train_batches=4, max_game_moves=64)
    family = module.FamilySpec("best_current_33", "cnn", "cnn", available=True)
    recipe = module.StaticRecipe(
        full_sims=512,
        pcr_low_sims=128,
        policy_top_k=96,
        candidate_budget=256,
        head_bundle="structural",
        temperature_family="slow_cool",
        train_batch_size=128,
    )

    cfg = module.Phase3Supervisor._make_config(
        supervisor,
        family,
        recipe,
        module.DynamicParams(),
        tmp_path,
        "3A_calibration",
    )
    replay = module.Phase3Supervisor._make_replay_buffer(supervisor, cfg, family)

    assert replay.max_policy_v2_entries == 256
    assert replay.max_policy_v2_entries < module.FULL_GLOBAL_POLICY_ROWS
    feature_groups = replay.memory_estimate()["feature_groups"]
    assert feature_groups["pair_policy"] is False
    assert feature_groups["opp_policy"] is False
    assert feature_groups["sparse_diagnostics"] is False


def test_replay_buffer_skips_disabled_optional_storage():
    module = _load_phase3_autotune_module()

    replay = module.RingBuffer(
        capacity=4,
        max_policy_entries=2,
        max_policy_v2_entries=16,
        store_opp_policy=False,
        store_pair_policy=False,
        store_sparse_diagnostics=False,
    )

    assert replay.memory_estimate()["feature_groups"] == {
        "opp_policy": False,
        "pair_policy": False,
        "sparse_diagnostics": False,
    }
    assert replay.memory_estimate()["optional_target_blob_mib"] == 0.0


def test_phase3_pair_policy_storage_is_only_for_global_pair_heads(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.host = SimpleNamespace(
        system="linux",
        cuda_available=True,
        cuda_name="NVIDIA GeForce RTX 4070 Ti",
        cuda_memory_gb=12.0,
        system_memory_gb=24.0,
        physical_cpus=16,
        logical_cpus=32,
    )
    supervisor.base_cfg = Config()
    supervisor.base_cfg.model.channels = 8
    supervisor.base_cfg.model.blocks = 1
    supervisor.base_cfg.train.batch_size = 8
    supervisor.base_cfg.buffer.capacity = 32
    supervisor.base_cfg.buffer.lookahead_horizons = []
    supervisor.base_cfg.buffer.lookahead_lambdas = []
    supervisor.args = SimpleNamespace(seed=9300, train_batches=1, max_game_moves=64)

    graph_hybrid = module.FamilySpec("graph_hybrid_0", "graph_hybrid_0", "graph_hybrid_0", graph=True, available=True)
    pair_global = module.FamilySpec(
        "global_pair_twostage_0",
        "global_pair_twostage_0",
        "global_pair_twostage_0",
        graph=True,
        global_graph=True,
        available=True,
    )
    recipe = module.StaticRecipe(
        full_sims=384,
        pcr_low_sims=128,
        policy_top_k=96,
        candidate_budget=256,
        head_bundle="graph_tactical",
        temperature_family="slow_cool",
        train_batch_size=128,
    )
    root_pair_recipe = module.StaticRecipe(
        full_sims=384,
        pcr_low_sims=128,
        policy_top_k=96,
        candidate_budget=256,
        head_bundle="graph_tactical",
        temperature_family="slow_cool",
        train_batch_size=128,
        pair_strategy="root_pair_mcts",
    )

    hybrid_cfg = module.Phase3Supervisor._make_config(
        supervisor,
        graph_hybrid,
        recipe,
        module.DynamicParams(),
        tmp_path / "hybrid",
        "3A_calibration",
    )
    global_cfg = module.Phase3Supervisor._make_config(
        supervisor,
        pair_global,
        recipe,
        module.DynamicParams(),
        tmp_path / "global",
        "3A_calibration",
    )
    root_pair_cfg = module.Phase3Supervisor._make_config(
        supervisor,
        pair_global,
        root_pair_recipe,
        module.DynamicParams(),
        tmp_path / "global_root_pair",
        "3A_calibration",
    )

    assert "pair_policy" not in hybrid_cfg.model.heads
    assert not (set(hybrid_cfg.model.heads) & module.GLOBAL_GRAPH_PAIR_HEADS)
    assert hybrid_cfg.model.pair_strategy == "none"
    assert hybrid_cfg.model.pair_strategy_max_pairs == 0
    assert set(global_cfg.model.heads) & module.GLOBAL_GRAPH_PAIR_HEADS
    assert global_cfg.model.pair_strategy == "none"
    assert global_cfg.model.pair_strategy_max_pairs == 0
    assert root_pair_cfg.model.pair_strategy == "root_pair_mcts"
    assert root_pair_cfg.model.pair_strategy_max_pairs == 256

    hybrid_replay = module.Phase3Supervisor._make_replay_buffer(supervisor, hybrid_cfg, graph_hybrid)
    global_replay = module.Phase3Supervisor._make_replay_buffer(supervisor, global_cfg, pair_global)

    assert hybrid_replay.memory_estimate()["feature_groups"]["pair_policy"] is False
    assert global_replay.memory_estimate()["feature_groups"]["pair_policy"] is True


def test_global_graph_candidate_recall_is_not_sparse_penalized():
    module = _load_phase3_autotune_module()
    services = module.EvaluationServices.__new__(module.EvaluationServices)
    services.args = SimpleNamespace(candidate_recall_gate=0.95)
    trial = SimpleNamespace(family=SimpleNamespace(sparse_policy=False, global_graph=True))

    candidate = module.EvaluationServices.candidate_recall(
        services,
        trial,
        {
            "candidate_discovery_top8": 0.0,
            "candidate_discovery_winning_move": 0.0,
            "candidate_discovery_forced_block": 0.0,
            "candidate_discovery_two_placement_cover": 0.0,
        },
    )

    assert candidate == {"applicable": False, "score": 1.0}


def test_static_candidates_are_family_balanced(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(seed=9300)
    supervisor.output_root = tmp_path
    supervisor.log = _CaptureLog()
    supervisor.host = SimpleNamespace(
        cuda_available=True,
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
        physical_cpus=16,
    )
    supervisor.blocked_families = {}
    supervisor.families = [
        module.FamilySpec(
            name="best_current_33",
            description="cnn",
            architecture="cnn",
            available=True,
        ),
        module.FamilySpec(
            name="best_restnet_33",
            description="restnet",
            architecture="restnet_crop_scout",
            attention_positions=(5, 10, 14),
            available=True,
        ),
        module.FamilySpec(
            name="graph_hybrid_0",
            description="graph",
            architecture="graph_hybrid_0",
            graph=True,
            sparse_policy=True,
            available=True,
        ),
    ]
    supervisor.bohb_sampler = BOHBSampler(
        module.Phase3Supervisor._bohb_search_space(supervisor),
        min_resource=8,
        max_resource=14,
        warmup_points=6,
    )

    candidates = module.Phase3Supervisor._generate_static_candidates(supervisor, 12)
    counts = {}
    for family, _recipe in candidates:
        counts[family.name] = counts.get(family.name, 0) + 1

    assert counts == {"best_current_33": 4, "best_restnet_33": 4, "graph_hybrid_0": 4}
    assert supervisor.log.events[-1][0] == "static_candidates_generated"
    assert supervisor.log.events[-1][1]["family_balanced"] is True


def test_static_candidates_stay_balanced_with_global_graph_families(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(seed=9300)
    supervisor.output_root = tmp_path
    supervisor.log = _CaptureLog()
    supervisor.host = SimpleNamespace(
        cuda_available=True,
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
        physical_cpus=16,
    )
    supervisor.blocked_families = {}
    supervisor.families = [
        module.FamilySpec("best_current_33", "cnn", "cnn", available=True),
        module.FamilySpec(
            "best_restnet_33",
            "restnet",
            "restnet_crop_scout",
            attention_positions=(5, 10, 14),
            available=True,
        ),
        module.FamilySpec(
            "graph_hybrid_0",
            "graph",
            "graph_hybrid_0",
            graph=True,
            sparse_policy=True,
            available=True,
        ),
    ] + [
        module.FamilySpec(name, "global", name, graph=True, global_graph=True, available=True)
        for name in module.GLOBAL_GRAPH_SCOUT_FAMILIES
    ]
    supervisor.bohb_sampler = BOHBSampler(
        module.Phase3Supervisor._bohb_search_space(supervisor),
        min_resource=8,
        max_resource=14,
        warmup_points=6,
    )

    candidates = module.Phase3Supervisor._generate_static_candidates(supervisor, 14)
    counts = {}
    for family, _recipe in candidates:
        counts[family.name] = counts.get(family.name, 0) + 1

    assert counts == {family.name: 2 for family in supervisor.families}
    assert supervisor.log.events[-1][1]["family_balanced"] is True


def test_phase3_static_asha_round_robins_epochs_within_rung(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(max_active_trials=3, target_epoch_seconds=10.0)
    supervisor.output_root = tmp_path
    supervisor.log = _CaptureLog()
    supervisor.trials = []
    supervisor.asha_table = ASHARungTable(resources=(2,), promotion_fraction=1.0)
    supervisor._within_stage = lambda stage: True
    supervisor._asha_resources = lambda: [2]
    supervisor._score_population = lambda current, stage: None
    supervisor._record_asha_rung = lambda current, resource: None
    supervisor._apply_asha_decision = lambda current, decision, stage: []
    supervisor._save_state = lambda: None
    supervisor._initial_dynamic = lambda family: module.DynamicParams()

    recipe = module.StaticRecipe(
        full_sims=512,
        pcr_low_sims=128,
        policy_top_k=96,
        candidate_budget=256,
        head_bundle="structural",
        temperature_family="slow_cool",
        train_batch_size=128,
    )
    families = [
        module.FamilySpec("best_restnet_33", "restnet", "restnet", available=True),
        module.FamilySpec("global_xattn_0", "global", "global_xattn_0", graph=True, global_graph=True, available=True),
        module.FamilySpec("global_pair_twostage_0", "global", "global_pair_twostage_0", graph=True, global_graph=True, available=True),
    ]
    supervisor._generate_static_candidates = lambda max_trials: [(family, recipe) for family in families]

    def create_trial(trial_id, family, recipe, dynamic, stage):
        return SimpleNamespace(
            trial_id=trial_id,
            family=family,
            static=recipe,
            dynamic=dynamic,
            epoch=0,
            pruned=False,
            checkpoint_path=tmp_path / trial_id / "checkpoint.pt",
            metrics_history=[],
            score_history=[],
        )

    train_order = []
    eval_order = []

    def train_one_epoch(trial, *, stage, target_epoch_seconds):
        trial.epoch += 1
        train_order.append((trial.trial_id, trial.epoch))
        trial.metrics_history.append({"epoch": trial.epoch})

    supervisor._create_trial = create_trial
    supervisor._train_trial_epoch = train_one_epoch
    supervisor._evaluate_trial = lambda trial, stage, force: eval_order.append((trial.trial_id, trial.epoch)) or {}

    module.Phase3Supervisor.phase_3b_static_asha(supervisor)

    assert train_order == [
        ("asha_00_best_restnet_33", 1),
        ("asha_01_global_xattn_0", 1),
        ("asha_02_global_pair_twostage_0", 1),
        ("asha_00_best_restnet_33", 2),
        ("asha_01_global_xattn_0", 2),
        ("asha_02_global_pair_twostage_0", 2),
    ]
    assert eval_order == [
        ("asha_00_best_restnet_33", 2),
        ("asha_01_global_xattn_0", 2),
        ("asha_02_global_pair_twostage_0", 2),
    ]


def test_low_memory_graph_config_extends_inference_start_timeout(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.base_cfg = module.Config()
    supervisor.host = SimpleNamespace(
        logical_cpus=32,
        physical_cpus=16,
        system="linux",
        cuda_available=True,
        cuda_name="test-gpu",
        cuda_memory_gb=12.0,
        system_memory_gb=23.5,
    )
    supervisor.args = SimpleNamespace(seed=9300, train_batches=100, max_game_moves=384)
    family = module.FamilySpec(
        name="graph_hybrid_0",
        description="graph",
        architecture="graph_hybrid_0",
        graph=True,
        sparse_policy=True,
        available=True,
    )
    recipe = module.StaticRecipe(
        full_sims=512,
        pcr_low_sims=128,
        policy_top_k=96,
        candidate_budget=256,
        head_bundle="structural",
        temperature_family="slow_cool",
        train_batch_size=128,
    )

    cfg = module.Phase3Supervisor._make_config(
        supervisor,
        family,
        recipe,
        module.DynamicParams(),
        tmp_path,
        "3A_calibration",
    )

    assert cfg.runtime.inference_start_timeout_s == 90.0
    assert cfg.buffer.capacity == 4096


def test_runtime_sweep_prunes_when_all_candidates_fail(tmp_path):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.args = SimpleNamespace(runtime_sweep_states=16)
    supervisor.output_root = tmp_path
    supervisor.runtime_sweep_cache = {}
    supervisor.log = _CaptureLog()
    supervisor._runtime_sweep_key = lambda trial: "key"
    supervisor._runtime_sweep_candidates = lambda trial: [{"workers": 1}]
    supervisor._within_stage = lambda stage: True
    supervisor._run_runtime_sweep_candidate = lambda *args, **kwargs: {
        "candidate": {"workers": 1},
        "ok": False,
        "positions": 0,
        "positions_per_min": 0.0,
        "memory": {"unsafe": True},
    }
    released = []
    saved_trials = []
    saved_supervisor_state = []
    supervisor._release_trial_runtime = lambda trial, reason: released.append((trial.trial_id, reason))
    supervisor._save_trial_state = lambda trial: saved_trials.append(trial.trial_id)
    supervisor._save_state = lambda: saved_supervisor_state.append(True)
    trial = SimpleNamespace(
        trial_id="cal_graph_hybrid_0",
        run_dir=tmp_path / "cal_graph_hybrid_0",
        runtime_sweep={},
        pruned=False,
        prune_reason="",
    )

    module.Phase3Supervisor._ensure_runtime_sweep(supervisor, trial, stage="3A_calibration")

    assert trial.pruned is True
    assert trial.prune_reason == "runtime_sweep_failed:all_probe_candidates_failed_or_memory_unsafe"
    assert supervisor.log.events[-2][0] == "runtime_sweep_failed"
    assert released == [("cal_graph_hybrid_0", trial.prune_reason)]
    assert saved_trials == ["cal_graph_hybrid_0"]
    assert saved_supervisor_state == [True]


def test_runtime_sweep_rejects_suboptimal_cached_selection():
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.host = SimpleNamespace(
        cuda_available=True,
        cuda_memory_gb=12.0,
        physical_cpus=16,
        system_memory_gb=23.5,
    )
    trial = SimpleNamespace(
        family=SimpleNamespace(graph=False, sparse_policy=False),
        static=SimpleNamespace(full_sims=512),
    )
    cached = {
        "selected": {"workers": 1, "batch_size_per_worker": 8, "max_batch_size": 72, "max_wait_us": 500},
        "selected_record": {
            "candidate": {"workers": 1, "batch_size_per_worker": 8, "max_batch_size": 72, "max_wait_us": 500},
            "ok": True,
            "positions_per_min": 141.0,
            "score": 141.0,
            "memory": {"unsafe": False},
        },
        "results": [
            {
                "candidate": {"workers": 1, "batch_size_per_worker": 8, "max_batch_size": 72, "max_wait_us": 500},
                "ok": True,
                "positions_per_min": 141.0,
                "score": 141.0,
                "memory": {"unsafe": False},
            },
            {
                "candidate": {"workers": 2, "batch_size_per_worker": 8, "max_batch_size": 80, "max_wait_us": 500},
                "ok": True,
                "positions_per_min": 381.0,
                "score": 381.0,
                "memory": {"unsafe": False},
            },
        ],
    }

    assert module.Phase3Supervisor._runtime_sweep_cached_selection_safe(supervisor, trial, cached) is False

    cached["selected"] = dict(cached["results"][1]["candidate"])
    cached["selected_record"] = dict(cached["results"][1])

    assert module.Phase3Supervisor._runtime_sweep_cached_selection_safe(supervisor, trial, cached) is True


def test_phase3_persists_trial_state_after_epoch(tmp_path, monkeypatch):
    module = _load_phase3_autotune_module()
    supervisor = module.Phase3Supervisor.__new__(module.Phase3Supervisor)
    supervisor.output_root = tmp_path
    supervisor.log = _CaptureLog()
    supervisor.calibration = {}
    supervisor.blocked_families = {}
    supervisor.baseline_loss_p75 = {True: 128.0, False: 128.0}
    supervisor.runtime_sweep_cache = {}
    supervisor.asha_table = ASHARungTable(resources=(10,), promotion_fraction=0.5)
    supervisor.bohb_sampler = SimpleNamespace(samples=[])
    supervisor.pb2_scheduler = SimpleNamespace(events=[])
    supervisor.trials = []
    supervisor.elapsed_s = lambda: 12.0
    supervisor._cleanup_shared_memory = lambda: None
    supervisor._apply_epoch_budget = lambda *args, **kwargs: None
    supervisor._apply_dynamic_to_config = lambda trial: None
    supervisor._apply_dynamic_to_trainer = lambda trial: None
    supervisor._ensure_runtime_sweep = lambda trial, stage: None
    supervisor._hard_prune_reason = lambda trial, record: None
    supervisor._release_trial_runtime = lambda trial, reason: None

    family = module.FamilySpec(
        name="best_restnet_33",
        description="test",
        architecture="restnet",
    )
    trial_dir = tmp_path / "trials" / "asha_00_best_restnet_33"
    trial_dir.mkdir(parents=True)
    trial = SimpleNamespace(
        trial_id="asha_00_best_restnet_33",
        family=family,
        static=module.StaticRecipe(
            full_sims=512,
            pcr_low_sims=128,
            policy_top_k=96,
            candidate_budget=256,
            head_bundle="structural",
            temperature_family="slow_cool",
            train_batch_size=128,
        ),
        dynamic=module.DynamicParams(lr=3e-4),
        cfg=SimpleNamespace(model=SimpleNamespace(heads=["policy", "value"])),
        run_dir=trial_dir,
        recorder=None,
        replay=SimpleNamespace(),
        trainer=None,
        checkpoint_path=None,
        epoch=0,
        wall_time_s=0.0,
        metrics_history=[],
        score_history=[],
        mutation_history=[],
        checkpoint_history=[],
        runtime_sweep={},
        pruned=False,
        prune_reason="",
    )
    supervisor.trials.append(trial)

    def public_state(t):
        return {
            "trial_id": t.trial_id,
            "epoch": t.epoch,
            "checkpoint_path": str(t.checkpoint_path) if t.checkpoint_path else None,
            "metrics_history": t.metrics_history,
            "pruned": t.pruned,
        }

    supervisor._trial_public_state = public_state

    result = SimpleNamespace(
        trainer=object(),
        checkpoint_path=trial_dir / "epoch_0001.pt",
        train_stats={"epoch": 1, "loss_policy": 1.25},
        buffer_stats={"size": 32},
        elapsed_s=3.0,
    )
    monkeypatch.setattr(module, "run_epoch", lambda *args, **kwargs: result)

    module.Phase3Supervisor._train_trial_epoch(
        supervisor,
        trial,
        stage="3B_static_asha",
        target_epoch_seconds=10.0,
    )

    trial_state = json.loads((trial_dir / "trial.json").read_text())
    run_state = json.loads((tmp_path / "state.json").read_text())
    assert trial_state["epoch"] == 1
    assert trial_state["checkpoint_path"].endswith("epoch_0001.pt")
    assert trial_state["metrics_history"][0]["train"]["loss_policy"] == 1.25
    assert run_state["trials"][0]["epoch"] == 1


def test_phase3_runtime_sweep_log_summary_names_quality_and_resource_signals():
    module = _load_phase3_autotune_module()
    summary = module._event_log_summary(
        "runtime_sweep_result",
        {
            "event": "runtime_sweep_result",
            "trial_id": "cal_graph_hybrid_0",
            "stage": "3A_calibration",
            "ok": True,
            "candidate": {
                "workers": 2,
                "batch_size_per_worker": 8,
                "max_batch_size": 80,
                "max_wait_us": 500,
            },
            "positions": 500,
            "positions_per_min": 120.8,
            "elapsed_s": 248.9,
            "score": 121.4,
            "selfplay": {
                "games_done": 1,
                "truncation_rate": 1.0,
                "terminal_reason_max_game_moves": 1,
            },
            "gpu_after": {
                "gpu_util_pct": 38,
                "memory_used_mb": 3437,
                "memory_total_mb": 12282,
            },
            "memory": {
                "min_available_gb": 20.1,
                "unsafe": False,
            },
        },
    )

    assert "Runtime sweep result" in summary
    assert "trial=cal_graph_hybrid_0" in summary
    assert "workers=2" in summary
    assert "trunc_rate=1.00" in summary
    assert "max_move_games=1" in summary
    assert "gpu_mem=3437/12282MiB" in summary
    assert "unsafe_memory=False" in summary


def test_phase3_trial_created_log_summary_names_model_contract():
    module = _load_phase3_autotune_module()
    summary = module._event_log_summary(
        "trial_created",
        {
            "event": "trial_created",
            "trial_id": "cal_global_pair_twostage_0",
            "stage": "3A_calibration",
            "family": {
                "name": "global_pair_twostage_0",
                "architecture": "global_pair_twostage_0",
            },
            "heads": ["policy_place", "value", "policy_pair_first", "policy_pair_joint", "policy_pair_second"],
            "model_contract": {
                "outputs": ["policy_place", "value", "policy_pair_first", "policy_pair_joint", "policy_pair_second"],
                "pair_capabilities": ["graph_pair_first", "graph_pair_joint", "graph_pair_second"],
            },
            "pair_strategy": {
                "strategy": "root_pair_mcts",
                "max_pairs": 256,
            },
            "static": {
                "full_sims": 512,
                "pcr_low_sims": 128,
                "graph_token_budget": 256,
                "graph_layers": 1,
                "candidate_budget": 256,
            },
        },
    )

    assert "Trial created" in summary
    assert "family=global_pair_twostage_0" in summary
    assert "runtime_outputs=[policy_place, value, policy_pair_first" in summary
    assert "pair_capabilities=[graph_pair_first, graph_pair_joint, graph_pair_second]" in summary
    assert "pair_strategy=root_pair_mcts" in summary
    assert "max_pairs=256" in summary


class _CaptureLog:
    def __init__(self):
        self.events = []

    def write(self, event, payload):
        self.events.append((event, payload))


def _obs(trial_id, resource, score, hard_failure=False):
    return TrialObservation(
        trial_id=trial_id,
        resource=resource,
        score=score,
        completed_epochs=resource,
        wall_time_seconds=resource * 10.0,
        selfplay_positions=resource * 100,
        hard_failure=hard_failure,
    )


def _space():
    return SearchSpace(
        {
            "model_family": {"type": "categorical", "choices": ["cnn", "graph"]},
            "batch_size": {"type": "categorical", "choices": [32, 64]},
            "graph_token_budget": {
                "type": "int",
                "low": 128,
                "high": 512,
                "condition": {"key": "model_family", "values": ["graph"]},
            },
        }
    )


def _pb2_with_observations():
    scheduler = PB2Scheduler(
        {
            "learning_rate": (1e-4, 0.1),
            "c_puct": (0.5, 3.0),
        }
    )
    for idx, lr in enumerate([0.001, 0.005, 0.01, 0.03, 0.08]):
        scheduler.observe(
            PB2Observation(
                trial_id=f"t{idx}",
                epoch=12 + idx,
                params={"learning_rate": lr, "c_puct": 1.0 + idx * 0.3},
                score=1.0 - abs(lr - 0.03) * 10.0 + idx * 0.05,
                compatible_group="graph",
            )
        )
    return scheduler


def _load_phase3_autotune_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_phase3_48h_autotune.py"
    spec = importlib.util.spec_from_file_location("phase3_autotune_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
