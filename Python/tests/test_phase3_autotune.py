import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_phase3_sparse_candidate_gate_uses_discovery_metrics():
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
    assert reason == "candidate_discovery_below_gate:0.4000"


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


def test_graph_low_memory_runtime_sweep_includes_one_worker():
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
    supervisor._release_trial_runtime = lambda trial, reason: released.append((trial.trial_id, reason))
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
