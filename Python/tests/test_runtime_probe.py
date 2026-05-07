import json
from types import SimpleNamespace

from hexorl.tuning.debug_bundle import write_runtime_failure_debug_bundle
from hexorl.tuning.quarantine import CandidateQuarantineRecord
from hexorl.tuning.runtime_probe import (
    RuntimeCalibrationCache,
    RuntimeKnobs,
    RuntimeProbe,
    RuntimeProbeIdentity,
    RuntimeProbeResult,
    apply_runtime_knobs,
    identity_from_config,
    semantic_config_hash,
)


def _identity(*, candidate_id="candidate-a", config_hash="cfg-a", code_hash="code-a", trial=1):
    return RuntimeProbeIdentity(
        candidate_id=candidate_id,
        architecture_id="global_pair_twostage_0",
        heads=("policy", "value", "policy_pair_joint"),
        pair_mode="root_pair_mcts",
        pair_row_cap=128,
        full_sims=800,
        pcr_sims=192,
        graph_token_set="graph512_turn_pair_prior",
        graph_token_budget=512,
        graph_layers=3,
        host_profile={"system": "linux", "cuda_name": "test-gpu", "cuda_memory_gb": 12.0},
        config_hash=config_hash,
        code_hash=code_hash,
        architecture_contract_version="graph-v1",
        recipe_schema_version="scout-v1",
        optuna_trial_number=trial,
    )


def _result(knobs, *, pps, ok=True, unsafe=False, score=None):
    return RuntimeProbeResult(
        candidate=knobs,
        ok=ok,
        positions=max(1, int(pps * 10)) if ok else 0,
        elapsed_s=10.0,
        positions_per_second=pps,
        score=pps if score is None else score,
        memory={"unsafe": unsafe},
        gpu_after={"gpu_util_pct": 55.0},
    )


def test_runtime_probe_cache_reuses_equivalent_identity_and_rejects_stale(tmp_path):
    cache = RuntimeCalibrationCache(tmp_path / "runtime_cache.json")
    slow = RuntimeKnobs(1, 4, 64, 200)
    fast = RuntimeKnobs(2, 8, 96, 500)
    calls = []

    def runner(knobs, index):
        calls.append((knobs, index))
        return _result(knobs, pps=3.0 if knobs == fast else 2.2)

    first = RuntimeProbe(
        identity=_identity(candidate_id="candidate-a", trial=7),
        candidates=[slow, fast],
        runner=runner,
        cache=cache,
    ).run()
    assert first.selected == fast
    assert first.cache_hit is False
    assert len(calls) == 2

    same_recipe_new_trial = RuntimeProbe(
        identity=_identity(candidate_id="candidate-b", trial=42),
        candidates=[slow, fast],
        runner=runner,
        cache=cache,
    ).run()
    assert same_recipe_new_trial.selected == fast
    assert same_recipe_new_trial.cache_hit is True
    assert len(calls) == 2
    assert _identity(trial=1).cache_key() == _identity(trial=999).cache_key()

    stale_config = RuntimeProbe(
        identity=_identity(candidate_id="candidate-c", config_hash="cfg-b", trial=42),
        candidates=[slow],
        runner=runner,
        cache=cache,
    ).run()
    assert stale_config.cache_hit is False
    assert len(calls) == 3


def test_runtime_probe_speed_threshold_creates_candidate_quarantine(tmp_path):
    knobs = RuntimeKnobs(1, 4, 64, 200)
    decision = RuntimeProbe(
        identity=_identity(candidate_id="too-slow"),
        candidates=[knobs],
        runner=lambda candidate, index: _result(candidate, pps=2.0),
        cache=RuntimeCalibrationCache(tmp_path / "runtime_cache.json"),
        debug_bundle_root=tmp_path / "debug_bundles",
        repro_command=["python", "probe.py", "--candidate", "too-slow"],
    ).run()

    assert decision.quarantined
    assert decision.selected is None
    assert decision.quarantine is not None
    assert decision.quarantine.candidate_id == "too-slow"
    assert decision.quarantine.state == "quarantined"
    assert "not_above_2" in decision.quarantine.reason
    assert decision.debug_bundle_path is not None
    assert (decision.debug_bundle_path / "runtime_probe_results.jsonl").exists()


def test_runtime_probe_rejects_memory_unsafe_fast_candidate():
    unsafe_fast = RuntimeKnobs(4, 8, 192, 200)
    safe_slower = RuntimeKnobs(2, 8, 96, 500)

    def runner(knobs, index):
        if knobs == unsafe_fast:
            return _result(knobs, pps=20.0, unsafe=True)
        return _result(knobs, pps=3.0)

    decision = RuntimeProbe(
        identity=_identity(),
        candidates=[unsafe_fast, safe_slower],
        runner=runner,
    ).run()
    assert not decision.quarantined
    assert decision.selected == safe_slower


def test_quarantine_retest_transitions_preserve_old_evidence():
    record = CandidateQuarantineRecord.quarantined(
        candidate_id="candidate-a",
        reason="runtime_probe_speed_quarantine:slow",
        reason_category="inference latency",
        evidence={"old_probe": {"positions_per_second": 1.5}},
        config_hash="old-cfg",
        code_hash="old-code",
    )
    old_evidence = list(record.evidence)

    record.mark_ready_for_retest(
        config_hash="new-cfg",
        code_hash="new-code",
        evidence={"manifest": "approved"},
    )
    assert record.state == "ready_for_retest"
    record.begin_retest(evidence={"run_id": "retest-1"})
    assert record.state == "retesting"
    assert record.retest_attempts == 1
    assert record.evidence[0] == old_evidence[0]
    assert len(record.evidence) == 3

    record.finish_retest(healthy=False, reason="runtime_probe_speed_quarantine:still_slow")
    assert record.state == "quarantined"
    assert record.evidence[0] == old_evidence[0]


def test_runtime_failure_debug_bundle_contains_required_files(tmp_path):
    bundle = write_runtime_failure_debug_bundle(
        tmp_path,
        candidate_id="candidate-a",
        reason="runtime_probe_speed_quarantine:slow",
        repro_command="python scripts/probe.py --candidate candidate-a",
        runtime_telemetry={"positions_per_second": 1.25},
        runtime_probe_results=[{"candidate": {"workers": 1}, "ok": True}],
        dashboard_links={"run": "http://localhost:8000/run/candidate-a"},
        legal_rows={"available": True, "rows": 10},
    )

    expected = {
        "repro_command.txt",
        "runtime_telemetry.json",
        "runtime_probe_results.jsonl",
        "dashboard_links.json",
        "legal_rows.json",
        "pair_rows.json",
        "replay.json",
        "model_output_summary.json",
        "failing_history.json",
        "manifest.json",
    }
    assert expected <= {path.name for path in bundle.iterdir()}
    assert (bundle / "repro_command.txt").read_text(encoding="utf-8").startswith("python scripts/probe.py")
    assert json.loads((bundle / "legal_rows.json").read_text(encoding="utf-8"))["available"] is True
    assert json.loads((bundle / "pair_rows.json").read_text(encoding="utf-8"))["available"] is False
    assert len((bundle / "runtime_probe_results.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_apply_runtime_knobs_preserves_semantic_identity():
    cfg = {
        "model": {
            "architecture": "global_graph_full_0",
            "heads": ["policy", "value"],
            "pair_strategy": "none",
            "pair_strategy_max_pairs": 0,
            "graph_token_set": "graph512_turn",
            "graph_token_budget": 512,
            "graph_layers": 2,
        },
        "selfplay": {
            "num_workers": 1,
            "batch_size_per_worker": 4,
            "mcts_simulations": 800,
            "pcr_low_sims": 192,
        },
        "inference": {"max_batch_size": 64, "max_wait_us": 200},
        "train": {"loss_weights": {"policy": 1.0, "value": 1.0}},
    }
    before = semantic_config_hash(cfg)
    apply_runtime_knobs(cfg, RuntimeKnobs(3, 8, 128, 700))
    assert semantic_config_hash(cfg) == before
    assert cfg["selfplay"]["num_workers"] == 3
    assert cfg["selfplay"]["batch_size_per_worker"] == 8
    assert cfg["inference"]["max_batch_size"] == 128
    assert cfg["inference"]["max_wait_us"] == 700


def test_identity_from_config_includes_required_cache_fingerprints():
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            architecture="global_graph768_champion",
            heads=["policy", "value"],
            pair_strategy="none",
            pair_strategy_max_pairs=0,
            graph_token_set="graph768_champion",
            graph_token_budget=768,
            graph_layers=4,
        ),
        selfplay=SimpleNamespace(
            num_workers=1,
            batch_size_per_worker=4,
            mcts_simulations=1200,
            pcr_low_sims=256,
        ),
        inference=SimpleNamespace(max_batch_size=64, max_wait_us=200),
    )
    identity = identity_from_config(
        candidate_id="candidate-a",
        config=cfg,
        host_profile={"system": "linux", "cuda_name": "test"},
        code_hash="abc123",
        optuna_trial_number=11,
    )
    payload = identity.cache_payload()
    assert payload["architecture_id"] == "global_graph768_champion"
    assert payload["heads"] == ["policy", "value"]
    assert payload["pair_mode"] == "none"
    assert payload["full_sims"] == 1200
    assert payload["pcr_sims"] == 256
    assert payload["graph_token_budget"] == 768
    assert payload["host_profile"]["cuda_name"] == "test"
    assert payload["config_hash"]
    assert payload["code_hash"] == "abc123"
    assert "candidate_id" not in payload
    assert "optuna_trial_number" not in payload
