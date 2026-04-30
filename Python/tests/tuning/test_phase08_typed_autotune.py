import pytest

from hexorl.models.factory import get_model_registry
from hexorl.tuning import (
    AutotuneScheduler,
    HostProfile,
    ModelRecipe,
    RecipeTransform,
    RuntimeSpec,
    default_runtime_spec,
    dry_run_validate_recipe,
    family_space,
    poor_learning_report,
    score_trial,
    simulate_no_progress,
    trial_lifecycle_report,
)


def test_family_spaces_and_dry_run_cover_every_registered_family():
    host = HostProfile(cpu_count=16, memory_gb=32.0)
    runtime = default_runtime_spec(host)
    for family in get_model_registry().names():
        recipe = family_space(family).default_recipe(seed=3)
        results = dry_run_validate_recipe(recipe, runtime, host)
        assert all(item["ok"] for item in results)
    with pytest.raises(ValueError, match="pair_strategy_max_pairs"):
        ModelRecipe(
            recipe_id="bad",
            model_family=get_model_registry().names()[0],
            pair_strategy="none",
            pair_strategy_max_pairs=2,
        )


def test_recipe_transforms_are_typed_and_raw_config_is_rejected():
    recipe = family_space("dense_cnn").default_recipe()
    wider = recipe.transform(RecipeTransform("wide", {"channels": 32}))
    assert wider.channels == 32
    assert recipe.channels != wider.channels
    with pytest.raises(TypeError):
        RecipeTransform.from_raw_config({"model": {"architecture": "cnn"}})
    with pytest.raises(ValueError, match="model_family"):
        recipe.transform(RecipeTransform("switch", {"model_family": "restnet"}))


def test_runtime_sweep_watchdogs_scheduler_and_reports_are_actionable():
    host = HostProfile(cpu_count=4, memory_gb=8.0)
    runtime = RuntimeSpec(
        selfplay_workers=8,
        rust_threads=8,
        torch_threads=8,
        dataloader_workers=2,
        inference_max_batch=32,
        microbatch_wait_ms=2.0,
        leaf_batch_size=16,
        record_queue_capacity=1,
        replay_prefetch=2,
        train_batch_size=64,
    )
    failures = runtime.validate(host)
    assert {failure["kind"] for failure in failures} >= {"runtime_budget", "backpressure"}
    watchdog = simulate_no_progress(default_runtime_spec(host), "inference", trace_id="trace-1")
    assert watchdog["event"] == "watchdog_abort"
    score = score_trial({"win_rate": 0.6, "positions_per_sec": 500.0, "idle_fraction": 0.1})
    decision = AutotuneScheduler().decide(
        "trial-a",
        score,
        validation_results=({"ok": True, "owner": "unit"},),
        runtime_budget=default_runtime_spec(host).to_manifest(),
        progress_signals={"stalled": False},
        trace_ids=("trace-1",),
    )
    assert decision.action in {"promote", "early_stop"}
    lifecycle = trial_lifecycle_report([decision.to_log()])
    assert lifecycle["promoted"] or lifecycle["stopped"]
    report = poor_learning_report(
        trace_ids=["trace-1"],
        debug_bundles=["bundle.json"],
        failure_hints={"rust_failure_class": "MCTS prior validation", "policy_mapping": "low legal mass"},
    )
    assert report["likely_failure_classes"]["engine"] == "MCTS prior validation"


def test_tuning_import_audit_rejects_legacy_raw_config_modules():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hexorl" / "tuning"
    assert not (root / "asha.py").exists()
    assert not (root / "bohb.py").exists()
    assert not (root / "pb2.py").exists()
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
    assert "SearchSpace" not in text
    assert "raw config" not in text.lower()


def test_runtime_scripts_use_typed_recipe_transforms():
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[3]
    script_paths = [
        repo / "scripts" / "run_restnet_sparse_epoch10.py",
        repo / "scripts" / "run_ablation_suite.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in script_paths)
    assert "cfg.model.architecture =" not in text
    assert "setattr(" not in text
    assert ".split(\".\")" not in text
    assert "RecipeTransform" in text
    assert "ConfigSectionTransform" in text
