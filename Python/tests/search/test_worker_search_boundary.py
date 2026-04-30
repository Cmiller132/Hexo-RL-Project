from pathlib import Path


def _worker_source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "src" / "hexorl" / "selfplay" / "worker.py").read_text(encoding="utf-8")


def _runner_source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "src" / "hexorl" / "selfplay" / "game_runner.py").read_text(encoding="utf-8")


def test_selfplay_worker_contains_no_architecture_string_checks():
    source = _worker_source()
    assert "architecture.startswith" not in source
    assert 'startswith("global_' not in source
    assert "global_graph_enabled" not in source


def test_worker_does_not_call_rust_mcts_directly():
    source = _worker_source()
    assert "mcts_engine_class" not in source
    assert "PyMCTSEngine" not in source
    assert "RealMCTSEngine" not in source
    assert "MockMCTSEngine" not in source


def test_worker_does_not_score_pair_chunks_directly():
    source = _worker_source()
    assert "_score_graph_pair_chunks" not in source
    assert "_score_crop_pair_chunks" not in source
    assert "policy_pair_first" not in source
    assert "policy_pair_joint" not in source
    assert "policy_pair_second" not in source


def test_evaluation_uses_policy_provider_path():
    source = _runner_source()
    assert "create_policy_provider" in source
    assert "SearchEvaluation" in source
    assert "commit_root(engine, evaluation, pair_eval)" in source
