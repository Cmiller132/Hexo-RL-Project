from pathlib import Path


RUNTIME_ROOTS = [
    Path("Python/src/hexorl/selfplay"),
    Path("Python/src/hexorl/replay"),
    Path("Python/src/hexorl/train"),
    Path("Python/src/hexorl/epoch"),
]


def test_phase07_runtime_has_no_buffer_imports():
    offenders = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "hexorl.buffer" in text or "from hexorl.buffer" in text or "import hexorl.buffer" in text:
                offenders.append(str(path))
    assert offenders == []


def test_phase07_projector_is_the_training_batch_source():
    trainer = Path("Python/src/hexorl/train/adapters.py").read_text(encoding="utf-8")
    sampler = Path("Python/src/hexorl/replay/sampler.py").read_text(encoding="utf-8")
    assert "replay/projector.py" in trainer
    assert "ReplayProjector" in sampler


def test_phase07_train_adapter_rejects_raw_tuple_batch_path():
    trainer = Path("Python/src/hexorl/train/adapters.py").read_text(encoding="utf-8")
    assert "len(batch) == 5" not in trainer
    assert "len(batch) == 4" not in trainer
    assert "ProjectedReplayBatch" in trainer


def test_phase07_runtime_has_no_reference_pair_row_helper_or_python_legal_fallback():
    graph_builder = Path("Python/src/hexorl/graph/semantic_builder.py").read_text(encoding="utf-8")
    epoch_pipeline = Path("Python/src/hexorl/epoch/pipeline.py").read_text(encoding="utf-8")
    engine_legal = Path("Python/src/hexorl/engine/legal.py").read_text(encoding="utf-8")

    forbidden_helper = "graph_batch_with_" + "reference_pair_rows"
    forbidden_legal = "_fallback_" + "bootstrap_legal_moves"
    forbidden_game = "_make_" + "fallback_" + "bootstrap_game"
    forbidden_raw_legal = "game." + "legal_moves()"
    assert f"def {forbidden_helper}" not in graph_builder
    assert forbidden_legal not in epoch_pipeline
    assert forbidden_game not in epoch_pipeline
    assert forbidden_raw_legal not in engine_legal
    assert "encode_board_and_legal legal-byte protocol" in engine_legal
