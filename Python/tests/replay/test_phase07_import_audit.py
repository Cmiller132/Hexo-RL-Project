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
