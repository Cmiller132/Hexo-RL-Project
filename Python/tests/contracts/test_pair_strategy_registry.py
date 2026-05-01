import pathlib

from hexorl.contracts.pair_strategy import (
    PAIR_STRATEGY_REGISTRY,
)


def test_builtin_pair_strategy_descriptors_build_specs():
    none = PAIR_STRATEGY_REGISTRY.build_spec("none", max_pairs=0)
    assert none.name == "none"
    assert none.root_enabled is False
    assert none.leaf_enabled is False
    assert none.chunk_size == 0

    two_stage = PAIR_STRATEGY_REGISTRY.build_spec("two_stage_root", max_pairs=32)
    assert two_stage.name == "two_stage_root_only"
    assert two_stage.root_enabled is True
    assert two_stage.leaf_enabled is False
    assert two_stage.max_root_pair_rows == 32
    assert two_stage.max_full_pair_rows == 0
    assert two_stage.chunk_size == 32

    tactical = PAIR_STRATEGY_REGISTRY.build_spec("tactical", max_pairs=16)
    assert tactical.name == "tactical_only"
    assert tactical.root_enabled is True
    assert tactical.max_root_pair_rows == 16

    diagnostic = PAIR_STRATEGY_REGISTRY.build_spec("diagnostic_full_pair", max_pairs=64)
    assert diagnostic.name == "diagnostic_full_root"
    assert diagnostic.diagnostic is True
    assert diagnostic.root_enabled is True
    assert diagnostic.leaf_enabled is False
    assert diagnostic.max_full_pair_rows == 64
    assert diagnostic.max_root_pair_rows == 0


def test_descriptor_builds_pair_table_strategy():
    descriptor = PAIR_STRATEGY_REGISTRY.resolve("diagnostic_full_pair")
    table_strategy = descriptor.build_table_strategy(max_pairs=7)
    assert table_strategy.generation_mode == "full_capped"
    assert table_strategy.max_pairs == 7
    assert table_strategy.allow_full is True


def test_pair_strategy_grep_gate_has_no_runtime_ladders():
    root = pathlib.Path(__file__).resolve().parents[3]
    offenders: list[str] = []
    for base in ("Python/src/hexorl/selfplay", "Python/src/hexorl/search", "Python/src/hexorl/contracts"):
        for path in (root / base).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "pair_strategy" not in text:
                continue
            if "pair_strategy/registry.py" in str(path) or "pair_strategy/descriptors.py" in str(path):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                compact = line.replace(" ", "")
                if "pair_strategy==" in compact or "pair_strategyin{" in compact:
                    offenders.append(f"{path}:{line_number}:{line.strip()}")
    assert offenders == []
