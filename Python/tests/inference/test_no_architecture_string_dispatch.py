from pathlib import Path


def test_inference_server_has_no_architecture_string_dispatch():
    root = Path(__file__).resolve().parents[2] / "src" / "hexorl" / "inference"
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for banned in ('architecture.startswith', 'startswith("global_"', "startswith('global_'"):
            if banned in text:
                offenders.append(f"{path.name}:{banned}")
    assert offenders == []
