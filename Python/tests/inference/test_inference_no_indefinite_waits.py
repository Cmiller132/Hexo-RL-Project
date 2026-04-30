from pathlib import Path


def test_inference_transport_waits_are_bounded():
    root = Path(__file__).resolve().parents[2] / "src" / "hexorl" / "inference"
    offenders = []
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            compact = line.split("#", 1)[0].replace(" ", "")
            if line.strip().startswith("server.join()"):
                continue
            if ".wait()" in compact or ".join()" in compact:
                offenders.append(f"{path.name}:{line_no}:{line.strip()}")
    assert offenders == []
