"""Phase 09 architecture policy audit.

The audit is intentionally deterministic and local. It checks the final V2
deletion gates that are cheap enough to enforce on every PR, and emits a JSON
artifact suitable for CI upload and phase evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]

ABSENT_RUNTIME_PATHS = (
    Path("Python/src/hexorl/model"),
    Path("Python/src/hexorl/buffer"),
    Path("Python/src/hexorl/action_contract"),
)

SCAN_ROOTS = (
    Path("Python/src/hexorl"),
    Path("Python/scripts"),
    Path("scripts"),
    Path("benches"),
    Path(".github/workflows"),
)

PYTHON_SCAN_ROOTS = (
    Path("Python/src/hexorl"),
    Path("Python/scripts"),
    Path("scripts"),
    Path("benches"),
)

ARCHITECTURE_GATE_ALLOWLIST = {
    Path("Python/src/hexorl/config/schema.py"),
    Path("Python/src/hexorl/models/specs.py"),
    Path("Python/src/hexorl/models/factory.py"),
}

MCTS_ALLOWLIST = {
    Path("Python/src/hexorl/engine/rust.py"),
    Path("Python/src/hexorl/search/engine_adapter.py"),
}

DASHBOARD_PRIVATE_ALLOWLIST = {
    Path("Python/src/hexorl/dashboard/contract_inspector.py"),
}

BANNED_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+hexorl\.(?:model|buffer|action_contract)(?:\.|\s+import)"
    r"|import\s+hexorl\.(?:model|buffer|action_contract)(?:\.|\s|$))"
)

DASHBOARD_PRIVATE_PATTERNS = (
    "CandidateContractBuilder",
    "PairActionTableBuilder",
    "build_graph_batch_from_history",
    "transform_history",
)

RUNTIME_ARCH_PATTERNS = (
    "architecture.startswith",
    'startswith("global_"',
    "startswith('global_'",
    "architecture ==",
    "architecture in",
)

MCTS_PATTERNS = (
    "mcts_engine_class(",
    "MCTSEngine",
    "PyMCTSEngine",
)


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    line: int
    detail: str


def _rel(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def _files(roots: Iterable[Path], suffixes: tuple[str, ...]) -> Iterable[Path]:
    for root in roots:
        absolute = ROOT / root
        if not absolute.exists():
            continue
        if absolute.is_file():
            candidates = [absolute]
        else:
            candidates = [p for p in absolute.rglob("*") if p.is_file()]
        for path in candidates:
            if any(part == "__pycache__" for part in path.parts):
                continue
            if path.suffix.lower() in suffixes:
                yield path


def _line_number(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text.count("\n", 0, idx) + 1


def _check_absent_paths(findings: list[Finding]) -> None:
    for rel in ABSENT_RUNTIME_PATHS:
        path = ROOT / rel
        if path.exists():
            findings.append(Finding("absent-runtime-path", rel.as_posix(), 1, "runtime legacy package exists"))


def _check_banned_imports(findings: list[Finding]) -> None:
    for path in _files(SCAN_ROOTS, (".py", ".yml", ".yaml")):
        rel = _rel(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        match = BANNED_IMPORT_RE.search(text)
        if match:
            findings.append(Finding("banned-runtime-import", rel.as_posix(), _line_number(text, match.group(0)), match.group(0).strip()))


def _check_architecture_gates(findings: list[Finding]) -> None:
    for path in _files(PYTHON_SCAN_ROOTS, (".py",)):
        rel = _rel(path)
        if rel in ARCHITECTURE_GATE_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in RUNTIME_ARCH_PATTERNS:
            if pattern in text:
                findings.append(Finding("architecture-string-gate", rel.as_posix(), _line_number(text, pattern), pattern))


def _check_direct_mcts(findings: list[Finding]) -> None:
    for path in _files((Path("Python/src/hexorl"),), (".py",)):
        rel = _rel(path)
        if rel in MCTS_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in MCTS_PATTERNS:
            if pattern in text:
                findings.append(Finding("direct-rust-mcts-call", rel.as_posix(), _line_number(text, pattern), pattern))


def _check_dashboard_private_rebuilds(findings: list[Finding]) -> None:
    root = ROOT / "Python/src/hexorl/dashboard"
    if not root.exists():
        return
    for path in root.glob("*.py"):
        rel = _rel(path)
        if rel in DASHBOARD_PRIVATE_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in DASHBOARD_PRIVATE_PATTERNS:
            if pattern in text:
                findings.append(Finding("dashboard-private-reconstruction", rel.as_posix(), _line_number(text, pattern), pattern))


def _check_duplicate_protocol_decoders(findings: list[Finding]) -> None:
    owner = Path("Python/src/hexorl/engine/legal.py")
    allowed = {owner, Path("Python/src/hexorl/contracts/history.py"), Path("Python/src/hexorl/engine/history.py")}
    for path in _files((Path("Python/src/hexorl"),), (".py",)):
        rel = _rel(path)
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "np.frombuffer" in text and ("legal" in text.lower() or "history" in text.lower() or "pair" in text.lower()):
            findings.append(Finding("duplicate-ffi-byte-decoder", rel.as_posix(), _line_number(text, "np.frombuffer"), "np.frombuffer"))


def _check_skipped_phase09_tests(findings: list[Finding]) -> None:
    for path in _files((Path("Python/tests"),), (".py",)):
        rel = _rel(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        if "phase09" in rel.as_posix().lower() and ("@pytest.mark.skip" in text or "@pytest.mark.xfail" in text):
            findings.append(Finding("phase09-skipped-test", rel.as_posix(), _line_number(text, "@pytest.mark"), "skip/xfail in Phase 09 test"))


def run_audit() -> dict[str, object]:
    findings: list[Finding] = []
    _check_absent_paths(findings)
    _check_banned_imports(findings)
    _check_architecture_gates(findings)
    _check_direct_mcts(findings)
    _check_dashboard_private_rebuilds(findings)
    _check_duplicate_protocol_decoders(findings)
    _check_skipped_phase09_tests(findings)
    return {
        "schema_version": 1,
        "cwd": str(ROOT),
        "ok": not findings,
        "checks": {
            "absent_runtime_paths": [p.as_posix() for p in ABSENT_RUNTIME_PATHS],
            "scan_roots": [p.as_posix() for p in SCAN_ROOTS],
            "architecture_gate_allowlist": [p.as_posix() for p in sorted(ARCHITECTURE_GATE_ALLOWLIST)],
            "mcts_allowlist": [p.as_posix() for p in sorted(MCTS_ALLOWLIST)],
            "dashboard_private_allowlist": [p.as_posix() for p in sorted(DASHBOARD_PRIVATE_ALLOWLIST)],
        },
        "findings": [asdict(item) for item in findings],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_audit()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
