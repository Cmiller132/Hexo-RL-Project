"""Validate final refactor artifact packet structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "MANIFEST.md",
    "ci/ci_policy_checks.json",
    "import_audits/phase09_policy_audit.json",
    "deletion_manifest/deletion_manifest.md",
    "final_smoke/summary.json",
    "final_smoke/debug_bundle.json",
    "telemetry_samples/phase09_trace_samples.jsonl",
    "verification/mutation_corruption_report.json",
    "verification/rust_suspicion_report.json",
    "performance/performance_comparison.json",
    "ci_tiers/ci_tier_inventory.json",
    "ci_tiers/artifact_retention_policy.json",
    "ci_tiers/flaky_quarantine_report.json",
    "final_conformance_report.md",
    "agent_completion_packet.md",
    "evidence_reconciliation.md",
    "exit_gate_report.md",
)


def validate(phase_dir: Path) -> dict[str, object]:
    missing = [item for item in REQUIRED if not (phase_dir / item).exists()]
    invalid_json = []
    for path in phase_dir.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            invalid_json.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})
    for path in phase_dir.rglob("*.jsonl"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_json.append({"path": str(path.relative_to(ROOT)), "line": line_no, "error": str(exc)})
    return {"ok": not missing and not invalid_json, "missing": missing, "invalid_json": invalid_json}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_dir", type=Path, default=ROOT / "Docs/refactor/artifacts/phase_09", nargs="?")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    phase_dir = args.phase_dir if args.phase_dir.is_absolute() else ROOT / args.phase_dir
    report = validate(phase_dir)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
