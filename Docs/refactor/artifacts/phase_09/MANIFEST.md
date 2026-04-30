# Phase 09 Artifact Manifest

Phase: Final Deletion And CI Enforcement

Base SHA at start: `26eae99`

Scope:
- Close V2 rows `V2-090` through `V2-100`.
- Delete final old runtime package paths: `Python/src/hexorl/action_contract/` and `Python/src/hexorl/buffer/`.
- Enforce final V2 architecture policy in CI.
- Archive final smoke, policy audit, mutation/corruption, Rust suspicion, CI tier, and performance evidence.

Generated artifacts:
- `import_audits/phase09_policy_audit.json`
- `final_smoke/summary.json`
- `final_smoke/debug_bundle.json`
- `final_smoke/autotune_dry_run.json`
- `telemetry_samples/phase09_trace_samples.jsonl`
- `verification/mutation_corruption_report.json`
- `verification/rust_suspicion_report.json`
- `performance/performance_comparison.json`
- `ci_tiers/ci_tier_inventory.json`
- `ci_tiers/artifact_retention_policy.json`
- `ci_tiers/flaky_quarantine_report.json`
- `ci/ci_policy_checks.json`
- `deletion_manifest/deletion_manifest.md`
- `final_conformance_report.md`
- `agent_completion_packet.md`
- `evidence_reconciliation.md`
- `exit_gate_report.md`

Primary commands:
- `python tools\refactor\phase09_policy_audit.py --output Docs\refactor\artifacts\phase_09\import_audits\phase09_policy_audit.json`
- `python tools\refactor\phase09_final_smoke.py --output-dir Docs\refactor\artifacts\phase_09\final_smoke`
- `python tools\refactor\phase09_performance_probe.py --output Docs\refactor\artifacts\phase_09\performance\performance_comparison.json`

No migration tooling remains under `Python/src/hexorl/` for deleted packages.
