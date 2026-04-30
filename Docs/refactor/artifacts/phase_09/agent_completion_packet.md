# Phase 09 Agent Completion Packet

Closed V2 rows:
- `V2-090` through `V2-100`

Runtime consumers changed:
- Tactical oracle runtime imports now consume `hexorl.engine.tactical`.
- Long-run scripts consume `ReplayStorage` instead of deleted `hexorl.buffer`.
- Tuning trial manifest uses registered global model family membership instead of string-prefix behavior.

Files changed:
- CI workflow: `.github/workflows/ci.yml`
- Policy/smoke tooling: `tools/refactor/phase09_policy_audit.py`, `phase09_final_smoke.py`, `phase09_performance_probe.py`, `phase09_artifact_validator.py`
- Runtime cleanup: deleted `Python/src/hexorl/action_contract/`, deleted `Python/src/hexorl/buffer/`, added `Python/src/hexorl/engine/tactical.py`
- Tests/docs/artifacts updated under `Python/tests/phase09/`, `Docs/refactor/artifacts/phase_09/`, `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`

Legacy paths deleted or quarantined:
- Deleted from runtime: `hexorl.action_contract`, `hexorl.buffer`
- No `hexorl.model` runtime package exists
- No migration tool remains under `Python/src/hexorl/`

Tests and commands run with exit status:
- `python tools\refactor\phase09_policy_audit.py --output Docs\refactor\artifacts\phase_09\import_audits\phase09_policy_audit.json` -> 0
- `python tools\refactor\phase09_final_smoke.py --output-dir Docs\refactor\artifacts\phase_09\final_smoke` -> 0
- `python tools\refactor\phase09_performance_probe.py --output Docs\refactor\artifacts\phase_09\performance\performance_comparison.json` -> 0
- `python -m pytest Python\tests -q` -> 0 (`293 passed`)
- `cargo test --workspace` -> 0
- `cargo test --workspace --release` -> 0
- `cargo clippy --workspace --release -- -D warnings` -> 0
- `npm run build` in `Python/dashboard_frontend` -> 0

Artifacts produced:
- `Docs/refactor/artifacts/phase_09/`

Performance/utilization evidence:
- `performance/performance_comparison.json`
- final smoke storage throughput stats in `final_smoke/summary.json`

Contract examples/docs:
- `contract_examples/contract_examples_audit.md`
- final debug bundle and tuning dry-run artifacts

Known blockers:
- None.

Skipped/deferred/manual-only statement:
- No skipped, deferred, xfailed, flaky-only, or manual-only requirement is being claimed complete.
