# Phase 00 Command Index

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`
- Allowed status values used: `passed`, `failed-known-baseline`.
- Nonzero baseline attempts are retained with superseding passing transcripts; no failed command is used as closure proof.

| Command ID | V2 rows | CI tier | Owner | Status | Exit | Transcript | Config hash | Notes |
|---|---|---|---|---|---:|---|---|---|
| `P00-CMD-001` | V2-000,V2-005 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_fmt_check.txt` | `none` |  |
| `P00-CMD-002` | V2-000,V2-005 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_test_workspace.txt` | `none` |  |
| `P00-CMD-003` | V2-000,V2-005 | deep | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_test_workspace_release.txt` | `none` |  |
| `P00-CMD-004` | V2-000,V2-005 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_clippy_workspace_release.txt` | `none` |  |
| `P00-CMD-005A` | V2-000,V2-005 | pr_required | S2 | failed-known-baseline | 1 | `Docs/refactor/artifacts/phase_00/commands/maturin_develop_hexgame_py.txt` | `P00-HASH-RUST-PYO3-MANIFEST` | Initial maturin attempt recorded the missing active venv; P00-CMD-005B supersedes it. |
| `P00-CMD-005B` | V2-000,V2-005 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/maturin_develop_hexgame_py_venv.txt` | `P00-HASH-RUST-PYO3-MANIFEST` |  |
| `P00-CMD-006A` | V2-000,V2-003,V2-005 | pr_required | S2 | failed-known-baseline | 1 | `Docs/refactor/artifacts/phase_00/commands/pytest_engine_smoke_invariants_inference.txt` | `P00-HASH-PYTHON-PROJECT` | Initial pytest run saw stale Windows shared-memory segments from a timed-out local run; P00-CMD-006B supersedes it. |
| `P00-CMD-006B` | V2-000,V2-003,V2-005 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/pytest_engine_smoke_invariants_inference_rerun.txt` | `P00-HASH-PYTHON-PROJECT` |  |
| `P00-CMD-007` | V2-000,V2-005 | deep | Orchestrator | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/pytest_full_python_tests.txt` | `P00-HASH-PYTHON-PROJECT` |  |
| `P00-CMD-008` | V2-000 | local | S5 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/python_hexorl_cli_help.txt` | `P00-HASH-PYTHON-PROJECT` |  |
| `P00-SMOKE-INFERENCE` | V2-003,V2-006 | pr_required | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/phase00_inference_smoke.txt` | `P00-HASH-RUNTIME-SMOKE-SCRIPT` |  |
| `P00-SMOKE-SELFPLAY` | V2-001,V2-002,V2-003,V2-006 | pr_required | S2/S3 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/phase00_selfplay_smoke.txt` | `P00-HASH-RUNTIME-SMOKE-SCRIPT` |  |
| `P00-SMOKE-TRAINING` | V2-000,V2-006 | deep | S4 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/phase00_training_smoke.txt` | `P00-HASH-RUNTIME-SMOKE-SCRIPT` |  |
| `P00-SMOKE-AUTOTUNE` | V2-003,V2-006 | deep | S4 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/phase00_autotune_runtime_dry_run.txt` | `P00-HASH-RUNTIME-SMOKE-SCRIPT` |  |
| `P00-SMOKE-DASHBOARD` | V2-000,V2-006 | pr_required | S4 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/dashboard_frontend_build.txt` | `P00-HASH-DASHBOARD-DEPS` |  |
| `P00-WATCHDOG` | V2-003,V2-006 | pr_required | S2 | passed | 2 | `Docs/refactor/artifacts/phase_00/commands/phase00_watchdog_smoke_expected_abort.txt` | `P00-HASH-BASELINE-SCRIPT` | Underlying command exited 2 by design; wrapper transcript records passed_expected_abort. |
| `P00-PERF-THREATS` | V2-006 | scheduled | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_bench_threats_short.txt` | `none` |  |
| `P00-PERF-MCTS` | V2-006 | scheduled | S2 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_bench_mcts_short.txt` | `none` |  |
| `P00-PERF-ENCODE` | V2-006 | scheduled | S2/S4 | passed | 0 | `Docs/refactor/artifacts/phase_00/commands/cargo_bench_encode_short.txt` | `none` |  |
| `P00-FINALIZE` | V2-000,V2-003,V2-004,V2-005,V2-006 | final | Orchestrator | passed | 0 | `scripts/phase00_finalize_artifacts.py` | `P00-HASH-FINALIZE-SCRIPT` | Generated final control-plane artifacts from command transcripts and reviewable evidence. |

## Mandatory Check Resolution

- `maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python` first failed because maturin requires an active venv; `.venv` was created for Windows and the same develop build passed in `P00-CMD-005B`.
- Focused engine/inference pytest first failed because stale local shared-memory state remained from a timed-out run; stale local pytest processes were stopped and `P00-CMD-006B` passed.
- The watchdog smoke intentionally exits 2 from the underlying script and is transcripted as `passed_expected_abort` because predictable abort is the required behavior.

## CI Routing

| Tier | Rows | Commands | Promotion rule |
|---|---|---|---|
| local | V2-000 | CLI/help and developer reruns | Supports closure only with transcript and manifest evidence. |
| pr_required | V2-000,V2-001,V2-002,V2-003,V2-005,V2-006 | Rust fmt/test/clippy, maturin, focused pytest, self-play/inference/dashboard/watchdog | Missing or non-passing superseded rows block closure. |
| deep | V2-000,V2-005,V2-006 | Rust release tests, full Python tests, training/autotune smokes | Required for Phase 00 signoff. |
| scheduled | V2-006 | Short Rust benches and later stable-runner benchmarks | Phase 00 records local baseline; final V2 schedules compare on stable runners. |
| artifact_only | V2-003,V2-004,V2-005 | Logs, traces, inventories, verification plan | Must be linked to a command or reviewable artifact; cannot replace deterministic checks. |
