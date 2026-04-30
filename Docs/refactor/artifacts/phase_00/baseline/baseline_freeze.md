# Phase 00 Baseline Freeze

- Created: `2026-04-30T03:31:49Z`
- Branch: `main`
- Baseline SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`
- Baseline tag: `v2-phase-00-pre-python-foundation`
- Tag SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`
- Rust baseline: current tree contains the completed Rust Phase 2 hardening state referenced by `Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md`.
- Current runtime oracle policy: the current Python runtime is inventory input only and is not accepted as sole proof for later refactor boundaries.
- Known local instability: a stale Windows shared-memory segment caused one focused pytest attempt to fail; stale local processes were stopped and the deterministic rerun passed.

Functional and smoke evidence is indexed in `commands/COMMAND_INDEX.md`. Host and performance evidence is indexed in `performance/performance_summary.md`.
