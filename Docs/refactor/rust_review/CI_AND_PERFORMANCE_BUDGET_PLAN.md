# CI And Performance Budget Plan

Date: 2026-04-29

Purpose: define fast pull-request gates, separated deep CI, and concrete performance budgets for the current Rust/Python engine slice.

This is the Rust-specific annex to `Docs/refactor/CI_STRATEGY.md` and `Docs/refactor/PERFORMANCE_STRATEGY.md`. The central refactor CI policy owns tiering, artifact retention, flaky/quarantine rules, and final V2 closure. This annex supplies the Rust engine checks and benchmark areas that feed those central gates.

## Fast PR Gates

Fast CI should run on push and pull request:

| Gate | Command |
| --- | --- |
| Rust formatting | `cargo fmt --all -- --check` |
| Rust workspace tests | `cargo test --workspace` |
| Rust release fast tests | `cargo test --workspace --release` |
| Rust clippy | `cargo clippy --workspace --release -- -D warnings` |
| Python extension smoke/invariants/inference | Build `_engine` with `maturin develop`, then run `pytest` for `test_engine_smoke.py`, `test_engine_invariants.py`, and `test_inference_server.py` |

Python inference tests require the Python project dependencies, including CPU Torch. If a dependency outage makes the inference test infeasible on CI, the failure should be explicit in CI output rather than silently skipped by workflow logic.

## Deep CI Separation

Scheduled and manual workflows own expensive tests:

| Gate | Event | Timeout budget |
| --- | --- | --- |
| Ignored Rust oracle/deep tests | `schedule`, `workflow_dispatch` | 60 minutes |
| Future benchmark baseline check | `workflow_dispatch` first, then `schedule` after stable thresholds | 30 minutes |

Deep oracle tests remain separate from PR gates so a pull request gets quick correctness feedback while longer stochastic/oracle coverage still runs regularly.

Artifact retention and flaky-test handling follow `Docs/refactor/CI_STRATEGY.md`. Required Rust suspicion checks may not be silently skipped; if a deep Rust oracle becomes flaky, it needs an owner, issue, expiry, continued scheduled execution, and a deterministic PR replacement that covers the same invariant.

## Performance Budget Matrix

These are planning budgets for the benchmark suite and profiling scripts. They are concrete enough to wire into gates once baseline data is captured on stable hardware.

| Area | Existing or intended measurement | Budget target | Gate path |
| --- | --- | --- | --- |
| Candidate generation | Criterion `legal_moves_near_radius2`, `legal_moves_near_radius8`, `legal_moves_near_into_radius8`, `candidates_near2` | No more than 10% regression from checked-in baseline median on the same runner profile. | Add Criterion JSON export in `crates/hexgame-bench` and compare medians in a workflow script. |
| Encoding | Criterion `encode_board_into`, `encode_board_into_radius8` | No more than 10% regression; zero extra allocation in steady-state `encode_board_into` buffers. | Extend `benches/encode.rs` with allocation-sensitive scenario or add a small Rust perf harness. |
| Tactical status | Criterion `tactical_status` plus multi-threat fixtures | No more than 10% regression and no fixture over 2x baseline. | Capture Criterion baseline after tactical fixtures are added. |
| MCTS select/backprop | Criterion `single_mcts_full_sim` and a future split select/backprop bench | No more than 10% regression for full sim; split phases get separate baselines before gating. | Add benches for `select_leaves` and `expand_and_backprop` using deterministic mock policies. |
| Tree extraction | Python smoke/profile around `extract_tree_node_states` | No more than 15% regression for fixed min-visits/tree size once baseline is recorded. | Add a Python benchmark script beside existing `benches/threaded_inference_benchmark.py`. |
| Python/Rust inference boundary | Maturin build plus engine/inference smoke profile | No unbounded wait; no stale-token or malformed-byte acceptance; latency/throughput recorded with HostProfile. | Feed Phase 04/05/09 performance artifacts and central scheduled comparison. |

## Why Hard Perf Gates Are Deferred

Hard performance gates are not enabled in this slice because the repository has Criterion benches but no checked-in machine-normalized baselines, no JSON comparison script, and no stable runner-specific threshold metadata. Enabling hard gates without that baseline would produce noisy failures instead of actionable regression signals.

This deferral is acceptable only before final V2 closure. Phase 09 must not close until benchmark metadata, runner profiles, JSON comparison tooling, scheduled artifacts, and threshold ownership exist for every promoted budget.

The next executable path is:

1. Add a benchmark metadata file under `crates/hexgame-bench/` that records runner profile, command, benchmark ids, and accepted regression percentage.
2. Add a script that reads Criterion estimates and compares them with that metadata.
3. Run the script only on `workflow_dispatch` until two scheduled runs establish stability.
4. Promote candidate generation, encoding, tactical status, and MCTS split benches to scheduled gates.
5. Add tree extraction after the Python benchmark has a deterministic fixture size.
