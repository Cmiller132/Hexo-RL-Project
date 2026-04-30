# Robust Engine CI, Performance, And Completeness Plan

Date: 2026-04-29

## 5. CI And Performance Gates

### Pull Request CI

PR CI must run:

- `cargo fmt --check`
- `cargo test --workspace`
- `cargo test --workspace --release`
- `cargo clippy --workspace --release -- -D warnings`
- `maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python`
- `pytest Python/tests/test_engine_smoke.py -v`
- `pytest Python/tests/test_engine_invariants.py -v`
- `pytest Python/tests/test_inference_server.py -v`

If a platform cannot link a PyO3 release test harness directly, CI must still
test the Rust workspace in release mode for non-PyO3 targets and verify the
extension through `maturin`.

### Deep CI

Scheduled/manual CI owns:

- ignored oracle tests;
- long property tests;
- benchmark sweeps;
- performance-budget drift reports.

Deep jobs use explicit timeout budgets so an infinite or unexpectedly explosive
test fails loudly.

### Performance Budgets

Record baseline timings for:

- candidate generation;
- encoding at radius 2 and radius 8;
- tactical status and tactical masks;
- MCTS selection;
- MCTS backpropagation;
- tree extraction.

Budgets become hard gates only after baselines are recorded in CI artifacts.

## 6. Implementation Sequence

1. Centralize FFI protocols and remove duplicated byte logic.
2. Add malformed/stale protocol tests.
3. Make `WindowKey` release-safe.
4. Add `HexGameState::assert_consistent()` and recompute tests.
5. Narrow public docs and exports to stable facades.
6. Expand CI fast and deep gates.
7. Record benchmark baselines and then enable drift detection.

## 7. Completeness Review

Before the refactor is accepted:

- every requested section has a source change, test, or explicit evidence that
  the existing implementation already satisfies it;
- no legacy compatibility path remains when an active robust path exists;
- every public/FFI invalid input path returns an error instead of panicking;
- every byte protocol has exactly one Rust implementation;
- every incremental cache has a recompute assertion path;
- CI covers Rust, Python, lint, formatting, and deep verification separation;
- performance-sensitive paths have named budgets or recorded baselines.
