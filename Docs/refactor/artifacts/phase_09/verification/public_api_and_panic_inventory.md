# Public API And Panic Inventory

Rust public API drift check:
- PR tier runs `cargo test --workspace`, `cargo test --workspace --release`, `cargo clippy --workspace --release -- -D warnings`.
- Scheduled tier runs ignored oracle tests with `cargo test --workspace --release -- --ignored --test-threads=1`.
- Phase 09 policy requires direct Rust MCTS ownership through `Python/src/hexorl/search/engine_adapter.py` only.

Panic/assert/unwrap classification:
- Public/FFI misuse is covered by Rust `MCTSError` and Python `EngineAdapterError` structured paths.
- Stale root token, stale batch token, invalid policy length, non-finite policy/value, and illegal tree action are classified as structured engine/search errors.
- Remaining Rust `assert!`/`debug_assert!` usage is internal invariant checking and is not a Python fallback or compatibility path.

Artifact references:
- `verification/rust_suspicion_report.json`
- `Python/tests/search/test_engine_adapter.py`
- `Python/tests/engine/test_phase01_engine_contract_parity.py`
