# Public API Inventory

## Workspace

Root workspace members:

- `crates/hexgame-core`
- `crates/hexgame-py`
- `crates/hexgame-bench`
- `crates/hexgame-cli`

## `hexgame-core`

Public modules currently exported from `src/lib.rs`:

- `board`
- `core`
- `encoder`
- `eval`
- `mcts`
- `search`
- `threats`

Convenience re-exports:

- `GameError`
- `HexGameState`
- `Hex`
- `Turn`
- `PLACEMENT_RADIUS`
- `WIN_LENGTH`
- `MCTSEngine`
- `MCTSError`
- `live_cells`
- `threat_status`
- `ThreatStatus`

Review implication: implementation modules are part of the public API, so internal refactors may require compatibility planning unless the API surface is narrowed.

## `hexgame-py`

PyO3 extension crate:

- crate name: `hexgame-py`
- library name: `_engine`
- crate type: `cdylib`

Prominent Python-exposed surfaces observed:

- `HexGame`
- legal move list/bytes exports
- move history list/bytes exports
- board/tensor encoding
- tactical oracle
- classical search
- `MCTSEngine` root init, root expansion, leaf selection, backprop, pair priors, reroot, results, sampling, telemetry

Review implication: Python/Rust protocol shape is broad and accepts byte buffers, numpy arrays, offsets, logits, legal rows, and metadata that should be guarded by stronger identity checks.

## CI/Public Tooling Surface

CI currently runs:

- `cargo build --release`
- `cargo test --release`
- `cargo test --release -- --ignored`
- `cargo clippy --release -- -D warnings`
- `maturin develop --features python`
- `pytest Python/tests/test_engine_smoke.py -v`

Review implication: the Python integration command appears to name an undeclared Cargo feature; Phase 2 should verify the workflow command directly.

