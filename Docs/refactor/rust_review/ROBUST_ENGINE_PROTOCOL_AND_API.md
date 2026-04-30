# Robust Engine Protocol And API Plan

Date: 2026-04-29

## 1. Public API Narrowing

### Target Shape

The public API is organized around explicit facades:

- `hexgame_core::rules`
- `hexgame_core::encoding`
- `hexgame_core::tactics`
- `hexgame_core::classical`

Root-level re-exports are allowed only when they are intentionally stable and
actively used by first-party callers.  Everything else remains private or
`pub(crate)`.

### Required Changes

- Remove public documentation that presents compatibility threat APIs as active
  engine contracts.
- Keep `TacticalStatus` as the tactical source of truth.
- Keep MCTS fallible APIs canonical; do not reintroduce panic wrappers.
- Audit root re-exports and keep only the intentionally stable set.
- Update Rust docs/examples to prefer facade paths.

### Completion Tests

- `cargo doc --workspace --no-deps` succeeds.
- `cargo test --workspace` succeeds after facade updates.
- `rg` shows no public panic-wrapper names or old `try_`/compatibility MCTS
  protocol paths.

## 2. FFI Protocol Centralization

### Target Shape

`crates/hexgame-py/src/protocol.rs` owns the binary wire formats used by PyO3:

- legal rows: repeated little-endian `(q: i32, r: i32)`;
- pair rows: repeated little-endian `(q1: i32, r1: i32, q2: i32, r2: i32)`;
- board pieces: repeated little-endian `(q: i32, r: i32, player: i32)`;
- compact histories: repeated little-endian `(player: i32, q: i32, r: i32)`.

Every binding uses these helpers.  No binding hand-rolls byte loops or duplicate
length checks.

### Required Changes

- Add typed protocol errors mapped to Python `ValueError`.
- Decode malformed byte lengths before any state mutation.
- Validate pair rows for distinct coordinates and canonicalize where the Rust
  MCTS contract expects unordered pairs.
- Route `encode_compact_record`, `move_history_bytes`, `board_pieces_bytes`,
  `legal_moves_near_bytes`, `encode_board_and_legal`, root expansion, sparse
  expansion, and pending leaf metadata through the shared helpers.
- Keep Python method names stable for active self-play/inference callers.

### Completion Tests

- Python malformed legal bytes fail before root expansion.
- Python malformed compact history bytes fail before encode replay.
- Python stale root token submissions fail.
- Python stale batch token submissions fail.
- Python malformed pair rows fail before prior application.
- Rust PyO3 source contains one implementation of each binary protocol.
