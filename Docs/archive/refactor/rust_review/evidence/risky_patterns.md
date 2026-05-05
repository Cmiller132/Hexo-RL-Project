# Risky Pattern Inventory

This inventory is not a defect list. It names patterns that should receive targeted verification in Phase 2.

## Input/Protocol Validation

- Dense/sparse root expansion accepts caller-provided legal rows and offsets.
- Leaf batch submission is tied to current pending state, not an explicit batch token.
- Legal/history bytes are hand-encoded/decoded in multiple PyO3 methods.
- Dense policy paths appear less strict about non-finite logits than global prior paths.

## State Mutation And Restoration

- `set_position` mutates the game while validating tuple-by-tuple.
- `HexGameState` has multiple mutation paths: normal placement, undo, reset, custom position setup.
- MCTS temporarily mutates its internal game during selection and extraction.
- Eval/hot windows and candidate sets are derived caches that must stay aligned with stones.

## Approximation Boundaries

- Board rules are sparse/infinite.
- Eval/hot-window threat detection is finite-grid bounded.
- Tactical verification can use a conservative radius-3 oracle around existing stones; radius-8 legality is not needed for ordinary win/block oracle checks.
- Winning/blocking oracle checks can ignore 3-windows because two placements cannot turn a 3-window into a win this turn.
- MCTS action storage uses `i16` while `Hex` uses `i32`.
- Encoder crops to a fixed 33x33 window and can clip legal/hot cells.

## Panic/Assert Sites

- Several invalid inputs are currently guarded by `assert!`, `debug_assert!`, `expect`, or `panic!`.
- Release profile uses `panic = "abort"`, which makes production panics worker-fatal.
- A Phase 2 panic inventory should classify each non-test panic/assert as user input, FFI input, internal invariant, or impossible.

## Test Coverage Shape

- Strong oracle machinery exists, but some deep oracle tests are ignored/slow.
- The oracle shares some production APIs such as candidates and place/unplace.
- Existing MCTS tests cover many happy paths and pair-prior cases, but not stale batch/root misuse.
- Benchmarks exist but are not connected to explicit CI budgets.
