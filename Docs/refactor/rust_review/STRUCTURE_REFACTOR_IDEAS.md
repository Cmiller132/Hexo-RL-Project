# Rust Structure Refactor Ideas

These are maintainability/code-quality recommendations. They do not need the same defect verification as behavior hypotheses, but they should still be implemented with tests where they touch public behavior.

Status note: this file is historical design input. The final active public API decision is recorded in `API_AND_FFI_PROTOCOL_PLAN.md`: stable facades are `rules`, `encoding`, `tactics`, and `classical`; `mcts` remains public only as an FFI exception for `hexgame-py`; the crate root should not preserve convenience re-exports of implementation details.

## Narrow The Public API

`hexgame-core` currently publishes implementation modules directly and also re-exports selected public types. Prefer a smaller stable public surface:

- keep stable facade modules for `rules`, `encoding`, `tactics`, and `classical`
- make implementation modules private or `pub(crate)` where downstream callers do not need them
- keep `mcts` public only for the active Python FFI crate until a stable MCTS facade is intentionally designed

Benefit: future internal refactors stop becoming public API compatibility problems.

## Separate Safe Protocol Facades From Internal Engines

`MCTSEngine` exposes a multi-step protocol where some misuse is guarded by asserts/panics. Add a fallible facade for external callers:

- root generation token returned by `init_root`
- batch generation token returned by `select_leaves`
- fallible root expansion and backprop calls
- explicit stale-token and wrong-shape errors

Benefit: Python and future Rust callers cannot accidentally mix stale legal rows, tensors, metadata, or results.

## Centralize FFI Byte Protocols

Legal/action bytes and history bytes are encoded/decoded in several locations. Add small shared helpers:

- `encode_hex_pairs`
- `decode_hex_pairs`
- `encode_move_history`
- `decode_move_history_triples`

Benefit: malformed input handling, duplicate checks, row identity checks, and wire-format documentation live in one place.

## Clarify Board Loader Vs History Loader

Split or rename `set_position` semantics:

- `set_fixture_position`: board loader for synthetic tests, explicit that history is synthetic
- `load_history`: chronological legal-history loader, validates turn order and can replay exactly

Benefit: encoder recency, opponent-last-turn, replay, and debug tools no longer guess whether history is real.

## Add Consistency Assertion Hooks

Add debug/test helpers that recompute derived data from authoritative stones:

- full legal/candidate sets
- zobrist hash
- current winner/winning line
- eval counts/hot windows where in-bounds
- search root state after temporary MCTS traversals

Benefit: subtle cache drift becomes a local test failure instead of a later training-quality mystery.

## State The Panic Policy

Create a short Rust policy document or module-level section:

- public/FFI bad input returns `Result`
- internal impossible invariants may panic
- debug assertions are not the only guard for externally reachable invalid data
- release `panic = "abort"` means production panics terminate workers

Benefit: future contributors know when to add validation instead of assertions.

## Separate PR CI From Deep Verification

Keep fast correctness checks in PR CI and move slow oracle/property suites into a named deep gate or nightly job if they become expensive.

Benefit: high-value oracle tests remain available without making everyday feedback brittle.

## Make Performance Budgets Visible

Add lightweight benchmark/smoke budgets for correctness-sensitive hot paths:

- legal move generation
- encoder radius 2 and radius 8
- threat status/live cells
- MCTS select/backprop
- tree-node extraction

Benefit: robustness work can be balanced against self-play throughput.
