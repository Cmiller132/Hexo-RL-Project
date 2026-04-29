# Rust Review Phase 2 Verification Plan

Phase 2 should verify or falsify Phase 1 hypotheses. It should not rely on the current runtime as the only oracle.

## Tactical Oracle Bound

Use a radius-3 tactical oracle for winning/blocking verification:

- Generate candidate tactical placements from empty cells within 3 hexes of existing stones.
- Treat radius 3 as the conservative verification bound until a formal proof shows radius 2 is sufficient.
- Do not brute-force all radius-8 legal cells for tactical oracle checks unless a test is specifically about radius-8 legality or far-placement behavior.
- Limit winning and blocking window analysis to 4+ windows. With two placements available, a 3-window cannot become a win in one turn.
- Keep a small number of explicit radius-8 legality tests separate from tactical-win/block verification.

This bound should make Phase 2 oracle tests faster and easier to reason about while still covering tactical correctness.

## Golden And Corrupt Fixtures

Add a small Rust fixture module for:

- opening position
- normal two-placement turn
- second-placement state with one placement remaining
- near-terminal immediate win
- multi-win position with more than one winning move
- mandatory single block
- mandatory pair block
- far-coordinate tactical threat near/outside eval grid bounds
- non-chronological `set_position` state
- invalid duplicate, invalid player, post-terminal, and out-of-radius `set_position` inputs

Each fixture should state expected board state, current player, placements remaining, legal rows, terminal state, and whether history is real or synthetic.

## Core Rules Verification

- Test `set_position` transactionality by preserving an existing game across every validation failure class.
- Decide `set_position` semantics. If it is a board loader, validate all input before mutation and document synthetic history. If it is a history loader, enforce chronological players and turn phases.
- Add candidate-set proptests that compare cached legal rows with brute-force scans after every place and unplace.
- Add a debug/test-only `assert_consistent()` that recomputes hash, candidates, winner, and selected eval/hot data from stones.
- Add far-coordinate tests for MCTS `i16` truncation and `WindowKey` 15-bit packing bounds.

## Encoding And FFI Verification

- Add stale `legal_bytes` tests for dense, sparse, and global root expansion.
- Add stale offset tests that prove shifted offsets are rejected or cannot alter mapping silently.
- Add malformed legal/history byte tests with odd lengths, duplicate legal rows, illegal cells, stale rows, and wrong sortedness assumptions.
- Add non-finite dense policy tests for root and leaf expansion.
- Add batch-generation misuse tests: select batch A, select batch B, submit A's outputs/metadata.
- Verify D6 tensor symmetry preserves scalar planes, distance channel, masks, and legal-row semantics for all 12 transforms.

## MCTS And Search Verification

- Test repeated `select_leaves` without backprop and require virtual loss rollback.
- Test caught/unwindable shape failures only if the API intends recoverability; otherwise document panic-only behavior and keep Python wrappers fallible.
- Inject or construct stale/illegal child actions and assert selection/extraction fails without corrupting root state.
- Add state-snapshot assertions around temporary MCTS traversals: history, current player, placements remaining, winner, zobrist.
- Test reversed unordered pair rows and define reject/merge semantics.
- Verify pair-first telemetry reports candidate counts consistently.
- Use the radius-3 threat oracle to prove search candidate/pair caps do not omit mandatory winning or blocking moves from 4+ windows.
- Add a pair-turn first-placement-win test for alpha-beta score and exact undo behavior.

## Eval/Threat Verification

- Add a radius-3 tactical scanner independent of `CandidateSet`, `EvalState`, `live_cells`, and hot windows.
- Keep the scanner limited to 4+ windows for winning/blocking searches.
- Compare incremental eval/hot windows/threat status against the full scanner after every random place/unplace.
- Include release-mode or optimized differential runs so debug-only assertions are not the only safety net.
- Test far-grid 4/5/6-in-row positions to classify finite-grid behavior as defect, accepted approximation, or bounded-contract requirement.
- Test multi-winning-turn positions and decide whether `ThreatStatus` exposes a complete tactical mask or one sufficient search move.

## Structure And CI Verification

- Run the Python integration command from CI: `maturin develop --features python`.
- Inventory all non-test panic/assert/unwrap sites and classify them as user input, FFI input, internal invariant, or impossible.
- Run `cargo public-api` or an equivalent API inventory to identify deep public module paths that can be made private/re-exported.
- Capture current timings for fast tests vs ignored oracle tests, then decide PR vs nightly placement.
- Compare benchmark fixtures against self-play telemetry and add at least one representative hot-path benchmark if needed.

## Phase 2 Exit Criteria

- Every Phase 1 item is marked confirmed, disproven, accepted tradeoff, structure-only, or no-action.
- Confirmed bugs have regression tests before fixes are proposed.
- Accepted tradeoffs have explicit documentation and boundaries.
- Structure recommendations are split into small, independently actionable refactors.
