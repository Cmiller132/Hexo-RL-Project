# Rust Review Phase 2 Verification Report

Date: 2026-04-29

## Executive Summary

Phase 2 confirmed the highest-confidence Phase 1 defects and fixed the issues
that had clear, low-risk resolutions.  The continuation pass also re-checked
each Phase 1 subagent section, applied additional unambiguous fixes, and split
remaining work into explicit refactor/design buckets.

Phase 2 is now complete as a verification and implementation pass for the
highest-risk engine correctness items.  The continuation pass resolved the
largest remaining correctness decisions by adding a chronological history API,
tokenizing asynchronous MCTS inference, widening MCTS-owned coordinates to
`i32`, replacing bounded tactical-hot-window dependence with a complete sparse
tactical scanner, and routing first-party MCTS callers through fallible APIs.

The remaining work is now architectural cleanup rather than known correctness
poisoning: narrowing the public crate API, centralizing FFI byte protocols,
expanding performance budgets, and deciding how aggressive to be about removing
older convenience wrappers that still panic on misuse in direct Rust tests.

## Completed Fixes

- `set_position` is transactional on validation failures.
- `set_position` is documented as a synthetic fixture board loader, not
  chronological proof.
- `set_position` rejects post-terminal extra stones.
- `PLACEMENT_RADIUS` rustdoc now matches the implemented rule: non-opening
  stones must be near an existing stone, not near the origin.
- public Rust/Python Zobrist docs now state that the hash is board-only and is
  not sufficient alone for full-state transposition keys.
- `HexGameState::eval()` rustdoc now states that eval/hot-window data is bounded
  by the finite eval grid.
- repeated `select_leaves` rolls back abandoned virtual loss before selecting a
  new batch.
- `select_leaves` now fails loudly instead of silently continuing if an expanded
  child action is illegal.
- unordered root pair-prior rows now reject reversed duplicate pairs.
- pair-first root policy telemetry now reports consumed root legal rows.
- Dirichlet noise is validated as finite/non-negative, normalized before
  blending, and root priors are renormalized after blending.
- `extract_tree_node_states` now selects candidates from the reachable subtree
  below the current root instead of the whole arena.
- `extract_tree_node_states` cleanup now uses edge-depth from the current root,
  avoiding off-by-one unplace cleanup on error paths.
- single-placement root and inner search turn generation now applies
  `turn_satisfies_tactical`, so must-block/winning constraints are not bypassed
  when `placements_remaining() == 1`.
- Python MCTS root expansion validates `legal_bytes` and tensor offsets against
  the latest `init_root`.
- Python dense root/leaf policy inputs reject non-finite logits before entering
  Rust MCTS.
- CI Python integration builds `hexgame-py` through the crate manifest and
  activates a venv before `maturin develop`.
- `hexgame-py` declares the `python` feature requested by CI.
- ignored slow oracle tests have been moved out of normal push/PR CI into a
  scheduled/manual `deep-oracle` job.
- `load_history` now replays real chronological `(q, r, player)` move histories
  from an empty board, validates turn order, rejects trailing post-terminal
  moves, and commits transactionally.
- Python exposes `load_history` while keeping `set_position` for synthetic
  fixtures.
- MCTS action storage, sampled actions, re-root actions, and extracted tree
  histories now use `i32` coordinates instead of narrowing to `i16`.
- MCTS root and leaf inference now has generation tokens:
  `init_root` returns a root token, `select_leaves` returns a batch token, and
  root/backprop APIs reject stale tokens.
- Python self-play and the Rust CLI/bench callers now pass those MCTS tokens
  through the inference loop.
- `TacticalStatus` is now the complete tactical source of truth, preserving all
  immediate winning turns and complete block constraints.
- tactical detection now scans sparse full-board windows touching actual stones
  instead of depending on bounded eval-grid hot windows.
- search, quiescence, root candidate generation, and encoder legal masks now
  consume complete tactical constraints.
- mandatory tactical win/block turns are appended beyond normal candidate and
  pair caps, preventing caps from deleting required legal responses.
- `HexGameState::unplace` now returns `Result<(), GameError>` instead of
  panicking on an empty history.
- `MCTSEngine::add_dirichlet_noise` now returns `Result<(), MCTSError>` instead
  of asserting on malformed noise.
- `hexgame-core` now exposes stable `rules`, `encoding`, `tactics`, and
  `classical` facades; implementation modules are no longer public API.

## Verification Commands

Continuation verification:

- `cargo fmt --check` passed.
- `cargo test --workspace` passed:
  - core lib: 118 passed, 6 ignored;
  - core board integration: 45 passed;
  - core encoder integration: 9 passed;
  - core doctests: 2 passed;
  - CLI/Python crate test harnesses: no Rust tests.
- `cargo clippy --workspace --release -- -D warnings` passed.
- `cargo test -p hexgame-core --release` passed:
  - core lib: 117 passed, 6 ignored;
  - core board integration: 45 passed;
  - core encoder integration: 9 passed;
  - core doctests: 2 passed.
- `cargo test -p hexgame-cli --release` passed.
- `cargo test -p hexgame-bench --release` passed.
- `cargo bench -p hexgame-bench --bench threats -- --warm-up-time 1 --measurement-time 2`
  passed; `tactical_status` measured about 5.8 microseconds on the local machine.
- `cargo bench -p hexgame-bench --bench mcts -- --warm-up-time 1 --measurement-time 2`
  passed; `single_mcts_full_sim` measured about 0.50 milliseconds on the local
  machine.
- `maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python`
  passed against the local `.venv`.
- `pytest Python/tests/test_engine_smoke.py -v` passed: 8 passed.
- `pytest Python/tests/test_engine_invariants.py -v` passed: 13 passed.
- `pytest Python/tests/test_inference_server.py -v` passed: 7 passed.

Local macOS caveat:

- `cargo test --workspace --release` failed while linking the `hexgame-py` lib
  test binary because the local macOS linker did not resolve Python symbols for
  PyO3 (`_PyBaseObject_Type`, `_PyBytes_*`, etc.).  This is separate from the
  extension workflow: `maturin develop` plus Python smoke tests passed, and the
  non-PyO3 release test targets passed.

Prior Phase 2 verification:

- `cargo test --workspace`
- `cargo test --workspace --release`
- `cargo clippy --workspace --release -- -D warnings`
- WSL Python integration:
  - `maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python`
  - `pytest Python/tests/test_engine_smoke.py -v`

Deep verification:

- `cargo test --workspace --release -- --ignored` exceeded 15 minutes locally
  during the first Phase 2 pass.  It is now a scheduled/manual deep gate rather
  than a PR gate.

## Consolidated Phase 1 Status

| ID | Status | Result |
|---|---|---|
| D1 | Fixed | `set_position` reset-before-validation was real. Transactional loading and regression tests are in place. |
| D2 | Fixed | repeated MCTS selection leaked virtual loss. Pending rollback and regression tests are in place. |
| D3 | Fixed | Python root expansion validates dense/sparse/global root legal rows and offsets against `init_root`, and root generation tokens reject stale root inference responses. |
| D4 | Fixed | CI requested an undeclared `python` feature and ambiguous maturin build. Feature and manifest-path workflow are in place. |
| D5 | Fixed | reversed unordered pair rows were accepted. Duplicate rejection and test are in place. |
| D6 | Fixed | `PLACEMENT_RADIUS` docs contradicted behavior; rustdoc now matches implementation. |
| H1 | Fixed | `set_position` remains synthetic and rejects post-terminal stones; `load_history` now provides chronological replay with player/turn validation and transactional commit. |
| H2 | Mostly verified | Candidate-set brute-force property coverage exists for place/unplace, and transactional failure tests check state preservation. A broader `assert_consistent()` hook remains useful. |
| H3 | Fixed for MCTS-owned paths | MCTS node actions, re-root, sample_action, and extracted histories now use `i32`; far-coordinate roundtrip coverage was added. `WindowKey` bounds remain a separate eval-structure question. |
| H4 | Mostly fixed | Python dense policy paths reject non-finite root/leaf logits. New tokenized MCTS APIs, `unplace`, and Dirichlet noise return `Result`; some older direct Rust convenience wrappers still panic and should be retired or hidden during public API narrowing. |
| H5 | Fixed | Root and batch generation tokens are implemented in Rust and Python. Stale root and stale batch responses return errors. |
| H6 | Fixed for tactical correctness | Tactical status no longer depends on bounded eval-grid hot windows; far-grid 4/5/6 fixtures cover sparse full-board tactical detection. Eval scoring itself remains a bounded incremental heuristic. |
| H7 | Fixed for engine tactical filtering | The tactical scanner now independently scans sparse full-board windows touching stones rather than depending on `CandidateSet`, `EvalState`, `live_cells`, or hot windows. |
| H8 | Fixed | `TacticalStatus` carries complete masks/turn sets and the legacy `ThreatStatus` compatibility model has been removed from active core code. |
| H9 | Fixed | single-placement filtering is fixed and mandatory tactical turns bypass normal candidate/pair caps for search and masking paths. |
| H10 | Partially verified | Release-mode fast tests pass; full ignored oracle suite is now a deep gate. Counts/hot-window full recompute coverage should be expanded. |

## Subagent Completion Review

### S1 - Core Rules And Board State

Confirmed fixed:

- `set_position` transactionality.
- post-terminal extra-stone rejection.
- placement radius docs.
- Zobrist documentation now clarifies board-only semantics.
- eval accessor docs now clarify finite-grid semantics.

Accepted current behavior:

- `set_position` remains a synthetic fixture loader.  Its move history is useful
  for fixtures but is not chronological proof.

Still open:

- Decide whether synthetic histories should be excluded from encoder
  recency/opponent-last-turn workflows, or whether fixture docs plus
  `load_history` are sufficient.
- Add a debug/test `assert_consistent()` helper that recomputes hash,
  candidates, winner, and selected eval/hot data from stones.

### S2 - Encoding And Python FFI

Confirmed fixed:

- root stale legal row/offset validation for dense, sparse, and global root
  expansion.
- Python dense non-finite root/leaf policy rejection.

Partially fixed:

- legal byte decoding is centralized, but byte encoding and history triple
  decoding are still split across binding paths.
- sparse root expansion is protected against stale root legal rows, but sparse
  auxiliary rows still do not have a full row-identity protocol.

Still open:

- Add malformed legal/history byte tests from Python.
- Decide whether unsorted legal rows are an intentional coordinate-keyed
  contract or should be sorted at FFI boundaries.
- Add D6 scalar-channel invariance tests for phase/color/distance channels.
- Rebuild `_engine` before local Python smoke verification; stale local
  extensions can otherwise hide or falsely fail source-level fixes.

### S3 - MCTS And Search

Confirmed fixed:

- virtual-loss rollback on repeated selection.
- unordered pair-prior duplicate rejection.
- pair-first telemetry.
- Dirichlet noise validation/normalization.
- extraction reachable-candidate selection.
- extraction cleanup edge-depth.
- single-placement search threat filtering.
- illegal child traversal now fails loudly instead of silently corrupting search
  state.

Still open:

- Retire or hide older direct Rust MCTS convenience wrappers that still panic on
  misuse, now that tokenized `Result` APIs exist.
- Add a search-level first-placement-win regression for score and exact undo
  restoration.

### S4 - Eval And Threats

Confirmed fixed:

- single-placement turn generation no longer bypasses threat filtering.
- public eval docs now mention bounded grid semantics.

Accepted current behavior:

- `TacticalStatus::WinningTurns` is the source of truth for tactical masks,
  diagnostics, and search filtering.
- `Unblockable` makes `turn_satisfies_tactical` unconstrained because the branch
  is already tactically lost; callers that need labels should preserve the
  `Unblockable` status itself rather than treating the predicate as a mask.

Still open:

- Extend release/full-recompute tests to validate `ThreatCounts`, score, and hot
  windows after intermediate place/unplace states.
- Add adversarial overlap fixtures for two-axis crossings and mixed 4/5-window
  block constraints.
- Decide whether/when Python tactical-oracle helpers should delegate to the Rust
  complete `TacticalStatus` scanner to eliminate duplicate semantics.

### S5 - Structure, CI, Tests, Performance

Confirmed fixed:

- Python CI feature/workflow.
- ignored slow oracle tests removed from normal PR/push CI and moved to a
  scheduled/manual deep job.
- stable public facades added and first-party downstream callers moved off deep
  implementation module paths.

Still open:

- Produce a panic/assert inventory and classify public/FFI misuse vs internal
  invariants.
- Resolve `WindowKey` 15-bit release behavior with a fallible constructor,
  runtime assertion, or explicit coordinate bound.
- Centralize history and pair byte protocols.
- Add benchmark budgets and at least one representative training-batch
  benchmark.

## Completed Refactor Decisions

1. Board loading split
   - Keep `set_position` as synthetic fixture loading.
   - Added `load_history` that replays chronological legal moves and rejects
     impossible turn order.
   - Mark synthetic-history encoder assumptions prominently.

2. MCTS protocol hardening
   - Added `root_generation` and `batch_generation` tokens.
   - Root expansion and backprop now require matching generation tokens.
   - Python, CLI, and bench callers now use the tokenized/fallible path.

3. Coordinate contract
   - Widened MCTS action storage and extracted histories to `i32`.
   - Applied the same decision to Python `sample_action` and `re_root`.
   - Added far-coordinate MCTS roundtrip coverage.

4. Tactical oracle and search caps
   - Implemented a sparse full-board tactical scanner over stone-touched
     windows.
   - Added complete `TacticalStatus` for winning-turn sets and block masks.
   - Search, quiescence, root candidates, and encoder legal masks now use the
     complete tactical constraints.
   - Mandatory tactical turns bypass normal candidate/pair caps.

5. Eval-grid classification
   - Tactical correctness no longer depends on the finite eval grid.
   - Eval scoring remains a bounded incremental heuristic and is documented as
     such.

## Recommended Follow-up Refactor Plan

1. Public API narrowing
   - Audit external users outside this workspace for old deep module imports.
   - Keep `rules`, `encoding`, `tactics`, and `classical` as the stable facade
     modules.
   - Retire or hide old MCTS convenience wrappers that still panic on misuse.

2. FFI protocol centralization
   - Move legal-row, pair-row, and history byte encoding/decoding into one
     shared Rust helper module.
   - Add Python tests for malformed legal/history bytes and stale token
     submissions.

3. Consistency and invariant hooks
   - Add a debug/test `assert_consistent()` helper for hash, candidates,
     winner, move history, and selected eval/hot data.
   - Extend release/full-recompute tests for `ThreatCounts`, score, and hot
     windows after intermediate place/unplace states.

4. WindowKey and eval bounds
   - Resolve `WindowKey` 15-bit release behavior with a fallible constructor,
     runtime assertion, or explicit coordinate bound.
   - Keep eval-grid bounds separate from tactical correctness guarantees.

5. CI and performance gates
   - Keep PR CI fast: workspace tests, release fast tests, clippy, Python smoke.
   - Keep ignored oracle tests in scheduled/manual deep CI, or shard them with
     explicit timeout budgets.
   - Record timing budgets for candidate generation, encoding, threat status,
     MCTS selection/backprop, and tree extraction.

## Remaining Nuanced Issues

These should not be fixed casually without choosing the public contract first:

- whether far-coordinate support is fully unbounded or explicitly bounded;
- whether synthetic histories should be allowed to drive recency/last-turn
  encoder features;
- whether direct Rust MCTS convenience wrappers are kept for tests only, hidden,
  or removed in favor of tokenized `Result` APIs;
- whether Python tactical-oracle helpers should delegate to Rust
  `TacticalStatus`;
- what explicit coordinate bounds apply to `WindowKey` and bounded eval
  internals.
