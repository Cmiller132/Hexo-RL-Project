# S4 Eval / Threats / Oracle Hypotheses

Scope reviewed: `crates/hexgame-core/src/eval/*`, `crates/hexgame-core/src/threats.rs`, and oracle/threat/eval tests under `crates/hexgame-core/src/tests/*`.

## Hypothesis: finite win-grid can create false tactical quiet states

- **Area:** Incremental evaluator state, hot windows, false tactical data risks.
- **Risk:** Threat detection can silently miss windows whose origins fall outside `WIN_GRID_RADIUS`, causing `threat_status`, `live_cells`, search pruning, and encoder threat planes to see `Quiet` or incomplete hot cells while the board has real tactical threats.
- **Why It Might Be Hard To Catch:** The code documents this as an approximation, so failures may look like expected evaluator degradation rather than a tactical correctness bug. CI oracle tests use short random games and `candidates_near2()` generation, while the long 40-turn oracle tests are ignored.
- **Evidence Observed:** `WIN_GRID_RADIUS = 30` is explicitly finite, and the docs say windows are clipped after roughly 3-4 moves along an axis (`crates/hexgame-core/src/eval/grid.rs:21`, `crates/hexgame-core/src/eval/grid.rs:31`, `crates/hexgame-core/src/eval/grid.rs:79`). `EvalState::place` and `unplace` skip out-of-bounds windows (`crates/hexgame-core/src/eval/state.rs:265`, `crates/hexgame-core/src/eval/state.rs:322`). `threat_status` exits early from incremental `has_any_threats()` (`crates/hexgame-core/src/threats.rs:201`), and `live_cells` exits from incremental `has_threats()` (`crates/hexgame-core/src/threats.rs:437`).
- **Phase 2 Verification:** Build targeted positions with a contiguous 4/5/6-in-row whose relevant window origins cross radius 30, then compare `EvalState` hot/count data and `threat_status` against a full-board scanner independent of the win grid. Include encoder channel 9/10 checks if those planes are safety-critical.
- **Severity Guess:** High if tactical legality depends on this path outside the radius; Medium if this approximation is acceptable for search only.
- **Confidence:** High that the blind spot exists by design; Medium on practical impact.

## Hypothesis: oracle is not fully independent of production candidate/update state

- **Area:** Oracle coverage, update/undo consistency.
- **Risk:** The oracle can agree with the fast path while both are wrong if the shared candidate set or board place/unplace path omits a tactical cell or corrupts state.
- **Why It Might Be Hard To Catch:** The oracle is described as brute force and independent of `live_cells`, but it still uses `game.candidates_near2()`, `game.legal_moves()`, `place_unchecked`, and `unplace`. A candidate-maintenance bug or radius assumption can therefore enter both the "fast" and "oracle" sides.
- **Evidence Observed:** `player_candidates_near2` starts from `game.candidates_near2()` (`crates/hexgame-core/src/tests/oracle.rs:75`). `analyse` derives `near2_me`, `near2_opp`, and `near2_all` from candidate APIs (`crates/hexgame-core/src/tests/oracle.rs:111`, `crates/hexgame-core/src/tests/oracle.rs:149`) and mutates via `place_unchecked`/`unplace` (`crates/hexgame-core/src/tests/oracle.rs:125`, `crates/hexgame-core/src/tests/oracle.rs:132`). The public board candidate cache is also reused by `candidates_near2()` (`crates/hexgame-core/src/board.rs:665`).
- **Phase 2 Verification:** Add a test-only full-board scanner that enumerates legal cells without `CandidateSet` and detects wins with direct geometry, then compare it to both oracle and fast path on generated and hand-built tactical positions.
- **Severity Guess:** Medium.
- **Confidence:** High on shared dependencies; Medium on whether existing bugs exploit them.

## Hypothesis: multiple winning turns are collapsed to one tactical answer

- **Area:** Threat constraints, oracle coverage, false tactical data risks.
- **Risk:** `ThreatStatus::WinningTurn` and `turn_satisfies_status` accept only one exact winning turn, even when multiple immediate wins exist. That is fine for search pruning if any win is equivalent, but it can produce false negatives for legality masks, policy targets, or diagnostics that interpret the status as the complete set of allowed tactical moves.
- **Why It Might Be Hard To Catch:** The property tests intentionally allow the fast path to expose only the first winning turn, so this behavior is treated as acceptable in oracle comparisons.
- **Evidence Observed:** `threat_status` returns the first 5-window empty as `WinningTurn(Turn::single(...))` (`crates/hexgame-core/src/threats.rs:213`, `crates/hexgame-core/src/threats.rs:222`) or the first remembered 4-window pair (`crates/hexgame-core/src/threats.rs:224`, `crates/hexgame-core/src/threats.rs:233`). `turn_satisfies_status` requires exact equality for `WinningTurn` (`crates/hexgame-core/src/threats.rs:317`). The oracle test explicitly expects two winning singles for one position (`crates/hexgame-core/src/tests/oracle.rs:326`), while property tests allow non-reported winning turns to be skipped (`crates/hexgame-core/src/tests/threats.rs:313`). The encoder threat-constrained legal mask uses `threat_status(game)` and a single winning turn's cells (`crates/hexgame-core/src/encoder.rs:252`).
- **Phase 2 Verification:** Decide whether `ThreatStatus` is a search-only "one sufficient move" API or a complete tactical legality API. If the latter, add a multi-win fixture and require all oracle winning turns to satisfy the fast constraint.
- **Severity Guess:** Medium.
- **Confidence:** High.

## Direct issue: single-placement turn generation bypasses threat filtering outside the opening

- **Area:** Threat constraints, update/undo consistency consumers.
- **Risk:** Generated single-placement turns can bypass `turn_satisfies_status`, so a position with `placements_remaining() == 1` and active threats may produce unconstrained candidate turns.
- **Why It Might Be Hard To Catch:** Normal game flow may hit `placements_remaining == 1` mainly at the opening, while tests often call `threat_status` directly rather than route through search generation. Synthetic positions created by `set_position(..., remaining = 1)` are exactly where this can surface.
- **Evidence Observed:** Root generation handles empty/opening positions, but later returns all single candidates before the threat filter (`crates/hexgame-core/src/search.rs:480`, `crates/hexgame-core/src/search.rs:538`). Inner generation also returns single candidates before applying the precomputed status (`crates/hexgame-core/src/search.rs:573`). In contrast, pair paths retain by `turn_satisfies_status` (`crates/hexgame-core/src/search.rs:545`, `crates/hexgame-core/src/search.rs:583`). The threat module supports single-placement `MustBlock` (`crates/hexgame-core/src/threats.rs:276`, `crates/hexgame-core/src/threats.rs:321`), and tests cover that semantic directly (`crates/hexgame-core/src/tests/threats_internal.rs:219`).
- **Phase 2 Verification:** Add or run a search-level fixture with `remaining = 1`, opponent five/four threats, and verify generated turns are filtered to the blocking cell or winning move. If this is intentionally unreachable in production search except opening, document the invariant at the generator boundary.
- **Severity Guess:** Medium.
- **Confidence:** Medium, because this crosses beyond the assigned files into the search consumer.

## Hypothesis: hot-window duplicate/missing-key errors are debug-only detectable

- **Area:** Hot windows, update/undo consistency.
- **Risk:** `HotWindows` duplicate inserts are guarded only by `debug_assert!`, and removals silently no-op if a key is absent. In release, a subtle incremental mismatch could leave duplicates or stale hot windows that inflate live cells, distort block constraints, or make threat status look urgent when it is not.
- **Why It Might Be Hard To Catch:** The strong invariant rebuild only runs inside `EvalState::unplace` in debug builds, not in release search. Existing tests cover simple insertion/removal and roundtrips, but not adversarial overlapping hot-window churn.
- **Evidence Observed:** Duplicate prevention is a debug assertion (`crates/hexgame-core/src/eval/hot.rs:68`), `remove` ignores missing keys (`crates/hexgame-core/src/eval/hot.rs:83`), and the full hot-window recomputation is `#[cfg(debug_assertions)]` (`crates/hexgame-core/src/eval/state.rs:350`). Incremental hot updates occur on both placement and unplacement (`crates/hexgame-core/src/eval/state.rs:290`, `crates/hexgame-core/src/eval/state.rs:344`).
- **Phase 2 Verification:** Run a release-mode differential fuzzer comparing incremental hot windows/counts to a full recompute after every random place/unplace, including overlapping 4/5-window transitions and opponent blocks.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Hypothesis: threat-count invariants do not validate counts against a full recompute

- **Area:** Incremental evaluator state, update/undo consistency.
- **Risk:** `assert_invariants` rebuilds hot-window sets but not `ThreatCounts` or `score`, so counts could drift while hot windows remain correct. Since `has_threats` is based on counts, drift could cause false `Quiet`, false `MustBlock`, or unnecessary quiescence.
- **Why It Might Be Hard To Catch:** Tests assert roundtrip-to-zero and simple monotonic examples, but not count equality after every intermediate board state against a full recompute.
- **Evidence Observed:** `EvalState::place` applies count deltas (`crates/hexgame-core/src/eval/state.rs:286`, `crates/hexgame-core/src/eval/state.rs:297`), while `unplace` applies the negated deltas (`crates/hexgame-core/src/eval/state.rs:315`). The debug invariant recomputes only expected hot windows (`crates/hexgame-core/src/eval/state.rs:356`) and compares hot sets (`crates/hexgame-core/src/eval/state.rs:360`). `has_threats` uses `counts` directly (`crates/hexgame-core/src/eval/state.rs:417`).
- **Phase 2 Verification:** Add a test helper that recomputes score, fives, fours, threes, and hot windows from board stones after each placement/unplacement in random and targeted sequences.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Structure recommendation: make oracle "all legal moves block when no opponent threats" explicit in the type

- **Area:** Oracle coverage, threat constraints.
- **Risk:** `TurnAnalysis.blocking_single` / `blocking_pairs` are overloaded: when there are no opponent winning turns, vacuous `.all()` makes many moves appear as blocking. The property tests compensate by checking `opp_has_threats`, but future callers may misread the oracle result as literal block evidence.
- **Why It Might Be Hard To Catch:** The behavior is mathematically correct for "opponent has no winning response" but semantically misleading for a field named `blocking_*`.
- **Evidence Observed:** Blocking classification uses `.all(...)` over `opp_winning` (`crates/hexgame-core/src/tests/oracle.rs:171`, `crates/hexgame-core/src/tests/oracle.rs:181`, `crates/hexgame-core/src/tests/oracle.rs:213`). The property helper notes that when the opponent has no immediate threats, the oracle includes all legal moves and should not be compared as `MustBlock` (`crates/hexgame-core/src/tests/threats.rs:122`).
- **Phase 2 Verification:** Consider adding an `opp_has_winning_response` boolean or renaming fields to `safe_single` / `safe_pairs`, then derive literal blocking sets only when the opponent has threats.
- **Severity Guess:** Low.
- **Confidence:** High.

## Question: should `Unblockable` really make all turns satisfy the threat filter?

- **Area:** Threat constraints, false tactical data risks.
- **Risk:** `turn_satisfies_status(Unblockable, turn)` returns true for every turn. That matches comments saying the branch is hopeless, but it means any downstream consumer treating the function as a legality mask cannot distinguish tactical hopelessness from unconstrained legality.
- **Why It Might Be Hard To Catch:** Search handles `Unblockable` separately in quiescence, but encoder or training-data consumers may use constrained masks differently.
- **Evidence Observed:** `turn_satisfies_status` returns true for `Unblockable` (`crates/hexgame-core/src/threats.rs:342`). The unit test asserts this behavior, including for an arbitrary far-away cell (`crates/hexgame-core/src/tests/threats_internal.rs:280`). Search quiescence treats `Unblockable` as a terminal tactical loss (`crates/hexgame-core/src/search.rs:727`), while encoder mask construction maps `Unblockable` to no constraint (`crates/hexgame-core/src/encoder.rs:252`).
- **Phase 2 Verification:** Clarify whether `turn_satisfies_status` is a search-pruning helper or a general "allowed tactical moves" predicate. If general, expose a separate status-to-mask policy for `Unblockable`.
- **Severity Guess:** Low to Medium.
- **Confidence:** Medium.

## Structure recommendation: add adversarial overlap fixtures for block constraints

- **Area:** Threat constraints, oracle coverage.
- **Risk:** Current hand tests cover straight-axis simple runs, disjoint threats, and one common-cell case, but block constraints can become tricky with overlapping windows across axes, shared empties, and mixed 4-window/5-window combinations.
- **Why It Might Be Hard To Catch:** Random tests are broad but slow/ignored at high depth, and deterministic fixtures are mostly one-axis. Pair enumeration depends on intersections and all-pairs coverage of `all_cells`.
- **Evidence Observed:** Constraint construction intersects all threat-window empties (`crates/hexgame-core/src/threats.rs:258`) and enumerates pairs from all unique cells (`crates/hexgame-core/src/threats.rs:287`). The hand tests assert one-axis exact pairs and disjoint runs (`crates/hexgame-core/src/tests/threats_internal.rs:92`, `crates/hexgame-core/src/tests/threats_internal.rs:116`, `crates/hexgame-core/src/tests/threats_internal.rs:141`).
- **Phase 2 Verification:** Add table-driven fixtures with two-axis crossings, one shared single blocker plus valid/invalid pairs, and mixed own-winning/opponent-must-block positions. Compare against the oracle and a full-board scanner.
- **Severity Guess:** Medium.
- **Confidence:** Medium.
