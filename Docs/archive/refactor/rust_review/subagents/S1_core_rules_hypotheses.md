# S1 Core Rules Hypotheses

Scope: `crates/hexgame-core` core rules, with emphasis on board state, move legality, placement phases, undo, `set_position`, candidate sets, win detection, hashes, and direct consumers where they can expose core-rule drift.

This is Phase 1 hypothesis formation. Entries below are not final findings unless classified as `Direct issue`, and even those should be verified in Phase 2 with targeted tests.

## 1. Direct issue: `set_position` is not transactional on validation failure

- Area: Custom `set_position`, board state integrity.
- Risk: A failed `set_position` call can discard the previous game and leave a partially loaded replacement position.
- Why It Might Be Hard To Catch: Existing tests only assert the returned error for duplicate cells; they do not assert that the caller's previous state is preserved after an error.
- Evidence Observed: `HexGameState::set_position` validates `player` and `remaining`, then calls `self.reset()` before per-stone validation and mutation (`crates/hexgame-core/src/board.rs:443`, `crates/hexgame-core/src/board.rs:450`, `crates/hexgame-core/src/board.rs:455`). Later errors such as duplicate cell, invalid stone player, non-origin first stone, or out-of-radius return after earlier stones may already have been inserted (`crates/hexgame-core/src/board.rs:457`, `crates/hexgame-core/src/board.rs:460`, `crates/hexgame-core/src/board.rs:463`, `crates/hexgame-core/src/board.rs:466`). The duplicate-cell test only checks `Err` (`crates/hexgame-core/tests/board.rs:501`).
- Phase 2 Verification: Add a test that starts from a non-empty valid game, calls `set_position` with an invalid later tuple, then asserts all public state, hash, history, winner, eval-dependent legal moves, and candidates are unchanged. Also test the same from an empty game to decide whether partial mutation is acceptable API behavior.
- Severity Guess: High.
- Confidence: High.

## 2. Hypothesis: `set_position` can load post-terminal stones and make winner depend on input order

- Area: Custom `set_position`, terminal-state semantics, win detection.
- Risk: Synthetic positions may contain stones after the first detected win, while `winner` and `winning_line` reflect whichever winning line is first encountered in input order rather than a canonical board fact.
- Why It Might Be Hard To Catch: Normal `place` prevents moves after game over, but `set_position` intentionally bypasses normal turn rules. Tests cover terminal detection and no remaining placements, but not extra stones after a terminal line or two players already having six-in-a-row.
- Evidence Observed: `set_position` continues iterating over every tuple even after `self.winner` becomes `Some`; it only skips additional winner detection with `if self.winner.is_none()` (`crates/hexgame-core/src/board.rs:495`). At the end it sets `placements_remaining` to `0` if any winner exists (`crates/hexgame-core/src/board.rs:509`). Normal placement rejects further moves after terminal state via `validate_move` (`crates/hexgame-core/src/board.rs:581`).
- Phase 2 Verification: Construct two `set_position` inputs with the same final stones but different ordering where both players have a six-line, and check whether `winner` changes. Also load a six-line plus trailing stones and decide whether the API should reject, truncate, or explicitly allow impossible terminal snapshots.
- Severity Guess: Medium to High.
- Confidence: Medium.

## 3. Hypothesis: `set_position`'s history snapshots are inconsistent for arbitrary custom positions

- Area: Custom `set_position`, undo, placement phases.
- Risk: After `set_position`, `move_history` records synthetic `current_player_before` and `placements_remaining_before` values that may not match the supplied stone owners or final `(player, remaining)`. `unplace` may therefore restore a plausible chronological game only when the input tuple list was already chronological.
- Why It Might Be Hard To Catch: Most test fixtures use `set_position` as a static board loader and do not undo back through the custom setup. Undo tests mostly exercise normal `place` or one move after setup.
- Evidence Observed: `set_position` initializes `sim_player` and `sim_remaining` from a fresh reset state (`crates/hexgame-core/src/board.rs:452`) and writes those values into each `MoveRecord` regardless of `stone_player` (`crates/hexgame-core/src/board.rs:475`). It then assigns the caller-provided final `current_player` and `placements_remaining` only after all stones are inserted (`crates/hexgame-core/src/board.rs:509`). `unplace` restores directly from the popped record (`crates/hexgame-core/src/board.rs:409`).
- Phase 2 Verification: Add round-trip tests that call `set_position` with non-chronological but otherwise accepted stone owners, call `unplace` repeatedly, and compare against either documented expected behavior or a freshly replayed chronological construction. Clarify whether `set_position` is a position loader or a history loader.
- Severity Guess: Medium.
- Confidence: Medium.

## 4. Hypothesis: `opponent_last_turn_cells` can over-report after `set_position`

- Area: Placement phases, move history, encoder channel 12.
- Risk: The encoder may mark more than the opponent's last completed turn if a custom position has more than two consecutive records for the non-current player.
- Why It Might Be Hard To Catch: In normal gameplay, `place` enforces one opening placement and then two placements per player. `set_position` does not enforce player alternation, and many fixtures intentionally list several same-player stones in a row.
- Evidence Observed: `opponent_last_turn_cells` groups by contiguous `MoveRecord.player` values, not by stored turn boundaries (`crates/hexgame-core/src/board.rs:696`). It keeps collecting while the previous record's player differs from `current_player` (`crates/hexgame-core/src/board.rs:706`). Encoder channel 12 consumes this directly and assumes one or two cells (`crates/hexgame-core/src/encoder.rs:329`). `set_position_basic` demonstrates multiple same-player records are accepted (`crates/hexgame-core/tests/board.rs:447`).
- Phase 2 Verification: Build a `set_position` fixture with current player 0 and three trailing player-1 stones, then encode and inspect channel 12. Decide whether `set_position` should synthesize turn boundaries, reject impossible histories, or whether `opponent_last_turn_cells` should clamp/derive from placement counts.
- Severity Guess: Medium.
- Confidence: Medium.

## 5. Hypothesis: Public `zobrist()` is a board-only hash but is exposed as if it may identify full game state

- Area: Hashes, transposition/deduplication interfaces.
- Risk: Callers using `zobrist()` outside `search::tt_hash` may collide positions with identical stones but different `current_player` or `placements_remaining`, which are different legal states.
- Why It Might Be Hard To Catch: Internal alpha-beta handles this correctly with a private side/phase mix, while Python exports the raw board hash under documentation saying it is suitable for transposition tables.
- Evidence Observed: `zobrist` is XORed only with `(player, cell)` on place/unplace (`crates/hexgame-core/src/board.rs:547`, `crates/hexgame-core/src/board.rs:548`, `crates/hexgame-core/src/board.rs:402`, `crates/hexgame-core/src/board.rs:403`). The alpha-beta TT separately mixes current player and placements remaining (`crates/hexgame-core/src/search.rs:238`). The Python getter returns raw `self.inner.zobrist()` and documents it as suitable for transposition tables (`crates/hexgame-py/src/engine.rs:166`).
- Phase 2 Verification: Create two states via `set_position` with the same stones and different `(player, remaining)`, assert raw `zobrist()` equality and private-equivalent full-state hash inequality. Audit Python and replay-buffer users for assumptions.
- Severity Guess: Medium.
- Confidence: High.

## 6. Hypothesis: Candidate-set correctness is under-tested across long undo and error paths

- Area: Move legality, candidate sets, undo.
- Risk: A stale reference count in either `candidates` or `placement_candidates` could make a legal move disappear or an illegal move validate, and most callers rely on these sets rather than scanning `stones`.
- Why It Might Be Hard To Catch: The candidate maps are private. Public behavior tests cover common paths, but not exhaustive place/unplace interleavings, terminal undo, or failed `set_position` mutation paths.
- Evidence Observed: Normal move validation uses `placement_candidates.contains(cell)` instead of a scan (`crates/hexgame-core/src/board.rs:597`). Legal moves use cached sets for radius 2 and placement radius when available (`crates/hexgame-core/src/board.rs:635`). Undo decrements reference counts through `CandidateSet::on_unplace` (`crates/hexgame-core/src/board.rs:197`, `crates/hexgame-core/src/board.rs:406`). There is one focused brute-force comparison after a small populated game and two undos (`crates/hexgame-core/src/tests/board.rs:10`), plus property tests that mainly assert hash reset and move count reset (`crates/hexgame-core/tests/board.rs:569`).
- Phase 2 Verification: Add proptests that after each successful place and unplace compare `legal_moves_near_into(2)` and `legal_moves()` against a brute-force scan, including wins followed by undo. Add explicit checks after rejected `set_position` if transactional behavior is required.
- Severity Guess: Medium to High.
- Confidence: Medium.

## 7. Hypothesis: MCTS tree actions truncate `Hex` coordinates to `i16`

- Area: Board coordinate range, move legality consumers, hashes/history export.
- Risk: The board is modeled with `i32` infinite-grid coordinates, but MCTS stores child actions and re-root inputs as `i16`. A sufficiently long drifting game could wrap or misapply actions, corrupting state or returning an action different from the legal `Hex`.
- Why It Might Be Hard To Catch: Typical games may stay near the origin, and encoder windows also focus on local board regions. The conversion is silent with `as i16`.
- Evidence Observed: `Hex` uses `i32` coordinates (`crates/hexgame-core/src/core.rs:45`). `MCTSNode::new` stores action as `(i16, i16)` and expansion casts legal `Hex` coordinates with `as i16` (`crates/hexgame-core/src/mcts.rs:140`, `crates/hexgame-core/src/mcts.rs:2007`). `re_root` accepts `i16` and applies the cast-back coordinate to the game (`crates/hexgame-core/src/mcts.rs:1512`, `crates/hexgame-core/src/mcts.rs:1537`). Tree-node history extraction also casts `i32` cells to `i16` (`crates/hexgame-core/src/mcts.rs:1806`).
- Phase 2 Verification: Create a legal chain that reaches coordinates beyond `i16::MAX` or below `i16::MIN`, initialize/expand MCTS, and assert child actions round-trip to the original legal moves. If this is intentionally out of scope, document a hard coordinate bound and enforce it before truncation.
- Severity Guess: Medium.
- Confidence: High.

## 8. Direct issue: MCTS selection ignores illegal-placement errors during traversal

- Area: Move legality consumers, undo safety.
- Risk: If an expanded child action ever becomes illegal because of stale tree state, coordinate truncation, root misuse, or candidate corruption, `select_leaves` continues after a failed `place`. It still pushes the child onto `search_path`, so later unplace depth can diverge from actual applied placements.
- Why It Might Be Hard To Catch: Under ideal invariants, every child was generated from legal moves and remains legal. This only appears when another invariant is already violated, which makes the original cause harder to diagnose.
- Evidence Observed: `select_leaves` traverses expanded children and assigns the `Result` of `self.game.place(...)` to `_` (`crates/hexgame-core/src/mcts.rs:1140`, `crates/hexgame-core/src/mcts.rs:1142`). It then records the node in `search_path` regardless (`crates/hexgame-core/src/mcts.rs:1145`) and later unplaces once per path depth (`crates/hexgame-core/src/mcts.rs:1213`).
- Phase 2 Verification: Force an illegal child action in a small test-only hook or by constructing an out-of-range/truncated action, then verify whether `select_leaves` panics, corrupts state, or silently produces tensors for the wrong board. Preferred behavior may be to assert or mark the batch failed immediately.
- Severity Guess: High when triggered, low expected frequency.
- Confidence: Medium.

## 9. Question: Is `set_position` intended to be ordered-history loading or unordered-board loading?

- Area: Custom `set_position`, move legality, fixture ergonomics.
- Risk: The current API description says it sets pieces directly while bypassing normal turn rules, but implementation still requires first tuple at origin and each later tuple within placement radius of earlier tuples. That makes validity depend on tuple order, not only final board geometry.
- Why It Might Be Hard To Catch: Most current uses pass chronological `move_history` or hand-written ordered fixtures. A caller loading from a map/database may provide the same stones in a different order and get `MustPlaceAtOrigin` or `OutOfRadius`.
- Evidence Observed: `set_position` rejects a non-origin first tuple (`crates/hexgame-core/src/board.rs:463`) and checks radius against only stones already inserted (`crates/hexgame-core/src/board.rs:466`). The doc comment says pieces are placed directly regardless of current player (`crates/hexgame-core/src/board.rs:415`), while the implementation enforces order-sensitive placement geometry.
- Phase 2 Verification: Ask maintainers to classify the API. If it is a history loader, rename/document it and enforce player/turn chronology. If it is a board loader, pre-validate unordered connectivity or remove order-sensitive checks.
- Severity Guess: Medium.
- Confidence: High.

## 10. Structure recommendation: Separate authoritative rules from derived/cache policy

- Area: Board state, candidate sets, eval/hot-window caches, hashes.
- Risk: `HexGameState` owns authoritative stones and several derived caches (`eval`, radius-2 candidates, placement-radius candidates, zobrist). Each mutation path must update all of them in the right order. Future changes to one path can create subtle divergence.
- Why It Might Be Hard To Catch: Bugs can manifest far away: stale placement candidates become move-legality errors, stale hot windows become threat-filter errors, and board-only hashes can look correct while phase-sensitive callers disagree.
- Evidence Observed: `commit_placement` updates stones, hash, history, placements, both candidate sets, eval, win state, and turn phase in one function (`crates/hexgame-core/src/board.rs:535`). `unplace` separately reverses eval before popping the move record and then reverses stones, hash, candidates, and phase (`crates/hexgame-core/src/board.rs:397`). `set_position` has a third mutation path that manually duplicates most of the placement update sequence (`crates/hexgame-core/src/board.rs:484`).
- Phase 2 Verification: Consider a debug-only `assert_consistent()` on `HexGameState` that recomputes hash, placement candidates, near-2 candidates, winner from all stones, and possibly eval/hot windows within the known eval grid. Run it after every mutation in debug/property tests.
- Severity Guess: Medium.
- Confidence: High.

## 11. Structure recommendation: Make finite evaluation bounds explicit at core-rule boundaries

- Area: Win detection, threat detection, eval caches.
- Risk: Core win detection is sparse/infinite and should still work far from origin, but eval/hot-window threat logic intentionally clips to a finite `WIN_GRID_RADIUS`. Search and threat filters may become approximate in positions that are still legal under board rules.
- Why It Might Be Hard To Catch: The finite grid is documented as an accepted approximation, and most tests stay near origin. A user may reasonably expect all core tactical detection to match infinite-board rules because `HexGameState::find_winning_line` does.
- Evidence Observed: Board win detection scans actual `stones` along three axes (`crates/hexgame-core/src/board.rs:726`). Eval skips windows whose origins are outside `WIN_GRID_RADIUS` (`crates/hexgame-core/src/eval/state.rs:265`, `crates/hexgame-core/src/eval/grid.rs:88`). The grid docs state out-of-bounds windows are skipped as a known approximation (`crates/hexgame-core/src/eval/grid.rs:30`). Threat status relies on `game.eval().has_any_threats()` and hot windows (`crates/hexgame-core/src/threats.rs:195`, `crates/hexgame-core/src/threats.rs:213`).
- Phase 2 Verification: Construct legal far-from-origin four/five threats outside the eval grid and compare `find_winning_line` after completing the line, `threat_status` before completion, encoder hot channels, and search behavior. Decide whether to enforce a gameplay coordinate bound, widen/recenter eval, or document threat/search approximation separately from rules.
- Severity Guess: Medium.
- Confidence: High.

## 12. Direct issue: `PLACEMENT_RADIUS` documentation contradicts implemented rule

- Area: Move legality documentation.
- Risk: Contributors may implement callers or tests assuming placement is bounded by origin, while the actual rule is distance from any existing stone.
- Why It Might Be Hard To Catch: Code and board-level docs enforce the intended "near any tile" rule, so only readers of `core.rs` see the wrong description.
- Evidence Observed: `core.rs` documents `PLACEMENT_RADIUS` as "Maximum distance from the origin" (`crates/hexgame-core/src/core.rs:205`). `validate_move` checks membership in `placement_candidates`, which is built around every placed stone (`crates/hexgame-core/src/board.rs:597`, `crates/hexgame-core/src/board.rs:810`). Board module rules correctly say within radius of any previously placed tile (`crates/hexgame-core/src/board.rs:9`).
- Phase 2 Verification: Confirm intended public wording with maintainers, then update only the source docs in the implementation phase if assigned. No behavior test required beyond existing radius tests.
- Severity Guess: Low.
- Confidence: High.
