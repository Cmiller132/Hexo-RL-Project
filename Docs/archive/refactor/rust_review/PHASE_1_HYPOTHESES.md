# Rust Review Phase 1 Hypotheses

This is a consolidated hypothesis repository. Detailed evidence is in `subagents/`.

## Direct Issues

### D1 - `set_position` Is Not Transactional On Validation Failure

- **Area:** `crates/hexgame-core/src/board.rs`, `HexGameState::set_position`.
- **Risk:** A failed load can reset the previous game and leave a partially loaded replacement state.
- **Why It Might Be Hard To Catch:** Existing tests check returned errors but not state preservation after an error.
- **Evidence Observed:** `set_position` calls `reset()` before validating all supplied stones, then mutates as it iterates.
- **Phase 2 Verification:** Start from a non-empty game, call `set_position` with an invalid later tuple, and assert all public state, hash, history, legal moves, and eval-derived data are unchanged.
- **Severity Guess:** High.
- **Confidence:** High.

### D2 - MCTS Can Drop Pending Virtual Loss On Repeated Selection

- **Area:** `crates/hexgame-core/src/mcts.rs`, `select_leaves`.
- **Risk:** Calling `select_leaves` twice before `expand_and_backprop` can clear pending leaves without rolling back virtual loss.
- **Why It Might Be Hard To Catch:** Normal control flow is strict select-then-backprop; retry/interruption paths are not covered.
- **Evidence Observed:** `select_leaves` clears `self.pending` directly, while a rollback helper exists separately for pending cleanup.
- **Phase 2 Verification:** Record visits/Q after one selection, call `select_leaves` again without backprop, and assert prior virtual visits/value loss were removed.
- **Severity Guess:** High.
- **Confidence:** High.

### D3 - Root Expansion Accepts Stale Or Mutated `legal_bytes` And Offsets

- **Area:** `crates/hexgame-py/src/engine.rs`, dense/sparse MCTS root expansion.
- **Risk:** Python can pass legal rows or offsets from another root, causing children and priors to refer to the wrong state.
- **Why It Might Be Hard To Catch:** The engine still normalizes priors and returns plausible search results; symptoms look like weak model behavior.
- **Evidence Observed:** Dense/sparse root expansion validates length/shape but does not prove rows/offsets match the current root.
- **Phase 2 Verification:** Mutate/shuffle/reuse `legal_bytes` and offsets after `init_root`; assert stale rows are rejected before expansion.
- **Severity Guess:** High.
- **Confidence:** High.

### D4 - Python CI Appears To Request A Nonexistent Cargo Feature

- **Area:** `.github/workflows/ci.yml`, `crates/hexgame-py/Cargo.toml`.
- **Risk:** Python integration CI may fail before smoke tests because `maturin develop --features python` names a feature that is not declared.
- **Why It Might Be Hard To Catch:** Rust-only CI and local extension build commands can still work.
- **Evidence Observed:** CI uses `--features python`; inspected Cargo manifests do not define `[features] python`.
- **Phase 2 Verification:** Run the Python integration workflow command or remove/add the intended feature.
- **Severity Guess:** High.
- **Confidence:** High.

### D5 - Unordered Pair Priors Are Not Canonicalized Or Deduplicated

- **Area:** `crates/hexgame-core/src/mcts.rs`, `apply_root_pair_priors`.
- **Risk:** `(a,b)` and `(b,a)` can be accepted as separate rows even though the API documents unordered pairs, double-counting pair mass.
- **Why It Might Be Hard To Catch:** Upstream may usually deduplicate; existing tests cover self-pairs and illegal coordinates, not reversed duplicate legal pairs.
- **Evidence Observed:** Pair rows are pushed to `valid_pairs` without canonicalization/deduplication.
- **Phase 2 Verification:** Submit reversed duplicates with asymmetric logits and define expected reject/merge behavior.
- **Severity Guess:** Medium.
- **Confidence:** High.

### D6 - `PLACEMENT_RADIUS` Documentation Contradicts The Implemented Rule

- **Area:** `crates/hexgame-core/src/core.rs`.
- **Risk:** Contributors may think placement radius is origin-bounded; implementation is near any existing stone.
- **Why It Might Be Hard To Catch:** Behavior tests pass; the bug is in public documentation.
- **Evidence Observed:** Constant doc says distance from origin, while `validate_move` uses placement candidates around existing stones.
- **Phase 2 Verification:** No behavioral verification needed beyond confirming intended wording; fix rustdoc in implementation phase.
- **Severity Guess:** Low.
- **Confidence:** High.

## High-Value Hypotheses

### H1 - `set_position` May Load Impossible Terminal Or Ambiguous Histories

- **Area:** custom position loading, move history, terminal semantics.
- **Risk:** Synthetic positions can contain post-terminal stones or same-board/different-order winner outcomes; history may not represent real gameplay.
- **Why It Might Be Hard To Catch:** `set_position` is convenient for fixtures, and most tests do not undo through arbitrary custom histories.
- **Phase 2 Verification:** Use same final stones in different input orders, post-terminal extra stones, and non-chronological players. Decide whether `set_position` is a board loader or history loader.
- **Severity Guess:** Medium to High.
- **Confidence:** Medium.

### H2 - Candidate Sets May Drift Across Long Undo/Error Paths

- **Area:** move legality caches.
- **Risk:** Stale reference counts can hide legal moves or allow illegal moves.
- **Why It Might Be Hard To Catch:** Candidate maps are private and most tests exercise short common paths.
- **Phase 2 Verification:** Proptest place/unplace sequences and compare cached legal sets to brute-force scans after every operation.
- **Severity Guess:** Medium to High.
- **Confidence:** Medium.

### H3 - MCTS Action Coordinates Truncate `i32` To `i16`

- **Area:** MCTS node actions and histories.
- **Risk:** Far-coordinate legal moves can wrap or export incorrect actions.
- **Why It Might Be Hard To Catch:** Normal games stay near origin; casts are silent.
- **Phase 2 Verification:** Create legal chains past `i16` bounds, expand MCTS, and assert actions/history round-trip.
- **Severity Guess:** Medium.
- **Confidence:** High.

### H4 - Dense Policy Paths May Not Reject Non-Finite Logits

- **Area:** dense root/leaf policy normalization.
- **Risk:** `NaN`/`Inf` model outputs can become degraded priors instead of explicit errors.
- **Why It Might Be Hard To Catch:** Search still runs; quality degrades statistically.
- **Phase 2 Verification:** Feed `NaN`, `+Inf`, and `-Inf` policies through root and leaf expansion and inspect errors/priors.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

### H5 - MCTS Batch Identity Is Implicit

- **Area:** leaf selection, metadata, backprop.
- **Risk:** Stale tensors/metadata from one batch can be submitted for another batch if async/retry code is introduced.
- **Why It Might Be Hard To Catch:** Happy-path workers submit immediately; misuse looks like search noise.
- **Phase 2 Verification:** Call `select_leaves` twice and submit stale tensors/metadata; consider root/batch generation IDs.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

### H6 - Finite Eval Grid Can Produce False Tactical Quiet States

- **Area:** `eval/grid.rs`, `EvalState`, `threat_status`, encoder hot channels.
- **Risk:** Tactical threats outside `WIN_GRID_RADIUS` can be missed while board win detection remains exact.
- **Why It Might Be Hard To Catch:** The approximation is documented and typical tests stay near origin.
- **Phase 2 Verification:** Build far-from-origin 4/5/6-in-row fixtures and compare incremental eval/threat status with a full-board scanner.
- **Severity Guess:** High if tactical legality depends on it; Medium if accepted approximation.
- **Confidence:** High that the blind spot exists by design.

### H7 - Oracle Is Not Fully Independent Of Production Candidate/Mutation State

- **Area:** threat oracle tests.
- **Risk:** Oracle and fast path can share candidate or place/unplace bugs.
- **Why It Might Be Hard To Catch:** The oracle is independent of `live_cells`, but still uses board candidate and mutation APIs.
- **Phase 2 Verification:** Add a radius-3 tactical scanner that does not use `CandidateSet` or production tactical helpers. Limit winning/blocking analysis to 4+ windows because two placements cannot convert a 3-window into a current-turn win. Radius 3 is the conservative bound until radius 2 has a formal proof.
- **Severity Guess:** Medium.
- **Confidence:** High on shared dependencies.

### H8 - Multiple Winning Turns Collapse To One Tactical Answer

- **Area:** `ThreatStatus::WinningTurn`, constrained legal masks.
- **Risk:** A search helper that returns one sufficient winning turn may be misused as a complete tactical legality mask.
- **Why It Might Be Hard To Catch:** Current oracle tests intentionally allow one reported win.
- **Phase 2 Verification:** Decide whether threat status is complete or sufficient-only; add multi-win mask fixtures.
- **Severity Guess:** Medium.
- **Confidence:** High.

### H9 - Search Pair Generation Can Miss Mandatory Blocks Due Candidate/Pair Caps

- **Area:** alpha-beta search candidate caps and pair caps.
- **Risk:** Required blocking pairs can be omitted before threat filtering, leading to empty/fallback moves.
- **Why It Might Be Hard To Catch:** Tactical ranking often saves obvious cases.
- **Phase 2 Verification:** Use radius-3 oracle-generated mandatory win/block positions from 4+ windows and assert generated root/inner turns include at least one satisfying turn.
- **Severity Guess:** High.
- **Confidence:** Medium.

### H10 - Hot-Window And Threat-Count Drift May Be Release-Silent

- **Area:** incremental eval/hot windows.
- **Risk:** Debug-only checks may catch duplicate/stale hot windows, but release search can continue with distorted tactical data.
- **Why It Might Be Hard To Catch:** Full recompute invariant is debug-only and does not validate all count fields.
- **Phase 2 Verification:** Release-mode differential fuzzer comparing incremental counts/hot windows to full recompute after every place/unplace.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Questions To Resolve Before Phase 2 Implementation

- Is `set_position` a fixture board loader, a chronological history loader, or both with separate APIs?
- Is `ThreatStatus` intended to expose a complete set of tactical legal moves or one sufficient search action?
- Should internal MCTS leaf expansion use threat-constrained legal rows or document the root-only constraint?
- Are unsorted legal rows a stable coordinate-keyed contract, or should FFI sort/canonicalize them?
- Are far-coordinate games supported by all engine components, or should the engine enforce a gameplay coordinate bound?
- Should Phase 2 prove the suspected radius-2 tactical bound mathematically, or keep radius 3 as the documented conservative oracle radius?
