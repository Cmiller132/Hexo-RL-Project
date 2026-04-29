# S3 MCTS/Search Phase 1 Hypotheses

Scope reviewed: `crates/hexgame-core/src/mcts.rs`, `crates/hexgame-core/src/search.rs`, and `crates/hexgame-core/src/tests/mcts.rs`.

This is hypothesis formation only. Entries below are not final findings.

## 1. Direct issue - stale pending virtual loss can be dropped by a second selection

Area: MCTS pending leaves / virtual loss

Risk: Calling `select_leaves` twice before `expand_and_backprop` appears to discard the first pending batch without rolling back virtual loss, leaving inflated visits and depressed values on the prior search paths.

Why It Might Be Hard To Catch: The normal loop is strict `select_leaves` then `expand_and_backprop`; current tests cover this path and a reroot cleanup path, but not double selection after an interrupted or retried inference batch.

Evidence Observed: `select_leaves` clears `self.pending` directly at `crates/hexgame-core/src/mcts.rs:1121-1122`, while virtual loss is applied later at `crates/hexgame-core/src/mcts.rs:1155-1159`. The rollback helper exists in `clear_pending_leaves` at `crates/hexgame-core/src/mcts.rs:1481-1495`, but `select_leaves` does not use it. The reroot test checks cleanup via `re_root` at `crates/hexgame-core/src/tests/mcts.rs:167-189`, not repeated selection.

Phase 2 Verification: Add a test that records root/child visits and Q after one `select_leaves`, calls `select_leaves` again without backprop, then verifies the first batch's virtual visits/value loss were removed. Also test `get_results` before and after retry.

Severity Guess: High

Confidence: High

## 2. Hypothesis - failed backprop length validation leaves virtual loss in place if execution continues

Area: MCTS pending leaves / panic path

Risk: If an embedding layer or FFI boundary catches a panic from `expand_and_backprop` length validation and attempts to keep using the engine, pending leaves and virtual loss remain active.

Why It Might Be Hard To Catch: Rust unit tests treat the panic as test success and terminate the path; production wrappers may convert panics to errors or recover at a higher layer.

Evidence Observed: Length assertions run before pending leaves are swapped out or virtual loss is removed at `crates/hexgame-core/src/mcts.rs:1239-1269` and `crates/hexgame-core/src/mcts.rs:1362-1405`. The test at `crates/hexgame-core/src/tests/mcts.rs:120-137` confirms the panic, but does not verify engine state after a caught unwind.

Phase 2 Verification: Use `std::panic::catch_unwind` in a targeted test, then either call `clear_pending_leaves` or continue search and inspect visits/Q. Decide whether the public API should be panic-only or recoverable.

Severity Guess: Medium

Confidence: Medium

## 3. Hypothesis - traversal ignores illegal placement errors and can over-restore root state

Area: MCTS state restoration / stale tree safety

Risk: `select_leaves` ignores the result of applying child actions during traversal. If a reused/stale child ever becomes illegal, the engine can continue with the board at a shallower state and then unplace by the full logical depth.

Why It Might Be Hard To Catch: Normal tree construction should only create legal children. This would likely require stale subtree reuse, manual corruption, a threat-constrained reroot edge, or future changes to legality rules.

Evidence Observed: Traversal discards the `Result` from `game.place` at `crates/hexgame-core/src/mcts.rs:1140-1145`, then computes depth from `search_path.len()` and unplaces that many moves at `crates/hexgame-core/src/mcts.rs:1161-1216`.

Phase 2 Verification: Build a white-box test in the module that makes a root child illegal after expansion, or injects an illegal child action, then checks that `select_leaves` reports/handles it rather than corrupting `self.game`.

Severity Guess: Medium

Confidence: Medium

## 4. Direct issue - `extract_tree_node_states` error cleanup depth appears off by one

Area: MCTS training extraction / state restoration

Risk: On extraction error paths, the cleanup loop may unplace one move too many, corrupting the engine's root state.

Why It Might Be Hard To Catch: The affected paths are error paths: terminal expanded sampled nodes or illegal tree edges during extraction. Happy-path extraction can look fine.

Evidence Observed: `depth_from_root` starts at the node and increments before following the parent, so the root itself has depth 1 at `crates/hexgame-core/src/mcts.rs:1763-1772`. Error cleanup uses that value as the number of placed moves to undo at `crates/hexgame-core/src/mcts.rs:1781-1788` and `crates/hexgame-core/src/mcts.rs:1830-1839`. In the traversal model, only edges below the root are placed.

Phase 2 Verification: Add a test that triggers the illegal-child extraction path and verifies `move_history`, `current_player`, `placements_remaining`, and zobrist are unchanged after the error. Consider changing depth to edge count from current root.

Severity Guess: Medium

Confidence: High for the off-by-one shape; Low/Medium for reachable impact

## 5. Hypothesis - extraction candidate selection after reroot can be biased by unreachable old subtrees

Area: MCTS result extraction / rerooting

Risk: After rerooting, `extract_tree_node_states` chooses top candidates from the entire arena before traversal. High-visit nodes in dead sibling subtrees can occupy the 128 candidate cap, reducing or eliminating reachable samples from the current root.

Why It Might Be Hard To Catch: The traversal only emits reachable nodes, so the function still returns a consistent count; the issue would show up as missing training states after subtree reuse, not a crash.

Evidence Observed: Candidate collection scans `self.arena.iter()` globally at `crates/hexgame-core/src/mcts.rs:1724-1735` and truncates before traversal at `crates/hexgame-core/src/mcts.rs:1737-1743`. Actual traversal starts only from `self.root_idx` at `crates/hexgame-core/src/mcts.rs:1774-1778`. `re_root` leaves old sibling subtrees in the arena by design at `crates/hexgame-core/src/mcts.rs:1509-1511`.

Phase 2 Verification: Search a root, reroot into a lower-visit child, call extraction with a low `min_visits`, and compare emitted histories against a traversal-first reachable candidate list.

Severity Guess: Medium

Confidence: Medium

## 6. Direct issue - unordered pair priors are not canonicalized or deduplicated

Area: MCTS pair-prior handling

Risk: `apply_root_pair_priors` documents unordered pairs, but accepts both `(a,b)` and `(b,a)` as separate rows. Duplicate/reversed rows can double-count a pair in the softmax input and alter the marginal first-placement priors.

Why It Might Be Hard To Catch: Tests cover duplicate same-coordinate rejection and illegal coordinates, but not reversed duplicate legal rows. If upstream pair generation already deduplicates, this remains latent until a data pipeline changes.

Evidence Observed: Pair rows are pushed directly into `valid_pairs` with no canonicalization/deduplication at `crates/hexgame-core/src/mcts.rs:816-839`; mass is then normalized over rows at `crates/hexgame-core/src/mcts.rs:841-864`. The test at `crates/hexgame-core/src/tests/mcts.rs:421-435` checks self-pairs and illegal pairs only.

Phase 2 Verification: Add pair rows `[(a,b), (b,a)]` with asymmetric logits and verify whether intended behavior is reject, merge by logsumexp, or canonical first occurrence.

Severity Guess: Medium

Confidence: High

## 7. Hypothesis - pair-first telemetry may underreport candidate count

Area: MCTS pair-prior telemetry / training diagnostics

Risk: Applying `policy_pair_first` marks root sources as pair-derived but does not update `root_pair_candidate_count`, making telemetry inconsistent with other pair-prior paths.

Why It Might Be Hard To Catch: The pair-first helper has no direct test in `mcts.rs`; existing telemetry assertions cover joint pair priors and second-placement pair priors.

Evidence Observed: `apply_root_pair_first_priors` marks all sources as `PRIOR_SOURCE_PAIR` at `crates/hexgame-core/src/mcts.rs:946-952`, but never sets `self.root_pair_candidate_count` before returning at `crates/hexgame-core/src/mcts.rs:895-954`. Joint and second-placement paths set the count at `crates/hexgame-core/src/mcts.rs:886` and `crates/hexgame-core/src/mcts.rs:1069`.

Phase 2 Verification: Add a telemetry test for `apply_root_pair_first_priors` and decide whether `root_pair_candidate_count` should equal root child count, logits row count, or remain zero by design.

Severity Guess: Low

Confidence: Medium

## 8. Question - should Dirichlet noise be renormalized and finite-checked?

Area: MCTS priors / result quality

Risk: `add_dirichlet_noise` blends arbitrary noise values into priors without finite checks, non-negativity checks, or renormalization. If caller-supplied noise is malformed or not normalized, PUCT exploration scale changes.

Why It Might Be Hard To Catch: Correct Dirichlet callers produce valid normalized vectors, so tests with well-formed noise would pass. Effects are statistical rather than obviously illegal.

Evidence Observed: The blend is direct at `crates/hexgame-core/src/mcts.rs:1078-1093`. Other prior paths explicitly normalize softmax or mixed priors, e.g. `crates/hexgame-core/src/mcts.rs:866-876` and `crates/hexgame-core/src/mcts.rs:934-944`.

Phase 2 Verification: Confirm caller contract. If the engine owns robustness, test non-normalized and NaN noise and either reject or renormalize.

Severity Guess: Low/Medium

Confidence: Medium

## 9. Hypothesis - internal MCTS expansions intentionally bypass threat constraints, but terminal/value impact needs an oracle check

Area: MCTS terminal handling / legal move filtering

Risk: Internal MCTS nodes use unconstrained legal moves even when threats are active. If threat constraints encode mandatory blocks rather than just pruning, value estimates may include strategically illegal continuations until terminal detection catches them.

Why It Might Be Hard To Catch: The code comments say this is intentional for cost reasons, so failures would appear as quality regressions in tactical positions rather than invariant violations.

Evidence Observed: Root `init_root` passes `self.constrain_threats` to the encoder at `crates/hexgame-core/src/mcts.rs:636-642`, while leaf expansion always passes `false` at `crates/hexgame-core/src/mcts.rs:1187-1198`. Reroot clears an expanded child only when constraints are enabled and active threats exist at `crates/hexgame-core/src/mcts.rs:1544-1562`.

Phase 2 Verification: Construct positions with mandatory pair blocks and compare MCTS internal legal leaves against the threat oracle. Measure whether unconstrained internal moves create optimistic values or just broader exploration.

Severity Guess: Medium

Confidence: Low/Medium

## 10. Structure recommendation - add explicit state-restore invariants around MCTS public calls

Area: MCTS state restoration / test structure

Risk: Several public methods mutate `self.game` temporarily (`select_leaves`, extraction) or permanently (`re_root`). Tests currently verify behavioral outcomes, but not root-state invariants after every temporary traversal.

Why It Might Be Hard To Catch: State corruption can surface many calls later as illegal placements, wrong current player, or bad tensor histories. Zobrist/move-history invariants are cheap sentinels but are not used in MCTS tests.

Evidence Observed: `select_leaves` applies/unplaces path moves at `crates/hexgame-core/src/mcts.rs:1136-1216`; `extract_tree_node_states` performs iterative place/unplace traversal at `crates/hexgame-core/src/mcts.rs:1774-1847`. Current MCTS tests focus on determinism, visit counts, bounded values, sparse/pair priors, and pending metadata at `crates/hexgame-core/src/tests/mcts.rs:10-515`.

Phase 2 Verification: Add a helper that snapshots `move_history`, `current_player`, `placements_remaining`, `winner`, and `zobrist` before temporary MCTS calls and asserts equality after success and selected error paths.

Severity Guess: Medium

Confidence: High

## 11. Hypothesis - search move generation can miss mandatory blocking pairs due candidate/pair caps

Area: Alpha-beta search / pair generation / terminal handling

Risk: `search.rs` filters generated pairs by threat status, but the valid blocking pair must first survive candidate truncation and the pair-sum cap. A required pair outside those caps can be omitted, making the node look empty or forcing the root fallback to an arbitrary move.

Why It Might Be Hard To Catch: Tactical bonuses make threat cells likely to rank high, so many obvious cases pass. Edge cases with dispersed threats, history bias, or colony/candidate ordering may be needed.

Evidence Observed: Candidates are truncated at `ROOT_CANDIDATE_CAP` or `CANDIDATE_CAP` in `crates/hexgame-core/src/search.rs:365-393`; pairs are generated only when `i + j <= PAIR_SUM_CAP` at `crates/hexgame-core/src/search.rs:402-412`; threat filtering retains only generated satisfying turns at `crates/hexgame-core/src/search.rs:542-548` and `crates/hexgame-core/src/search.rs:580-585`. If root turns are empty, iterative deepening falls back to `c[0], c[1]` without checking the threat status at `crates/hexgame-core/src/search.rs:1120-1138`.

Phase 2 Verification: Use the threat oracle to generate positions where at least one valid blocking pair exists, then assert `generate_root_turns`/inner generation include a satisfying turn. Consider making mandatory threat pairs bypass normal caps.

Severity Guess: High

Confidence: Medium

## 12. Question - search terminal score perspective may need documentation/tests for win-on-first-placement of a two-placement turn

Area: Alpha-beta search / terminal handling / value signs

Risk: `make_turn` stops after the first placement if it wins, so a two-placement `Turn::pair` can effectively execute as one placement. Search scoring appears to handle this, but tests should pin the intended score/undo behavior.

Why It Might Be Hard To Catch: The behavior is correct for game rules if the game ends immediately, but easy to break during refactors because search is turn-based while the board is placement-based.

Evidence Observed: `make_turn` returns `(true, 1)` after a first-placement win at `crates/hexgame-core/src/search.rs:260-265`, and callers unmake only `placed` at `crates/hexgame-core/src/search.rs:276-280`, `crates/hexgame-core/src/search.rs:824-833`, and `crates/hexgame-core/src/search.rs:1025-1048`. MCTS terminal value similarly relies on board state after placement at `crates/hexgame-core/src/mcts.rs:1164-1181`.

Phase 2 Verification: Add alpha-beta tests for a pair turn where the first cell wins and the second would also be legal, asserting score, best turn, and exact state restoration.

Severity Guess: Medium

Confidence: Medium

