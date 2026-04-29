# S2 Encoding / Python FFI Phase 1 Hypotheses

Scope: tensor encoding and Python FFI surfaces in `crates/hexgame-core/src/encoder.rs`, `crates/hexgame-py/src/encode.rs`, and `crates/hexgame-py/src/engine.rs`, with supporting references to `crates/hexgame-core/src/mcts.rs` where the FFI forwards data into the tree. These are Phase 1 hypotheses and questions, not final findings.

## Direct Issue: Root expansion accepts stale or mutated `legal_bytes` and offsets

- **Area:** Python MCTS root expansion FFI.
- **Risk:** Python can pass `legal_bytes`, `offset_q`, or `offset_r` that do not correspond to the current root tensor. Dense and sparse root expansion will then create children for caller-supplied coordinates and/or gather logits from the wrong tensor cells.
- **Why It Might Be Hard To Catch:** The failure can look like bad policy quality instead of an exception. Length validation succeeds as long as `legal_bytes.len() % 8 == 0`, and the dense path has no root-state cross-check. Mutations are easy in Python because byte buffers can be cached across games or across re-rooted positions.
- **Evidence Observed:** `PyMCTSEngine.init_root` returns `(tensor, oq, or_, legal_bytes)` from Rust (`crates/hexgame-py/src/engine.rs:744`). `expand_root` validates policy length and byte length only, then decodes arbitrary `Hex` values and forwards caller-provided offsets (`crates/hexgame-py/src/engine.rs:766`, `crates/hexgame-py/src/engine.rs:784`, `crates/hexgame-py/src/engine.rs:790`, `crates/hexgame-py/src/engine.rs:796`). Core `MCTSEngine::expand_root` directly expands using the supplied `legal` and offsets (`crates/hexgame-core/src/mcts.rs:658`). `gather_policy` maps each supplied move through those offsets and falls back to `-10.0` for out-of-window moves (`crates/hexgame-core/src/mcts.rs:268`, `crates/hexgame-core/src/mcts.rs:280`). The global-prior path validates `legal` against `global_actions`, but not against current root state (`crates/hexgame-core/src/mcts.rs:722`, `crates/hexgame-core/src/mcts.rs:744`).
- **Phase 2 Verification:** Add adversarial Python/Rust tests that call `init_root`, then alter one legal row, shuffle rows, use stale offsets, or reuse `legal_bytes` from another game before `expand_root` and `expand_root_with_sparse_priors`. Verify whether the engine accepts illegal/stale children and whether `sample_action` can return a coordinate not legal in the root game.
- **Severity Guess:** High.
- **Confidence:** High for the missing validation; medium on observed user-facing impact until tested.

## Hypothesis: Root sparse expansion is less validated than global-prior expansion

- **Area:** `expand_root_with_sparse_priors` FFI and sparse policy mixing.
- **Risk:** Sparse root expansion can blend priors using arbitrary `legal_bytes`, sparse coordinates, and sparse logits with only shape/length checks. This may silently accept sparse rows for a different position, unlike `expand_root_with_global_priors`, which checks row-for-row equality between legal rows and graph rows.
- **Why It Might Be Hard To Catch:** The mixed policy still normalizes, and unmatched sparse rows merely fall back to dense/default sources depending on stage. Telemetry may report dense/default/sparse source changes, but gameplay degradation would be statistical.
- **Evidence Observed:** `expand_root_with_sparse_priors` checks `policy.len() == BOARD_AREA`, `legal_bytes.len() % 8 == 0`, sparse shape `(N, 2)`, and `sparse_logits.len() >= rows`, then forwards all data (`crates/hexgame-py/src/engine.rs:802`, `crates/hexgame-py/src/engine.rs:814`, `crates/hexgame-py/src/engine.rs:824`, `crates/hexgame-py/src/engine.rs:830`, `crates/hexgame-py/src/engine.rs:834`, `crates/hexgame-py/src/engine.rs:844`, `crates/hexgame-py/src/engine.rs:851`). Core sparse gathering matches sparse rows by coordinate, with no legality or freshness validation beyond matching coordinates to supplied `moves` (`crates/hexgame-core/src/mcts.rs:340`, `crates/hexgame-core/src/mcts.rs:365`, `crates/hexgame-core/src/mcts.rs:395`). In contrast, global priors require `legal.len() == global_actions.len()` and row equality (`crates/hexgame-core/src/mcts.rs:730`, `crates/hexgame-core/src/mcts.rs:744`).
- **Phase 2 Verification:** Compare root sparse and global-prior behavior with shuffled, truncated, duplicated, and stale rows. Confirm whether sparse rows outside root legal moves are ignored, incorrectly mixed, or detectable through prior-source telemetry.
- **Severity Guess:** Medium-High.
- **Confidence:** Medium.

## Hypothesis: Dense policy logits can silently degrade to fallback priors on non-finite input

- **Area:** Dense policy array validation and normalization.
- **Risk:** `NaN`/`Inf` policy logits from Python may not be rejected on dense paths. Depending on normalization behavior, priors can become uniform or otherwise lose signal without surfacing an error.
- **Why It Might Be Hard To Catch:** Value arrays are explicitly checked for finite values during backprop, but dense policy arrays are only checked for length and contiguity at the FFI boundary. A training/inference bug could be masked as weaker search.
- **Evidence Observed:** `expand_root` and `expand_and_backprop` validate policy length but not finite policy contents (`crates/hexgame-py/src/engine.rs:774`, `crates/hexgame-py/src/engine.rs:777`, `crates/hexgame-py/src/engine.rs:1053`, `crates/hexgame-py/src/engine.rs:1060`). Core `expand_and_backprop` checks `v.is_finite()` for values only (`crates/hexgame-core/src/mcts.rs:1279`). `gather_policy` exponentiates gathered logits and falls back to uniform only when `sum > 0.0` is false (`crates/hexgame-core/src/mcts.rs:289`). The global-prior path is stricter and rejects non-finite logits (`crates/hexgame-core/src/mcts.rs:755`).
- **Phase 2 Verification:** Feed `NaN`, `+Inf`, and `-Inf` dense policies through root and leaf expansion from Python. Inspect root child priors, leaf priors, and whether any panic or error occurs. Decide whether dense policy finite validation should match global-prior validation.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Hypothesis: `last_non_terminal_count` can hide stale batch-result sequencing mistakes

- **Area:** Batched leaf selection and Python result submission.
- **Risk:** The Python wrapper uses `last_non_terminal_count` to validate result lengths, while core uses the current `pending` leaves. A caller that invokes `select_leaves` twice before submitting results will clear the original pending leaves and make the first batch's tensors stale. Length checks may still pass for the second batch if counts happen to match.
- **Why It Might Be Hard To Catch:** Normal worker code calls `select_leaves` then submits immediately. Misordered async or retry code could corrupt search quality without a structural type distinction between batch handles.
- **Evidence Observed:** Core `select_leaves` clears previous pending state at entry (`crates/hexgame-core/src/mcts.rs:1117`, `crates/hexgame-core/src/mcts.rs:1121`). Python stores only a count after selection (`crates/hexgame-py/src/engine.rs:1000`, `crates/hexgame-py/src/engine.rs:1009`). `expand_and_backprop` validates against `last_non_terminal_count`, copies arrays, then core consumes whatever is currently in `self.pending` (`crates/hexgame-py/src/engine.rs:1059`, `crates/hexgame-py/src/engine.rs:1075`, `crates/hexgame-core/src/mcts.rs:1239`, `crates/hexgame-core/src/mcts.rs:1240`).
- **Phase 2 Verification:** Add a misuse test: `select_leaves(A)`, retain tensors, `select_leaves(B)`, then submit results for A. Check whether counts can align and whether the tree expands the wrong leaves. Consider generation IDs or explicit pending-state errors.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Hypothesis: `pending_leaf_metadata` correctly copies bytes, but metadata has no batch identity

- **Area:** Legal/history bytes for sparse and graph leaf priors.
- **Risk:** The metadata buffers themselves are immutable Python `bytes`, but there is no batch identifier tying returned `(offset, legal_bytes, history_bytes)` to the exact tensor batch used for inference. Stale metadata can be mixed with a later batch if Python caches it or if retry logic is added.
- **Why It Might Be Hard To Catch:** The current self-play path immediately consumes metadata and converts `PyBytes` to owned Python `bytes`, so standard smoke tests will pass. Future asynchronous inference code could reorder metadata independently from tensors.
- **Evidence Observed:** Core `PendingLeaf` stores `offset_q`, `offset_r`, `legal_moves`, and packed `move_history` (`crates/hexgame-core/src/mcts.rs:180`). `pending_leaf_metadata` clones this data (`crates/hexgame-core/src/mcts.rs:1466`). Python converts each leaf's legal moves and history into fresh `PyBytes` (`crates/hexgame-py/src/engine.rs:1024`, `crates/hexgame-py/src/engine.rs:1031`, `crates/hexgame-py/src/engine.rs:1040`). `select_leaves` clears `pending` on the next call (`crates/hexgame-core/src/mcts.rs:1121`).
- **Phase 2 Verification:** Review the intended concurrency model for inference. If async/multi-request inference is allowed, test metadata/result reordering and add a monotonically increasing batch token to `select_leaves`, `pending_leaf_metadata`, and `expand_and_backprop`.
- **Severity Guess:** Medium.
- **Confidence:** Medium.

## Question: Should internal MCTS leaf expansion use threat-constrained legal moves?

- **Area:** Encoder legal channel and internal MCTS leaf legal sets.
- **Risk:** Root encoding can use threat-constrained legal moves, while internal leaf encoding explicitly disables threat constraints. If policy targets or graph metadata assume channel 3 always represents tactically constrained legal moves, internal batched inference may see broader legal masks than root inference.
- **Why It Might Be Hard To Catch:** This may be an intentional performance tradeoff. The difference is positional and tactical, so it may only appear in forced-block or winning-turn positions inside the tree.
- **Evidence Observed:** `encode_board_into` can constrain channel 3 and returned `legal_out` via `threat_status` (`crates/hexgame-core/src/encoder.rs:250`, `crates/hexgame-core/src/encoder.rs:251`, `crates/hexgame-core/src/encoder.rs:282`). `PyMCTSEngine` defaults `constrain_threats=true` at construction (`crates/hexgame-py/src/engine.rs:715`). Root init passes `self.constrain_threats` (`crates/hexgame-core/src/mcts.rs:639`, `crates/hexgame-core/src/mcts.rs:642`). Internal leaf selection passes `false` with a comment saying the O(n^2) unblockable check is too expensive (`crates/hexgame-core/src/mcts.rs:1187`, `crates/hexgame-core/src/mcts.rs:1191`, `crates/hexgame-core/src/mcts.rs:1194`).
- **Phase 2 Verification:** Confirm desired semantics with search owners. Construct a forced-threat position reachable after one tree step and compare root `legal_bytes`/channel 3 with leaf metadata/channel 3 for the same board.
- **Severity Guess:** Medium if unintended; Low if documented design.
- **Confidence:** High that behavior differs; low that it is a defect.

## Hypothesis: `encode_compact_record` is safe against buffer mutation but narrowly validates replay semantics

- **Area:** Compact history bytes replay into tensors.
- **Risk:** `encode_compact_record` copies input bytes before releasing the GIL, so stale/mutated Python buffer risk is low. The remaining risk is semantic: it only validates byte length, current-player match, and `game.place`; it cannot represent custom positions made through `set_position`, alternate `placements_remaining`, or histories with missing prefix context.
- **Why It Might Be Hard To Catch:** Most histories originate from Rust `move_history_bytes`, so replay works. Failures are likely around dashboard/debug/custom-state workflows rather than normal self-play.
- **Evidence Observed:** The FFI copies `history_bytes` into `bytes_owned` before `allow_threads` (`crates/hexgame-py/src/encode.rs:21`, `crates/hexgame-py/src/encode.rs:30`, `crates/hexgame-py/src/encode.rs:32`). Replay checks the stored player against `game.current_player()` and then calls validated `place` (`crates/hexgame-py/src/encode.rs:39`, `crates/hexgame-py/src/encode.rs:42`, `crates/hexgame-py/src/encode.rs:51`). The packed format is only `(player, q, r)` (`crates/hexgame-py/src/encode.rs:9`), while `MoveRecord` has additional pre-move turn fields internally (`crates/hexgame-core/src/board.rs:117`). Python `set_position` can create a board with explicit `current_player` and `placements_remaining` but no move history (`crates/hexgame-py/src/engine.rs:636`, `crates/hexgame-py/src/engine.rs:647`).
- **Phase 2 Verification:** Encode tensors from a game created via `set_position`, from its `move_history_bytes`, and from direct `encode_board_and_legal`; verify whether consumers ever expect equivalence. Test malformed but legal-looking histories that start mid-game.
- **Severity Guess:** Low-Medium.
- **Confidence:** Medium.

## Structure Recommendation: Centralize packed byte parsing and validation

- **Area:** Legal/history byte FFI utilities.
- **Risk:** Legal move byte parsing is repeated in several methods. Inconsistent hardening is already visible: root/global/sparse paths all parse bytes separately, and only some downstream paths validate row identity or non-finite logits.
- **Why It Might Be Hard To Catch:** Small validation fixes can land in one FFI method and miss the others. The duplication is not large, but it sits on a boundary that accepts untyped Python data.
- **Evidence Observed:** `legal_bytes` is decoded independently in `expand_root`, `expand_root_with_sparse_priors`, and `expand_root_with_global_priors` (`crates/hexgame-py/src/engine.rs:790`, `crates/hexgame-py/src/engine.rs:844`, `crates/hexgame-py/src/engine.rs:892`). Legal bytes are encoded independently in `legal_moves_near_bytes`, `encode_board_and_legal`, `init_root`, and `pending_leaf_metadata` (`crates/hexgame-py/src/engine.rs:426`, `crates/hexgame-py/src/engine.rs:512`, `crates/hexgame-py/src/engine.rs:758`, `crates/hexgame-py/src/engine.rs:1032`). History packing is duplicated between `move_history_bytes` and core `pack_move_history` (`crates/hexgame-py/src/engine.rs:453`, `crates/hexgame-core/src/mcts.rs:191`).
- **Phase 2 Verification:** Prototype local helpers such as `decode_legal_bytes`, `encode_hex_pairs`, and `encode_move_history`, then audit every caller for consistent validation: length, finite logits, duplicate legal rows, current-root consistency, and optional sortedness expectations.
- **Severity Guess:** Medium as maintainability/risk reduction.
- **Confidence:** High.

## Structure Recommendation: Add explicit stale-data guards for root and batch lifecycles

- **Area:** MCTS Python API design.
- **Risk:** The current API relies on Python preserving tuple coherence: tensor, offsets, legal bytes, metadata, policies, and values must all belong to the same root or leaf batch. Rust can cheaply make this contract explicit with generation IDs.
- **Why It Might Be Hard To Catch:** Current smoke tests follow the happy path. Bugs from cached bytes or async inference reordering would be rare and hard to minimize.
- **Evidence Observed:** `init_root` returns four independent Python objects (`crates/hexgame-py/src/engine.rs:745`). `select_leaves` returns tensor batch and count, while `pending_leaf_metadata` is fetched by a separate call (`crates/hexgame-py/src/engine.rs:1000`, `crates/hexgame-py/src/engine.rs:1024`). `expand_and_backprop*` accepts arrays without a batch token (`crates/hexgame-py/src/engine.rs:1047`, `crates/hexgame-py/src/engine.rs:1085`, `crates/hexgame-py/src/engine.rs:1165`).
- **Phase 2 Verification:** Evaluate adding `root_generation` and `batch_generation` values returned from Rust and required on expansion/submission calls. Tests should prove stale root `legal_bytes` and stale leaf metadata are rejected with `ValueError`.
- **Severity Guess:** Medium.
- **Confidence:** Medium-High.

## Question: Are unsorted legal rows intentional for tensors and leaf metadata?

- **Area:** Legal move ordering across encoder, Python bytes, and graph/sparse consumers.
- **Risk:** `legal_moves_near_bytes` exports sorted moves for Python, but encoder/MCTS legal buffers use `legal_moves_near_into`, which may return hash-map order from candidate caches. If Python graph/sparse components assume stable order beyond row equality checks, behavior can vary across runs or platforms.
- **Why It Might Be Hard To Catch:** Many consumers treat legal rows as sets or check equality against their own rows. Ordering issues appear only when logits are aligned by row rather than coordinate.
- **Evidence Observed:** `legal_moves_near_sorted` sorts only for tests/Python export (`crates/hexgame-core/src/board.rs:680`). `legal_moves_near_into` extends from hash-map keys when using candidate caches (`crates/hexgame-core/src/board.rs:625`, `crates/hexgame-core/src/board.rs:636`). `encode_board_into` uses `legal_moves_near_into` directly (`crates/hexgame-core/src/encoder.rs:250`). Root and leaf metadata then return the unsorted legal buffer from the encoder/MCTS path (`crates/hexgame-core/src/mcts.rs:648`, `crates/hexgame-core/src/mcts.rs:1208`, `crates/hexgame-py/src/engine.rs:758`, `crates/hexgame-py/src/engine.rs:1031`). The global-prior core path does require row equality, which partly mitigates graph row-order drift (`crates/hexgame-core/src/mcts.rs:744`).
- **Phase 2 Verification:** Determine whether row order is part of any Python contract. Run deterministic tests across repeated games/processes for `init_root` and `pending_leaf_metadata`; if order is not guaranteed, document coordinate-keyed semantics or sort at the FFI boundary.
- **Severity Guess:** Low-Medium.
- **Confidence:** Medium.

## Hypothesis: D6 tensor symmetry may not preserve non-spatial channel semantics

- **Area:** `apply_d6_symmetry` augmentation.
- **Risk:** The transform applies spatial coordinate mapping to every channel, including constant planes and centroid-distance/phase/color channels. Constants survive, but any future non-spatial or orientation-sensitive channel would be transformed silently. Existing channel 11 is precomputed relative to tensor center, so it probably survives D6 symmetry, but the function encodes no channel-level policy.
- **Why It Might Be Hard To Catch:** Current tests may verify shape or a subset of channels. A future channel addition could break augmentation without a compile-time signal.
- **Evidence Observed:** `apply_d6_symmetry` copies input to an owned vec, validates shape, then applies the same spatial transform for every channel `0..ch` (`crates/hexgame-py/src/encode.rs:81`, `crates/hexgame-py/src/encode.rs:89`, `crates/hexgame-py/src/encode.rs:96`, `crates/hexgame-py/src/encode.rs:101`). Encoder channel documentation includes global constant planes and a distance-from-centroid plane (`crates/hexgame-core/src/encoder.rs:155`, `crates/hexgame-core/src/encoder.rs:157`, `crates/hexgame-core/src/encoder.rs:162`).
- **Phase 2 Verification:** Add channel-invariant tests for phase/color planes and distance channel under all 12 symmetries. Add a comment or helper table documenting which channels are spatial masks versus scalar planes.
- **Severity Guess:** Low.
- **Confidence:** Medium.
