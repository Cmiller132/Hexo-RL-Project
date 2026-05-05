# S5 Structure, Tests, Performance Hypotheses

Scope: Rust project structure, crate boundaries, public APIs, tests, benchmarks, panic/unwrap policy, documentation/invariants, and performance transparency. This is Phase 1 hypothesis formation only.

## 1. Structure recommendation: split public API from internal modules

- Area: Public API / crate boundaries
- Risk: `hexgame-core` exports every implementation module as public (`board`, `core`, `encoder`, `eval`, `mcts`, `search`, `threats`) while also providing a smaller re-export surface. This makes internal organization harder to change without downstream breakage.
- Why It Might Be Hard To Catch: Rust will happily compile downstream callers against deep paths such as `hexgame_core::eval::state::EvalState`; later refactors become semver/API problems rather than local cleanup.
- Evidence Observed: [crates/hexgame-core/src/lib.rs](../../../../crates/hexgame-core/src/lib.rs:58) publishes all modules; [crates/hexgame-core/src/lib.rs](../../../../crates/hexgame-core/src/lib.rs:70) then re-exports only selected types as the convenient API.
- Phase 2 Verification: Run `cargo public-api` or inspect downstream Python/Rust uses to identify which deep module paths are actually required. Decide whether modules can become `pub(crate)`/private with explicit re-exports.
- Severity Guess: Medium
- Confidence: High

## 2. Structure recommendation: isolate neural MCTS protocol behind a safer facade

- Area: Public API / Python boundary / MCTS protocol
- Risk: `MCTSEngine` exposes a multi-step protocol (`init_root`, `expand_root`, `select_leaves`, `expand_and_backprop`, `re_root`) where call ordering and slice dimensions are enforced by panics/asserts in core rather than typed states or fallible APIs.
- Why It Might Be Hard To Catch: The Python wrapper validates many cases, but Rust callers can still call methods in the wrong order or with wrong-sized buffers. In release, this workspace uses `panic = "abort"`, so an assertion failure can kill a long-running training process.
- Evidence Observed: Release profile sets aborting panics in [Cargo.toml](../../../../Cargo.toml:23) and [Cargo.toml](../../../../Cargo.toml:28). Public protocol methods are exposed at [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:631), [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:658), [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:1117), and [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:1239). Shape checks use `assert!` at [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:1241), [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:1249), [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:1373), and nearby sparse-path checks.
- Phase 2 Verification: Build a small misuse matrix for direct Rust and Python calls. Consider returning `Result` for all externally reachable invalid-input cases and reserving `assert!` for impossible internal invariants.
- Severity Guess: High
- Confidence: High

## 3. Hypothesis: public `set_position` can create histories that are valid fixtures but ambiguous game records

- Area: Board invariants / public API / documentation
- Risk: `HexGameState::set_position` accepts arbitrary `(q, r, player)` tuples plus explicit player/remaining values. It validates duplicate cells, player IDs, opening origin, radius, and winner state, but it does not require the supplied stone ownership sequence to match normal alternating Hexo turns. The resulting `move_history` is therefore a synthetic fixture history, not necessarily a legal chronological game.
- Why It Might Be Hard To Catch: Most tests intentionally use `set_position` to construct tactical states quickly, so the ambiguity is useful. Bugs would appear only when later code treats history order as real gameplay, for example encoder recency channels, `opponent_last_turn_cells`, or tree extraction histories.
- Evidence Observed: Public setter begins at [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:437). The encoder reads `game.move_history()` for recency/turn features at [crates/hexgame-core/src/encoder.rs](../../../../crates/hexgame-core/src/encoder.rs:231) and [crates/hexgame-core/src/encoder.rs](../../../../crates/hexgame-core/src/encoder.rs:307). Board history access is public at [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:333).
- Phase 2 Verification: Add tests that construct intentionally non-chronological `set_position` states and assert expected encoder/history behavior, or document `set_position` as fixture-only/synthetic. Consider a separate `set_fixture_position` naming/API if external callers use it.
- Severity Guess: Medium
- Confidence: Medium

## 4. Direct issue: Python CI appears to request a nonexistent Cargo feature

- Area: CI / crate metadata
- Risk: Python integration CI may fail before smoke tests because it runs `maturin develop --features python`, but no `[features]` section defining `python` was observed in the root or crate manifests.
- Why It Might Be Hard To Catch: Local developers may build the extension with a different command, and the Rust-only CI job would still pass.
- Evidence Observed: CI command is in [.github/workflows/ci.yml](../../../../.github/workflows/ci.yml:68). Workspace/crate manifests list members and dependencies but no `[features]`; observed feature strings are dependency features only in [Cargo.toml](../../../../Cargo.toml:14), while `hexgame-py` declares only its `cdylib` in [crates/hexgame-py/Cargo.toml](../../../../crates/hexgame-py/Cargo.toml:10).
- Phase 2 Verification: Run the `python-integration` workflow command locally or in CI. Either add the intended feature or change CI to target/package the PyO3 crate without `--features python`.
- Severity Guess: High
- Confidence: High

## 5. Hypothesis: panic/unwrap policy is mostly intentional but not centrally stated

- Area: Panic/unwrap policy
- Risk: The codebase mixes public fallible APIs, documented panics, runtime `assert!`, `debug_assert!`, and a few `unwrap`/`expect` sites. Some are clearly structural invariants, but there is no single policy explaining which failures may abort production training.
- Why It Might Be Hard To Catch: Individual sites look reasonable in isolation; risk emerges at the process level because `panic = "abort"` makes all panics fatal in release.
- Evidence Observed: `HexGameState::unplace` documents and uses `expect("no move to undo")` at [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:397) and [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:400). `EvalState::unplace` similarly panics on an empty stack at [crates/hexgame-core/src/eval/state.rs](../../../../crates/hexgame-core/src/eval/state.rs:309) and [crates/hexgame-core/src/eval/state.rs](../../../../crates/hexgame-core/src/eval/state.rs:313). MCTS includes public-path assertions and `sample_action` panic at [crates/hexgame-core/src/mcts.rs](../../../../crates/hexgame-core/src/mcts.rs:2035). Python wrapper prevents at least that sample-action case at [crates/hexgame-py/src/engine.rs](../../../../crates/hexgame-py/src/engine.rs:1343).
- Phase 2 Verification: Produce a panic inventory for non-test Rust code, classify each as user input, FFI input, internal invariant, or impossible proof. Convert user/FFI cases to `Result`; document invariant panics in public rustdoc.
- Severity Guess: Medium
- Confidence: High

## 6. Hypothesis: `WindowKey` coordinate packing is a latent boundary contract

- Area: Invariants / release behavior / board extent
- Risk: `WindowKey::new` packs signed 15-bit coordinates and validates bounds only with `debug_assert!`. Release builds can silently wrap out-of-range window origins into different keys if the board/eval search ever reaches beyond `-16384..=16383`.
- Why It Might Be Hard To Catch: Current placement-radius and practical game lengths may keep coordinates small, so property tests are unlikely to exercise far-out coordinates. If future loaders or search fixtures create large coordinates, hot-window bookkeeping could become wrong without an immediate panic.
- Evidence Observed: `WindowKey` bit layout is documented at [crates/hexgame-core/src/core.rs](../../../../crates/hexgame-core/src/core.rs:224), and `new` uses debug-only checks at [crates/hexgame-core/src/core.rs](../../../../crates/hexgame-core/src/core.rs:234), [crates/hexgame-core/src/core.rs](../../../../crates/hexgame-core/src/core.rs:235), [crates/hexgame-core/src/core.rs](../../../../crates/hexgame-core/src/core.rs:239), and [crates/hexgame-core/src/core.rs](../../../../crates/hexgame-core/src/core.rs:243).
- Phase 2 Verification: Add boundary tests for large coordinate fixtures, decide whether the range is a game-level invariant, and consider a fallible constructor or runtime assert if external loaders can exceed it.
- Severity Guess: Medium
- Confidence: Medium

## 7. Structure recommendation: make benchmark coverage and budgets explicit

- Area: Benchmarks / performance transparency
- Risk: Criterion benchmarks exist, but CI does not appear to run or compare them against a budget. Performance-sensitive claims may regress silently unless developers run benches manually.
- Why It Might Be Hard To Catch: Correctness tests and clippy can stay green while allocations, branching factor, or MCTS throughput degrade. Criterion HTML is useful locally but does not by itself enforce thresholds.
- Evidence Observed: Bench crate is a separate workspace member in [Cargo.toml](../../../../Cargo.toml:2) and depends only on `hexgame-core`/Criterion in [crates/hexgame-bench/Cargo.toml](../../../../crates/hexgame-bench/Cargo.toml:9). Benchmarks cover encoder/legal moves/MCTS/threats at [crates/hexgame-bench/benches/encode.rs](../../../../crates/hexgame-bench/benches/encode.rs:28), [crates/hexgame-bench/benches/encode.rs](../../../../crates/hexgame-bench/benches/encode.rs:77), [crates/hexgame-bench/benches/mcts.rs](../../../../crates/hexgame-bench/benches/mcts.rs:26), and [crates/hexgame-bench/benches/threats.rs](../../../../crates/hexgame-bench/benches/threats.rs:34). CI runs build/test/ignored tests/clippy but not `cargo bench` in [.github/workflows/ci.yml](../../../../.github/workflows/ci.yml:40).
- Phase 2 Verification: Decide whether to add a lightweight performance smoke job, store Criterion baselines, or add explicit manual benchmark gates to release docs. Track wall-clock budgets for `select_leaves`, `expand_and_backprop`, `legal_moves_near`, and threat filtering.
- Severity Guess: Medium
- Confidence: High

## 8. Question: should search and MCTS be separate crates from core rules?

- Area: Crate boundaries / compile times / ownership
- Risk: `hexgame-core` currently contains rules, encoder, eval, classical search, and neural MCTS. That is convenient, but it ties low-level board correctness to higher-level engine experiments and exposes all of it through one crate version.
- Why It Might Be Hard To Catch: The current workspace split already solves the biggest PyO3 rebuild problem (`hexgame-core` rlib, `hexgame-py` cdylib, separate bench crate). The remaining boundary issue is maintainability rather than an immediate build failure.
- Evidence Observed: Workspace members are split in [Cargo.toml](../../../../Cargo.toml:2). `hexgame-core` is an `rlib` at [crates/hexgame-core/Cargo.toml](../../../../crates/hexgame-core/Cargo.toml:9), while Python and benches depend on it from [crates/hexgame-py/Cargo.toml](../../../../crates/hexgame-py/Cargo.toml:13) and [crates/hexgame-bench/Cargo.toml](../../../../crates/hexgame-bench/Cargo.toml:9). Core still publishes `search` and `mcts` modules at [crates/hexgame-core/src/lib.rs](../../../../crates/hexgame-core/src/lib.rs:62).
- Phase 2 Verification: Measure build times and dependency graph churn after touching search/MCTS vs board/eval. If MCTS evolves rapidly, consider `hexgame-engine` or feature-gated modules; otherwise keep the current split and narrow public modules.
- Severity Guess: Low
- Confidence: Medium

## 9. Direct issue: slow ignored oracle tests are currently part of every CI run

- Area: Tests / CI runtime
- Risk: CI runs `cargo test --release -- --ignored`, which includes tests explicitly labeled slow oracle tests. This may be correct for a nightly/deep gate, but it can make normal PR feedback expensive or flaky as property-test cases grow.
- Why It Might Be Hard To Catch: The test suite is valuable and may still be fast today. The risk is scaling: new ignored tests inherit the same PR gate automatically.
- Evidence Observed: CI runs ignored tests at [.github/workflows/ci.yml](../../../../.github/workflows/ci.yml:46). Several threat oracle tests are marked ignored with slow-oracle messages at [crates/hexgame-core/src/tests/threats.rs](../../../../crates/hexgame-core/src/tests/threats.rs:223), [crates/hexgame-core/src/tests/threats.rs](../../../../crates/hexgame-core/src/tests/threats.rs:259), [crates/hexgame-core/src/tests/threats.rs](../../../../crates/hexgame-core/src/tests/threats.rs:355), and [crates/hexgame-core/src/tests/threats.rs](../../../../crates/hexgame-core/src/tests/threats.rs:430).
- Phase 2 Verification: Capture current CI timing for fast vs ignored tests. Decide whether ignored oracle tests belong in PR CI, nightly CI, or a label-triggered deep verification workflow.
- Severity Guess: Low to Medium
- Confidence: High

## 10. Hypothesis: Python byte-protocol decoding is safe today but repetitive and easy to fork incorrectly

- Area: FFI/API maintainability
- Risk: Legal/action bytes are hand-decoded in several Python binding methods. Each current loop has a preceding length check, so the `try_into().unwrap()` calls should be structurally safe, but repeated parsing logic raises the chance of future methods missing a guard or changing the wire format inconsistently.
- Why It Might Be Hard To Catch: The byte protocol crosses Python/Rust; failures may show up as Python value errors, Rust panics, or silent action mismatches depending on which method diverges.
- Evidence Observed: Repeated legal-byte decoding appears in [crates/hexgame-py/src/engine.rs](../../../../crates/hexgame-py/src/engine.rs:792), [crates/hexgame-py/src/engine.rs](../../../../crates/hexgame-py/src/engine.rs:846), and [crates/hexgame-py/src/engine.rs](../../../../crates/hexgame-py/src/engine.rs:894). Bulk encoder history decoding similarly uses fixed chunks at [crates/hexgame-py/src/encode.rs](../../../../crates/hexgame-py/src/encode.rs:39).
- Phase 2 Verification: Add a small `decode_hex_pairs`/`decode_history_triples` helper returning `PyResult<Vec<_>>`, plus malformed-byte tests from Python. This would make the unwrap policy obvious and keep the protocol in one place.
- Severity Guess: Low
- Confidence: High

## 11. Structure recommendation: document approximate evaluation bounds next to public accessors

- Area: Documentation / invariants / performance transparency
- Risk: Incremental eval deliberately skips windows outside a finite win grid; that is documented in `eval::grid`, but public callers can access `HexGameState::eval()` directly and may assume it is exact for all infinite-board coordinates.
- Why It Might Be Hard To Catch: Board rules are infinite, while eval uses a finite approximation for performance. Most positions stay near origin, so the mismatch will not appear in normal tests.
- Evidence Observed: Public eval accessor is [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:303). The finite grid guard and known approximation are documented at [crates/hexgame-core/src/eval/grid.rs](../../../../crates/hexgame-core/src/eval/grid.rs:73) and [crates/hexgame-core/src/eval/grid.rs](../../../../crates/hexgame-core/src/eval/grid.rs:81). The board module describes an infinite board in [crates/hexgame-core/src/board.rs](../../../../crates/hexgame-core/src/board.rs:7).
- Phase 2 Verification: Add rustdoc to `HexGameState::eval()` or `EvalState` explaining exactness bounds, and add one far-coordinate regression test that shows win detection remains exact even if heuristic eval skips out-of-grid windows.
- Severity Guess: Medium
- Confidence: High

## 12. Question: are benchmark fixtures representative of training hot paths?

- Area: Benchmarks / performance transparency
- Risk: Existing Criterion benches use fixed handcrafted midgame positions and mock uniform policies. They are useful microbenches, but may not represent late-game branching, threat-constrained roots, sparse/global/pair prior paths, or Python batch sizes used in training.
- Why It Might Be Hard To Catch: Microbenchmarks can improve while end-to-end throughput regresses because the expensive workload moved to an unbenchmarked path.
- Evidence Observed: MCTS bench runs 10 simulations with uniform mock policy in [crates/hexgame-bench/benches/mcts.rs](../../../../crates/hexgame-bench/benches/mcts.rs:26). Encoder/legal-move benches cover radius 2 and radius 8 at [crates/hexgame-bench/benches/encode.rs](../../../../crates/hexgame-bench/benches/encode.rs:28) and [crates/hexgame-bench/benches/encode.rs](../../../../crates/hexgame-bench/benches/encode.rs:77). Threat bench covers one static board at [crates/hexgame-bench/benches/threats.rs](../../../../crates/hexgame-bench/benches/threats.rs:5).
- Phase 2 Verification: Compare benchmark fixtures against real self-play telemetry: average stones, legal count, threat-constrained count, sparse prior density, batch size, and late-game extraction use. Add one representative "training batch" benchmark if current fixtures miss common hot paths.
- Severity Guess: Medium
- Confidence: Medium
