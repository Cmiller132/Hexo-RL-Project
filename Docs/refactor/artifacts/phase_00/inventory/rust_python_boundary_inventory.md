# Phase 00 S2 Rust/Python Boundary Inventory

This artifact is an inventory, not a runtime implementation. It supports
`V2-003`, `V2-004`, `V2-005`, `V2-011`, `V2-012`, `V2-013`, `V2-016`,
`V2-040`, `V2-051`, `V2-056`, and `V2-095`, but it does not close any row by
itself because runtime consumption, deletion/import proof, structured telemetry,
and CI gates still belong to later phases.

## Assignment Frame

Goal: map the post-Rust-Phase-2 engine/runtime boundary, legal fallback paths,
MCTS token lifecycle, panic-wrapper suspicion points, structured error gaps, and
owner phases.

Success criteria: every listed boundary has a current source location, risk,
owner phase, and required deletion or replacement evidence.

Constraints: documentation-only inventory; no runtime code or tests changed; no
new compatibility path claimed complete.

Required evidence: command-backed source searches and Rust review docs.

Stop rules: if resolving a gap requires Phase 01+ implementation, record the
gap and owning phase rather than claiming completion.

## Command Evidence

`rg --files` was attempted first, per project instructions, but failed with
`Access is denied` from the bundled `rg.exe`. PowerShell search/read commands
were used as the fallback.

| Command | Exit | Evidence used |
| --- | ---: | --- |
| `git status --short` | 0 | Worktree was clean before edits. |
| `Get-Content -Raw Docs/refactor/phases/PHASE_00.md` | 0 | Active Phase 00 gate and S2 requirements. |
| `Get-Content -Raw Docs/RUST_API.md` | 0 | Published Rust/Python API reference. |
| `Get-Content -Raw Docs/refactor/rust_review/README.md` | 0 | Rust review crosswalk and suspicion semantics. |
| `Get-Content -Raw Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md` | 0 | Completed Rust Phase 2 baseline and open items. |
| `rg --files` | 1 | Tool unavailable in this desktop session: access denied. |
| `Get-ChildItem -Recurse -File Python/src/hexorl ... Select-String '_engine|HAS_ENGINE|_fallback_legal_moves|legal_bytes|root_generation|batch_generation'` | 0 | Direct Python engine imports, fallback paths, token plumbing, byte decoding. |
| `Get-ChildItem -Recurse -File crates/hexgame-py/src,crates/hexgame-core/src ... Select-String 'root_generation|batch_generation|MCTSError|legal_bytes|history_bytes|decode_*|expect|assert|panic'` | 0 | Rust protocol owners, token lifecycle, panic/assert inventory seeds. |
| `Select-String -Path Cargo.toml,... -Pattern 'panic'` | 0 | Workspace release/profile panic policy includes `panic = "abort"` in `Cargo.toml:28`. |
| `Get-ChildItem -Recurse -File Python/src/hexorl ... Select-String 'selfplay_no_progress|runtime_sweep_no_progress|watchdog|heartbeat|phase_transition'` | 0 | No matching structured no-progress/heartbeat event names in runtime source. |
| `Get-ChildItem -Recurse -File Python/tests ... Select-String 'root_generation|batch_generation|stale|legal_bytes|history_bytes|pair_rows'` | 0 | Existing Python smoke coverage for stale root/batch tokens and malformed bytes. |

## Completed Rust Phase 2 State

Treat this as the current baseline, not as proof that Rust outputs can skip
Python validation.

| Area | Current Phase 2 state | Source evidence |
| --- | --- | --- |
| Stable Rust facades | `hexgame_core::rules`, `encoding`, `tactics`, and `classical` are documented as stable facades. `MCTSEngine` and `MCTSError` remain public for the PyO3 crate but are not re-exported from the crate root. | `Docs/RUST_API.md`; `Docs/refactor/rust_review/PHASE_2_VERIFICATION_REPORT.md` |
| Chronological history | `load_history` exists for chronological replay while `set_position` remains synthetic fixture loading. | Rust review report; `_engine` history tests in `Python/tests/test_engine_invariants.py`. |
| FFI legal/history byte protocol | `crates/hexgame-py/src/protocol.rs` owns `encode_legal_rows`, `decode_legal_rows`, `encode_compact_history_rows`, `decode_compact_history_rows`, and `decode_pair_rows`. | `protocol.rs:10`, `protocol.rs:19`, `protocol.rs:42`, `protocol.rs:52`, `protocol.rs:60`. |
| MCTS root token | `init_root` increments and returns `root_generation`; root expansion validates `root_generation`, offsets, and legal rows. | `crates/hexgame-core/src/mcts.rs:733`, `mcts.rs:704`, `mcts.rs:750`; `crates/hexgame-py/src/engine.rs:803`, `engine.rs:845`, `engine.rs:870`. |
| MCTS batch token | `select_leaves` returns `batch_generation`; backprop APIs reject stale or absent pending batch tokens and clear pending state after use. | `mcts.rs:1355`, `mcts.rs:1470`, `mcts.rs:1498`, `mcts.rs:1504`, `mcts.rs:1565`; `engine.rs:1076`, `engine.rs:1093`, `engine.rs:1162`, `engine.rs:1171`. |
| Tactical source of truth | `TacticalStatus` is the complete tactical label source; tactical filtering no longer depends on bounded eval hot windows. | Rust review report; `Docs/RUST_API.md`. |
| Negative Python smoke coverage | Existing tests cover stale root token, stale batch token, malformed legal bytes, malformed history bytes, malformed pair rows, and non-finite root policy inputs. | `Python/tests/test_engine_smoke.py:83`, `:89`, `:98`, `:104`, `:110`, `:122`, `:132`, `:139`. |

## Boundary Owners And Owning Phases

| Boundary | Current owner | Required V2 owner | Owning phase | Required evidence before closure |
| --- | --- | --- | ---: | --- |
| Rust rule semantics: legal rows, move legality, turn phases, terminal/winner state | `crates/hexgame-core/src/board.rs`, `core.rs`, `encoder.rs`, `threats.rs` | Rust remains production source, wrapped by Python engine/contracts | 01 | Engine wrapper tests; legal-row semantic validation; import audit proving runtime no longer imports `_engine` directly; negative tests for occupied, duplicate, terminal, current-player mismatch. |
| Legal/history/pair FFI byte protocol | `crates/hexgame-py/src/protocol.rs` | Single protocol source recorded in engine/protocol contracts | 01, 04, 09 | Malformed byte tests; protocol source/version/hash in debug bundles and inference protocol manifests; duplicate parser/encoder audit. |
| Python-facing engine API | Many direct imports of `_engine` across runtime packages | `Python/src/hexorl/engine/` boundary only | 01 | Runtime import audit banning direct `_engine` outside engine and fixture/test quarantine; deletion manifest for fallback legal/history parsers. |
| Graph legal/action rows | `Python/src/hexorl/graph/batch.py` rebuilds legal rows and optionally imports `_engine` | Candidate/pair/graph contracts with Rust-derived legal source validation | 02 | Graph builder consumes canonical legal rows; D6 and corruption tests; import audit for private graph legal reconstruction. |
| MCTS lifecycle | Rust core + PyO3 + `Python/src/hexorl/selfplay/worker.py` wrapper | `search/EngineAdapter` is only Python MCTS caller | 05 | Stale root/batch token tests through adapter; direct MCTS import audit; no panic wrapper or string fallback use. |
| Self-play runtime ownership | `SelfPlayWorker` owns engine wiring, MCTS loop, graph/candidate/pair scoring, replay assembly, and crash loop | `GameRunner` composes explicit provider/adapter/pipeline outputs; worker is lifecycle/IPC only | 06 | Self-play integration smoke, heartbeat/no-progress telemetry, import audit proving worker does not own canonical builders or direct Rust MCTS calls. |
| Replay compact history and projection | `buffer/sampler.py`, `selfplay/records.py`, dashboard replay utilities | New replay codec/projector with Rust protocol identity | 07 | New replay records include FFI protocol source, compact history hash, legal-row hash; old buffer decode absent from runtime imports. |
| Dashboard/eval inspection | Dashboard/eval import `_engine` or decode bytes directly | Inspector services over read-only contracts | 08 | Dashboard/eval import audit; inspector route tests; no private model-input or legal-row reconstruction. |
| Panic/assert and public API drift gates | Rust source plus workspace `Cargo.toml` | CI public API drift, panic inventory, malformed FFI/stale-token gates | 09 | Panic inventory with classifications; CI job evidence for public API drift, malformed FFI bytes, stale MCTS tokens, panic/unwrap inventory, and structured engine error checks. |

## Protocol Locations And Duplicate Decode/Encode Sites

Canonical Rust locations:

| Protocol | Canonical source | Wire shape |
| --- | --- | --- |
| Legal rows | `crates/hexgame-py/src/protocol.rs:10`, `:19` | little-endian `i32` `(q, r)` rows, 8 bytes per row. |
| Board piece rows | `crates/hexgame-py/src/protocol.rs:27` | little-endian `i32` `(q, r, player)` rows, 12 bytes per row. |
| Compact history rows | `crates/hexgame-py/src/protocol.rs:42`, `:52` | little-endian `i32` `(player, q, r)` rows, 12 bytes per row. |
| Pair rows | `crates/hexgame-py/src/protocol.rs:60` | NumPy `int32` array with shape `(N, 4)` as `(q1, r1, q2, r2)`. |

Python duplicate or private protocol handling found by search:

| File | Evidence | Risk | Owner phase | Replacement/deletion evidence needed |
| --- | --- | --- | ---: | --- |
| `Python/src/hexorl/selfplay/worker.py` | `_legal_bytes_from_qr` returns `int32` `(N,2)` bytes at `worker.py:474`; mock engine builds legal bytes manually at `:592-:600`; root/leaf paths decode legal bytes with `np.frombuffer(...).reshape(-1, 2)` at `:1329`, `:1389`, `:1592`, `:1691`, `:1783`. | Hot runtime decodes/constructs Rust protocol bytes outside an engine contract. Mock path can mask malformed byte and token behavior. | 01, 05, 06 | Engine contract API owns byte decode/hash/source; worker consumes typed legal table and token objects; mock path quarantined to tests or replaced by fixture-only adapter. |
| `Python/src/hexorl/buffer/sampler.py` | Imports `_engine`; `_fallback_legal_moves`; pure Python fallback for `encode_compact_record`; pure Python fallback for D6; fallback legal bytes encoded at `:941-:946`. | Replay/training can silently continue without Rust and emit protocol-shaped bytes from Python. | 01, 07 | Sampler consumes new replay projector/engine contract; fallback import/code audit proves no runtime fallback remains. |
| `Python/src/hexorl/graph/batch.py` | `parse_history` unpacks `<iii>`; `transform_history` repacks `<iii>`; `legal_moves_for_stones` recomputes legal rows; `_engine_state_from_history` optional. | Graph semantics can be reconstructed from Python stones instead of canonical Rust legal rows. | 01, 02 | Graph builder receives canonical legal/history contract or explicit fixture-only fallback; D6 parity/corruption tests cover history and row identity. |
| `Python/src/hexorl/dashboard/replay.py` | Direct `_engine` import; fallback legal moves; byte move parsing and legal-byte decode. | Dashboard inspection may become a second semantic implementation. | 08 | Dashboard consumes `ContractInspector`; direct engine/fallback import audit passes. |
| `Python/src/hexorl/eval/arena.py`, `eval/players.py`, `eval/classical.py` | Direct `_engine` imports and legal-byte decode. | Eval may bypass `PolicyProvider` and canonical legal contracts. | 05, 08 | Eval consumes provider/engine adapter; `_engine` imports removed or test-only. |
| `Python/src/hexorl/epoch/pipeline.py` | Bootstrap path imports `_engine`; fallback bootstrap legal rows exist. | Synthetic data generation can supply old-runtime legal rows into training. | 07 | Bootstrap fixture path is explicit and quarantined; train sampler consumes new replay records only. |
| `Python/src/hexorl/action_contract/tactical_oracle.py` | Late direct `_engine` import and Python tactical/legal helpers. | Tactical facts can be reconstructed outside Rust `TacticalStatus` contract. | 01, 02 | Contract routes Rust tactical status with fixture-only oracle comparison; duplicate runtime helper removed. |

## Legal Fallback Paths

These are current runtime or quasi-runtime paths that can replace Rust legal
semantics when `_engine` is unavailable or bypassed. They must not be preserved
as production compatibility once their owning phase closes.

| Path | Current behavior | Risk | Owner | Required deletion/replacement proof |
| --- | --- | --- | ---: | --- |
| `SelfPlayWorker` `HAS_ENGINE=False` selects `MockMCTSEngine` | Full self-play loop can run against a mock engine. | Masks Rust import/linkage failures, stale-token failures, malformed bytes, and panic/abort behavior. | 05, 06 | Runtime refuses production self-play without engine adapter unless explicit fixture/test mode; import audit and smoke prove Rust/adapter path is used. |
| `ReplayDataset._encode_tensor_meta` fallback | If `_engine` is missing, Python builds legal rows and bytes. | Training can proceed on non-Rust legal rows. | 01, 07 | New replay projector uses Rust-derived protocol identity; fallback path deleted or fixture-only outside runtime. |
| `graph.batch.legal_moves_for_stones` | Recomputes all legal rows from Python stones when `_engine_state_from_history` is unavailable. | Graph contracts can diverge from Rust turn/tactical legality. | 01, 02 | Graph builder consumes canonical legal contract; fallback only for explicitly marked hand fixtures. |
| `dashboard.replay._fallback_legal_moves` | Dashboard can reconstruct legal rows without Rust. | Debug UI may report misleading legal data. | 08 | Inspector uses contract/replay records; fallback absent from dashboard runtime imports. |
| `epoch._make_fallback_bootstrap_game` | Synthetic bootstrap data can be generated without Rust. | Legacy bootstrap data may leak into training baseline as if canonical. | 07 | Bootstrap is fixture-only with source label, or replay cutover removes old sampler path. |
| `tactical_oracle` Python helpers | Provides tactical/legal helper behavior around `_engine` and Python fallback. | Rust `TacticalStatus` is not the only semantic authority. | 01, 02 | Rust tactical status wrapped in contract; Python oracle is test/probe only. |

## MCTS Token Lifecycle

Current token flow from Rust through Python:

1. `PyMCTSEngine.init_root()` calls Rust `MCTSEngine::init_root()` and returns
   `(tensor_3d, offset_q, offset_r, legal_bytes, root_generation)`.
2. `crates/hexgame-py/src/engine.rs` stores a `RootSnapshot` containing offset,
   legal rows, and `root_generation`.
3. Root expansion APIs decode `legal_bytes`, validate the snapshot and root
   token, validate policy/value shapes and finiteness, call Rust, then clear the
   root snapshot. Dense, sparse, and global root expansion follow this pattern.
4. `select_leaves(batch_size)` calls Rust and returns `(tensor_4d, count,
   batch_generation)` while PyO3 stores `last_batch_generation`.
5. `pending_leaf_metadata()` returns leaf offsets, legal bytes, and compact
   history bytes for non-terminal pending leaves. It does not return the batch
   token in the metadata payload; callers must pair metadata with the previously
   returned `batch_generation`.
6. Backprop APIs (`expand_and_backprop`, sparse variants, source variants)
   validate PyO3 `last_batch_generation`, call Rust validation, and clear the
   pending token after success.
7. `re_root(q, r, new_num_simulations)` clears pending leaves, applies the
   selected action, increments the root generation in Rust, and clears PyO3 root
   snapshot/last batch state.

Remaining suspicion points:

| Suspicion point | Evidence | Risk | Owner | Required evidence |
| --- | --- | --- | ---: | --- |
| Pair-prior APIs are not tokenized in PyO3 signatures | `engine.rs:991`, `:1014`, `:1027` accept pair rows/logits/mix, but no `root_generation` parameter. | Published docs say root pair-prior methods must reject stale root tokens, but source signatures cannot receive a root token. Current identity checks may be action-based only. | 05 | Adapter contract either tokenizes pair-prior submission or proves pair priors are synchronous root-local calls with stale-root negative tests. |
| `pending_leaf_metadata()` has no explicit batch token field | `engine.rs:1108` returns metadata only; user code pairs it by call order. | Metadata from a stale selection could be consumed if Python stores or reorders it incorrectly. | 05 | `EngineAdapter` binds metadata to a token object and stale metadata tests fail before inference/search consumption. |
| Mock MCTS increments tokens but does not validate them | `worker.py:545`, `:577`, `:599`, `:685`, `:700`. | Tests or fallback runtime can pass while real token validation would fail. | 05, 06 | Mock removed from production runtime or replaced with strict test double that rejects stale tokens and malformed rows. |
| Direct `RealMCTSEngine` wrapper lives inside worker | `worker.py:851` onward wraps Rust MCTS directly. | Worker owns search protocol instead of lifecycle only; broad exception catches string-log failures. | 05, 06 | `EngineAdapter` owns every MCTS call; worker/game runner consumes adapter methods with structured errors. |
| Root/leaf structured errors are stringified | PyO3 maps `MCTSError` via `PyValueError(format!("{:?}", e))` at root/backprop calls. Python catches broad `Exception` in root/leaf loops. | Logs cannot reliably classify stale token, malformed bytes, invalid prior, panic/abort, inference mismatch, or contract failure. | 04, 05, 06 | Structured `MCTSError` mapping with code, phase, token ids, counts, source hashes, and debug bundle samples. |

## Structured Error Gaps

| Gap | Current evidence | Required replacement |
| --- | --- | --- |
| `MCTSError` loses type and fields at PyO3 boundary | `engine.rs` uses `PyValueError::new_err(format!("{:?}", e))` around Rust MCTS calls. | Python engine layer exposes a structured error type with `code`, `phase`, `root_generation`, `batch_generation`, `legal_count`, `history_hash`, and `source`. |
| No protocol source/hash attached to legal/history bytes | Byte buffers cross Python as raw `bytes` or NumPy arrays. | Engine/contracts carry `protocol_owner`, `protocol_version`, byte length, row count, source label, and hash. |
| Python catches broad exceptions in root/leaf loops | `worker.py:1552` and `worker.py:1809` log warning strings and continue or mark graph invalid. | Game runner/search adapter records structured failure events and aborts/retries predictably. |
| Inference timeouts are plain exceptions | `InferenceClient` raises `TimeoutError` without a self-play no-progress event or last request payload. | Transport owns request/response sequence counters, last request telemetry, timeout events, and fail-fast structured errors. |
| Crash/abort is not contained as an event | Workspace has `panic = "abort"` and source contains Rust `expect`/`assert`/`panic` seeds; no `catch_unwind` evidence was found. | Parent process records worker exit code/signal, last phase, last engine op/token, and restart/abort decision in structured telemetry. |

## Panic/Abort Crash-Containment Requirements

Source evidence:

- `Cargo.toml:28` sets `panic = "abort"`.
- Rust source still contains panic/assert/expect seeds in non-test code:
  `board.rs:883`, `board.rs:957`, `board.rs:986`, `board.rs:997`,
  `board.rs:1009`, `board.rs:1010`, `board.rs:1027`,
  `core.rs:232`, `mcts.rs:1386`, `mcts.rs:1466`, `mcts.rs:1794`,
  `mcts.rs:2049`, `mcts.rs:2098`, `mcts.rs:2109`, `mcts.rs:2262`,
  `mcts.rs:2310`, `search.rs:283`, `eval/state.rs:294`,
  `eval/state.rs:334`, `eval/state.rs:350`, `eval/state.rs:355`,
  `eval/state.rs:437`, `eval/state.rs:515`.
- Search for `catch_unwind`, `set_hook`, and Rust panic containment returned no
  crash wrapper evidence in source; only classical search `aborted` fields were
  found.

Required containment before V2 closure:

1. Public/FFI misuse must return structured Rust/Python errors, not panic.
2. Internal invariant panics must be classified and covered by debug/test
   invariant probes.
3. If Rust aborts a worker process, the parent must record a structured crash
   event with worker id, process exit code or signal, last successful phase,
   last engine operation, root/batch tokens when known, progress counters, and
   restart/abort decision.
4. CI must run a panic/unwrap inventory and public API drift check before final
   closure.

## Deletion And Replacement Evidence Needed

| Evidence item | Phase |
| --- | ---: |
| Runtime import audit proving direct `_engine` imports are absent outside `Python/src/hexorl/engine/` and explicit fixture/test quarantines. | 01 |
| Legal rows from Rust are wrapped in a contract with semantic validation for row identity, ordering, duplicates, occupied cells, current player, terminal status, source, schema, hash, and mutation safety. | 01 |
| Compact history contract round-trips through Rust protocol and rejects malformed bytes, impossible turn order, duplicate occupancy, post-terminal moves, and mutation. | 01 |
| D6 transforms have one owner and parity-test coordinates, histories, legal rows, and dense tensors against Rust. | 01 |
| Candidate/pair/graph builders consume canonical legal/history contracts and delete private legal reconstruction from runtime. | 02 |
| Inference protocol manifest carries request-kind schemas, Rust FFI protocol identities, row hashes, and fails fast on mismatch. | 04 |
| `EngineAdapter` is the only Python caller of Rust MCTS APIs, preserves token objects, binds pending metadata to batch tokens, and exposes structured `MCTSError` ownership. | 05 |
| Self-play watchdog/heartbeat/debug bundle samples identify stalls, slow phases, pair scoring, inference waits, engine errors, and contract mismatches. | 06 |
| Replay records store Rust FFI protocol source, compact history hash, reconstructed legal-row hash, and exclude transient MCTS root/batch tokens from replay semantics. | 07 |
| Dashboard/eval consume inspector/provider contracts rather than reconstructing legal/history/model input facts. | 08 |
| Final CI keeps malformed FFI bytes, stale MCTS tokens, parity, invariant probes, public API drift, panic inventory, and structured engine error checks active. | 09 |

## Current Blockers

- This inventory cannot delete Python legal fallbacks or direct `_engine` imports
  without Phase 01+ implementation.
- A real no-progress watchdog event cannot be generated from the current runtime
  without code changes; see
  `Docs/refactor/artifacts/phase_00/watchdog/no_progress_smoke.md`.
- This artifact does not claim that Rust is an unquestioned oracle. Later phases
  must keep Rust-derived rows, compact histories, pair rows, MCTS tokens, and
  FFI bytes under semantic validation and negative tests.
