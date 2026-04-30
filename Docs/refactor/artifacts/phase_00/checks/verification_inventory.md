# Phase 00 S1 Verification Inventory

Status: evidence only for V2-004 and V2-005 orchestration reconciliation. This file does not mark either row complete and does not implement Phase 01+ architecture.

Old-runtime comparison policy: comparing new output to the current Python runtime is allowed only as a weak smoke signal. It is forbidden as the sole proof for legal rows, compact history, D6 transforms, candidate rows, pair rows, graph tensors, inference mapping, MCTS lifecycle state, replay records, trainer targets, dashboard inspection, or autotune recipes. Closure evidence must include independent or cross-validated oracles, negative/corrupt cases, row identity, schema/source/hash checks, and mutation detection.

## Audit Command Evidence

| Command | Exit | Purpose |
| --- | ---: | --- |
| `git status --short` | 0 | Checked working tree before and during inventory; unrelated source/test modifications were present before doc edits. |
| `rg --version` | 1 | `rg.exe` was blocked by `Access is denied`; inventory used fallback tooling. |
| `git grep -n -E "load_history\|set_position\|move_history_bytes\|legal_moves\|tactical_status\|cover_pairs\|D6\|symmetry\|transform\|duplicate\|malformed\|stale\|non-finite\|mutated\|mutation\|corrupt\|roundtrip\|round-trip\|oracle"` | 0 | Found existing golden, D6, oracle, negative, and mutation-risk tests. |
| `git grep -n -E "build_candidate\|Candidate\|PairCandidateBatch\|build_pair\|pair_candidate\|policy_pair\|graph_batch\|GraphBatch\|GRAPH_SCHEMA\|RELATION_SCHEMA\|schema_version\|legal_qr\|pair_token"` | 0 | Found candidate/pair/graph contract surfaces. |
| `git grep -n -E "class ReplayDataset\|def __iter__\|include_graph_policy\|include_pair_policy\|mutation\|writeable\|frombuffer\|copy\|ascontiguousarray\|torch.from_numpy\|np.asarray\|np.frombuffer"` | 0 | Found mutation-risk payloads and projection/copy sites. |
| `git grep -n -E "sha256\|config_hash\|model_dump\|cfg_json\|model_metadata\|action_contract_metadata\|recipe\|manifest\|runtime_sweep_key\|sort_keys"` | 0 | Found config, recipe, checkpoint, and hash identity inputs. |

## Golden Position Inventory

Compact history rows are little-endian `(player, q, r)` i32 triples unless a row explicitly says it is a synthetic board fixture. Golden fixtures must be promoted to checked-in fixture files or executable examples by their owner phases before closure.

| ID | Boundary coverage | Position / payload | Expected semantic facts | Existing evidence seed | Owner phase | Blocking tests/evidence |
| --- | --- | --- | --- | --- | ---: | --- |
| G00 | Opening legal/history/graph | Empty compact history `b""`. | Board empty; current player 0; placements remaining 1; legal rows exactly `{(0,0)}`; terminal false; pair rows masked at opening. | `Python/tests/test_engine_invariants.py:39-48`, `Python/tests/test_global_graph_contract.py:250-265`. | 01, 02 | Hand-audited compact fixture; legal-row hash/source; graph opening pair mask; illegal non-origin negative. |
| G01 | Post-opening two-placement turn | History `[(0,0,0)]`. | Player 1 to move with 2 placements; occupied origin excluded; legal rows are full Rust radius-8 table; pair rows possible only through explicit strategy/cap. | `Python/tests/test_engine_invariants.py:69-82`, `Python/tests/test_global_graph_contract.py:250-267`. | 01, 02, 05 | Rust legal rows plus hand radius oracle; row ordering/hash; no implicit pair scoring evidence. |
| G02 | Known-first second-placement rows | History `[(0,0,0),(1,1,0)]`. | Player 1 still has 1 placement; pair-second targets condition on known first `(1,0)` or the actual last move; first coordinate is not currently legal. | `Python/tests/test_global_graph_contract.py:272-302`, `Python/src/hexorl/buffer/sampler.py:766-848`. | 02, 03, 05, 07 | Pair-row contract example; illegal first-action negative; row identity from graph and replay projector. |
| G03 | Terminal no-legal state | Synthetic fixture with six stones on one axis for player 0, or chronological compact history equivalent once available. | Winner 0; terminal true; placements remaining 0; legal rows empty; post-terminal extra move rejected. | `Python/tests/test_engine_invariants.py:84-110`, `crates/hexgame-core/tests/board.rs:648-658`. | 01 | Convert to chronological compact fixture or mark synthetic-only; terminal legal hash; post-terminal negative. |
| G04 | Forced block / cover pair | History from engine invariant fixture: `[(0,0,0),(1,0,-1),(1,0,-2),(0,1,0),(0,2,0),(1,1,-1),(1,1,-2),(0,3,0),(0,2,-2)]`. | Tactical oracle reports forced blocks including `(-1,0)` and `(4,0)` and cover pair `((-1,0),(4,0))`; critical actions remain legal. | `Python/tests/test_engine_invariants.py:212-268`. | 01, 02, 05 | Rust tactical status plus independent hand window oracle; D6 pair canonicalization; candidate/pair preservation. |
| G05 | Far outside crop tactical cell | Synthetic stones `{(30+i,0): player}` with legal rows such as `(29,0)`, `(35,0)`, `(0,0)`. | Full-board oracle finds win/block outside crop; outside-crop cells must be represented in candidate/graph contracts. | `Python/tests/test_tactical_oracle.py:12-60`. | 01, 02 | Hand-audited far-coordinate fixture; Rust invariant hook; candidate overflow/critical inclusion proof. |
| G06 | Global graph target identity | Non-empty history with policy target, pair policy target, opponent legal rows, and opponent target. | Graph schema version present; legal rows unique; target masses normalize over their own legal rows; pair targets reject duplicates/illegal rows. | `Python/tests/test_global_graph_contract.py:77-150`, `250-335`, `697-750`. | 02, 04 | Graph semantic/tensor split tests; schema/source/hash identity; D6 relation invariance; corruption tests. |
| G07 | Replay projection with D6 | `PositionRecord` containing `policy_target_v2`, `pair_policy_target_v2`, `opp_policy_target_v2`, and compact history, with `ReplayDataset(use_symmetry=True)`. | Dense, sparse, pair, graph, opponent, axis, and moves-left projections preserve intended target mass after D6. | `Python/tests/test_training_data_pipeline.py:71-130`, `788-825`, `999-1048`. | 03, 07 | New replay record round-trip; projector identity hash; mutation/corruption tests; sampler no old-buffer proof. |
| G08 | MCTS root/leaf protocol | `PyHexGame` after origin placement; `init_root()` returns tensor, offsets, legal bytes, root token. | Root expansion validates offset, legal rows, finite policy, root token; leaf expansion validates batch token and finite values/policies. | `Python/tests/test_engine_smoke.py:46-141`. | 04, 05 | Token stale negatives; legal row hash/source in protocol; MCTS debug section; no panic/tokenless wrapper audit. |
| G09 | Inference graph IPC payload | `GraphBatch` submitted through graph IPC slot with schema versions, token/legal/opp/pair counts, relation tensors. | Counts match payload arrays; capacity failures are loud; stale slots/buffers cannot be consumed silently. | `Python/src/hexorl/inference/client.py:301-348`, `Python/src/hexorl/inference/server.py:500-530`. | 04 | Protocol manifest fixture; stale-ready/stale-slot tests; response telemetry with schema/source/hash/timing. |
| G10 | Config/recipe/checkpoint identity | Raw TOML config, resolved `Config.model_dump()`, `StaticRecipe`, family descriptor, runtime sweep key, checkpoint `cfg_json`/metadata. | Hashes distinguish raw config, resolved defaults, recipe fields, runtime knobs, model family, and checkpoint manifest. | `Configs/*.toml`, `scripts/run_phase3_48h_autotune.py:200-245,833-941,1362-1372`, `Python/src/hexorl/train/trainer.py:437-451`. | 00, 03, 08, 09 | Config hash index; typed recipe hash; checkpoint manifest round-trip; raw-config mutation audit. |

## D6 Variant Inventory

| Fixture set | Required D6 variants | Expected invariant | Owner phase | Blocking evidence |
| --- | --- | --- | ---: | --- |
| G00-G04 engine/legal/history | Apply all 12 transforms to compact history; include identity and inverse transform for each variant. | Current player, placements remaining, terminal/winner status, legal row set, row count, and source/hash policy remain explainable; transformed rows equal transformed original rows. | 01 | Rust/Python D6 parity; inverse/composition table; legal row ordering and hash assertions. |
| G04 tactical cover pairs | Transform forced block cells, cover cells, and unordered cover pairs under all 12 variants. | Pair rows remain canonical after transform; no reversed duplicate or self-pair appears. | 01, 02 | Existing tactical D6 seed plus contract-level pair identity tests. |
| G05 far-coordinate cells | Transform far outside-crop tactical cells. | Far coordinates stay legal/representable; no crop-only clipping or 15-bit/bounds truncation hides them. | 01, 02, 09 | Far-coordinate invariant hooks; release-mode bounds tests; debug bundle coordinates. |
| G06 graph payload | Transform history, legal rows, policy targets, pair targets, opponent legal rows, relation samples. | Token type counts stable; legal token mapping transformed; relation bias samples and target masses preserved. | 02, 04 | Graph D6 tests; relation schema/hash assertions; corruption negatives for mismatched transformed target. |
| G07 replay/train targets | Transform dense policy, sparse candidate rows, pair rows, axis labels/maps, graph targets. | Tensor/policy coordinates agree; axis labels permute one-to-one; pair and candidate target mass preserved. | 03, 07 | Replay projector D6 round-trip; single-position train adapter bundle. |
| G08/G09 inference and MCTS | Submit transformed legal rows/tensors through policy and MCTS mapping. | Raw model outputs map to the transformed legal rows before MCTS; stale original hashes fail. | 04, 05 | PolicyProvider debug bundle; stale row/hash negative tests. |

## Negative And Corrupt Case Inventory

| Boundary | Negative/corrupt cases | Existing seed | Owner phase | Required proof before closure |
| --- | --- | --- | ---: | --- |
| Compact history | Byte length not multiple of 12; wrong player/order; duplicate occupancy; illegal non-origin opening; far illegal placement; trailing post-terminal moves; synthetic `set_position` mistaken for chronological proof. | `Python/tests/test_engine_smoke.py:24-35,139-141`, `Python/tests/test_tactical_oracle.py:95-110`, `crates/hexgame-core/tests/board.rs:586-658`. | 01 | Contract decode rejects malformed bytes and impossible histories; synthetic fixtures labeled; Rust replay parity and transactionality. |
| Legal rows | Malformed legal bytes; row order mutation; duplicate/occupied rows; terminal rows non-empty; sub-radius global graph rows; stale legal rows with valid shape. | `Python/tests/test_engine_smoke.py:89-101`, `Python/tests/test_global_graph_contract.py:64-97`. | 01, 02, 04 | Legal contract schema/source/hash; row id/order tests; bad hash/version/source negatives. |
| D6 transforms | Invalid symmetry index; inverse not original; composition mismatch; scalar channels changed incorrectly; pair target canonicalization lost. | `Python/tests/test_engine_invariants.py:170-197,237-268`, `Python/tests/test_training_data_pipeline.py:71-130`. | 01, 02, 07 | D6 module parity with Rust; inverse/composition fixture; dense/history/legal/graph target round-trips. |
| Candidate rows | Duplicate legal rows; target outside legal; critical tactical cell overflow; target mass missing silently; post-validation mutation of NumPy arrays. | `Python/tests/test_tactical_oracle.py:43-60`, `Python/src/hexorl/action_contract/candidates.py:151-269`. | 02 | Immutable/mutation-guarded contract; source/hash changes on mutation; extension cannot bypass validation. |
| Pair rows | Duplicate coordinates; illegal pair action; duplicate active candidate row; reversed duplicate unordered pair; second-placement known-first mismatch; full pair enumeration without diagnostic cap. | `Python/tests/test_training_data_pipeline.py:999-1048`, `Python/tests/test_global_graph_contract.py:272-335`, `Python/tests/test_engine_smoke.py:121-129`. | 02, 05 | `PairActionTable` row identity; `PairStrategySpec` cap validation; no implicit pair scoring tests; D6 pair identity tests. |
| Graph payload | Policy/opp target not in its legal table; occupied opponent legal row; pair rows would truncate; relation tensors wrong shape; schema version mismatch; graph target mutation after validation. | `Python/tests/test_global_graph_contract.py:100-150,323-335,697-750`. | 02, 04 | Semantic builder vs tensorize/collate split; graph schema/hash/source tests; malformed graph IPC negatives. |
| Inference transport | Stale ready flag; stale slot generation; wrong request kind; wrong counts; malformed graph rows; non-finite logits; response count mismatch; protocol version mismatch; post-validation buffer mutation. | `Python/tests/test_inference_server.py`, `Python/src/hexorl/inference/client.py:301-348`. | 04 | Manifest handshake; slot sequence counters; timeout/backpressure fail-fast tests; response telemetry assertions. |
| MCTS lifecycle | Stale root token; stale batch token; wrong legal bytes; non-finite priors/values; malformed pair rows; tokenless/panic wrapper use. | `Python/tests/test_engine_smoke.py:83-136`, `crates/hexgame-core/src/mcts.rs`. | 05, 09 | `EngineAdapter` only caller audit; stale-token tests; panic/unwrap inventory; MCTS debug section. |
| Replay records | Truncated blobs; corrupt policy/pair/opp targets; mutated cached tensor; stale compact history hash; old buffer decode path used by train. | `Python/src/hexorl/buffer/sampler.py:637-952`, `Python/tests/test_training_data_pipeline.py`. | 07 | New replay codec corruption suite; trace-to-record-to-projector identity; old buffer import/deletion proof. |
| Training targets | Pair first/second/joint semantics swapped; graph legal masks wrong; loss consumes inactive heads; model outputs non-finite; target arrays mutated after collation. | `Python/src/hexorl/train/losses.py`, `Python/src/hexorl/train/trainer.py`. | 03 | TrainAdapter single-position bundle; one-batch every family; target alignment and mutation/corruption tests. |
| Config/recipe/checkpoint | Raw config mutation changes family behavior; missing resolved-config hash; checkpoint loads with mismatched model family/protocol; deprecated prefix cleanup hides drift. | `scripts/run_phase3_48h_autotune.py`, `Python/src/hexorl/train/trainer.py:437-451`, `Python/src/hexorl/dashboard/checkpoints.py:78-119`. | 03, 08, 09 | Typed recipe hash examples; strict checkpoint manifest; recipe dry-run tests; raw mutation audit. |

## Mutation-Risk Payloads

| Payload | Current mutation surface | Risk | Owner phase | Required mutation evidence |
| --- | --- | --- | ---: | --- |
| `CandidateBatch` / `PairCandidateBatch` arrays | Frozen dataclass holds mutable NumPy arrays. | Row identity and target mass can change after validation. | 02 | Read-only arrays or mutation-detecting hash/version; post-validation mutation negative. |
| `GraphBatch` arrays | Frozen dataclass holds mutable NumPy arrays and collator copies into padded arrays. | Token/legal/pair/relation identity can change between validation and inference/train. | 02, 04 | Immutable graph contract/projection hash; corruption test for changed legal rows or relation shape. |
| Replay sampler caches | `_tensor_cache` and `_meta_cache` store tensors and legal bytes keyed by history. | Cached tensor/legal bytes can become stale or mutable across projection consumers. | 07 | Cache identity/hash tests; copy/read-only policy; stale history hash rejection. |
| `np.frombuffer(legal_bytes)` views | Legal rows are views over bytes when decoded in sampler/MCTS tests. | Shape-valid mutated copies can reorder rows and bypass shape-only tests. | 01, 04, 05 | Legal-row hash/order tests; malformed and reordered row negatives. |
| Shared-memory inference slots | Client writes NumPy arrays into reusable IPC slots; server converts them to Torch tensors. | Stale counts or slot reuse can mix old/new request data. | 04 | Slot generation/sequence tests; stale ready/reset tests; request/response telemetry. |
| Torch tensors from NumPy | `torch.from_numpy` shares host memory in server batching. | Host buffer mutation can affect tensors after validation. | 04 | Copy/lifetime policy tests; post-validation buffer mutation test. |
| MCTS priors and tokens | Root/legal/pair priors and root/batch tokens cross Python/Rust boundary. | Stale tokens or row hashes can apply outputs to wrong root/leaf. | 05 | Token/hash ownership in `EngineAdapter`; stale root/batch tests. |
| `PositionRecord` / replay record targets | Mutable Python containers feed dense, sparse, pair, graph, opp, axis, and moves-left targets. | Replay projection can silently alter training semantics. | 07 | New record contract immutability; round-trip and corrupt-record rejection. |
| Checkpoint `cfg_json` / `model_metadata` | JSON dicts are loaded and reused for model construction/inspection. | Mutable raw config can be treated as identity-bearing recipe proof. | 03, 08 | Strict manifest hash; inspect without weights; mutation of metadata invalidates identity. |
| Dashboard/debug JSON payloads | Dashboard reconstructs graph/D6 summaries from request payloads and checkpoint metadata. | UI can normalize or hide mismatches instead of surfacing contract failure owner. | 08 | Inspector service read-only tests; mismatch owner tests; screenshots/artifacts where relevant. |

## Independent Oracle Options

| Boundary | Acceptable oracle mix | Old-runtime comparison rule |
| --- | --- | --- |
| Engine legal/history | Rust replay plus hand-audited histories, placement-radius reference, terminal/current-player assertions, and invariant hooks. | Current Python fallback is weak only and forbidden as sole proof. |
| Compact byte protocol | `crates/hexgame-py/src/protocol.rs` decode/encode, Python contract validation, malformed-byte negatives, source/version/hash examples. | Comparing decoded Python helper output to old sampler parser is forbidden as sole proof. |
| D6 | Rust `apply_d6_symmetry`, coordinate transform reference, inverse/composition table, legal/history/tensor/target round-trips. | Comparing transformed old replay output to new replay output is weak only. |
| Candidates | Hand target sets, Rust legal rows, tactical oracle critical cells, extension-proof registered selector. | Existing candidate builder output is weak only. |
| Pairs | Hand canonical pair rows, Rust legal rows, known-first second-placement fixture, `PairStrategySpec` caps, D6 pair transform. | Existing pair mini-contract output is weak only. |
| Graph | Semantic builder reference, graph schema/hash/source, relation sample invariants, D6 graph target preservation, IPC capacity checks. | Existing `GraphBatch` tensor output is weak only. |
| Inference | Manifest handshake, request-kind schema validation, synthetic stale-slot and wrong-version inputs, response telemetry. | Old inference server output equality is weak only and cannot prove protocol correctness. |
| MCTS/search | Rust fallible tokenized APIs, constructed stale-token cases, legal-row hash validation, finite prior checks, `EngineAdapter` audit. | Old Python MCTS loop success is weak only. |
| Replay | Codec round-trip, corrupt blob rejection, trace-to-record-to-projector hash identity, Rust replay validation. | Old buffer sampler equality is weak only and forbidden as sole replay proof. |
| Training | Single-position debug bundle, manually computed targets/loss inputs, adapter-owned required fields, non-finite output negatives. | Old trainer loss parity is weak only. |
| Dashboard/autotune | Inspector examples, typed recipe hashes, route/report tests with injected mismatches and no-progress causes. | Existing UI/report similarity is weak only. |

## Planned Single-Position Debug Bundle Sections

| Section | Required fields |
| --- | --- |
| Engine | `history_bytes_hex`, history hash, current player, placements remaining, terminal/winner, legal row count, legal row hash, Rust protocol source/version, invariant hook result. |
| Contracts | Contract type, schema version, source label, content hash, row ids/order, construction validation mode, mutation policy. |
| D6 | Symmetry index, inverse index, transformed history hash, transformed legal hash, composition check, target mass preservation. |
| Candidates | Candidate row count, row ids, mask count, critical actions, missing mass, overflow examples, selector/feature version. |
| Pairs | Pair strategy, cap, pair row count, known-first if any, row ids, duplicate/reversed/self-pair checks, scored vs possible rows. |
| Graph | Graph schema/relation versions, token/legal/opp/pair counts, relation shape, target masses, capacity report, tensorization/collation source. |
| Targets | Dense, sparse, pair, graph, opponent, value, lookahead, regret, axis, moves-left target summaries and source record hash. |
| Model inputs | Model family, recipe hash, protocol manifest id, input shapes/dtypes, masks, finite checks, device/batch metadata. |
| Model outputs | Head names, output shapes, finite checks, legal-row alignment, warning list, timing spans. |
| Policy priors | Raw output-to-legal mapping, priors before/after masking, source labels, pair-prior blend, non-finite/illegal rejection result. |
| MCTS | Root token, batch token, root legal hash, selected leaves, prior source counters, stale-token checks, MCTS error owner. |
| Replay | Record id, compact history hash, replay schema version, codec round-trip status, projector output hash, corruption check result. |
| Dashboard | Inspector route, displayed contract/source/hash/protocol/model/recipe identities, mismatch owner if present. |
| Autotune | Trial id, typed recipe hash, runtime spec hash, scheduler decision, watchdog/no-progress context, subsystem cause hypothesis. |

## Reconciliation Notes

- V2-004 evidence: stale fallback/parser/projection paths and implicit data shapes are named here and in `Docs/refactor/artifacts/phase_00/inventory/architecture_string_inventory.md`.
- V2-005 evidence: every boundary has required golden, D6, negative/corrupt, mutation, oracle, and debug-bundle evidence mapped to owner phases.
- No skipped, deferred, flaky, or manual-only requirement is claimed complete by this inventory.
