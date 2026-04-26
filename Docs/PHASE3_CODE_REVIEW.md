# Phase 3 — Complete Code Review

**Reviewer:** Kimi Code CLI (multi-subagent deep review)  
**Date:** 2026-04-26  
**Scope:** All implemented phases (1–3) across the Rust workspace and Python `hexorl` package  
**Method:** Six independent subagents reviewed source line-by-line against `SYSTEM_DESIGN.md` and `FINALIZATION_PASS.md`. This document synthesizes their findings.

---

## TL;DR

| Subsystem | Completeness | Verdict |
|---|---|---|
| Rust core engine (`hexgame-core`) | ~90% | Solid foundation; most FINALIZATION_PASS blockers fixed. One residual NaN defense and one process-killing `assert!` remain. |
| Rust PyO3 bridge (`hexgame-py`) | ~75% | 5 FFI entry points functionally present but not zero-copy. GIL not released in encode/augment paths. |
| Python inference server | ~45% | Shared-memory architecture works, but adaptive batching is inverted, `max_batch` unenforced, events are busy-poll, and race conditions exist. |
| Python self-play & buffer | ~55% | Skeleton is complete, but **policy targets use NN priors instead of MCTS visits**, resignation is checked after `re_root`, and buffer reads are unlocked. |
| Python model & config | ~80% | Correct shapes, working forward pass, good Pydantic schema. Residual block lacks batch norm and ignores `heads` config. |
| Tests & build system | ~65% | Excellent Rust test/proptest/bench coverage. Python tests are thin. CI missing benchmark runs, Python type checks, and E2E. |

**Bottom line:** Phase 3 is structurally complete but contains **critical functional bugs** that would produce garbage training data if training began today. Do not start self-play data generation until the P0 items in §9 are fixed.

---

## 1. Rust Core Engine (`crates/hexgame-core`)

### 1.1 Verification of FINALIZATION_PASS.md Blockers

| ID | Fix | Verdict | Evidence |
|---|---|---|---|
| T1-1 | Backprop sign flip uses depth parity | ✅ | `mcts.rs:636-641` — `parity_value` flips each ply; no `node.player` comparison. |
| T1-2 | Virtual loss adjusts `visit_count` and `total_value` | ✅ | Apply `mcts.rs:498-502`; revert `mcts.rs:602-606`. Both fields touched with correct signs. |
| T1-3 | `sims_done` only incremented in `expand_and_backprop` | ✅ | Single increment at `mcts.rs:598`. |
| T1-4 | Slice bounds asserted in `expand_and_backprop` | ✅ | Two `assert!` at `mcts.rs:582-592`. |
| T1-5 | GIL released in MCTS hot path | ✅ | `pybridge/mcts.rs` (now `engine.rs`) wraps calls in `py.allow_threads`. |
| T1-6 | `set_position` per-stone validation | ✅ | `board.rs:455-473` checks `InvalidPlayer`, `MustPlaceAtOrigin`, `OutOfRadius`. |
| T1-7 | Pattern index underflow guard promoted to `assert!` | ✅ | `eval/state.rs:323-328` — **is now `assert!`** (verified directly). |
| NaN-safe ingestion | `assert!(v.is_finite())` in `expand_and_backprop` | ✅ | `mcts.rs:616-618`. |
| `re_root` returns `Result` | `Result<(), MCTSError>` | ✅ | `mcts.rs:662`. PyO3 bridge propagates as `PyValueError`. |
| Tree-traversal panic demoted | `let _ = self.game.place(...)` | ✅ | `mcts.rs:486-487` — no `.expect`. |
| Temperature sampling | `sample_action(temperature, rng_state)` | ✅ | `mcts.rs:1076-1114`. |
| Resign threshold | `should_resign(threshold)` | ✅ | `mcts.rs:1116-1127`. |
| Deterministic seeding | `seed: u64` accepted | ✅ | `MCTSEngine::new(..., seed)` at `mcts.rs:303-310`. |

### 1.2 Remaining Critical Issues

#### RUST-CRIT-1: NaN-Vulnerable PUCT Scorer (Release Builds)
**File:** `mcts.rs:1005-1009`

```rust
debug_assert!(
    score.is_finite(),
    "PUCT score non-finite for child {i}: ..."
);
```

This is a **`debug_assert!`** — stripped in release builds. If `total_value` becomes `NaN` (adversarial NN output, arithmetic edge case), `score` becomes `NaN`, and `score > best_score` is always `false`. Every subsequent simulation silently biases toward the first legal child. The `expand_and_backprop` NaN guard catches *incoming* NN values but not *accumulated* state corruption.

**Fix:** Promote to `assert!(score.is_finite(), ...)` or clamp after computation.

#### RUST-CRIT-2: Process-Killing `assert!` in `re_root`
**File:** `mcts.rs:663-666`

```rust
assert!(
    self.pending.is_empty(),
    "re_root: pending leaves must be flushed"
);
```

With `panic = "abort"`, this kills the **entire Python process**. The PyO3 bridge cannot catch it. A worker calling `re_root` after `select_leaves` but before `expand_and_backprop` (e.g., on timeout or crash) will SIGABRT and lose the in-progress game.

**Fix:** Return `Err(MCTSError::PendingLeaves)` and propagate through the bridge as `PyValueError`.

### 1.3 Major Issues

#### RUST-MAJ-1: Per-Leaf Heap Allocation (`legal_buf.clone()`)
**File:** `mcts.rs:550`

`PendingLeaf.legal_moves` is `Vec<Hex>`. Every leaf triggers `self.legal_buf.clone()`. At batch_size=8 and 800 sims/move, this is ~800 small Vec allocations.

**Fix:** Replace `Vec<Hex>` with `SmallVec<[Hex; 32]>` in `PendingLeaf` and use `SmallVec::from_slice(&self.legal_buf)`.

#### RUST-MAJ-2: Orphaned Root `src/` Directory
**File:** `src/mcts.rs`, `src/pybridge/`, etc.

The workspace root `Cargo.toml` has no `[package]` section, so `src/` is **not compiled**. However, it is still tracked in git and contains an old monolithic implementation with **unfixed** hard panics:
- `src/mcts.rs:654` — `panic!("re_root: no child found...")`
- `src/mcts.rs:463` — `.expect("MCTS: illegal place...")`

A developer or IDE jumping to definition can land on stale code. This is a severe foot-gun.

**Fix:** Delete the entire root `src/` directory.

### 1.4 Minor Issues

- **`extract_tree_node_states` uses `&'static str` errors** — opaque for Python callers. Should include node indices.
- **`q_value()` returns `0.0` for unvisited nodes** — FPU reduction uses `parent_q - FPU_REDUCTION`, which is fine, but unvisited leaf Q is not configurable.
- **Arena never shrinks** — acknowledged design choice; ~400 KB waste per turn is acceptable.

### 1.5 Positive Observations

- **Arena allocator is production-quality.** Clean child-range layout, no pointer chasing.
- **Incremental `EvalState` with full undo** is verified by 1,500+ proptest cases against a brute-force oracle.
- **Workspace split is clean.** `hexgame-core`, `hexgame-py`, `hexgame-bench`, `hexgame-cli` build independently.
- **13-channel encoder** matches AlphaZero/KataGo CNN inputs exactly.

---

## 2. Rust PyO3 Bridge (`crates/hexgame-py`)

### 2.1 FFI API Surface Assessment

| Design Doc Entry (§3.2) | Status | Location | Notes |
|---|---|---|---|
| `MCTSEngine.run_until_inference_needed()` | ⚠️ **Decomposed** | — | Design wanted a single entry point. Instead there is `init_root` → `expand_root` → loop(`select_leaves` → `expand_and_backprop`) → `done()`. Functionally adequate but places orchestration burden on Python. |
| `MCTSEngine.submit_inference(policies, values)` | ✅ | `engine.rs:690` | Named `expand_and_backprop`. Accepts flat `PyReadonlyArray1<f32>`. |
| `MCTSEngine.sample_action(temperature, seed) -> (q, r)` | ✅ | `engine.rs:769` | Returns `(i16, i16)`. |
| `encode_compact_record(history_bytes) -> ndarray` | ✅ | `encode.rs:16` | Validates `len % 12 == 0`. Returns owned `PyArray4<f32>` shape `(N, 13, 33, 33)`. |
| `apply_d6_symmetry(tensor, sym_idx) -> ndarray` | ✅ | `encode.rs:65` | All 12 transforms implemented. Bounds-checks output indices. |

### 2.2 Critical Issues

#### PYO3-CRIT-1: Zero-Copy FFI Not Implemented
**Files:** `engine.rs:672-675` (`select_leaves`), `engine.rs:702-703` (`expand_and_backprop`)

```rust
// select_leaves
let (count, tensor_vec) = py.allow_threads(|| {
    let (tensors, count) = self.inner.select_leaves(batch_size);
    (count, tensors.to_vec())  // ← copy #1
});

// expand_and_backprop
let p = policies_slice.to_vec();  // ← copy #2
let v = values_slice.to_vec();    // ← copy #3
```

At batch_size=16 and ~1k inferences/sec, this is ~1 GB/s of pure memcpy overhead. The design doc (§3.5) explicitly called for zero-copy.

**Fix for `select_leaves`:** Allocate an uninitialized `PyArray4` under the GIL, then write into it inside `allow_threads` via `unsafe { arr.as_slice_mut().unwrap() }`.

**Fix for `expand_and_backprop`:** Pass borrowed slices directly into the Rust call without `.to_vec()`. If lifetime issues prevent this across `allow_threads`, keep the GIL during the call (the Rust work dominates; the copy does not).

### 2.3 Major Issues

#### PYO3-MAJ-1: GIL Not Released in Encode/Augment Paths
**Files:** `encode.rs:16-52`, `encode.rs:65-117`

Neither `encode_compact_record` nor `apply_d6_symmetry` uses `py.allow_threads()`. The design doc (§3.4) requires the GIL released for both. `encode_compact_record` replays an arbitrary-length game history and encodes every position into a dense `(N, 13, 33, 33)` tensor — this can block the Python inference dispatcher.

**Fix:** Wrap the hot loops in `py.allow_threads(|| { ... })`.

#### PYO3-MAJ-2: `expand_and_backprop` Copies Inputs While Holding the GIL
The `.to_vec()` calls execute **before** `py.allow_threads()`. For large batches, this holds the GIL during a significant memcpy.

#### PYO3-MAJ-3: `buffer.rs` is a Stub
**File:** `buffer.rs:1-7`

Only an empty `register_module` function. No Rust kernels for compact-to-dense decode or recency-weighted sampling. Python fallbacks in `sampler.py` exist but are O(N²) and unvalidated.

### 2.4 Minor Issues

- `sample_action` returns `(i16, i16)` while `place()` takes `(i32, i32)` — inconsistent coordinate width.
- `expand_and_backprop` does not pre-validate numpy array lengths before copying; core `assert!` will abort on mismatch.
- Docstring parameter name mismatch: `rng_state` vs `seed`.
- Exception type inconsistency: `place` → `PyValueError`; `classical_search` → `PyRuntimeError` for the same `GameError` class.

### 2.5 Positive Observations

- `legal_bytes` parser is hardened (len % 8 check, descriptive `PyValueError`).
- Docstrings are extensive and accurate.
- `apply_d6_symmetry` correctly validates shape and bounds-checks transformed indices.
- `re_root` properly propagates `MCTSError` as catchable `PyValueError`.

---

## 3. Python Inference Subsystem (`hexorl/inference/`)

### 3.1 Completeness vs Design Spec

| Requirement (SYSTEM_DESIGN.md §5) | Status | Notes |
|---|---|---|
| Process topology (1 server, N workers) | ✅ | `mp.Process` used correctly. |
| Shared-memory queues | ✅ | `SharedMemory` + numpy ndarray views. |
| Adaptive batching (wait `max_wait_us` after first request) | ❌ **Inverted** | Server processes immediately when any worker is ready; only sleeps when **no** workers are ready. Stragglers are processed in singleton batches. |
| `max_batch_size` enforcement | ❌ **Missing** | `_drain_ready_workers` has no batch-size cap. 30 workers × 8 leaves = 240 positions can OOM GPU. |
| GPU stream pipelining (H2D / forward / D2H) | ❌ **Not implemented** | Single implicit stream; `run_in_executor` wraps synchronous forward. |
| Model EMA for inference | ❌ **Not implemented** | §5.5 entirely absent. |
| FP16 / mixed precision | ✅ | `torch.cuda.amp.autocast(dtype=torch.float16)` in `_forward`. |
| OS-level doorbell events | ❌ **Busy-poll** | Custom `SharedEvent` spins with `time.sleep(0.0001)`. Adds latency and burns CPU. |

### 3.2 Critical Bugs

#### INF-CRIT-1: Race Condition in Request/Response Signaling
**Files:** `server.py:177`, `client.py:105`

The server clears `req_ready` **after** the forward pass finishes. A worker that submits a new request *during* the forward pass will have its `req_ready` silently cleared. The server then sets `res_ready` with **stale results** from the previous batch. The worker returns immediately with incorrect data.

**Impact:** Silent data corruption in MCTS evaluation.

**Fix:** Clear `req_ready` immediately when the server begins processing a worker. Add a monotonic sequence number to the slot protocol so the worker can verify the response matches its current request.

#### INF-CRIT-2: No `max_batch_size` Enforcement
**File:** `server.py:191-206`

The server drains **all** ready workers regardless of total position count. If 30 workers each submit 8 leaves, the forward pass receives 240 positions. With `max_batch_size=128` this can OOM the GPU.

**Fix:** Accumulate workers until `sum(counts) >= max_batch`, stop draining, and truncate the last worker's slice if necessary.

#### INF-CRIT-3: Broken Cross-Process Event Semantics
**File:** `shm_queue.py:46-89`

`SharedEvent` is a busy-polling loop. The unit tests manually patch `client._slot.req_ready = s0.req_ready`, which is a direct workaround indicating that cross-process event visibility is unreliable with this primitive.

**Fix:** Replace `SharedEvent` with `multiprocessing.Event` (spawn-safe on macOS/Linux) or a `Condition` variable. Remove event-patching hacks from tests.

#### INF-CRIT-4: Adaptive Batching Inverted
**File:** `server.py:182-184`

The spec says: *"wait up to `max_wait_us` for more requests once any request arrives."* The implementation does the opposite — it processes whatever is ready immediately and only sleeps when **no** workers are ready. This defeats batching and fails to saturate the GPU.

**Fix:** After observing the first ready worker, start a `max_wait_us` timer and collect additional workers until the timer expires or `max_batch` is reached.

### 3.3 Major Issues

- **Zero-copy boundary violated:** `np.array(slot.req_tensor[:c], copy=True)`, `np.concatenate`, `torch.from_numpy(...).to(device)`, and client-side `np.array(self._slot.res_policy[:count], copy=True)` all introduce copies. Design doc §3.5 mandates no intermediate copies.
- **No CUDA stream pipelining:** Batch N+1 cannot upload while batch N computes. Up to ~2× throughput loss.
- **No graceful forward-failure recovery:** Any `RuntimeError` (OOM, bad shape) is caught, printed, and re-raised, killing the server. Workers timeout one by one.
- **`InferenceClient` clears `res_ready` after wait, not before:** If the server finishes the next batch before the worker reaches the clear, the worker consumes stale data on the subsequent call.

### 3.4 Minor Issues

- `print()` instead of structured logging in `server.py`.
- `asyncio.get_event_loop()` is deprecated; should use `asyncio.get_running_loop()`.
- `__del__` used for resource cleanup — unreliable.
- `client.submit` flattens policies with `.ravel()`; docstring claims 2D return shape.
- `_build_batch` allocates `torch.empty(0)` on no-tensor path without matching dtype/device.

### 3.5 Positive Observations

- Spawn-safe architecture: `__getstate__` strips non-picklable attributes; child reconnects to named shared memory.
- Crash recovery for SHM names: `_create_shm` cleans up leftover segments.
- FP16 support is correct: casts outputs back to `float32` before CPU transfer.
- Signal hygiene: `_run` ignores `SIGINT` so child survives Ctrl-C in parent.
- Stats instrumentation: `n_batches`, `n_positions`, `total_forward_ms`.

---

## 4. Python Self-Play & Buffer (`hexorl/selfplay/`, `hexorl/buffer/`)

### 4.1 Critical Bugs (Do Not Train Before Fixing)

#### SP-CRIT-1: Policy Target Uses NN Priors Instead of MCTS Visit Counts
**File:** `worker.py:508-530`

```python
moves_q, moves_r, visits, root_value = engine.get_results()
priors = engine.root_child_priors()
...
policy = sparsify_policy(np.array(priors), top_k=20)
```

The `visits` returned by `get_results()` are **never used**. The policy target is built from `root_child_priors()` (the raw neural-network prior probabilities), not the MCTS visit counts.

**Impact:** The network will be trained to predict its own prior policy, not the search-improved policy. MCTS self-play becomes pointless.

**Fix:** Build a dense `BOARD_AREA` array from `moves_q`, `moves_r`, and `visits`, using `action_to_board_index(q, r, offset_q, offset_r)` to map each child to its flat board index, then sparsify *that* visit distribution.

#### SP-CRIT-2: Resignation Checked After Move Is Played
**File:** `worker.py:549-559`

`engine.should_resign(self.resign_threshold)` is called **after** `engine.re_root(q, r, sims)`. `re_root` advances the internal game state to the **next** position (opponent's turn). Therefore:
- The threshold is evaluated from the **wrong player's perspective**.
- The position evaluated is the one **after** the move, not the one that was just searched.

**Fix:** Move `should_resign` check **before** `re_root`, immediately after `sample_action`.

#### SP-CRIT-3: Thread-Unsafe `RingBuffer.__getitem__`
**File:** `ring.py:153-180`

`__getitem__` reads multiple arrays **without acquiring `self._lock`**. Meanwhile `append`/`extend` write to the same arrays under the lock. Data races between DataLoader worker threads (reading) and the record collector thread (writing) can produce corrupted indices, torn floats, or mixed old/new records.

**Fix:** Acquire `self._lock` at the top of `__getitem__` and hold it for the duration of the read.

#### SP-CRIT-4: Bare `except Exception: pass` Masks Fatal Errors
**Files:** `worker.py:512-516`, `worker.py:549-553`

Both `get_results()` and `re_root()` are wrapped in bare `except Exception: pass`.
- `re_root` can raise `PyValueError` (e.g., `MCTSError::ChildNotFound`). Swallowing it leaves the engine in an undefined state.
- The `get_results()` fallback fabricates fake visits/priors (`[1]*10`, `[0.1]*10`), inserting garbage into the game record.

**Fix:** Let `re_root` exceptions propagate to the outer crash handler in `run()`, which respawns the worker. Remove the fake-data fallback in `get_results()`.

### 4.2 Major Issues

- **Config values `c_puct_init` and `constrain_threats` are silently ignored:** Loaded in worker but never passed to `RealMCTSEngine`.
- **MockMCTSEngine fallback is unsafe for production:** If `_engine` is not importable, the worker silently falls back to random games. There is no fatal error. Make fallback opt-in via config.
- **Lookahead values computed but discarded by `RingBuffer`:** `process_game_record` computes EMA lookahead targets and stores them in `pos.lookahead_values`, but `RingBuffer.append()` only stores `record.to_value_target()` (a single scalar).
- **`RingBuffer.__getitem__` hardcodes `root_value=0.0`:** MCTS root Q-values are lost.
- **`action_to_board_index` default offsets are inconsistent with Rust encoder:** Python defaults to `offset_q=16, offset_r=16` (giving `gi = q - 16`), while Rust encoder for an empty board uses `offset_q = -16` (giving `gi = q + 16`).
- **`from_compact_bytes` lacks bounds checking:** `struct.unpack_from` on a truncated record will crash the dataloader.
- **`_monitor_workers` list swap-and-pop is fragile:** Confusing index surgery; use a `dict` mapping `worker_id → process`.

### 4.3 Minor Issues

- PCR is per-game, not per-position (reduces diversity but not strictly wrong).
- Dirichlet noise is flat, not shaped by existing child priors.
- Player access via private attribute: `engine._game.current_player`.
- `loader.py` default path uses wrong case (`configs` vs `Configs`) — fails on Linux.
- `get_batch` silently drops `None` records, causing batch size shrinkage.
- Quality weight ratio in `sample_indices` is `4.0 / 0.25 = 16:1`, much stronger than the spec's stated "4×".

### 4.4 Positive Observations

- Clean separation of concerns: worker, orchestrator, buffer, targets well-modularized.
- `MockMCTSEngine` is thorough — realistic game lengths, branching factors, state transitions.
- `RealMCTSEngine` wrapper is thin and type-compatible.
- Ring buffer uses struct-of-arrays layout for vectorized numpy access.
- Config schema with Pydantic provides runtime validation.
- Crash recovery logic exists: orchestrator monitors health and respawns dead processes.
- Temperature schedule interpolation is correct.
- `sparsify_policy` correctness: `np.argpartition` + renormalization is correct.

---

## 5. Python Model, Config & Training Stubs (`hexorl/model/`, `hexorl/config/`)

### 5.1 Completeness vs Phase 3 Spec

The model and config are largely complete for Phase 3. Training, evaluation, epoch, and dashboard modules are correctly empty stubs awaiting Phase 4.

### 5.2 Critical Issues

*None found in this subsystem. Architectural concerns below are major but not crash-level.*

### 5.3 Major Issues

#### MODEL-MAJ-1: Residual Block Lacks Batch Normalization
**File:** `network.py:107-119`

The docstring claims "Pre-activation residual block" but the implementation is:
```python
x = torch.relu(self.conv1(x))
x = torch.relu(self.conv2(x))
return x + residual
```

This is **not** a pre-activation block (BN→ReLU→Conv→BN→ReLU→Conv). It has **no batch normalization at all**. For a 16-block, 128-channel ResNet, omitting BN is a significant deviation from AlphaZero/KataGo literature and will likely cause training instability or vanishing/exploding gradients at depth. The skip connection technically works, but without normalization the network may fail to train effectively.

**Fix:** Add `nn.BatchNorm2d` after each conv (or switch to a modern norm-free architecture with explicit justification).

#### MODEL-MAJ-2: `from_config` Silently Ignores `cfg.model.heads`
**File:** `network.py:122-154`

The config schema allows `heads = ["policy", "value", "lookahead_short", ...]` but `from_config` only passes `channels` and `blocks` to `HexNet`. The `heads` list is completely ignored. `production.toml` configures 5 heads; the network will still output only policy + value with no warning.

**Fix:** Raise an error for unsupported heads or add a TODO comment that head configuration is Phase 4.

#### CONFIG-MAJ-1: `default_config.toml` Is Legacy with Incompatible Schema
**File:** `Configs/default_config.toml`

Uses old section names (`[self_play]`, `[training]`, `[rgsc]`, etc.) that do **not** validate against the current Pydantic `Config` schema. It is a trap for anyone trying to load it.

**Fix:** Delete `default_config.toml` or migrate it to the new schema.

### 5.4 Minor Issues

- `value_fc2` (final value layer) uses Kaiming init with `nonlinearity='relu'`, but it's followed by `tanh`. Large initial weights may cause saturation. Prefer Xavier or small/zero init for tanh outputs.
- `model/__init__.py` is just a docstring; should export `HexNet`, `from_config`.
- `cli.py` still says "Phase 1 stub" even though we're in Phase 3.
- No `[project.scripts]` in `pyproject.toml`; `hexorl` CLI is not installable as a console script.
- `half()` method is redundant but harmless.

### 5.5 Positive Observations

- Config schema is well-designed: Pydantic v2, good defaults, covers all sections from `SYSTEM_DESIGN.md` §8.1.
- Shape discipline is good: `(B, 13, 33, 33)` → policy `(B, 1089)`, value `(B, 1)` consistently enforced.
- Pure Python decoders exist in `sampler.py` as fallbacks before Rust FFI kernels are ready.
- `forward` assertion `x.shape[1:] == (13, 33, 33)` is correct — allows variable batch size.

### 5.6 Gaps for Phase 4

| Module | Expected Files | Current State |
|---|---|---|
| `train/` | `trainer.py`, `losses.py`, `schedule.py`, `ema.py` | Only `__init__.py` |
| `eval/` | `arena.py`, `elo.py`, `classical.py` | Only `__init__.py` |
| `epoch/` | `pipeline.py`, `autotune.py`, `stats.py` | Only `__init__.py` |
| `dashboard/` | `tui.py`, `tb.py` | Only `__init__.py` |
| `model/heads.py` | `BaseHead`, `PolicyHead`, `ValueHead`, `LookaheadHead`, `AxisHead`, `RegretHead` | Does not exist |
| `model/checkpoint.py` | Save/load with metadata | Does not exist |

---

## 6. Tests & Build System

### 6.1 Rust Tests — Strong

| Module | Tests | Notes |
|---|---|---|
| `core` | 16 | Hex coords, distances, `WindowKey`, `Turn` |
| `patterns` | 11 | Incremental eval consistency, pattern table integrity |
| `grid` | 4 | Win-grid bijection, bounds |
| `hot` | 4 | `HotWindows` insert/remove/clear/len |
| `mcts` | 5 | Deterministic replay, re-root, root-Q bounding, wrong-length panic, `done()` |
| `oracle` | 6 | Brute-force winning/blocking/unblockable detection |
| `threats` | ~15 | Property-based (500-case) threat/oracle equivalence |
| `threats_internal` | 18 | Winning turns, blocking singles/pairs, unblockable |
| `eval_state` | 8 | Place/unplace round-trip, threat counts, hypothetical delta |
| `board.rs` (integration) | 23 + 2 proptests | Opening rules, win detection, `set_position`, zobrist |
| `encoder.rs` (integration) | 9 + 1 proptest | Feature correctness, output range |

**Verdict:** Excellent coverage of core engine primitives.

### 6.2 Rust Benchmarks — Complete

All 3 Criterion benchmarks exist (contrary to outdated notes in `FINALIZATION_PASS.md`):
- `bench_encode_board_into`
- `bench_legal_moves_near_radius2`
- `bench_candidates_near2`
- `bench_single_mcts_sim` (full MCTS loop, 10 sims/iter, mock uniform policy)
- `bench_threat_status` (mid-game position threat classification)

### 6.3 Python Tests — Very Thin

- `test_engine_smoke.py`: 4 tests (constants, game basic, encode shape, MCTS completion). Does not exercise threat constraints, Dirichlet noise, re-rooting, or error paths.
- `test_inference_server.py`: 3 test classes covering server start/stop, single client round-trip, adaptive batching two clients, MCTS round-trip with engine. Uses `unittest` rather than `pytest`. Contains event-patching hacks that mask real SHM bugs.

**Missing entirely:**
- Buffer tests (`RingBuffer`, `sampler`, `targets`)
- Self-play worker unit tests (`MockMCTSEngine` is present but not tested)
- Model tests (`HexNet` forward pass, shape checks, FP16 conversion)
- Training loop tests
- Config validation tests

### 6.4 CI / Build System — Partial

**`ci.yml` Rust job:**
- ✅ Builds release
- ✅ Runs fast tests + ignored oracle tests
- ✅ Runs clippy (`-D warnings`)
- ❌ Does **not** run benchmarks (`cargo bench` missing — regression detection gap)
- ❌ Does **not** run debug-profile tests (some `#[should_panic]` tests are `#[cfg(debug_assertions)]`)

**`ci.yml` Python job:**
- ✅ Builds extension via `maturin develop`
- ✅ Runs `test_engine_smoke.py`
- ❌ Does **not** run `test_inference_server.py` (needs PyTorch)
- ❌ Does **not** run `ruff` or `mypy` (listed in `pyproject.toml` dev deps but never invoked)

**Missing workflows:**
- `e2e.yml`
- `cargo audit` / `cargo deny`
- Python formatting / type-checking gate

### 6.5 Build Config Issues

- **`.cargo/config.toml` clippy allow is global:** `-A clippy::module_name_repetitions` applies to dependency crates. Should be scoped via `[lints.clippy]` in workspace `Cargo.toml`.
- **`pyproject.toml` build-system mismatch:** Declares `setuptools` backend but CI uses `maturin`. Align the build backend or document the maturin workflow.
- **Duplicate `proptest-regressions` directories:** Both `./proptest-regressions/` and `./crates/hexgame-core/proptest-regressions/` exist.
- **Legacy dead code:** `Python/Model/`, `Python/Epoch/`, `Python/GameRunner/`, `Python/Core/`, `Python/Database/` contain obsolete files. Recommend cleanup PR before Phase 4.

### 6.6 Positive Observations

- MCTS test suite matches `FINALIZATION_PASS.md` T4-1 spec exactly.
- Property-based testing is extensive: 1,500+ proptest cases with tracked regression seeds.
- Workspace split is clean with correct workspace inheritance.
- CI runs ignored oracle tests — slow brute-force proptests are validated on every PR.
- Pattern-value checksum guard in `eval_state.rs` catches silent table corruption.

---

## 7. Cross-Cutting Issues

### 7.1 Determinism Is Partial

- Rust `MCTSEngine` accepts a `seed: u64` and uses XOR-shift RNG for `sample_action`.
- Python `SelfPlayWorker` seeds each game with `cfg.run.seed + worker_id * 10000 + game_counter`.
- However, **Dirichlet noise** is sampled via `np.random.dirichlet` with no seed plumbing.
- **Inference server** has no deterministic batch ordering (sort by `worker_id` before forward pass).
- **D6 symmetry** index is sampled from a global RNG, not a per-sample seed derived from `(game_id, ply)`.

**Score:** 6/10. Better than the 3/10 in `FINALIZATION_PASS.md`, but not yet fully reproducible.

### 7.2 Python/Rust Coordinate Indexing Mismatch

- Rust encoder uses `gi = q - offset_q` where `offset_q = centroid_q - HALF_BOARD` (e.g., `-16` for empty board).
- Python `action_to_board_index` defaults to `offset_q=16, offset_r=16`, giving `gi = q - 16`.
- These are **inverses**. Any Python-side code using default offsets will map coordinates to the wrong tensor indices relative to the Rust engine.

**Fix:** Unify on the Rust convention (`offset_q = -16` for centered board) and document clearly.

### 7.3 Policy Target Shape Mismatch

- `InferenceClient.submit` docstring says return shape is `(count, 1089)`.
- Actual return is `(count * 1089,)` due to `.ravel()` in `client.py:112`.
- `worker.py` expects the flat shape for `expand_and_backprop`. This is internally consistent but brittle.

---

## 8. Action Items — Prioritized

### P0 — Block Before Any Training Run

| # | Issue | File | Fix |
|---|---|---|---|
| 1 | **Policy target uses NN priors, not MCTS visits** | `worker.py` | Build dense visit distribution from `get_results()` and sparsify that. |
| 2 | **Resignation checked after `re_root`** | `worker.py` | Move `should_resign` before `re_root`. |
| 3 | **Thread-unsafe `RingBuffer.__getitem__`** | `ring.py` | Acquire `self._lock` for the entire read. |
| 4 | **Bare `except Exception: pass` masks fatal errors** | `worker.py` | Propagate `re_root` errors to crash handler; remove fake-data fallback. |
| 5 | **PUCT `debug_assert!` for NaN stripped in release** | `mcts.rs` | Promote to `assert!(score.is_finite(), ...)`. |
| 6 | **Process-killing `assert!` in `re_root`** | `mcts.rs` | Return `Err(MCTSError::PendingLeaves)`. |
| 7 | **Inference race condition (req_ready cleared late)** | `server.py` | Clear `req_ready` at start of processing; add sequence numbers. |
| 8 | **Inference `max_batch_size` unenforced** | `server.py` | Cap drained workers at `max_batch` total positions. |
| 9 | **Inference adaptive batching inverted** | `server.py` | Wait `max_wait_us` after first request before processing. |
| 10 | **Orphaned root `src/` with unfixed panics** | `src/` | Delete entire root `src/` directory. |

### P1 — Fix Before Production Scaling

| # | Issue | File | Fix |
|---|---|---|---|
| 11 | **Zero-copy FFI not implemented** | `engine.rs` | Write `select_leaves` directly into `PyArray4`; pass slices to `expand_and_backprop`. |
| 12 | **GIL not released in encode/augment** | `encode.rs` | Wrap hot loops in `py.allow_threads()`. |
| 13 | **SharedEvent busy-polls instead of OS events** | `shm_queue.py` | Use `multiprocessing.Event` or `Condition`. |
| 14 | **MockMCTSEngine fallback is silent** | `worker.py` | Make fallback opt-in via config; default to hard error. |
| 15 | **Config `c_puct_init` and `constrain_threats` ignored** | `worker.py` | Pass them to `RealMCTSEngine`. |
| 16 | **Lookahead values computed but discarded** | `ring.py`, `targets.py` | Add `lookahead_values` storage to `RingBuffer`. |
| 17 | **`RingBuffer` loses `root_value`** | `ring.py` | Store and reconstruct `root_value` in `__getitem__`. |
| 18 | **Coordinate indexing mismatch** | `records.py` | Unify offsets with Rust encoder convention. |
| 19 | **`_ResBlock` lacks batch normalization** | `network.py` | Add `nn.BatchNorm2d` or justify norm-free design. |
| 20 | **`default_config.toml` is incompatible legacy** | `Configs/` | Delete or migrate to new schema. |
| 21 | **Config loader path case sensitivity** | `loader.py` | Use `"Configs"` (capital C). |

### P2 — Polish & Phase 4 Readiness

| # | Issue | File | Fix |
|---|---|---|---|
| 22 | **Per-leaf heap allocation (`legal_buf.clone`)** | `mcts.rs` | Use `SmallVec<[Hex; 32]>` in `PendingLeaf`. |
| 23 | **No CUDA stream pipelining** | `server.py` | Create H2D / D2H `torch.cuda.Stream` objects. |
| 24 | **No graceful forward-failure recovery** | `server.py` | Catch `RuntimeError`, return dummy uniform policies, keep server alive. |
| 25 | **`from_compact_bytes` lacks bounds checking** | `records.py` | Validate buffer length before unpacking. |
| 26 | **Worker monitor list swap-and-pop** | `orchestrator.py` | Use `dict[worker_id, Process]`. |
| 27 | **No Python buffer/model/worker tests** | `Python/tests/` | Add `pytest` tests for `RingBuffer`, `HexNet`, `MockMCTSEngine`. |
| 28 | **CI does not run benchmarks** | `.github/workflows/` | Add `cargo bench` step (fail on >5% regression). |
| 29 | **CI does not run Python type checks** | `.github/workflows/` | Add `ruff` and `mypy` gates. |
| 30 | **Legacy dead code** | `Python/Model/`, etc. | Delete obsolete directories. |
| 31 | **`.cargo/config.toml` clippy allow is global** | `.cargo/config.toml` | Move to `[lints.clippy]` in workspace `Cargo.toml`. |
| 32 | **Model EMA for inference** | New files | Implement `train/ema.py` + server hot-swap. Phase 4. |

---

## 9. Phase 4 Readiness Assessment

| Criterion | Status | Blocker |
|---|---|---|
| Rust engine produces correct MCTS visit distributions | ❌ | SP-CRIT-1 |
| Rust engine does not abort on adversarial input | ⚠️ Partial | RUST-CRIT-1, RUST-CRIT-2 |
| Inference server batches safely without OOM | ❌ | INF-CRIT-2 |
| Inference server returns correct results under load | ❌ | INF-CRIT-1, INF-CRIT-4 |
| Self-play worker produces valid training targets | ❌ | SP-CRIT-1, SP-CRIT-2, SP-CRIT-4 |
| Ring buffer is thread-safe for concurrent train/load | ❌ | SP-CRIT-3 |
| Model forward pass is stable for training | ⚠️ Partial | MODEL-MAJ-1 (no BN) |
| Config system loads and validates correctly | ⚠️ Partial | CONFIG-MAJ-1, loader case |
| Zero-copy FFI eliminates memcpy bottleneck | ❌ | PYO3-CRIT-1 |
| Deterministic/reproducible runs possible | ⚠️ Partial | Dirichlet seeding, batch ordering |

**Recommendation:** Fix all P0 items and at least P1 items 11–21 before beginning Phase 4 implementation. The current codebase is an excellent skeleton, but the critical functional bugs in policy-target construction, inference batching, and buffer thread-safety will produce corrupted training data and silent failures that are extremely difficult to debug once a full training loop is running.

---

## Appendix — Files Touched During Phase 3

Verified to exist and contain expected content:

| File | Purpose |
|---|---|
| `crates/hexgame-core/src/mcts.rs` | T1 fixes, `sample_action`, `should_resign`, `re_root` Result |
| `crates/hexgame-core/src/eval/state.rs` | T1-7 underflow guard fix |
| `crates/hexgame-core/src/board.rs` | T1-6 per-stone validation |
| `crates/hexgame-core/src/encoder.rs` | Zero-alloc encode path |
| `crates/hexgame-core/src/threats.rs` | Threat detection |
| `crates/hexgame-core/src/search.rs` | Classical alpha-beta |
| `crates/hexgame-py/src/engine.rs` | PyMCTSEngine, PyHexGame, `classical_self_play` |
| `crates/hexgame-py/src/encode.rs` | `encode_compact_record`, `apply_d6_symmetry` |
| `crates/hexgame-py/src/buffer.rs` | Stub (Phase 4) |
| `Python/src/hexorl/inference/server.py` | GPU inference server |
| `Python/src/hexorl/inference/client.py` | Worker-side inference client |
| `Python/src/hexorl/inference/shm_queue.py` | Shared-memory queue primitives |
| `Python/src/hexorl/selfplay/worker.py` | Self-play worker + MockMCTSEngine |
| `Python/src/hexorl/selfplay/orchestrator.py` | Supervisor process |
| `Python/src/hexorl/selfplay/records.py` | Game record format |
| `Python/src/hexorl/buffer/ring.py` | Ring buffer |
| `Python/src/hexorl/buffer/sampler.py` | Recency-weighted sampler |
| `Python/src/hexorl/buffer/targets.py` | Target computation |
| `Python/src/hexorl/model/network.py` | HexNet CNN |
| `Python/src/hexorl/config/schema.py` | Pydantic config schema |
| `Python/src/hexorl/config/loader.py` | TOML config loader |
