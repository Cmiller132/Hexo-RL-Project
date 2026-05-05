# Hexagon vs. Hexo-RL-Project: Comprehensive Implementation Comparison

> **Generated:** 2026-04-26  
> **Original:** `/Users/coltonmiller/Documents/GitHub/Hexagon`  
> **Refactor:** `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project`  
> **Design Plan:** `Docs/SYSTEM_DESIGN.md`

This document compares every major subsystem between the original **Hexagon** project and its planned refactor **Hexo-RL-Project** (hereafter "Refactor"). The Refactor is currently a partial implementation following the architecture laid out in `SYSTEM_DESIGN.md`.

---

## 1. Repository Layout & Build System

### 1.1 Top-Level Structure

| Aspect | Original Hexagon | Refactor (Hexo-RL) |
|--------|------------------|-------------------|
| **Build system** | Single crate (`cdylib` + `rlib`); built with `maturin develop` | Cargo **workspace** with 4 crates (`hexgame-core`, `hexgame-py`, `hexgame-bench`, `hexgame-cli`) |
| **Rust source** | `src/*.rs` (7 files) | `crates/*/src/*.rs` + **legacy** `src/*.rs` (uncompiled mirror with drift) |
| **Python source** | `python/hexgame/` (deep nested package) | `Python/src/hexorl/` (flat, capability-based modules) + **legacy** `Python/Epoch/`, `Python/Model/`, `Python/GameRunner/` (mostly empty stubs) |
| **Root Cargo.toml** | Single package (`hexgame` v0.2.0) | Workspace manifest only (no `[package]`) |
| **pyproject.toml** | One file at root (maturin) | `crates/hexgame-py/pyproject.toml` (maturin) + `Python/pyproject.toml` (hexorl package) |
| **Configs** | `configs/*.toml` (model + training presets) | `Configs/*.toml` (unified schema: default, production, reproducible, small_test) |
| **Docs** | `docs/` (game rules, model architecture, proposals, compliance reviews) | `Docs/` (system design, architecture, perf budgets, code reviews, rust project docs) |
| **Tests** | `tests/` (pytest suite + standalone diagnostics) | `tests/` (Rust integration), `Python/tests/` (Python smoke + inference server), `crates/hexgame-core/src/tests/` (extensive Rust unit/property tests) |
| **Benchmarks** | `benchmarks/` (Python profiling scripts) | `benches/` (cross-language: `engine.rs`, `inference_latency.py`, `inference_throughput.py`) + `crates/hexgame-bench/benches/` (Criterion) |
| **Scripts** | PowerShell (`sync_logs.ps1`) | Bash (`scripts/` — bootstrap, run_epoch, profile) |
| **CI** | `.github/workflows/` (not inspected in detail) | `.github/workflows/rust.yml`, `python.yml`, `e2e.yml` |

### 1.2 Build-Time Design Difference

The Refactor's workspace split exists specifically to solve a build-time problem: in Hexagon, `cargo bench` rebuilds 30+ seconds of PyO3 every iteration because the single crate compiles both `cdylib` and `rlib` together. In the Refactor:
- `hexgame-core` is `rlib` only — fast rebuilds for benchmarks and downstream Rust consumers.
- `hexgame-py` is `cdylib` only — slow PyO3 build isolated.
- `hexgame-bench` depends only on `hexgame-core`.
- `hexgame-cli` is a standalone binary.

**Current state:** The Refactor's `src/` directory at the project root is a **legacy mirror** of `crates/hexgame-core/src/` with minor drift (e.g., extra `.clone()` calls, missing `MCTSError`). It is **not compiled** by the workspace Cargo.toml.

---

## 2. Rust Engine Differences

### 2.1 Crate Organization

| Original Hexagon | Refactor (Hexo-RL) |
|------------------|-------------------|
| `src/lib.rs` — crate root | `crates/hexgame-core/src/lib.rs` — core crate root |
| `src/core.rs` — axial coords | `crates/hexgame-core/src/core.rs` — axial coords + `Turn` canonicalization + `WindowKey` |
| `src/game.rs` — **monolithic** rules engine (1,764 lines) | Split into `board.rs` (state/rules) + `eval/` submodule + `threats.rs` + `encoder.rs` |
| `src/eval.rs` — classical eval + feature extraction | `eval/mod.rs`, `eval/grid.rs`, `eval/hot.rs`, `eval/patterns.rs`, `eval/state.rs` |
| `src/mcts.rs` — MCTS engine | `crates/hexgame-core/src/mcts.rs` (similar but cleaned up) |
| `src/search.rs` — alpha-beta | `crates/hexgame-core/src/search.rs` (similar) |
| `src/pybridge.rs` — PyO3 bindings | `crates/hexgame-py/src/engine.rs`, `encode.rs`, `buffer.rs` (split by concern) |

### 2.2 Game State (`game.rs` vs. `board.rs` + `eval/`)

**Original Hexagon (`src/game.rs`):**
- Single 1,764-line file containing `HexGameState` with ~25 fields.
- All incremental state (Zobrist, eval, threat counters, hot windows, candidate sets, window indices) lives in one struct.
- `window_indices: Vec<u16>` with size `61*61*3 = 11163` stored directly on `HexGameState`.
- `candidate_rc: FxHashMap<Hex, u32>` — reference-counted radius-2 candidate set.
- `hot_windows: [FxHashSet<(i32, i32, u8)>; 2]` — windows with 4+ stones and 0 opponent stones.
- Threat constraint methods (`compute_threat_constrained_moves`, `is_player_win_unblockable`, etc.) are methods on `HexGameState`.
- `compute_axis_influence` and `compute_tactical_targets` are methods on `HexGameState` for generating training targets.

**Refactor (`board.rs` + `eval/state.rs` + `eval/hot.rs` + `threats.rs`):**
- `HexGameState` in `board.rs` is **slimmed down** — eval/threat state is factored out.
- `EvalState` in `eval/state.rs` owns: `score`, `ThreatCounts` per player, `HotWindows`, `Box<[u16; 11163]>`, `delta_stack: Vec<EvalDelta>`.
- `HotWindows` uses `SmallVec<[WindowKey; 32]>` per player — zero-allocation inline buffer (never spills to heap in practice).
- `WindowKey(u32)` packs `(q, r, dir)` into 32 bits — used as cheap `HashSet`/`HashMap` keys without heap allocation.
- `TacticalStatus` enum in `threats.rs` cleanly models: `Quiet`, `WinningTurns`, `MustBlock(BlockConstraint)`, `Unblockable`.
- `board.rs` introduces **two** candidate sets: `candidates` (radius 2) and `placement_candidates` (radius 8) — the latter enables O(1) legality checks.
- `tactical_status()` is a **standalone function** (not a method), computed once per node and reused.

### 2.3 MCTS Engine

**Original Hexagon (`src/mcts.rs`):**
- 1,580 lines.
- `MCTSEngine` owns `game: HexGameState` directly.
- Three selectors: `Puct`, `UctVP`, `PuctV`.
- Gumbel Sequential Halving with `gumbel_sigmas`, `gumbel_candidates`, `gumbel_num_rounds`.
- `encode_board` / `encode_board_slice` monolithic encoding (13-channel 33×33 tensor).
- `extract_tree_node_states` returns `(Vec<f32>, Vec<Vec<(u8, i16, i16)>>, usize)` for RGSC training data.
- `re_root` with threat purge: if new position has active threats and `constrain_threats=true`, clears children recursively.
- Welford online variance (`m2`) on every node.

**Refactor (`crates/hexgame-core/src/mcts.rs`):**
- 1,128 lines (smaller, cleaner).
- Similar arena allocation (`Vec<MCTSNode>`), children as contiguous slices.
- Same three selectors + FPU reduction + virtual loss.
- Returns `MCTSError` (enum) instead of panicking on some error conditions — aligns with `FINALIZATION_PASS.md` §6 (no-panic re_root).
- `extract_tree_node_states` capped to 128 candidates.
- Subtree reuse (`re_root`) present.
- `sample_action` supports temperature=0 (argmax) or >0 (visit-count power).
- **Missing from refactor (as of current state):** Gumbel SH logic is not present in the explored `mcts.rs` file. The original had extensive Gumbel state fields.

### 2.4 Alpha-Beta Search

Both implementations are very similar:
- Turn-based search (atomic unit = `Turn` with 1-2 placements).
- Iterative deepening, aspiration windows, PVS, LMR, killer moves, history heuristic.
- Transposition table (`FxHashMap<u64, TTEntry>`) with mate-distance scoring.
- Quiescence search (depth 6 turns) along tactical lines only.
- Colony candidate at root.
- Noise injection for training variety.

**Minor differences:**
- Refactor's `search.rs` uses `Turn::pair(a, b)` canonicalization (ensures `a <= b`) for TT consistency.
- Refactor uses `HotWindows` + `TacticalStatus` from the factored eval module rather than inline hot window logic.

### 2.5 Evaluation & Pattern System

**Original:**
- `PATTERN_VALUES: [i32; 729]` and `PATTERN_COUNTS: [(u8, u8); 729]` are const arrays in `game.rs`.
- `eval.rs` (358 lines) provides `evaluate()`, `extract_features()`, `score_move()`, `is_forcing_move()`.
- `evaluate()` is O(1) using incremental `window_eval` and threat counters.

**Refactor:**
- `eval/patterns.rs` has the same precomputed tables but adds a **checksum test** (`0x9f5d14a209044de4`) to detect accidental table corruption.
- `eval/state.rs` (468 lines) manages incremental updates with `EvalDelta` stack for perfect `unplace` restoration.
- `assert_invariants` (debug-only) brute-force recomputes hot windows after every `unplace`.
- `extract_features` moved to `encoder.rs` but uses the same 13-feature vector.

### 2.6 PyO3 / FFI Bridge

**Original (`src/pybridge.rs` — 1,037 lines):**
- Single file exposing `PyHexGame`, `PyMCTSEngine`, and free functions.
- Heavy use of `unsafe` `std::slice::from_raw_parts` to cast `Vec<f32>`/`Vec<i32>` to byte slices for zero-copy `PyBytes`.
- `encode_board_and_legal` returns packed `PyBytes` (not numpy arrays).
- `classical_self_play` free function for bootstrap data generation.
- Thread-local XOR-shift RNG for classical search noise.

**Refactor (`crates/hexgame-py/src/`):**
- Split across `engine.rs`, `encode.rs`, `buffer.rs`.
- `engine.rs` (882 lines): `PyHexGame` + `PyMCTSEngine` + `classical_self_play`.
- Uses `py.allow_threads(|| ...)` to release GIL during all heavy computation.
- Returns **numpy arrays** (`PyArray3<f32>`, `PyArray4<f32>`) instead of packed bytes — more idiomatic PyO3 + numpy integration.
- `encode.rs`: `encode_compact_record` (replays move history → dense tensor batch) and `apply_d6_symmetry` (12 transforms).
- `buffer.rs`: Currently a **stub** (Phase 3 placeholder).

### 2.7 Board Encoding / Tensor Channels

Both use the **same 13-channel 33×33 tensor**:

| Ch | Content |
|----|---------|
| 0 | Own stones |
| 1 | Opponent stones |
| 2 | Empty mask |
| 3 | Legal moves |
| 4 | Turn phase (second placement flag) |
| 5 | First stone of turn |
| 6 | Player colour |
| 7 | Own recency |
| 8 | Opponent recency |
| 9 | Opponent hot cells |
| 10 | Own hot cells |
| 11 | Distance from centre / 16.0 |
| 12 | Opponent's last turn |

**Differences:**
- Original encodes in `mcts.rs` + `pybridge.rs` with monolithic `encode_board` function.
- Refactor encodes in `encoder.rs` with `encode_board_into` writing into caller-owned buffers to eliminate allocations.
- Both use **banker's rounding** to match Python's `round()` behavior.
- Refactor's `encoder.rs` adds `extract_features` for classical 13-element feature vectors.

---

## 3. Python Layout & Package Structure

### 3.1 Package Organization

**Original Hexagon (`python/hexgame/`):**
- Deep nesting organized by **epoch phase**: `data/`, `model/`, `training/`, `game/`, `cli/`, `ui/`, `integrations/`.
- All modules under one package `hexgame`.
- Lazy public exports in `__init__.py`.

**Refactor (`Python/src/hexorl/`):**
- Flat layout organized by **capability**: `config/`, `model/`, `inference/`, `selfplay/`, `buffer/`, `train/`, `eval/`, `epoch/`, `dashboard/`.
- Single package `hexorl`.
- Many modules are still **empty stubs** (see below).

### 3.2 File Mapping (Old → New)

| Original File | Refactor Equivalent | Status |
|---------------|---------------------|--------|
| `python/hexgame/model/network.py` | `Python/src/hexorl/model/network.py` | **Implemented** (simpler — only policy+value heads) |
| `python/hexgame/model/features.py` | Rust `encoder.rs` + `encode_compact_record` | **Moved to Rust** |
| `python/hexgame/model/loading.py` | Not yet present | **Missing** |
| `python/hexgame/model/config.py` | `Python/src/hexorl/config/schema.py` | **Replaced** by Pydantic v2 schema |
| `python/hexgame/training/config.py` | `Python/src/hexorl/config/schema.py` | **Merged** into unified Pydantic config |
| `python/hexgame/training/trainer.py` | `Python/src/hexorl/train/__init__.py` | **Empty stub** |
| `python/hexgame/training/loop.py` | `Python/src/hexorl/epoch/__init__.py` | **Empty stub** |
| `python/hexgame/training/selfplay.py` | `Python/src/hexorl/selfplay/records.py` | **Partial** (record format only; no worker/orchestrator) |
| `python/hexgame/training/buffer.py` | `Python/src/hexorl/buffer/ring.py` | **Implemented** (new design: struct-of-arrays) |
| `python/hexgame/training/regret_buffer.py` | `Python/src/hexorl/buffer/__init__.py` (stub) | **Missing** |
| `python/hexgame/training/matchmaking.py` | `Python/src/hexorl/eval/__init__.py` (stub) | **Missing** |
| `python/hexgame/game/mcts.py` | `Python/src/hexorl/inference/server.py` + `client.py` | **Replaced** by inference server architecture |
| `python/hexgame/game/arena.py` | `Python/src/hexorl/eval/__init__.py` (stub) | **Missing** |
| `python/hexgame/game/players.py` | Not yet present | **Missing** |
| `python/hexgame/game/recorder.py` | `Python/src/hexorl/selfplay/records.py` | **Partial** (compact format only) |
| `python/hexgame/game/analysis.py` | Not yet present | **Missing** |
| `python/hexgame/data/schema.py` | Not yet present | **Missing** (no SQLite ORM yet) |
| `python/hexgame/data/database.py` | Not yet present | **Missing** |
| `python/hexgame/cli/train.py` | `Python/src/hexorl/cli.py` | **Stub** (prints "not yet implemented") |
| `python/hexgame/cli/play.py` | `crates/hexgame-cli/src/main.rs` (stub) | **Missing** |
| `python/hexgame/ui/dashboard.py` | `Python/src/hexorl/dashboard/__init__.py` (stub) | **Missing** |

### 3.3 Legacy Stub Files in Refactor

The Refactor has many **empty or docstring-only** files in the old `Python/` tree (outside `src/hexorl/`):
- `Python/Model/network.py` — docstring only (describes intended KataGo-style architecture)
- `Python/Model/features.py` — 0 lines
- `Python/Model/load_model.py` — 0 lines
- `Python/Epoch/do_epoch.py` — 0 lines
- `Python/Epoch/Training/train.py` — 0 lines
- `Python/Epoch/Buffer/buffer.py` — docstring-only spec
- `Python/Epoch/Buffer/regret_buffer.py` — docstring-only spec
- `Python/Epoch/Buffer/buffer_helper.py` — docstring-only spec
- `Python/GameRunner/*.py` — all empty or near-empty

These are **pre-Phase 2 placeholders** that have been superseded by the new `hexorl` package.

---

## 4. Model Architecture Differences

### 4.1 Original Hexagon (`python/hexgame/model/network.py`)

**HexNet** is a **multi-head** residual CNN with 8 outputs:

1. `policy` — `(B, 1089)` logits
2. `value` — `(B, num_value_bins)` **categorical** value head; scalar V(s) = Σ(p_i × v_i)
3. `axis_inf` — `(B, 3, 33, 33)` per-axis line influence scores
4. `opp_policy` — `(B, 1089)` opponent next-move policy
5. `regret_rank` — `(B, 1)` RGSC ranking score
6. `regret_value` — `(B, 1)` absolute regret estimate
7. `ownership` — `(B, 1, 33, 33)` per-cell ownership prediction
8. `moves_left` — `(B, 1)` predicted moves remaining

**Blocks:**
- `ResBlock` — pre-activation residual with hex-masked 3×3 conv
- `NBTBlock` — Nested Bottleneck block (down-conv → 2× inner ResBlock → up-conv)
- `HexConv2d` — 3×3 hex convolution with optional RepVGG-linear 1×1 branch
- `GlobalPooling` — squeeze-excitation-style (mean + max + stddev → FC)
- `FixScale` — fixed affine scaling instead of GroupNorm

**Other features:**
- Supports `block_type="standard"` or `"nbt"`
- Activations: Mish or ReLU
- Norm: GroupNorm (default) or FixScale
- RepVGG-linear branches can be merged for inference via `prepare_for_inference()`
- Custom truncated normal weight init with activation-specific gains

### 4.2 Refactor (`Python/src/hexorl/model/network.py`)

**HexNet** is currently a **minimal 2-head** network (Phase 2 stub):

1. `policy_logits` — `(B, 1089)` logits
2. `value` — `(B, 1)` **scalar** tanh-bounded in `[-1, 1]`

**Architecture:**
- `conv_in`: 13 → `channels` (3×3)
- `blocks`: N pre-activation `_ResBlock`s (two 3×3 convs with ReLU, residual add)
- Policy head: `Conv2d` 1×1 → 2 filters → Linear → 1089
- Value head: `Conv2d` 1×1 → 1 filter → Linear(33×33 → 64 → 1) → tanh
- Default: 128 channels, 16 blocks
- `_init_weights`: Kaiming normal for Conv2d and Linear
- `forward_batch(x, autocast=False)`: inference-only with optional `torch.cuda.amp.autocast`
- FP16-ready via `.half()`

### 4.3 Key Model Differences

| Feature | Original Hexagon | Refactor (Current) |
|---------|------------------|-------------------|
| **Heads** | 8 heads (policy, value, axis, opp_policy, regret_rank, regret_value, ownership, moves_left) | 2 heads (policy, value) |
| **Value head** | Categorical (binned) | Scalar tanh |
| **Block types** | ResBlock + NBTBlock | Simple ResBlock only |
| **Conv type** | HexConv2d (hex-masked 3×3 + optional RepVGG 1×1) | Standard `nn.Conv2d` 3×3 |
| **Norm** | GroupNorm or FixScale | None (no normalization layers in current stub) |
| **Activation** | Mish or ReLU | ReLU only |
| **Global pooling** | Inserted every N blocks | None |
| **Inference optimization** | RepVGG merge, JIT trace, `torch.compile` | FP16 + `autocast` only |
| **Config-driven heads** | Fixed in code | Config `model.heads` list intended to be dynamic (not yet wired) |

**Note:** The Refactor's `SYSTEM_DESIGN.md` §8.2 describes adding new heads as a config change ("No Rust changes. No FFI changes."), but this flexibility is **not yet implemented** in the current model code.

---

## 5. Training Pipeline Differences

### 5.1 Original Hexagon Training Loop (`training/loop.py`)

A **fully implemented** end-to-end AlphaZero pipeline:

```
Bootstrap (classical) → Self-play (GPU MCTS) → Sparring → Classical injection →
Train on buffer → Gating → Eval → Checkpointing → Dashboard push
```

**Key characteristics:**
- **Config reload:** Re-reads `config.toml` from disk at every epoch boundary (live tuning).
- **Self-play per move:** Deep-copies model, merges RepVGG, JIT-traces for inference.
- **PCR:** Per-move cheap search mixing (e.g., 75% cheap @ 192 sims / 25% full @ 1200-2048 sims).
- **RGSC:** Restarts from `PrioritizedRegretBuffer` with probability `rgsc_lambda` after start epoch.
- **Sub-tree reuse:** Turn-level search reuses tree across the two placements of a turn.
- **Sparring:** NN vs classical engine games; adaptive learning from classical positions when NN is weak.
- **Classical injection:** Optional teacher data mixed into buffer.
- **Gating:** Pits new model vs old; reverts if win-rate < threshold (~45%).
- **Elo tracking:** Updates Elo ratings vs classical engine every epoch.
- **Checkpointing:** Saves `latest.pt`, `epoch_N.pt`, buffer snapshot, PRB state.
- **Metrics DB:** Writes `EpochMetrics` to SQLite with ~60 fields.
- **Dashboard:** Pushes live metrics via WebSocket.

### 5.2 Refactor Training Loop (Current State)

**Not yet implemented.**
- `Python/src/hexorl/epoch/__init__.py` — empty stub
- `Python/src/hexorl/train/__init__.py` — empty stub
- `Python/src/hexorl/selfplay/__init__.py` — empty stub (only `records.py` exists)

The intended pipeline from `SYSTEM_DESIGN.md`:

```
Inference server (1 GPU proc) ←→ N self-play workers (CPU Rust MCTS)
                    ↓
            Buffer process (ring buffer)
                    ↓
            Trainer process (GPU time-shared)
```

**Key design differences from original:**
- **Inference server:** Separate process with adaptive batching across workers via shared memory. Original had Python `game/mcts.py` driving GPU inference directly in the training process.
- **Process topology:** 24-30 workers + 1 inference + 1 trainer + 1 buffer + 1 orchestrator. Original ran self-play sequentially or in threads within the training process.
- **No SQLite ORM yet:** The Refactor does not yet have `data/schema.py` or SQLAlchemy integration. Original had full SQLAlchemy ORM with `TrainingRun`, `Checkpoint`, `GameRecord`, `EpochMetrics`, `GameAnalysis`, etc.
- **No dashboard yet:** Original had FastAPI + WebSocket dashboard. Refactor plans Rich-based TUI + TensorBoard.

### 5.3 Trainer / Loss Computation

**Original (`training/trainer.py`):**
- `train_on_buffer()` — multi-pass training over replay buffer.
- Losses: value (categorical CE), policy (CE with axis-head boosting `AXIS_BOOST=0.4`), opp_policy, threat (MSE), regret_rank, regret_value, ownership, moves_left, entropy regularization.
- `WeightedRandomSampler` with recency decay + policy surprise weighting.
- AMP: BF16 on Ada Lovelace+, else FP16 + GradScaler.
- Gradient clipping at norm 1.0.
- `model.zero_hex_corners()` after every step.
- Optimizer: SGD+Nesterov or Adam.

**Refactor (intended, not yet implemented):**
- `train/trainer.py`, `train/losses.py`, `train/ema.py` — all empty stubs.
- `SYSTEM_DESIGN.md` describes: AdamW, cosine LR, per-head losses, model EMA for inference.
- Loss weights in config: `policy=1.0, value=1.5, lookahead_short=0.2, lookahead_mid=0.2`.

---

## 6. Inference System Differences

### 6.1 Original Hexagon (`game/mcts.py`)

- Python function `mcts_search_batched_rust()` creates `MCTSEngine` and drives it **synchronously** in the same process.
- Pipelined MCTS loop:
  1. `select_leaves_pipeline(leaf_batch_size)` → Rust CPU select
  2. GPU inference on batched non-terminal leaves (async)
  3. `expand_prev_and_backprop()` → Rust expand + backprop
- `trace_model()` — JIT-traces `inference_forward` for MCTS; freezes graph; warms up all batch sizes.
- Root exploration: Gumbel SH or shaped Dirichlet noise.
- Axis-head boost applied to policy logits at root.

### 6.2 Refactor (`Python/src/hexorl/inference/`)

- **`server.py`** — `InferenceServer` runs in a **separate `multiprocessing.Process`**.
  - Owns GPU + model weights.
  - Adaptive batching: drains ready workers, builds batch tensor, runs `_forward()` in thread pool.
  - Uses `asyncio` event loop with `await asyncio.sleep(max_wait_us / 1e6)` when idle.
  - FP16 inference via `torch.cuda.amp.autocast`.
  - Spawn-safe: queue created in parent, child reconnects by name.
- **`client.py`** — `InferenceClient` per worker.
  - `submit(tensor, count)` — copies tensor into SHM slot, sets event, busy-waits on response.
  - Returns flattened policies (1D) for `expand_and_backprop`.
- **`shm_queue.py`** — Low-level shared memory transport.
  - `WorkerSlots` with `req_tensor`, `req_count`, `res_policy`, `res_value`, `req_ready`, `res_ready`.
  - `SharedEvent` — single-byte SHM-backed event with busy-wait + 0.1ms sleep.
  - `_create_shm` cleans up stale segments from crashes.

### 6.3 Key Inference Differences

| Aspect | Original Hexagon | Refactor |
|--------|------------------|----------|
| **Process model** | Same-process Python driver | Separate inference server process |
| **Worker communication** | Direct Python function calls | Shared memory (`multiprocessing.shared_memory`) |
| **Batching** | Fixed `leaf_batch_size` per worker | Adaptive batching across **all** workers up to `max_batch_size` |
| **GPU ownership** | Training process time-shares GPU | Inference server owns GPU; trainer uses separate CUDA stream |
| **Async framework** | PyTorch CUDA streams + Python generator | `asyncio` + thread-pool executor |
| **Model copy per game** | Deep copy + RepVGG merge + JIT trace | Single model in inference server; EMA weight updates every K steps |
| **Idle behavior** | Blocks on leaf selection | Sleeps `max_wait_us` (default 200µs) waiting for more requests |

---

## 7. Replay Buffer Differences

### 7.1 Original Hexagon (`training/buffer.py`)

**`CompactReplayBuffer`:**
- Stores per sample:
  - `move_coords` — int16 `(max_size, 256, 2)`
  - `move_counts` — uint16
  - `policies` — float16 `(max_size, 1089)` dense policy
  - `opp_policies` — float16 `(max_size, 1089)`
  - `values` — float32
  - `regrets` — float32
  - `has_opp_policy`, `is_full_search`, `has_regret` — bool flags
  - `epochs` — int32 (for recency weighting)
  - `policy_surprise` — float32
- Memory: ~1 GB for 200k samples.
- `HexDataset` replays `move_history` through `HexGame` on-the-fly, calls Rust `encode_board_and_legal()` and `axis_influence()` for fresh tensors.
- Computes ownership and moves-left targets dynamically.
- D6 augmentation via precomputed numpy LUTs.
- `WeightedRandomSampler` with recency decay + policy surprise weighting.
- Serialization: versioned numpy save/load (v7 format) with backward compatibility.

**`PrioritizedRegretBuffer` (`training/regret_buffer.py`):**
- Fixed-capacity buffer (default 100).
- `try_insert(state)` — inserts if regret > current minimum.
- `sample()` — samples with P(s_i) ∝ R(s_i)^(1/τ), τ=0.1.
- `update_regret()` — EMA update.
- `remove_stale()` — evicts states older than N epochs.

### 7.2 Refactor (`Python/src/hexorl/buffer/ring.py`)

**`RingBuffer`:**
- Struct-of-arrays layout:
  - `_histories`: list of `bytes` (compact move history)
  - `_policies` / `_policy_probs`: `(capacity, max_policy_entries)` uint16 / f32 — **sparse** policy (top-K only)
  - `_policy_counts`: uint16
  - `_values`: f32
  - `_game_ids`: uint32
  - `_is_full`: bool (full-search flag)
  - `_players`: uint8
- Oldest-first eviction when full.
- `sample_indices()` — recency-biased + quality-gated:
  - `weight ∝ decay^(max_game_id - game_id) × (4.0 if full-search else pcr_weight)`
- Thread-safe via `threading.Lock`.
- No dense tensor storage — relies on Rust `encode_compact_record` for decode.

**`targets.py`:**
- `compute_value_targets()` — simple outcome assignment (Phase 3).
- `compute_ema_lookahead()` — KataGo-style backward EMA over future turn boundaries.
- `compute_policy_targets()` — converts dense MCTS visits to sparse top-K policy.
- `process_game_record()` — full pipeline: outcomes → EMA lookahead → sparse policy.

**`records.py`:**
- `PositionRecord` dataclass with `move_history: bytes`, sparse `policy_target: Dict[int, float]`, `root_value`, `lookahead_values`, etc.
- `GameRecord` with binary serialization (`to_compact_bytes()` / `from_compact_bytes()`).
- Compact format: header + per-position variable-length records.

### 7.3 Key Buffer Differences

| Feature | Original Hexagon | Refactor |
|---------|------------------|----------|
| **Policy storage** | Dense float16 `(1089,)` | Sparse top-K (default 20 entries) |
| **Board tensor storage** | None (on-the-fly re-encoding) | None (on-the-fly Rust decode) |
| **Opp policy storage** | Dense float16 `(1089,)` | Not present in current `RingBuffer` |
| **Regret fields** | `regrets`, `has_regret` bool | Not present yet |
| **Ownership/moves-left** | Computed dynamically in `HexDataset` | Not present yet |
| **Axis influence targets** | Computed dynamically via Rust `axis_influence()` | Not present yet |
| **Recency weighting** | `WeightedRandomSampler` with decay | `sample_indices()` with explicit weight computation |
| **Quality gating** | `is_full_search` flag + policy surprise weight | `is_full` bool + `pcr_weight` scalar |
| **Regret buffer** | `PrioritizedRegretBuffer` (100 capacity, τ=0.1) | **Missing** (stub only) |
| **Buffer capacity** | 80k-200k typical | 2M in design, 10k in `small_test.toml` |
| **Serialization** | Numpy `.npy` blob (v7 format) | Custom binary `to_compact_bytes()` |
| **Database integration** | SQLite `LargeBinary` for games | Not yet implemented |

---

## 8. Self-Play & MCTS Driver Differences

### 8.1 Original Hexagon (`training/selfplay.py` + `game/mcts.py`)

- `self_play_game()` — plays one game with NN + MCTS.
  - Supports PRB restart: with probability `rgsc_lambda`, replays from a high-regret state.
  - Per-move PCR: independently rolls cheap vs full search per move.
  - Turn-level subtree reuse option.
  - Collects trajectory snapshots with policy targets, MCTS values, selected-action Q-values.
  - Computes trajectory regret via squared-error suffix means.
  - Scores trajectory states and tree node states using `raw_model` (untraced, full heads).
- `mcts_search_turn_rust()` — turn-level search with subtree reuse via `engine.re_root()`.
- `select_move()` — samples move with temperature annealing.
- `build_training_policy_target()` — prunes policy to top-prob/64 threshold, renormalizes.

### 8.2 Refactor (Intended, Not Fully Implemented)

- `Python/src/hexorl/selfplay/records.py` — only record format exists.
- `SYSTEM_DESIGN.md` §6 describes:
  - Workers as separate `multiprocessing.Process`es.
  - Each worker owns one `MCTSEngine`, one `HexGameState`.
  - Inner loop: `run_until_inference_needed()` → `submit_inference()` (via `InferenceClient`) → backprop.
  - `sample_action()` in Rust for temperature sampling.
  - Crash recovery: orchestrator monitors `process.is_alive()`, respawns on crash.
  - Per-game config reload (allows mid-run hyperparameter changes).

### 8.3 Key Self-Play Differences

| Feature | Original Hexagon | Refactor (Design) |
|---------|------------------|-------------------|
| **Worker model** | Sequential or threaded within training process | Separate `multiprocessing.Process` per worker |
| **Inference coupling** | Python driver directly calls GPU in same process | Workers submit to separate inference server via SHM |
| **Subtree reuse** | Turn-level via `re_root()` | Turn-level via `re_root()` (same) |
| **RGSC** | `PrioritizedRegretBuffer` with trajectory regret scoring | Regret priority subset in buffer (design only) |
| **PCR** | Per-move roll: cheap vs full search | Same concept; `pcr_low_sim_prob` in config |
| **Crash recovery** | None (single process) | Orchestrator respawns workers; crash log in SHM slot |
| **Config reload** | Per-epoch boundary | Per-game (allows mid-run changes) |
| **Resignation** | Configurable threshold | Configurable threshold + `resign_disable_prob` |

---

## 9. Configuration System Differences

### 9.1 Original Hexagon (`training/config.py` + TOML files)

- **`TrainingConfig`** dataclass with 80+ fields.
- Sections: self-play, bootstrap, sparring, training, loss weights, RGSC, PCR, model architecture, infrastructure.
- Config precedence at startup:
  1. `--config` TOML file
  2. `checkpoint_dir/config.toml` (single source of truth for resume)
  3. Checkpoint `.pt` embedded cfg
  4. `TrainingConfig` bare defaults
- Protected fields (architecture, optimizer type, etc.) cannot change mid-run.
- Dashboard can atomically update `config.toml` (temp file + rename).
- Multiple model configs: `model_default.toml`, `model_nbt_fast.toml`, `model_nbt_quality.toml`, etc.

### 9.2 Refactor (`Python/src/hexorl/config/schema.py` + TOML files)

- **Pydantic v2** schema with nested sections: `RunConfig`, `ModelConfig`, `SelfPlayConfig`, `InferenceConfig`, `BufferConfig`, `TrainConfig`.
- Single unified `Config(BaseModel)` object.
- `load_config(path)` uses `tomllib` (or `tomli` for <3.11) + `Config.model_validate(raw)`.
- Default TOML: `Configs/default.toml`.
- Production: `Configs/production.toml`.
- Reproducible: `Configs/reproducible.toml` (`deterministic = true`, `seed = 42`).
- Small test: `Configs/small_test.toml` (minimal for CI).

### 9.3 Config Schema Comparison

| Section | Original Fields | Refactor Fields |
|---------|----------------|-----------------|
| **Run** | Not explicit | `seed`, `output_dir`, `log_level`, `deterministic` |
| **Model** | `num_res_blocks`, `channels`, `block_type`, `bottleneck_channels`, `activation`, `use_repvgg_linear`, `norm_kind` | `channels`, `blocks`, `heads` (list of strings) |
| **Self-play** | `mcts_sims`, `games_per_epoch`, `states_per_epoch`, `c_puct`, `temperature_schedule`, `root_exploration_mode`, `leaf_batch_size`, `subtree_reuse` | `num_workers`, `games_per_epoch`, `states_per_epoch`, `batch_size_per_worker`, `mcts_simulations`, `c_puct`, `c_puct_init`, `temperature_schedule`, `dirichlet_alpha`, `dirichlet_fraction`, `pcr_low_sim_prob`, `pcr_low_sims`, `resign_threshold`, `resign_disable_prob`, `near_radius`, `constrain_threats` |
| **Inference** | Implicit in training config | `max_batch_size`, `max_wait_us`, `fp16`, `ema_update_every` |
| **Buffer** | `replay_buffer_size`, `min_buffer_for_training` | `capacity`, `recency_decay`, `pcr_weight`, `regret_fraction`, `lookahead_horizons`, `lookahead_lambdas` |
| **Train** | `lr`, `lr_schedule`, `batch_size`, `train_epochs`, `optimizer_type`, `use_amp`, `use_compile` | `batch_size`, `batches_per_epoch`, `optimizer`, `lr_schedule`, `peak_lr`, `weight_decay`, `loss_weights` |
| **RGSC** | `rgsc_lambda`, `rgsc_buffer_capacity`, `rgsc_start_epoch` | `regret_fraction` only (in BufferConfig) |
| **Loss weights** | value, policy, threat, opp_policy, regret_rank, regret_value, ownership, moves_left, entropy | policy, value, lookahead_short, lookahead_mid |

**Key difference:** The Refactor uses a **capability-driven config** (inference has its own section, buffer has its own section) while the Original uses a **phase-driven config** (training section covers everything). The Refactor's `model.heads` list is intended to make adding heads a config change rather than a code change.

---

## 10. Data Storage & Persistence Differences

### 10.1 Original Hexagon

Full **SQLite + filesystem** persistence:

| Layer | Technology | Location |
|-------|------------|----------|
| Games (compact) | SQLite `LargeBinary` | `data/hexgame.db` |
| Games (JSON) | Filesystem | `games/game_{timestamp}.json` |
| Metrics | SQLite (SQLAlchemy ORM) | `data/hexgame.db` |
| Checkpoints | PyTorch `.pt` files | `checkpoints/latest.pt`, `epoch_N.pt` |
| Replay Buffer | Numpy `.npy` blob | `checkpoints/latest_buffer.pt` |
| Elo Ratings | JSON | `checkpoints/elo_ratings.json` |
| Config | TOML | `checkpoints/config.toml` |
| Logs | Text files | `training_log_*.txt` |

**SQLAlchemy ORM tables:**
- `TrainingRun` — run metadata
- `Checkpoint` — checkpoint records with metrics
- `GameRecord` — compact binary move data + metadata
- `EpochMetrics` — ~60 fields of per-epoch loss/stats/timing
- `BufferSnapshot` — buffer file references
- `GameAnalysis` — analysis results (blunders, forks, accuracy)
- `CanonicalOpening` — D6-normalized opening prefixes

### 10.2 Refactor (Current State)

**No persistent data layer yet.**
- No SQLite, no SQLAlchemy, no ORM.
- No `data/` directory.
- Checkpoints would be saved to `output_dir` (from config) but no loading/saving code exists yet.
- `RingBuffer` is RAM-only with no disk persistence.
- `GameRecord.to_compact_bytes()` provides a binary format but no database integration.

**Intended (from `SYSTEM_DESIGN.md`):**
- Buffer process owns continuous RAM ring buffer.
- Trainer mmap's buffer slices.
- Optional game record retention to SQLite/compressed files for offline analysis.

---

## 11. Dashboard & UI Differences

### 11.1 Original Hexagon (`python/hexgame/ui/dashboard.py`)

- **FastAPI + WebSocket** backend (`hexgame-dashboard`).
- Serves `static/dashboard.html` (vanilla HTML/JS frontend).
- Model cache: LRU cache of loaded JIT-traced models (max 3) with ref counting.
- Game sessions: interactive play with 1-hour TTL.
- Arena matches: background threads running NN vs NN/Classical with live WebSocket broadcast.
- Training control: start/stop/pause/resume training via subprocess management.
- Auto-resume: checks if training process died and relaunches.
- Config management: atomic TOML updates with validation.
- REST endpoints for runs, metrics, games, checkpoints, training control.
- **No authentication** (local dev only).

### 11.2 Refactor (Current State)

- `Python/src/hexorl/dashboard/__init__.py` — **empty stub**.
- `SYSTEM_DESIGN.md` §Phase 6 describes:
  - `dashboard/tui.py` — Rich-based live monitoring (terminal UI).
  - `dashboard/tb.py` — TensorBoard writer.
  - No web dashboard planned; TUI replaces FastAPI.

**Key difference:** Original has a full **web dashboard** (FastAPI + HTML/JS). Refactor plans a **terminal UI** (Rich TUI) + TensorBoard.

---

## 12. CLI Differences

### 12.1 Original Hexagon (`python/hexgame/cli/`)

Multiple CLI entry points registered in `pyproject.toml`:
- `hexgame-train` — `cli/train.py`
- `hexgame-evaluate` — `cli/evaluate.py`
- `hexgame-play` — `cli/play.py` (HTTP server for interactive play)
- `hexgame-axis` — `cli/axis.py`
- `hexgame-viewer` — `cli/viewer.py`
- `hexgame-sealbot` — `cli/sealbot.py`
- `hexgame-migrate` — `cli/migrate.py`

`train.py` supports:
- `--epochs`, `--lr`, `--sims`, `--games`, `--batch-size`, `--checkpoint-dir`, `--dashboard`, `--use-compile`, `--config`
- Config resolution hierarchy (explicit → checkpoint dir → checkpoint embedded → defaults)
- Resume validation (architecture fields must match)

### 12.2 Refactor (`Python/src/hexorl/cli.py` + `crates/hexgame-cli/`)

- `Python/src/hexorl/cli.py` — **stub only**. Argparse with subcommands `epoch`, `bench`, `arena`. Prints "Phase 1 stub (not yet implemented)".
- `crates/hexgame-cli/src/main.rs` — Rust CLI stub with three subcommands:
  - `play` — unimplemented
  - `bench` — runs 10 MCTS self-play games (50 sims each, uniform policy)
  - `perft` — counts depth-1 legal moves from initial position

**Key difference:** Original has a **rich, fully functional Python CLI** with multiple commands. Refactor has only **stubs** on both sides.

---

## 13. Testing Differences

### 13.1 Original Hexagon

- `tests/test_*.py` — standard pytest modules.
- `tests/standalone/test_*.py` — isolated diagnostics (not in default suite).
- `tests/helpers.py` — `seed_everything()`, `tiny_cfg(tmp_path)`.
- Tests cover: encoding pipeline, MCTS exploration refactor, model+MCTS integration, regret buffer, training pipeline, training capabilities, subtree reuse, dashboard config source of truth.
- Uses `ConstantModel` (uniform policy, neutral value) to avoid GPU dependency.

### 13.2 Refactor

**Rust tests (extensive):**
- `crates/hexgame-core/src/tests/` — unit tests:
  - `core` — Hex distance, ordering, hashing, WindowKey
  - `eval_state` — place/unplace round-trip, score consistency, hot windows
  - `grid` — win-grid index bijection
  - `hot` — HotWindows insert/remove/clear
  - `mcts` — determinism, re-root visit preservation, root Q boundedness
  - `oracle` — brute-force verifier (winning singles, blocking, unblockable)
  - `patterns` — ternary round-trip, checksum, incremental vs brute-force equality
  - `threats_internal` — exact threat status semantics
  - `threats` — **property-based tests (proptest)** comparing fast path against oracle
- `crates/hexgame-core/tests/board.rs` — integration tests for opening rules, win detection, Zobrist, proptest place/unplace identity.
- `crates/hexgame-core/tests/encoder.rs` — integration tests for feature extraction, proptest encode output range.
- `tests/board.rs` (root) — additional board tests.

**Python tests:**
- `Python/tests/test_engine_smoke.py` — smoke tests for Rust extension (constants, basic game, encode shape, MCTS completion).
- `Python/tests/test_inference_server.py` — end-to-end integration tests for `InferenceServer` + `InferenceClient`, including MCTS round-trip with real engine.

### 13.3 Key Testing Differences

| Aspect | Original Hexagon | Refactor |
|--------|------------------|----------|
| **Rust test tiers** | Basic unit tests | Smoke (10 cases), Medium (25 cases), Full (500 cases × 2 seeds, `#[ignore]`) |
| **Property-based testing** | None in Rust | `proptest` in `threats.rs` and `encoder.rs` |
| **Oracle verification** | None | Brute-force oracle in `tests/oracle.rs` bidirectionally verifies threat status |
| **Invariant checking** | None | `assert_invariants` debug-only brute-force recompute after every `unplace` |
| **Checksums** | None | `PATTERN_VALUES` FNV-1a checksum to catch table corruption |
| **Python tests** | ~8 test modules covering full pipeline | 2 test modules (engine smoke, inference server) |
| **CI** | Not inspected in detail | `rust.yml` (cargo test, clippy, bench), `python.yml` (pytest, ruff, mypy), `e2e.yml` (weekly full epoch) |

---

## 14. Benchmarking Differences

### 14.1 Original Hexagon (`benchmarks/`)

Python profiling scripts:
- `bench_2400sims.py` — MCTS with 2400 sims
- `bench_compare.py` — comparison benchmarks
- `bench_compile_test.py` — `torch.compile` testing
- `bench_continuation.py` — continuation path profiling
- `bench_mcts.py` / `bench_mcts_profile.py` — MCTS profiling
- `bench_nbt_profile.py` — NBT block profiling
- `bench_optimizations.py` — optimization comparison
- `bench_optimized.py` — optimized path
- `bench_phase3_5.py` — phase 3-5 profiling
- `bench_selfplay.py` / `bench_selfplay_profile.py` — self-play throughput
- `bench_step.py` — single step profiling
- `profile_continuation.txt`, `profile_phase3_5.txt`, `profile_results.txt`, `profile_results_v2.txt` — archived profiles

### 14.2 Refactor (`benches/` + `crates/hexgame-bench/benches/`)

**Cross-language benchmarks:**
- `benches/engine.rs` — Rust criterion benchmark
- `benches/inference_latency.py` — GPU batching characterization
- `benches/inference_throughput.py` — positions/sec

**Rust criterion benchmarks:**
- `crates/hexgame-bench/benches/encode.rs` — `encode_board_into` + `legal_moves_near`
- `crates/hexgame-bench/benches/mcts.rs` — full MCTS simulation loop (10 sims, uniform mock)
- `crates/hexgame-bench/benches/threats.rs` — `threat_status` on 20-stone position

**Performance budgets** (from `SYSTEM_DESIGN.md` §9.3):
- `MCTSEngine::run_until_inference_needed()` → < 200 µs at sim_count=800
- `encode_board_into()` → < 5 µs
- `apply_d6_symmetry()` → < 2 µs
- Inference dispatcher request→response → < 1 ms p50, < 5 ms p99
- Buffer decode 256 samples → < 10 ms

**Key difference:** Original benchmarks are **ad-hoc Python scripts** for profiling specific training runs. Refactor benchmarks are **structured Criterion + Python harnesses** with explicit pass/fail budgets enforced in CI.

---

## 15. MCTS Algorithm Differences

### 15.1 Selectors

Both support:
- **PUCT** — standard AlphaZero with dynamic `c_puct + ln((N + c_puct_init) / c_puct_init)`
- **UCT-V-P** — Weichart 2026 variance-aware with `O(√log N)` exploration
- **PUCT-V** — Heuristic variance-aware with `O(√N)` scaling

### 15.2 Root Exploration

**Original:**
- Three modes: `gumbel`, `dirichlet`, `none`.
- Gumbel Sequential Halving: samples Gumbel(0,1) per root child, computes `sigma = gumbel + ln(prior)`, selects top-m candidates, allocates simulations in rounds, halves by Q-value.
- Dirichlet: shaped Dirichlet noise applied to root priors.

**Refactor (Current):**
- `add_dirichlet_noise()` present in `PyMCTSEngine`.
- Gumbel SH **not present** in explored `mcts.rs`.
- `SYSTEM_DESIGN.md` does not mention Gumbel; it specifies Dirichlet noise in config.

### 15.3 Virtual Loss & FPU

Both:
- `VIRTUAL_LOSS_VISITS = 1` added to search path during batch selection.
- FPU reduction: unvisited children get `Q = parent_Q - 0.2`.

### 15.4 Training Data Extraction

**Original:**
- `extract_tree_node_states(min_visits)` — DFS of expanded nodes with ≥`min_visits`, returns board tensors + move histories for RGSC candidate scoring.
- Python `game/mcts.py` extracts tree node states and passes them to `raw_model` for regret scoring.

**Refactor:**
- `extract_tree_node_states` capped to 128 candidates.
- No RGSC wiring yet in Python.

---

## 16. Classical Search Differences

Both use the same underlying alpha-beta engine with minor variations:
- Turn-based search (atomic unit = `Turn`).
- Iterative deepening, aspiration windows, PVS, LMR, killer moves, history heuristic.
- Transposition table with mate-distance scoring.
- Quiescence search along tactical lines.

**Original-specific:**
- `classical_search()` and `classical_search_turn()` exposed via `pybridge.rs`.
- `classical_self_play()` free function for bulk bootstrap data generation.
- Noise injection via thread-local XOR-shift RNG.

**Refactor-specific:**
- `classical_search` and `classical_search_turn` present in `engine.rs`.
- `classical_self_play` present.
- Same noise mechanism.

---

## 17. Documentation Differences

### 17.1 Original Hexagon (`docs/`)

- `docs/game.md` — Game rules and strategy implications.
- `docs/model.md` — Neural architecture, data flow, MCTS integration, training loop.
- `docs/project_structure.md` — Repository layout, build system, CLI reference, config schema, checkpoint conventions, development workflow.
- `docs/proposal_mcts_exploration_refactor.md` — Design rationale for variance-aware selectors and Dirichlet noise.
- `docs/proposal_rgsc_mcts_review.md` — RGSC algorithm review and faithfulness analysis.
- `docs/rgsc_paper_compliance_review.md` — Detailed code-vs-paper compliance audit.
- `docs/profiling_report.md` — Performance measurements and optimization recommendations.
- `AGENTS.md` — Comprehensive agent guidance (build commands, test commands, conventions, security).
- `.forge/plan.md` — Active implementation plans.

### 17.2 Refactor (`Docs/`)

- `Docs/SYSTEM_DESIGN.md` — **Central architecture document** (699 lines): inference server, process topology, shared memory, adaptive batching, buffer design, config schema, migration path, performance budgets, hardware tuning.
- `Docs/ARCHITECTURE.md` — Diagrams + decision log (referenced, not inspected).
- `Docs/PERF_BUDGETS.md` — Measured perf targets per component (referenced, not inspected).
- `Docs/HISTORY.md` — Archived old review docs.
- `Docs/CODE_REVIEW_RUST.md` — Rust code review.
- `Docs/FINALIZATION_PASS.md` — Finalization checklist (includes 6 blocker fixes for Phase 1).
- `Docs/RUST_PROJECT.md` — Rust project documentation.
- `Docs/game.md` — Game rules.
- `Docs/2602.20809v1.txt` — RGSC paper text.
- `AGENTS.md` — Empty in root (no instructions).

**Key difference:** Original docs are **retrospective and operational** (how the system works, how to use it). Refactor docs are **prospective and architectural** (how the system *will* work, migration path, design rationale). The Refactor's `SYSTEM_DESIGN.md` is the single most important document — it is a comprehensive blueprint for the entire system.

---

## 18. Security & Deployment Differences

### 18.1 Original Hexagon

- FastAPI dashboard runs local HTTP/WebSocket with **no authentication**.
- Dashboard can spawn training subprocesses.
- SQLite DB is world-readable by default.
- No network ingress beyond localhost by default.
- PowerShell script (`sync_logs.ps1`) for log synchronization.

### 18.2 Refactor

- No web dashboard (plans TUI instead), so no HTTP server security concerns yet.
- Shared memory queue cleanup handles stale segments from crashes.
- Process crash recovery designed into architecture (orchestrator monitors workers).
- Bash scripts (`scripts/bootstrap.sh`, `run_epoch.sh`, `profile.sh`) instead of PowerShell.
- No SQLite DB yet.

---

## 19. Summary of Completeness

### 19.1 What Exists in Original Hexagon (Fully Implemented)

| Subsystem | Status |
|-----------|--------|
| Rust engine (game rules, eval, search, MCTS, PyO3) | ✅ Complete |
| Multi-head neural network (8 heads, NBT blocks, RepVGG) | ✅ Complete |
| Training loop (bootstrap, self-play, sparring, train, gating, eval) | ✅ Complete |
| Replay buffer (compact, on-the-fly encoding, recency weighting) | ✅ Complete |
| Regret buffer (RGSC prioritized restart) | ✅ Complete |
| MCTS driver (batched GPU inference, Gumbel SH, subtree reuse) | ✅ Complete |
| FastAPI dashboard (WebSocket live metrics, arena, game browser) | ✅ Complete |
| SQLite ORM (training runs, checkpoints, games, metrics, analysis) | ✅ Complete |
| CLI (train, evaluate, play, viewer, sealbot, migrate) | ✅ Complete |
| Elo tracking | ✅ Complete |
| Game analysis (blunder detection, accuracy, turn classification) | ✅ Complete |
| Config system (live reload, protected fields, multiple presets) | ✅ Complete |

### 19.2 What Exists in Refactor (Current State)

| Subsystem | Status |
|-----------|--------|
| Rust workspace (core, py, bench, cli) | ✅ Complete |
| Rust engine (board, eval, threats, search, MCTS, encoder) | ✅ Complete |
| PyO3 bridge (HexGame, MCTSEngine, encode, classical self-play) | ✅ Complete |
| Minimal neural network (2 heads, ResBlocks) | ✅ Implemented |
| Inference server (adaptive batching, SHM, asyncio) | ✅ Implemented |
| Inference client (worker-side SHM submit) | ✅ Implemented |
| Ring buffer (struct-of-arrays, recency + quality sampling) | ✅ Implemented |
| Record format (compact binary serialization) | ✅ Implemented |
| Target computation (outcome, EMA lookahead, sparse policy) | ✅ Implemented |
| Config schema (Pydantic v2, unified TOML) | ✅ Implemented |
| Rust CLI stub (bench, perft) | ✅ Stub |
| Python CLI stub | 🟡 Stub only |
| Self-play worker / orchestrator | ❌ Missing |
| Trainer loop / losses / EMA | ❌ Missing |
| Epoch orchestrator | ❌ Missing |
| Evaluation / arena / Elo | ❌ Missing |
| Dashboard (TUI / TensorBoard) | ❌ Missing |
| SQLite / ORM / persistence | ❌ Missing |
| Multi-head model (axis, opp_policy, regret, ownership, moves_left) | ❌ Missing |
| Regret buffer (RGSC) | ❌ Missing |
| Game analysis | ❌ Missing |

### 19.3 Migration Phases (from `SYSTEM_DESIGN.md` §11)

| Phase | Duration | Work | Status |
|-------|----------|------|--------|
| **Phase 1 — Foundation** | 1-2 weeks | Workspace split, 5 FFI entry points, 6 blocker fixes, benchmarks | ✅ **Mostly done** |
| **Phase 2 — Inference Server** | 1-2 weeks | SHM queues, adaptive batching, stub model, benchmark >30k pos/sec | ✅ **Done** |
| **Phase 3 — Self-Play & Buffer** | 2 weeks | Workers, orchestrator, buffer, Rust encode kernels, 1k-game test | 🟡 **Partial** (buffer done, workers missing) |
| **Phase 4 — Training Loop** | 1 week | Full model heads, trainer, losses, EMA, small_test epoch | ❌ **Not started** |
| **Phase 5 — Evaluation & Stability** | 1 week | Arena, Elo, classical opponent, 10-epoch stability | ❌ **Not started** |
| **Phase 6 — Polish** | Ongoing | Auto-tuning, dashboard, long-run benchmarking, bootstrap data | ❌ **Not started** |

---

## 20. Conclusion

The **original Hexagon** is a **mature, fully operational** AlphaZero-style training system with a rich feature set: multi-head neural networks, Gumbel Sequential Halving, RGSC, comprehensive SQLite persistence, a FastAPI web dashboard, and extensive CLI tooling. Its architecture is monolithic: single Rust crate, single Python package, single-process training driver.

The **Refactor (Hexo-RL-Project)** is a **clean-slate architectural redesign** focused on:
1. **Process isolation** — inference server, self-play workers, trainer, and buffer run as separate processes communicating via shared memory.
2. **Workspace modularity** — Rust split into core / py / bench / cli for faster builds and clearer boundaries.
3. **Capability-based Python layout** — flat `hexorl` package organized by subsystem rather than by training phase.
4. **Performance budgets** — explicit latency/throughput targets enforced from day one via Criterion + pytest-benchmark.
5. **Config-driven extensibility** — adding network heads or MCTS variants should be config changes, not refactors.

The Refactor currently has a **solid foundation** (Rust engine, inference server, buffer infrastructure, config system) but is **missing the upper layers** of the training pipeline (self-play orchestrator, trainer, evaluation, dashboard, persistence). These are explicitly planned in `SYSTEM_DESIGN.md` Phases 3-6.

The most consequential architectural shift is the **inference batching server**: in Hexagon, GPU inference is driven synchronously by Python in the training process. In the Refactor, a dedicated process owns the GPU and amortizes forward passes across 24-30 worker processes via shared memory. This is the same design used by KataGo and is intended to achieve much higher GPU saturation on consumer hardware (RTX 4070 Ti).
