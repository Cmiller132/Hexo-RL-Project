# System Design — Hexo-RL Training Pipeline

**Hardware target:** RTX 4070 Ti (12 GB VRAM), AMD 7950X (16C / 32T), 32 GB DDR5.
**Goal:** A maintainable, modular AlphaZero/KataGo-style training system that saturates both the GPU (inference) and CPU (self-play tree search) with minimum tail-latency and maximum future flexibility.

This document is a **design plan**, not an implementation. It defines the architecture, component boundaries, data formats, and migration path. Code lives in subsequent implementation tickets.

---

## 1. Executive Summary

The system is an **inference-server pipeline**:

```
┌────────────────────────────┐                   ┌────────────────────────┐
│  N self-play worker procs  │  inference req    │   GPU inference proc   │
│  (CPU-bound; Rust MCTS)    │ ────────────────► │   (one process, FP16,  │
│  Each owns 1 MCTSEngine    │ ◄──────────────── │    batches across all  │
│  + 1 HexGameState          │   (policy, value) │    workers)            │
└────────────────────────────┘                   └────────────────────────┘
            │ finished games
            ▼
┌────────────────────────────┐                   ┌────────────────────────┐
│  Buffer process            │  training chunks  │   Trainer process      │
│  Continuous RAM ring buf   │ ────────────────► │   (one process, GPU    │
│  Recency-weighted sampler  │                   │    time-shared with    │
│                            │                   │    inference)          │
└────────────────────────────┘                   └────────────────────────┘
```

The **central architectural decision** is to decouple CPU work (tree search, game state) from GPU work (NN inference) via a **batching inference server**. Workers do not own the GPU; they call into a queue. This allows N workers to amortize one GPU forward pass.

The **secondary architectural decision** is that the Rust engine stays single-threaded per instance. We get parallelism by running many engines, not by sharing a tree across threads. This avoids the complexity of lock-free arenas and matches the design of every successful AlphaZero descendant (Leela, KataGo, ELF).

The **third architectural decision** is workspace decomposition: the Rust crate splits into `core` (rlib only, fast to rebuild) and `py` (cdylib, slow to rebuild) so that benchmarks and downstream Rust consumers don't pay for PyO3 every time.

---

## 2. Architectural Principles

These principles drive every concrete decision below. When a future change conflicts with one, that's a signal to stop and re-evaluate.

1. **One owner per resource.** GPU has one owner (inference server). Each game state has one owner (its worker). The buffer has one owner (buffer process). No shared mutable state across processes.

2. **Batching at every boundary.** GPU calls are batched. Buffer writes are batched. Training reads are batched. The boundaries between processes are the slowest part of the system; the only mitigation is amortization.

3. **Compact representations on the wire, dense in the pipeline.** Move histories cross process boundaries. Tensors are reconstructed at the consumer. This trades CPU for memory bandwidth, which is correct on a 32-thread machine.

4. **No heap allocations on hot paths.** Both Rust and Python. Pre-allocated buffers, ring queues, fixed-size arrays. `cargo bench` and Python micro-benchmarks enforce this.

5. **Determinism is opt-in.** Production runs are non-deterministic for performance; reproduction runs use a master seed propagated through every RNG. Every randomness source is plumbed.

6. **Flexibility through configuration, not through code.** Hyperparameters live in `config.toml`. Module structure stays stable across experiments. Adding a new head to the network or a new sampling strategy should be a config change, not a refactor.

7. **Performance is benchmarked from day one.** Every PR runs `cargo bench` and a Python end-to-end throughput test in CI. Performance regressions block merges.

8. **Crash safety.** A worker dying must not corrupt the buffer or kill the trainer. A trainer crash must not lose buffered games. Use process supervision (systemd, supervisord, or a simple Python supervisor).

---

## 3. The Python ↔ Rust Interaction Model

### 3.1 What Goes Where

| Concern | Layer | Why |
|---|---|---|
| Game rules, move generation, win detection | **Rust** | Hot path, called millions of times per second |
| MCTS tree, PUCT, virtual loss, backprop | **Rust** | Hot path; arena allocator is fast in Rust |
| Tensor encoding (13-channel) | **Rust** | Called once per leaf; needs zero-alloc |
| Threat analysis, classical eval | **Rust** | Hot path during search |
| NN forward pass | **Python (PyTorch)** | Mature ecosystem, good GPU drivers |
| NN training (loss, backprop, optimizer) | **Python (PyTorch)** | No reason to do this in Rust |
| Replay buffer (storage + sampling) | **Python with Rust kernels** | Storage is Python; recency-weighted sampling and compact-to-dense decode are Rust |
| Self-play orchestration (worker pool, supervision) | **Python** | Process management is Python's strength |
| Inference batching dispatcher | **Python** | Coordinates GPU; no hot inner loop |
| Configuration, logging, dashboards | **Python** | Iterate fast, no perf reason for Rust |

**Rule of thumb:** if it's called >1000 times/sec, it's Rust. If it's called <100 times/sec, it's Python. The middle ground (NN forward pass) goes to Python because PyTorch's CUDA kernels are state of the art.

### 3.2 The Five FFI Entry Points

The PyO3 surface area is deliberately tiny. Five APIs cover everything:

1. **`MCTSEngine.run_until_inference_needed() -> InferenceRequest | TerminalResult`**
   The worker calls this in a tight loop. It runs MCTS as far as it can without an NN call, returns either:
   - `InferenceRequest` — a batch of board tensors (zero-copy numpy view) waiting for `policies, values`
   - `TerminalResult` — a finished game; the worker pulls the game record and resets

2. **`MCTSEngine.submit_inference(policies, values)`**
   Resumes the engine after the inference server returns. Backprops, applies virtual loss, and either selects the next action or continues batching.

3. **`MCTSEngine.sample_action(temperature, seed) -> (q, r)`**
   Pure Rust temperature sampling at the root. Returns the chosen move.

4. **`encode_compact_record(history_bytes) -> ndarray[N, 13, 33, 33]`**
   Decodes a stored compact game record into dense tensors for training. Called by the dataloader; runs in Rust to be fast and to avoid duplicating the encoder logic in Python.

5. **`apply_d6_symmetry(tensor, sym_idx) -> ndarray`**
   In-place hex-grid symmetry transform for data augmentation. Twelve transforms (6 rotations × 2 reflections). Rust because it's tight loops over 13×33×33.

That's the whole API. Everything else (game state queries for debug, config, etc.) is non-hot-path and can go through whatever PyO3 surface is convenient.

### 3.3 Why Not gRPC / Shared Memory / IPC

Inside one process, PyO3 with `py.allow_threads()` is the fastest option:
- No serialization (numpy arrays are zero-copy borrowed)
- No IPC syscalls
- The GIL is released during Rust work, so other Python threads (including the inference dispatcher) run concurrently

Across processes, we use **`multiprocessing.shared_memory`** for the inference queue (see §5.2). The hot data is large (numpy tensors); copying through pickle would dominate the budget.

### 3.4 GIL Strategy

| Code path | GIL state |
|---|---|
| Rust MCTS work | **Released** (`py.allow_threads`) |
| Rust tensor encoding | **Released** |
| Rust replay buffer decode/augment | **Released** |
| Python orchestration | Held (this is fine; Python orchestration is light) |
| PyTorch forward pass | Released by PyTorch internally |

The only Python-bound code in the inner loop is the inference dispatcher, which is mostly waiting on `asyncio.Event`s and CUDA streams. The GIL is not on the critical path.

### 3.5 Zero-Copy Boundaries

Three explicit zero-copy paths:

1. **Worker → inference server.** Worker writes its leaf-batch tensor into a pre-allocated slot in shared memory. Inference server reads from that slot directly into its GPU upload buffer.

2. **Inference server → worker.** Server writes (policy, value) into a shared-memory return slot. Worker reads via a numpy view.

3. **Buffer → trainer.** Trainer mmap's a slice of the buffer's storage and gets a numpy view. The Rust decode kernel writes the dense tensor batch directly into a pre-allocated training tensor on the trainer side.

No intermediate Vec, no `pickle.dumps`, no `numpy.copy`.

---

## 4. Concrete Project Structure

### 4.1 Repository Layout

```
Hexo-RL-Project/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── hexgame-core/             # rlib only — pure Rust engine
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── core.rs
│   │   │   ├── eval/
│   │   │   ├── board.rs
│   │   │   ├── threats.rs
│   │   │   ├── encoder.rs
│   │   │   ├── search.rs
│   │   │   └── mcts.rs
│   │   └── tests/                 # integration tests
│   ├── hexgame-py/                # cdylib only — PyO3 wrapper
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── engine.rs          # PyMCTSEngine
│   │   │   ├── encode.rs          # encode_compact_record, D6 symmetry
│   │   │   └── buffer.rs          # buffer kernels
│   │   └── pyproject.toml         # maturin build
│   ├── hexgame-bench/             # benchmarks (independent crate)
│   │   ├── Cargo.toml
│   │   └── benches/
│   │       ├── mcts.rs
│   │       ├── encode.rs
│   │       ├── threats.rs
│   │       └── buffer_decode.rs
│   └── hexgame-cli/               # standalone CLI for profiling/debugging
│       ├── Cargo.toml
│       └── src/main.rs            # `hexgame play`, `hexgame bench`, `hexgame perft`
│
├── python/
│   ├── pyproject.toml             # one Python package: `hexorl`
│   ├── src/hexorl/
│   │   ├── __init__.py
│   │   ├── config/                # configuration system
│   │   │   ├── schema.py          # Pydantic/dataclass schema
│   │   │   ├── default.toml
│   │   │   └── loader.py
│   │   ├── model/
│   │   │   ├── network.py         # KataGo-style CNN
│   │   │   ├── heads.py           # policy, value, lookahead, axis, regret
│   │   │   ├── blocks.py          # bottleneck, global pool, mish
│   │   │   ├── conv.py            # HexConv2d
│   │   │   └── checkpoint.py      # load/save with metadata
│   │   ├── inference/
│   │   │   ├── server.py          # GPU inference server (one process)
│   │   │   ├── client.py          # Worker-side inference client
│   │   │   ├── batcher.py         # Adaptive batching logic
│   │   │   └── shm_queue.py       # Shared-memory request/response queue
│   │   ├── selfplay/
│   │   │   ├── worker.py          # One self-play worker (driven by Rust)
│   │   │   ├── orchestrator.py    # Spawns workers, supervises
│   │   │   ├── game_record.py     # Compact game record format
│   │   │   └── pcr.py             # Playout cap randomization
│   │   ├── buffer/
│   │   │   ├── ring.py            # Continuous RAM ring buffer
│   │   │   ├── sampler.py         # Recency-weighted sampler
│   │   │   ├── targets.py         # KataGo-style EMA target generation
│   │   │   ├── regret.py          # RGSC priority subset
│   │   │   └── dataloader.py      # PyTorch DataLoader interface
│   │   ├── train/
│   │   │   ├── trainer.py         # Main training loop
│   │   │   ├── losses.py          # Per-head losses
│   │   │   ├── schedule.py        # LR, loss weights, aux ramp-up
│   │   │   └── ema.py             # Model EMA for inference
│   │   ├── eval/
│   │   │   ├── arena.py           # Match orchestration (model-agnostic)
│   │   │   ├── elo.py             # ELO tracking
│   │   │   └── classical.py       # Alpha-beta opponent
│   │   ├── epoch/
│   │   │   ├── pipeline.py        # do_epoch: bootstrap → selfplay → train → eval
│   │   │   ├── autotune.py        # Detect max_batch_size / num_parallel_games
│   │   │   └── stats.py           # Per-epoch metrics
│   │   ├── dashboard/
│   │   │   ├── tui.py             # Rich-based live monitoring
│   │   │   └── tb.py              # TensorBoard writer
│   │   └── cli.py                 # `hexorl epoch`, `hexorl bench`, `hexorl arena`
│   └── tests/
│       ├── test_inference_server.py
│       ├── test_buffer.py
│       ├── test_targets.py
│       └── test_e2e_smoke.py      # one-iteration end-to-end run
│
├── benches/                       # cross-language benchmarks
│   ├── e2e_throughput.py          # full pipeline samples/sec
│   ├── inference_latency.py       # GPU batching characterization
│   └── selfplay_throughput.py     # games/sec at various scales
│
├── configs/
│   ├── default.toml
│   ├── small_test.toml            # quick iteration
│   ├── production.toml            # full epoch
│   └── reproducible.toml          # deterministic
│
├── Docs/
│   ├── SYSTEM_DESIGN.md           # this document
│   ├── ARCHITECTURE.md            # diagrams + decision log
│   ├── PERF_BUDGETS.md            # measured perf targets per component
│   └── HISTORY.md                 # archived old review docs
│
├── .github/workflows/
│   ├── rust.yml                   # cargo test, clippy, bench
│   ├── python.yml                 # pytest, ruff, mypy
│   └── e2e.yml                    # weekly: full epoch on small config
│
└── scripts/
    ├── bootstrap.sh               # initial setup
    ├── run_epoch.sh               # production epoch
    └── profile.sh                 # py-spy + perf for a worker
```

### 4.2 Why a Cargo Workspace

Three concrete benefits over the current single-crate layout:

1. **Build time.** `cargo bench` rebuilds 30s of PyO3 every iteration today. With `hexgame-core` separate, benchmarks rebuild only the engine.
2. **Independent versioning.** `hexgame-core` can be published as a normal Rust crate. Downstream Rust consumers (a TUI debugger, a perft tool, a tournament runner) don't pay for PyO3.
3. **Clearer testing boundary.** Integration tests in `crates/hexgame-core/tests/` test the public Rust API. PyO3-specific tests live in `crates/hexgame-py/`.

### 4.3 Why a Single Python Package

The original sketch had folders like `Epoch/Buffer/regret_buffer.py` with deep nesting. A flat layout of submodules under one `hexorl` package is easier to import, easier to test, and lets us refactor without circular-dep dramas. Group by *capability* (inference, selfplay, buffer, train), not by *epoch phase* — the same module is used across phases.

---

## 5. Inference Server — The Most Important Subsystem

This is where GPU saturation lives or dies. Get it right.

### 5.1 Process Topology

Three process groups:

- **Inference server (1 process)** — owns the GPU, owns the model weights.
- **Self-play workers (N processes, N ≈ 24-30 on the 7950X)** — one MCTSEngine each.
- **Trainer (1 process)** — time-shares the GPU with inference (separate CUDA streams).
- **Buffer process (1 process)** — owns the ring buffer.
- **Orchestrator (1 process)** — supervisor, restarts on crash.

Total: 27-33 processes. Memory budget: ~5-7 GB for engines + model copies + ring buffer.

### 5.2 Shared-Memory Queues

Use `multiprocessing.shared_memory.SharedMemory` for the inference channel. Each worker has:

- **Request slot:** `shm_req[i]` — a `(MAX_BATCH, 13, 33, 33) f32` tensor + `(MAX_BATCH, 4) i32` legal-move list + a length field.
- **Response slot:** `shm_res[i]` — a `(MAX_BATCH, 1089) f32` policy + `(MAX_BATCH,) f32` value.
- **Doorbell:** `multiprocessing.Event` per worker for request and response signaling. (Avoid spinning; use OS events.)

Rust writes directly into the request slot (the numpy array is a borrowed view of shared memory, exposed to Python). The inference server reads from all worker slots, batches across workers if multiple have requests pending, runs forward pass, and writes results back.

**This is the same design KataGo uses** — they call it the `NNEvaluator`. It works.

### 5.3 Adaptive Batching

The server runs an `asyncio` loop:

```
while running:
    drain ready requests (up to MAX_BATCH total)
    if batch_size > 0:
        upload to GPU on stream A
        forward pass on stream B (overlap with next upload)
        download results on stream C
        write to response slots, signal doorbells
    else:
        wait briefly (microseconds) on any-doorbell
```

**Adaptive batching rule:** wait up to a configurable `max_wait_us` (default 200 µs) for more requests once any request arrives. Tune via auto-detect: run a grid of `(num_workers, max_wait_us)` at startup, pick the throughput-maximizing point.

**Why this matters:** the 4070 Ti's tensor cores need batch ≥ 32 to saturate. With one request per worker, we need ≥ 32 workers in flight to fill the batch. On 32 logical cores that's tight; the wait window lets us queue stragglers without starving the GPU.

### 5.4 GPU Stream Layout

Three CUDA streams in the inference server:

- **Stream A:** H2D copies (request slots → GPU upload buffer)
- **Stream B:** Forward pass
- **Stream C:** D2H copies (GPU output → response slots)

Pipelined: while batch N forwards on B, batch N+1 uploads on A. Doubles effective throughput when CPU-side queueing is fast enough.

The trainer uses **a fourth stream** (Stream D) for backward pass + optimizer step. CUDA's stream scheduler interleaves; with `MIG` not available on consumer GPUs we rely on stream priorities.

### 5.5 Model EMA for Inference

The inference server holds an **EMA copy** of the trainer's weights. Trainer pushes weight updates to a shared-memory weight slot every K steps; inference server hot-swaps. EMA reduces self-play→training noise feedback (KataGo does this; it's a measurable Elo win).

### 5.6 FP16 / Mixed Precision

- Inference server: FP16 forward pass (`torch.cuda.amp.autocast`) for ~2× throughput on Ada Lovelace tensor cores.
- Trainer: BF16 if numerical stability is a concern, otherwise FP16 with loss scaling.
- Encoder output: stays FP32 (encoder is CPU-side; conversion is free on GPU upload).

---

## 6. Self-Play Workers

### 6.1 Worker Lifecycle

```
spawn():
    load_engine_config(seed = master_seed + worker_id)
    open_shm_slots(worker_id)
    while not killed:
        engine = MCTSEngine(config, seed = next_game_seed())
        game_record = play_one_game(engine)
        push_to_buffer(game_record)
```

Each worker:
- Is a separate `multiprocessing.Process`.
- Owns one `MCTSEngine`, one `HexGameState`.
- Talks only to the inference server (via shared memory) and the buffer process (via a `multiprocessing.Queue` for completed game records).
- Re-reads config on each game (allows mid-run hyperparameter changes — KataGo idea).

### 6.2 Inner Loop

```python
def play_one_game(engine):
    engine.init_root()
    while not engine.is_terminal():
        # inner MCTS-over-batches loop, all in Rust
        while not engine.simulations_complete():
            req = engine.run_until_inference_needed()
            if req is None: break  # all sims done
            policies, values = inference_client.submit(req)
            engine.submit_inference(policies, values)

        # temperature sampling, all in Rust
        action = engine.sample_action(temperature, seed.next())
        record_state(action, engine.snapshot_for_record())
        engine.advance(action)

    return finalize_game_record()
```

The Python side is light: dispatch inference, sample, record. All the work is Rust + GPU.

### 6.3 Process Crash Recovery

Workers can die (SIGSEGV, OOM, pathological position). The orchestrator:

- Spawns workers as `multiprocessing.Process(daemon=False)`.
- Monitors via `process.is_alive()`.
- On crash: drops the in-progress game (it never made it to the buffer), respawns with the next worker_id seed.
- Logs the crash with the last 10 moves of the dying game (stored in the worker's shared-memory crash log slot, written before each move).

This is why §2.7 of `FINALIZATION_PASS.md` matters: replace hard panics in MCTS with `Result` types so a single bad position doesn't take down the worker.

---

## 7. Replay Buffer Architecture

### 7.1 Storage

A continuous ring buffer in **the buffer process**. Capacity ~2M samples (KataGo runs at 4-12M; 2M is conservative for our hardware).

Per sample, **compact format**, stored in struct-of-arrays layout:

| Field | Type | Bytes | Notes |
|---|---|---|---|
| `move_history` | `u8[]` (variable) | ~50-200 | Replay-able from initial state |
| `policy_target` | `(u16, f32)[]` (sparse) | ~50-300 | Top-K only |
| `value_target` | `f32` | 4 | From outcome + EMA |
| `lookahead_targets` | `f32[3]` | 12 | KataGo-style EMA at horizons 4/12/36 |
| `pcr_flag` | `u8` | 1 | Full-search vs. low-sim |
| `turn_boundary` | `u8` | 1 | For lookahead bootstrap |
| `game_id` | `u32` | 4 | For dedup, for loss-weighting by recency-of-game |

**Total: ~150-500 bytes per sample.** 2M × 350 avg = 700 MB. Comfortable in 32 GB.

**Decode is Rust:** `encode_compact_record` rebuilds the (13, 33, 33) tensor on the dataloader path. This is acceptable because (a) decode is fast (~10 µs per sample) and (b) it lets us add new channels without re-encoding the entire buffer.

### 7.2 Sampling

Three concerns:

1. **Recency weighting.** Sample probability ∝ `decay ^ (current_game - sample_game_id)`. KataGo uses `decay ≈ 0.9-0.99` per "epoch"; we adapt to per-game.
2. **PCR weighting.** Full-search samples weighted 4×, low-sim PCR samples 1× in the loss.
3. **Regret priority.** RGSC: a separate priority queue of high-loss samples; trainer pulls 5-10% of each batch from this queue.

Implementation: `WeightedRandomSampler` in PyTorch with weights recomputed every K steps. Or: explicit alias-method sampler in Rust if Python is too slow (likely not, but bench it).

### 7.3 Target Generation Pipeline

Game records arrive at the buffer process as completed games (full move history + per-move MCTS visits + per-move root value). Buffer process runs `targets.py` to:

1. Compute final value target from outcome.
2. Compute KataGo-style EMA lookahead targets at horizons {4, 12, 36} turn-boundaries.
3. Quality-weight EMA terms (full-search > PCR).
4. Sparsify policy to top-K (K=20 typical).
5. Push compact samples into the ring.

This is **Python**; it's not on the inner loop. CPU cost per game: ~1 ms. Negligible.

### 7.4 Dataloader Interaction

PyTorch `DataLoader` workers (4-8 threads) hit the buffer with a sampler. Each batch:

1. Sampler selects N indices via recency-weighted draw + RGSC.
2. Buffer process serves the compact records.
3. Rust decode kernel materializes (N, 13, 33, 33) tensor + (N, 1089) policy + (N,) value + lookahead targets.
4. D6 symmetry transform applied (random sym_idx per sample) in Rust.
5. Tensor is uploaded to GPU on Stream D (trainer's stream).

Decode + augment runs **in the dataloader worker thread**, not the trainer process. With 8 workers and a target of 256 samples/batch, that's 32 samples per worker × 10 µs = 320 µs per worker. Easily keeps up with training.

---

## 8. Configuration & Extensibility

### 8.1 Config Schema

One `config.toml` per run. Loaded into a Pydantic model so type errors fail at parse time, not at training time.

Top-level sections:

```toml
[run]
seed = 42                  # master seed; 0 = non-deterministic
output_dir = "./runs/{name}"
log_level = "INFO"

[model]
channels = 128
blocks = 16
heads = ["policy", "value", "lookahead_short", "lookahead_mid", "axis"]

[selfplay]
num_workers = 24
games_per_epoch = 4096
states_per_epoch = 400_000
batch_size_per_worker = 8
mcts_simulations = 800
c_puct = 1.5
c_puct_init = 19652.0
temperature_schedule = [[0, 1.0], [30, 0.0]]
dirichlet_alpha = 0.3
dirichlet_fraction = 0.25
pcr_low_sim_prob = 0.75
pcr_low_sims = 192
resign_threshold = -0.95
resign_disable_prob = 0.1

[inference]
max_batch_size = 128
max_wait_us = 200
fp16 = true
ema_update_every = 100

[buffer]
capacity = 2_000_000
recency_decay = 0.99
pcr_weight = 0.25
regret_fraction = 0.08
lookahead_horizons = [4, 12, 36]
lookahead_lambdas = [0.75, 0.90, 0.97]

[train]
batch_size = 256
batches_per_epoch = 2000
optimizer = "adamw"
lr_schedule = "cosine"
peak_lr = 3e-3
loss_weights = { policy = 1.0, value = 1.5, lookahead_short = 0.2, lookahead_mid = 0.2 }
```

### 8.2 Adding a New Network Head

To add e.g. a "regret_value" head:

1. Add to `[model].heads` in config.
2. Add a `RegretValueHead` class in `python/hexorl/model/heads.py` (5-line subclass of `BaseHead`).
3. Add a target generator in `python/hexorl/buffer/targets.py`.
4. Add a loss in `python/hexorl/train/losses.py`.

**No Rust changes.** No FFI changes. The compact buffer format already carries arbitrary scalar/sparse fields tagged by name.

### 8.3 Adding a New MCTS Variant

If you want to swap PUCT for AlphaZero-style UCB, or add LCB-based exploration:

1. New file `crates/hexgame-core/src/mcts/variant_xyz.rs`.
2. New constructor on `MCTSEngine` that selects the variant.
3. Config switch `[selfplay].mcts_variant = "xyz"`.

The `MCTSEngine` API stays the same. Workers don't change.

---

## 9. Performance Benchmarking

### 9.1 What to Benchmark

| Layer | Benchmark | Target | Owner |
|---|---|---|---|
| Rust hot paths | `cargo bench` (criterion) | regression < 5% | rust.yml CI |
| FFI roundtrip | `bench_ffi_roundtrip.py` | < 50 µs single, < 300 µs batch=64 | python.yml CI |
| Inference latency | `inference_latency.py` | batch=64 < 5ms FP16 | manual + nightly |
| Inference throughput | `inference_throughput.py` | > 30k positions/sec | manual + nightly |
| Self-play throughput | `selfplay_throughput.py` | > 200 games/min on 24 workers | manual + nightly |
| End-to-end | `e2e_throughput.py` | > 50k samples/sec into buffer | manual + nightly |

### 9.2 Benchmark Infrastructure

- **`cargo bench`** with criterion. Each Rust crate has its own benches in `crates/<crate>/benches/`.
- **`pytest-benchmark`** for Python micro-benchmarks. Runs under `python.yml` CI.
- **End-to-end harness** in `benches/e2e_throughput.py`. Spins up a tiny config (4 workers, 100 games) and measures samples/sec.
- **Continuous profiling** via `py-spy --duration 60 --output flamegraph.svg` integrated into `scripts/profile.sh`. Run weekly on a long self-play job; archive flamegraphs.

### 9.3 Performance Budgets

A `Docs/PERF_BUDGETS.md` lists every component's target latency/throughput. PRs that exceed a budget block on review.

Examples:

- `MCTSEngine::run_until_inference_needed()` → < 200 µs at sim_count=800
- `encode_board_into()` → < 5 µs
- `apply_d6_symmetry()` → < 2 µs
- inference dispatcher request→response → < 1 ms p50, < 5 ms p99
- buffer decode 256 samples → < 10 ms

These numbers are derived from "what we need to saturate the GPU at 30k positions/sec on 24 workers." Document the math.

### 9.4 Auto-Tuning at Epoch Start

The original spec mentions detecting `max_batch_size` and `num_parallel_games` automatically. Implement as `python/hexorl/epoch/autotune.py`:

1. Allocate the model. Try `forward(batch=N)` for N in `[16, 32, 64, 128, 256, 512]` until OOM. Pick the largest that fits.
2. For each `num_workers ∈ {16, 20, 24, 28, 32}`, run a 60-second self-play sample; measure positions/sec.
3. Save the (max_batch, num_workers) point with highest throughput to `runs/<name>/autotune.json`.
4. Subsequent epochs read this file and skip retuning unless the model architecture changed.

---

## 10. Determinism & Reproducibility

For RL research you will need to reproduce a run. The system supports it with one config flag.

```toml
[run]
seed = 42                      # master seed
deterministic = true           # disables non-determinism sources below
```

When `deterministic = true`:

- Worker `i` is seeded with `seed XOR worker_id`.
- Each game `g` within a worker uses seed `worker_seed XOR g`.
- The Rust RNG (currently from time XOR address) accepts a `u64` seed via `MCTSEngine::new(..., seed)`.
- Dirichlet noise sampled from a Python `numpy.random.Generator(seed=...)`.
- `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms(True)`.
- Inference batching becomes order-deterministic (sort requests by `worker_id` before the forward pass; otherwise floating-point reduction order varies).
- D6 symmetry index is sampled from a per-sample seed derived from `(game_id, ply)`, not a global RNG.

Cost: ~10-15% throughput. Off by default in production runs, on for ablations.

---

## 11. Migration Path from Current State

The current state: Rust engine is solid (per `FINALIZATION_PASS.md`). Python is empty stubs. Single-crate layout. No inference server.

### Phase 1 — Foundation (1-2 weeks)
1. Split Rust into a workspace (`crates/hexgame-core` + `crates/hexgame-py` + `crates/hexgame-bench` + `crates/hexgame-cli`).
2. Implement the 5 FFI entry points in `hexgame-py`.
3. Apply the 6 blocker fixes from `FINALIZATION_PASS.md` §6 (NaN-safe PUCT, no-panic re_root, deterministic seeding, temperature/resign in Rust, T1-7 underflow guard, zero-copy FFI tensor).
4. Add `bench_single_mcts_sim` and `bench_threat_status` to `hexgame-bench`.

### Phase 2 — Inference Server (1-2 weeks)
1. Implement `python/hexorl/inference/server.py` with shared-memory queues.
2. Implement `python/hexorl/inference/client.py` (worker side).
3. Stub model in `python/hexorl/model/network.py` (small CNN; correct shapes, untrained).
4. Benchmark inference throughput: target > 30k positions/sec FP16 batch=64.

### Phase 3 — Self-Play & Buffer (2 weeks)
1. Implement `selfplay/worker.py` and `selfplay/orchestrator.py`.
2. Implement `buffer/ring.py`, `buffer/sampler.py`, `buffer/targets.py`.
3. Implement Rust `encode_compact_record` and `apply_d6_symmetry` kernels.
4. Run 1k-game self-play with the stub model. Measure: games/min, samples/min, GPU utilization.

### Phase 4 — Training Loop (1 week)
1. Implement `model/network.py` properly (KataGo-style heads).
2. Implement `train/trainer.py`, `train/losses.py`, `train/ema.py`.
3. Run a single epoch end-to-end on `configs/small_test.toml`.

### Phase 5 — Evaluation & Stability (1 week)
1. Implement `eval/arena.py` and `eval/elo.py`.
2. Implement `eval/classical.py` (alpha-beta opponent for ELO anchor).
3. Run 10-epoch stability test.

### Phase 6 — Polish (ongoing)
1. Auto-tuning (`epoch/autotune.py`).
2. Dashboard (`dashboard/tui.py` + TensorBoard).
3. Long-run benchmarking + flamegraph regression tracking.
4. Bootstrap data generation (one-time, reused).

---

## 12. Hardware-Specific Tuning

### 12.1 7950X (32 logical cores, 16 P-cores)

- 24 self-play workers + 1 inference + 1 trainer + 1 buffer + 1 orchestrator = 28 processes. Leaves 4 cores for OS and dataloader workers.
- Pin self-play workers to specific cores with `os.sched_setaffinity` to reduce migration overhead. Pin inference and trainer to opposite NUMA halves (the 7950X is single-NUMA but core-complex affinity still matters for L3 cache).
- AMD's per-CCD L3 is 32 MB; an `MCTSEngine` arena fits in one CCD's L3, so pinning helps.

### 12.2 4070 Ti (12 GB VRAM, Ada Lovelace, 80 SMs)

- FP16 tensor cores: ~165 TFLOPS at boost. Useful budget for inference + training.
- 12 GB is comfortable: 200 MB model + 1 GB activations training + 1 GB inference EMA + headroom = ~3 GB used, 9 GB headroom.
- Use `cudnn_benchmark = True` (model architecture is fixed once tuned).
- Use channel-last memory format (`tensor.to(memory_format=torch.channels_last)`); ~10% boost on Ada for CNNs.
- Compile the model with `torch.compile(model, mode='reduce-overhead')` after the first epoch (skip if it breaks tracing — known issue with custom heads).

### 12.3 32 GB DDR5

- Ring buffer: 700 MB
- Per-worker (engine + game state + shm slots): ~100 MB × 24 = 2.4 GB
- Inference server (model + EMA + workspaces): ~2 GB
- Trainer (model + optimizer state + activations): ~3 GB
- OS, Python interpreters, CUDA runtime, etc.: ~4 GB
- **Total: ~13 GB. Plenty of headroom for larger buffer or more workers.**

If you decide to grow the buffer to 8M samples (~3 GB) or run 32 workers (~3.2 GB), still comfortable.

---

## 13. Open Design Questions

These are decisions to make during implementation, not now:

1. **Single trainer or async actor-critic split?** AlphaZero uses one trainer reading from a buffer. Some papers (R2D2-style) use an async setup. Start with the simple version.
2. **Curriculum learning?** KataGo uses a "playout doubling" trick for early training. Worth considering if early training is unstable.
3. **Tournament-based opponent pool vs. trainer EMA?** Eval setup; doesn't block self-play architecture.
4. **Distributed training?** Out of scope for single-machine. The architecture allows it (workers can run on different boxes hitting a remote inference server) but adds complexity.
5. **Game record retention beyond the buffer?** SQLite or compressed files for offline analysis. Cheap and easy; just make `buffer/ring.py` also write to disk on game completion.

---

## 14. Summary

The system is built around three load-bearing decisions:

1. **Inference batching server** decouples GPU from CPU and lets N workers amortize one forward pass. This is the single most important architectural choice.
2. **Single-threaded MCTSEngine, multi-process workers** matches the proven AlphaZero/KataGo pattern. No shared-tree complexity, full 7950X utilization through process parallelism.
3. **Compact buffer + Rust decode kernels** keeps memory usage low and lets the schema evolve without re-encoding the buffer.

Everything else (config schema, benchmarks, FFI surface, workspace layout, crash recovery, autotuning) follows from making those three things work well.

The Rust engine is ready. The Python pipeline is the next 6-8 weeks of work, broken down in §11. Performance targets in §9.3 give concrete pass/fail criteria for each phase.
