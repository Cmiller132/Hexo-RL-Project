# Finalization Pass — Hexo-RL-Project

**Reviewer:** Independent verification pass against the implementation done by DeepSeek V4 from `Docs/CODE_REVIEW_RUST.md`.
**Method:** Sonnet subagents read the actual source, not the `Docs/RUST_PROJECT.md` claim doc, and verified each item line-by-line.
**Hardware target:** RTX 4070 Ti (12 GB VRAM), AMD 7950X (16C/32T), 12 GB system RAM.
**Goal under evaluation:** A robust, performant, multithread-capable foundation for an AlphaZero / KataGo–style training pipeline.

---

## TL;DR

| Aspect | Verdict |
|---|---|
| **Tier 1 correctness fixes (7)** | 6 VERIFIED, 1 PARTIAL (T1-7 underflow guard) |
| **Tier 2 performance fixes (12)** | 12 VERIFIED (1 has a residual sub-issue) |
| **Tier 3 structure fixes (8)** | 8 VERIFIED |
| **Tier 4 test infrastructure (5)** | 5 VERIFIED |
| **AlphaZero readiness** | Engine: ~80%. Python pipeline: ~0% (stubs only). |
| **Multithreading** | Root-parallel via multiprocessing: ✅. Shared-tree or threaded: ❌. |
| **Stability under adversarial NN output** | NaN-vulnerable in PUCT, hard panics in 2 hot paths |
| **Hardware fit (4070 Ti + 7950X + 12 GB)** | RAM/VRAM comfortable; CPU underutilized; FFI copies cap GPU saturation |

The Rust engine is a solid, well-engineered foundation. **The reported implementation is genuine — DeepSeek V4 actually did the work, the claims in `RUST_PROJECT.md` are accurate.** What remains is a small set of finalization issues plus a much larger gap: the Python training pipeline is entirely unimplemented.

---

## Section 1 — Verification Results

### 1.1 Tier 1 (Correctness) — 6/7 VERIFIED, 1 PARTIAL

| ID | Fix | Verdict | Evidence |
|---|---|---|---|
| T1-1 | Backprop sign flip uses depth parity | ✅ VERIFIED | `mcts.rs:606-613` — `let mut parity_value = value; ... parity_value = -parity_value;`. No `node.player` comparison anywhere in the loop. |
| T1-2 | Virtual loss adjusts both `visit_count` and `total_value` | ✅ VERIFIED | Apply at `mcts.rs:474-478`, revert at `mcts.rs:578-582`. Both touch both fields with correct signs. |
| T1-3 | `sims_done` only incremented in `expand_and_backprop` | ✅ VERIFIED | Single increment at `mcts.rs:574` (`self.sims_done += leaves.len() as u32`). Not present in `select_leaves`. |
| T1-4 | Slice bounds asserted in `expand_and_backprop` | ✅ VERIFIED | Two `assert!` at `mcts.rs:558-568` with diagnostic messages. |
| T1-5 | GIL released in MCTS hot path | ✅ VERIFIED | `pybridge/mcts.rs:121-124` and `153-155` both wrap Rust calls in `py.allow_threads`. |
| T1-6 | `set_position` per-stone validation | ✅ VERIFIED | `board.rs:455-473` checks `InvalidPlayer`, `MustPlaceAtOrigin`, and `OutOfRadius` per stone. |
| T1-7 | Pattern index `assert!` (3 sites) | ⚠️ **PARTIAL** | Sites 1 (`state.rs:273`) and 3 (`state.rs:461`) are `assert!`. Site 2 has the bounds `assert!` at `state.rs:328` but the **underflow guard immediately preceding it** (`state.rs:323-326`) is still `debug_assert!`. In release builds, if `new_idx < cell_val * POW3[off]`, the subtraction wraps with no diagnostic. |

#### Open issue from T1-7

```rust
// src/eval/state.rs:323-328
debug_assert!(                                  // ← still debug_assert!
    new_idx >= cell_val * POW3[off as usize],
    "unplace: index underflow at gi={gi}"
);
let old_idx = new_idx - cell_val * POW3[off as usize];
assert!(old_idx < 729, "pattern index out of range on unplace: {}", old_idx);
```

**Fix:**
```rust
assert!(
    new_idx >= cell_val * POW3[off as usize],
    "unplace: index underflow at gi={gi}"
);
```

---

### 1.2 Tier 2 (Performance) — 12/12 VERIFIED

All twelve items match spec. One residual sub-issue worth noting:

**T2-7 residual:** `MCTSEngine` correctly threads a `legal_buf: Vec<Hex>` through `encode_board_into` (eliminating allocation inside the encoder), but the call sites at `mcts.rs:363` and `mcts.rs:526` still do `self.legal_buf.clone()` to populate `PendingLeaf.legal_moves`. Each pending leaf therefore still triggers a heap allocation at the engine boundary.

**Fix options (pick one):**
1. Store `legal_moves` in `PendingLeaf` as a `(start, len)` pair indexing into a per-engine flat `Vec<Hex>` (highest performance, structural change)
2. Replace `Vec<Hex>` in `PendingLeaf` with `SmallVec<[Hex; 32]>` (zero-alloc for typical batch sizes; one-line change)

Recommended: option 2.

```rust
// src/mcts.rs — PendingLeaf
struct PendingLeaf {
    node_idx: u32,
    search_path: SmallVec<[u32; 32]>,
    is_terminal: bool,
    terminal_value: f32,
    offset_q: i32,
    offset_r: i32,
    legal_moves: SmallVec<[Hex; 32]>,  // was Vec<Hex>
}
```
Then replace `self.legal_buf.clone()` with `SmallVec::from_slice(&self.legal_buf)`.

---

### 1.3 Tier 3 (Structure) — 8/8 VERIFIED

| ID | Fix | Verdict |
|---|---|---|
| T3-1 | `panic = "abort"`, `[profile.bench]`, criterion dev-dep, `[[bench]]` | ✅ |
| T3-2 | `c_puct_init` private and constructor-wired | ✅ |
| T3-3 | Legal-bytes parser returns `PyErr` | ✅ |
| T3-4 | Docstring uses real method names | ✅ |
| T3-5 | `BOARD_SIZE`, `NUM_CHANNELS`, `TENSOR_SIZE` exported | ✅ |
| T3-6 | `mcts`, `search`, `threats` are `pub(crate)` | ✅ |
| T3-7 | `rust-toolchain.toml` exists | ✅ |
| T3-8 | `rustfmt.toml`, `.clippy.toml`, `.cargo/config.toml` exist | ✅ |

---

### 1.4 Tier 4 (Tests & Bench) — 5/5 VERIFIED

| ID | Fix | Verdict |
|---|---|---|
| T4-1 | 5 MCTS tests in `src/tests/mcts.rs` | ✅ all 5 present, names match spec |
| T4-2 | Python smoke tests + CI `python-integration` job | ✅ 4 tests, CI runs maturin + pytest |
| T4-3 | Integration tests in top-level `tests/` | ✅ `tests/board.rs`, `tests/encoder.rs` exist; old `src/tests/board.rs` and `src/tests/encoder.rs` removed |
| T4-4 | `benches/engine.rs` with criterion | ✅ `bench_encode_board`, `bench_legal_moves` |
| T4-5 | Proptests broadened | ✅ `place_unplace_is_identity` in `tests/board.rs:540`; `encode_output_range` in `tests/encoder.rs:136` |

**Test gap:** The benchmark suite has only `bench_encode_board` and `bench_legal_moves`. The plan's `bench_threat_status` and `bench_single_mcts_sim` are missing. Without an MCTS-loop benchmark, throughput regressions in the search itself will not be caught.

---

## Section 2 — Remaining Issues for AlphaZero Readiness

The implementation is correct and complete relative to the original review. The following are issues **not** in the original review that surfaced during this finalization pass and **must** be addressed before training begins.

### 2.1 Critical — NaN-Vulnerable PUCT Scorer

**File:** `src/mcts.rs:928-980` (`select_child_puct`)

If a single `total_value` field becomes `NaN` (from a NaN value-head output during backprop), the score for that child becomes `NaN`. `score > best_score` is always `false` for NaN, so iteration leaves `best_idx = start` regardless of which child actually has the highest score. This silently biases every subsequent simulation toward the first legal move.

**Fix:** Defend at the source (in backprop) and at the scorer:

```rust
// In expand_and_backprop, before backprop loop (~mcts.rs:557):
assert!(
    !value.is_nan() && value.is_finite(),
    "expand_and_backprop: NN returned NaN/Inf value at leaf {}", leaf.node_idx
);

// In select_child_puct, after computing score (~mcts.rs:970):
let score = q + effective_c_puct * child.prior * sqrt_parent / (1.0 + vc as f32);
debug_assert!(score.is_finite(), "PUCT score non-finite for child {}", i);
```

The `gather_policy` softmax already has a NaN-safe uniform fallback (mcts.rs:184) — model that pattern at every other NN-output ingestion site.

### 2.2 Critical — Hard Panics in Hot Paths

**File:** `src/mcts.rs:463` (tree traversal)
```rust
self.game.place(child.action.0 as i32, child.action.1 as i32)
    .expect("MCTS: illegal place during tree traversal");
```

**File:** `src/mcts.rs:653-655` (re_root)
```rust
panic!("re_root: no child found for action ({}, {})", q, r);
```

With `panic = "abort"` set in the release profile, both terminate the entire process. In a long-running self-play daemon producing training data, this loses the in-progress game and (depending on the buffer flush cadence) potentially many recent games.

**Fix for `re_root`:** Convert to a result-returning method and surface as `PyErr`:
```rust
pub fn re_root(&mut self, q: i16, r: i16, new_num_simulations: u32) -> Result<(), MCTSError> {
    // ... find child ...
    let child_idx = match found {
        Some(idx) => idx,
        None => return Err(MCTSError::ChildNotFound { q, r }),
    };
    // ...
    Ok(())
}
```

**Fix for tree-traversal expect:** Make it a `debug_assert!` for invariants you trust under correct use:
```rust
debug_assert!(self.game.place(...).is_ok(), "tree-traversal place failed");
let _ = self.game.place(child.action.0 as i32, child.action.1 as i32);
```

### 2.3 Major — FFI Copy Tax

**Files:** `src/pybridge/mcts.rs:121-124` and `:153-155`

Every `select_leaves` round trip copies the batch buffer once (`tensors.to_vec()`), and every `expand_and_backprop` copies both `policies` and `values` (`.to_vec()` x2). At batch_size=16 and ~1k inferences/sec this is ~1 GB/s of pure memcpy overhead.

```rust
// Current (pybridge/mcts.rs:121-124):
let (count, tensor_vec) = py.allow_threads(|| {
    let (tensors, count) = self.inner.select_leaves(batch_size);
    (count, tensors.to_vec())
});
```

**The copy is forced because:**
1. `tensors: &[f32]` borrows `self.inner.batch_buf` and cannot cross the `allow_threads` boundary safely without `unsafe`.
2. `PyReadonlyArray1` borrows the GIL and cannot be sent across thread boundaries.

**Fix path:** Build the numpy array under the GIL but skip the intermediate Vec by writing directly into a pre-allocated `PyArray4`:

```rust
fn select_leaves<'py>(
    &mut self,
    py: Python<'py>,
    batch_size: u32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, u32)> {
    // Allocate uninit numpy first
    let arr = unsafe {
        PyArray4::<f32>::new(
            py,
            [batch_size as usize, NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize],
            false,
        )
    };
    let count = py.allow_threads(|| {
        let (tensors, count) = self.inner.select_leaves(batch_size);
        // SAFETY: arr is exclusively owned by this call, not yet visible to Python
        unsafe {
            let slice = arr.as_slice_mut().unwrap();
            slice[..tensors.len()].copy_from_slice(tensors);
        }
        count
    });
    Ok((arr, count))
}
```

This still has a single `copy_from_slice` (unavoidable — the numpy buffer is a different allocation), but eliminates the intermediate `Vec<f32>`. For `expand_and_backprop`, the existing `.to_vec()` for `policies`/`values` can be eliminated by passing slice references through `unsafe` lifetime extension, or by just keeping the GIL during the call (the alpha-beta search benchmark would tell us whether the GIL matters more than the copy).

### 2.4 Major — Missing AlphaZero Features in Rust

These are mentioned in `default_config.toml` but not implemented in the engine:

| Feature | Required | Current |
|---|---|---|
| Temperature-based move sampling | At root after MCTS | Not in Rust; visit counts returned raw |
| Resignation threshold | Per-game early termination | Not implemented |
| Playout cap randomization | Variable sim count per move | Not in Rust |
| D6 symmetry augmentation | 6× training data | Not implemented anywhere |
| Deterministic seeding | Reproducible runs | RNG seeded from time XOR addr (`pybridge/mod.rs:63-70`) — no `seed` parameter |
| Position deduplication | Prevent buffer poisoning | Zobrist hash exists but no dedup hook |

**Highest priority:** temperature sampling and resign threshold. Both belong in the Rust engine (called from Python with one FFI call per move):

```rust
// In MCTSEngine, add:
pub fn sample_move(&self, temperature: f32, rng_state: &mut u64) -> (i16, i16) {
    let root = &self.arena[self.root_idx as usize];
    let start = root.children_start as usize;
    let count = root.children_count as usize;

    if temperature == 0.0 {
        // Argmax visits.
        let best = (start..start + count)
            .max_by_key(|&i| self.arena[i].visit_count)
            .unwrap();
        return (self.arena[best].action.0, self.arena[best].action.1);
    }

    // Temperature sampling: visits^(1/T), then normalize, then sample.
    let inv_t = 1.0 / temperature;
    let weights: Vec<f32> = (start..start + count)
        .map(|i| (self.arena[i].visit_count as f32).powf(inv_t))
        .collect();
    let sum: f32 = weights.iter().sum();
    let r = next_uniform(rng_state) * sum;
    let mut acc = 0.0;
    for (offset, &w) in weights.iter().enumerate() {
        acc += w;
        if acc >= r {
            let n = &self.arena[start + offset];
            return (n.action.0, n.action.1);
        }
    }
    // Fallback: last child
    let last = start + count - 1;
    (self.arena[last].action.0, self.arena[last].action.1)
}

pub fn should_resign(&self, threshold: f32) -> bool {
    let root_q = self.arena[self.root_idx as usize].q_value();
    root_q < threshold
}
```

**D6 symmetry:** Hex on a hexagonal grid has 12-fold symmetry (6 rotations × 2 reflections). The centroid-offset encoder breaks translation invariance correctly but does not exploit rotational symmetry. Add a Python-side augmentation that, for each training example, samples one of 12 transforms and applies it to both the tensor and the policy target. This is best done in the Python data loader, not Rust.

### 2.5 Major — No Intra-Engine Parallelism

The 7950X has 32 threads. A single `MCTSEngine` uses one. Self-play parallelism today must be process-level (multiprocessing with one engine per process).

This is not a defect of the design — it is the deliberate choice made by AlphaZero's reference implementation — but it is a constraint to acknowledge in your training plan:

- **N processes × 1 engine each**: works today. Memory cost ~2.5 MB per engine. 32 processes ≈ 80 MB; comfortable on 12 GB system RAM.
- **1 process × N threads sharing a tree**: not feasible with current types. Would require:
  - `MCTSEngine` fields wrapped in lock-free structures (`crossbeam::deque` for pending, atomics on `MCTSNode.visit_count` and `total_value`).
  - Tree expansion serialized via `parking_lot::Mutex` on the arena.
  - This is a 2-3 week refactor and AlphaZero/KataGo do not require it for competitive play.

**Recommendation:** Stay process-parallel. Document the model in a `Docs/SELF_PLAY_ARCHITECTURE.md` so contributors don't reach for `Arc<Mutex<MCTSEngine>>` and waste time.

### 2.6 Minor — `assert!` in Hot Paths

The new `assert!` calls at `eval/state.rs:273`, `mcts.rs:1012` (`children_count`), and `mcts.rs:564,567` (`expand_and_backprop`) all fire in release builds and abort with `panic = "abort"`.

For the long-running self-play daemon, these are correctness firewalls — if they ever fire, training data is already corrupt and aborting is the right call. Keep them. But pair them with:

- A pre-flight check at engine creation that runs a few simulations and verifies no asserts fire (catches any seed-dependent issue early).
- A wrapper Python process that catches the SIGABRT and logs the last `n` moves of the dying game.

### 2.7 Minor — `tests/encoder.rs` Range Check Too Lax

The new proptest at `tests/encoder.rs:136` checks `v >= 0.0 && v <= 1.0` — but the existing test in the original review wrote `assert!(v >= 0.0)` for `[0.0, ∞)`. Verify which is correct: channel 11 (centroid distance) is normalized to `[0, 1]`, but other channels are also in `[0, 1]` by construction. The tighter bound is correct **provided** all channels are in `[0,1]`. Spot-check by searching for any channel that writes a non-binary value other than the bankers-rounding path. (Channels 0-6 binary; 7-8 are `1/(1+plies_ago)` ∈ (0,1]; 9-10 binary; 11 ∈ [0,1]; 12 binary.) The `[0, 1]` bound is correct.

---

## Section 3 — High-Level Structure Assessment

### 3.1 Rust Engine — Strong

```
core → eval → board → threats → {mcts, search, encoder} → pybridge
```

Clean layering, zero circular deps, deliberate `pub`/`pub(crate)` boundaries after T3-6. Arena allocator is the right choice for MCTS. Incremental `EvalState` with full undo is production-quality. The 13-channel encoder is expressive and matches typical AlphaZero CNN inputs.

The single critique: **this should be a Cargo workspace.** Today the `cdylib` (Python extension) and `rlib` (downstream Rust consumers) are built from the same crate. Splitting into:

```
hexgame-core/    # rlib only — pure Rust engine
hexgame-py/      # cdylib only — wraps core via pyo3
hexgame-bench/   # benchmarks against core
```

would let you:
- Iterate benchmarks without rebuilding pyo3 every time
- Develop Rust-only consumers (e.g., a TUI debugger) without pulling in numpy
- Pin separate semver tracks for the public Rust API and the Python wheel

This is not blocking; it's a hygiene upgrade.

### 3.2 Python Pipeline — Empty

`grep` of `Python/` finds 16 files. Excluding `test_engine_smoke.py`, all training/inference code is **stubs or specs only.** There is no:

- Working `self_play.py` (config exists, code does not)
- Working `train.py`
- Working `buffer.py` (replay buffer)
- PyTorch model definition (the 64-line file is a docstring)
- Checkpoint manager
- ELO arena for model evaluation

This is the dominant remaining work. The Rust engine cannot train a model on its own.

### 3.3 Repository Hygiene

- `Docs/CODE_REVIEW_2.md` … `CODE_REVIEW_5.md` — process artifacts that should be archived or merged into a single `Docs/HISTORY.md`.
- `proptest-regressions/` is tracked in `tests/` (good — that's how it's supposed to work).
- `.cargo/config.toml` clippy deny-list applies pedantic warnings globally; this is aggressive and may noise up downstream consumers if `hexgame` ever becomes a published crate. Consider scoping to `[lints.clippy]` in `Cargo.toml` instead.

---

## Section 4 — Hardware Fit

### 4.1 Memory Budget

| Resource | Per-engine cost | 32-process budget | Available | Headroom |
|---|---|---|---|---|
| `MCTSNode` arena (1024 sims) | ~1.6 MB | 51 MB | 12 GB | ✅ vast |
| `EvalState` (boxed `[u16; 11163]`) | 22 KB | 700 KB | — | ✅ |
| `batch_buf` (batch=16) | 900 KB | 29 MB | — | ✅ |
| Total per engine | ~2.5 MB | 80 MB | 12 GB | ✅ |
| GPU model weights (typical CNN) | 50–200 MB | — | 12 GB | ✅ |
| GPU activations (batch=64) | ~500 MB | — | 12 GB | ✅ |
| GPU peak | ~1 GB | — | 12 GB | ✅ |

You can run **dozens of parallel self-play processes** on this hardware. Memory will not be the bottleneck.

### 4.2 CPU Throughput

Single-engine MCTS at the current implementation: rough estimate from the file analysis, ~5,000–15,000 sims/sec depending on game-state complexity. The 7950X has 32 threads. Theoretical aggregate: ~150,000–500,000 sims/sec across 32 processes.

GPU inference at 4070 Ti FP16 (batch=64, typical AlphaZero-19b CNN): ~3,000–8,000 batches/sec ≈ ~200,000–500,000 positions/sec.

**The CPU-GPU ratio is roughly balanced.** The risk is that the FFI copy tax (Section 2.3) artificially caps throughput well below this ceiling. Fix the zero-copy path before scaling.

### 4.3 GPU Precision

Encoder produces `f32`. The 4070 Ti's FP16 tensor cores deliver ~40 TFLOPS, vs ~21 TFLOPS for FP32. To use FP16:

- Cast in PyTorch with `tensor.half()` before the forward pass.
- Use AMP (`torch.cuda.amp.autocast`) for mixed precision during training.
- Keep the policy/value head in FP32 to preserve numeric stability (PyTorch AMP handles this automatically).

Do not cast in Rust — the encoder is fine in FP32, the conversion is essentially free on GPU.

---

## Section 5 — Stability & Resilience Scorecard

| Category | Score | Notes |
|---|---|---|
| Memory safety | 10/10 | Zero `unsafe` code, no raw pointers, bounds checks on indexing |
| FFI safety | 9/10 | All numpy access checked; `legal_bytes` parser hardened in T3-3 |
| Adversarial NN input (NaN/Inf) | **5/10** | `gather_policy` defends, `select_child_puct` does not (Section 2.1) |
| Panic resistance | **6/10** | Two abort paths under tree-state inconsistency (Section 2.2); `assert!`s appropriate elsewhere |
| Memory leaks | 9/10 | Arena grows across re_root (acknowledged ~400 KB waste); engine drops on game end |
| Thread safety | 8/10 | Send + Sync, no global mutable state, OnceLock used correctly |
| Determinism | **3/10** | No seed parameter, RNG seeded from time XOR address |
| Long-run stability | 7/10 | Pending leaves correctly flushed if API is followed; abort paths could orphan in-progress games |

---

## Section 6 — Action Items, Prioritized

### Block before first training run

1. **NaN-safe PUCT** — assertion at NN ingestion + `debug_assert` after score (Section 2.1)
2. **Replace `re_root` panic with `Result`** + propagate as `PyErr` (Section 2.2)
3. **Demote tree-traversal `expect` to `debug_assert!`** (Section 2.2)
4. **Add deterministic seeding** — `MCTSEngine::new(..., seed: u64)` and `set_global_seed(seed)` for the pybridge thread-local RNG
5. **Implement temperature sampling and resign threshold in Rust** (Section 2.4)
6. **Fix T1-7 underflow guard** — promote `debug_assert!` to `assert!` at `eval/state.rs:323`

### Before scaling self-play to 32 processes

7. **Zero-copy FFI batch tensor** — write directly into pre-allocated `PyArray4` (Section 2.3)
8. **Add MCTS-loop benchmark** (`bench_single_mcts_sim`) to catch throughput regressions
9. **Add `bench_threat_status`** for the most-called function in classical search
10. **Eliminate `legal_buf.clone()` per leaf** (T2-7 residual; Section 1.2)

### Before merging downstream Rust consumers

11. **Cargo workspace split** — `hexgame-core` rlib + `hexgame-py` cdylib (Section 3.1)
12. **Scope clippy lints** to `[lints.clippy]` in `Cargo.toml` (Section 3.3)
13. **Archive old review docs** into `Docs/HISTORY.md`

### Required for AlphaZero pipeline (Python work, out of Rust scope)

14. Implement `self_play.py` driving `PyMCTSEngine` across multiprocessing
15. Implement `train.py` with policy + value loss
16. Implement `buffer.py` replay buffer with Zobrist-based dedup
17. Implement PyTorch model definition (the file is a docstring today)
18. Implement D6 symmetry augmentation in the data loader
19. Implement checkpoint manager + ELO arena

### Optional enhancements

20. **Playout cap randomization** (KataGo) — pass `pcr_low_sim_prob`, `pcr_low_sims` through to `MCTSEngine::run_one`
21. **Dual-tree MCTS** for opponent modelling
22. **Position hash export** for replay-buffer dedup (requires exposing `zobrist()` on `MCTSEngine`)

---

## Section 7 — Final Verdict

**Implementation completeness:** The work that DeepSeek V4 was tasked with is **substantively complete and correct**. Of 32 individual fix items, 31 fully verified and 1 partial (T1-7, a minor underflow guard). The implementation is faithful to the review specification.

**As a foundation for AlphaZero/KataGo training:**

- ✅ The core MCTS primitives (PUCT, virtual loss, batched leaves, re-rooting, Dirichlet) are implemented correctly.
- ✅ The performance budget is tight enough to saturate the target hardware once FFI copies are eliminated.
- ✅ The threading model (multiprocessing root-parallel) is sound and matches AlphaZero's reference design.
- ⚠️ Several adversarial-input and abort-path issues need fixing before a long-running self-play daemon can be trusted.
- ❌ The Python training pipeline does not exist beyond stubs and configs.

**The Rust engine is ready to be the foundation. The training pipeline on top of it is the next, larger, mostly-Python project.** Do the 6 blocker items in Section 6 and the engine is production-grade for self-play data generation.

---

## Appendix — Files Touched During Implementation

Verified to exist and contain the expected content:

| File | Purpose |
|---|---|
| `src/mcts.rs` | T1-1, T1-2, T1-3, T1-4, T2-10, T2-11, T3-2 |
| `src/eval/state.rs` | T1-7, T2-2 (added `clear()`) |
| `src/board.rs` | T1-6, T2-1, T2-2 (call site), T2-3, T2-9, T2-12 |
| `src/threats.rs` | T2-4, T2-5, T2-6 |
| `src/encoder.rs` | T2-7, T2-8 |
| `src/pybridge/mcts.rs` | T1-5, T3-2 (call site), T3-3 |
| `src/pybridge/mod.rs` | T3-4, T3-5 |
| `src/lib.rs` | T3-6 |
| `Cargo.toml` | T3-1 |
| `rust-toolchain.toml` | T3-7 |
| `rustfmt.toml`, `.clippy.toml`, `.cargo/config.toml` | T3-8 |
| `src/tests/mcts.rs` | T4-1 (new file, 5 tests) |
| `Python/tests/test_engine_smoke.py` | T4-2 (new file, 4 tests) |
| `.github/workflows/ci.yml` | T4-2 (added `python-integration` job) |
| `tests/board.rs`, `tests/encoder.rs` | T4-3 (moved from `src/tests/`), T4-5 |
| `benches/engine.rs` | T4-4 (new file, 2 benches) |

No regressions detected in the verification pass. Changes are surgical and consistent with the original review.
