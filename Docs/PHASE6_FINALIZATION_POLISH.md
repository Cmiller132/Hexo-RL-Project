# Phase 6 — Finalization & Polish Pass

**Goal:** Working V1. Wire everything end-to-end, fix remaining critical bugs, implement missing target pipelines, get arena evaluation operational.

**Reviewed:** commit `4eb9a79` (Phase 5) + working tree mods implementing PHASE4_IMPLEMENTATION_PLAN.

---

## Phase 4 Plan Verification — All 22 Fixes ✅

P0-1, P0-2, P0-3a/b/c/d, P1-1..6, P2-1..6, P3-1..4 all correctly implemented. Ring buffer stores lookahead, sampler returns 4-tuple, trainer unpacks per-head, server has correct adaptive batching with race fix, EMA tracks buffers, etc.

**However**, verification surfaced 3 critical bugs that block training entirely + several majors. Fix list below.

---

## Execution Order

Fix in this exact order — later fixes depend on earlier ones:

1. C1: Encoder/sampler tensor alignment
2. C2: Position[0] empty-history handling
3. C3: `apply_d6_symmetry` numpy-not-list
4. M1: `compute_losses` graceful missing-target skip + add target generation
5. M2: EMA skip integer buffers
6. M3: Worker_id ↔ slot_id alignment in orchestrator
7. M4: Eliminate remaining bare-except blocks
8. M5: Implement arena vs real engine
9. M6: `effective_decay` consistency
10. Cleanup batch (minors)

---

## C1 — Encoder/Sampler Tensor Alignment ★ BLOCKER

**Symptom:** Every training input is the empty board. Position[0] also crashes the sampler.

**Root cause:** `worker.py` records `position[i].move_history` with the `i` moves played BEFORE position `i`. The Rust encoder generates `N` tensors for `N` moves, where `tensor[k]` = state with `k` moves placed (since it encodes BEFORE each placement). `sampler.py:237-239` takes `tensor[0]` (always empty board).

**Pick ONE of these two equivalent fixes:**

### Option A (recommended) — Encoder generates N+1 tensors

**File:** `crates/hexgame-py/src/encode.rs`

Replace the empty-history check + loop with:

```rust
let num_moves = history_bytes.len() / 12;
// Note: num_moves == 0 is now legal (returns single empty-board tensor)

let bytes_owned: Vec<u8> = history_bytes.to_vec();

let positions = py.allow_threads(move || -> Result<Vec<f32>, String> {
    let mut game = HexGameState::new();
    let mut positions = Vec::with_capacity((num_moves + 1) * TENSOR_SIZE);
    for chunk in bytes_owned.chunks_exact(12) {
        let tensor = encoder::encode_board(&game, near_radius, false).tensor;
        positions.extend_from_slice(&tensor);
        let q = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let r = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
        game.place(q, r).map_err(|e| e.to_string())?;
    }
    // Final state AFTER all moves placed
    let tensor = encoder::encode_board(&game, near_radius, false).tensor;
    positions.extend_from_slice(&tensor);
    Ok(positions)
}).map_err(|e| PyValueError::new_err(e))?;

let shape = (num_moves + 1, NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize);
```

**File:** `Python/src/hexorl/buffer/sampler.py:237-239`

```python
# Before: tensor = tensor[0]
# After:
if tensor.ndim == 4:
    tensor = tensor[-1]   # final state = state at this position's decision
tensors[i] = tensor
```

Same change in the pure-python fallback `_py_decode_compact_record` (return all N+1 states; sampler picks `[-1]`).

### Option B — Worker stores i+1 moves per position

Less surgical (changes record creation order). If you go this route, you must also adjust `from_game_data` and `_split_history_bytes`. Recommend Option A.

---

## C2 — Position[0] Empty History (subsumed by C1)

C1 Option A removes the `num_moves == 0 → Err` early return. After C1, position[0] (empty bytes) returns a single empty-board tensor. No separate fix needed.

---

## C3 — `apply_d6_symmetry` numpy-not-list

**File:** `Python/src/hexorl/buffer/sampler.py:249-253`

```python
# Before:
tensors[i] = np.array(
    _engine.apply_d6_symmetry(tensors[i].tolist(), sym_idx),
    dtype=np.float32,
)
# After:
tensors[i] = np.array(
    _engine.apply_d6_symmetry(tensors[i], sym_idx),
    dtype=np.float32,
)
```

`PyReadonlyArray3<f32>` requires a numpy array. `.tolist()` produces nested Python lists which won't convert.

---

## M1 — Missing Target Pipelines + Graceful Skip

Two parts: (a) make `compute_losses` resilient, (b) implement the missing target generators.

### M1a — Graceful target lookup

**File:** `Python/src/hexorl/train/losses.py:212-237`

Replace the dispatch with key-aware skip:

```python
for head_name, pred in predictions.items():
    if head_name not in loss_weights:
        continue
    weight = loss_weights[head_name]

    # Required-target heads — skip cleanly if target missing.
    REQ = {
        "policy": "policy",
        "value": "value",
        "regret_rank": "regret_rank",
        "regret_value": "regret_value",
        "moves_left": "moves_left",
    }
    if head_name in REQ and REQ[head_name] not in targets:
        continue
    if head_name.startswith("lookahead_") and head_name not in targets:
        continue

    if head_name == "policy":
        loss = policy_loss(pred, targets["policy"])
    elif head_name == "opp_policy":
        loss = opp_policy_loss(pred, targets.get("opp_policy", targets["policy"]))
    elif head_name == "value":
        loss = binned_value_loss(pred, targets["value"], n_bins)
    elif head_name.startswith("lookahead_"):
        loss = binned_value_loss(pred, targets[head_name], n_bins)
    elif head_name == "regret_rank":
        loss = regret_rank_loss(pred.squeeze(-1), targets["regret_rank"])
    elif head_name == "regret_value":
        loss = regret_value_loss(pred, targets["regret_value"], n_bins)
    elif head_name == "axis":
        loss = axis_loss(pred, targets.get("axis"))
    elif head_name == "moves_left":
        loss = moves_left_loss(pred, targets["moves_left"])
    else:
        continue

    per_head[head_name] = weight * loss
```

Note: `pred.squeeze()` (no arg) collapses every singleton dim, including the batch dim if B==1. Use `pred.squeeze(-1)`.

### M1b — Implement target generators

Add to `PositionRecord` (`Python/src/hexorl/selfplay/records.py`):

```python
@dataclass
class PositionRecord:
    move_history: bytes
    policy_target: Dict[int, float]
    root_value: float
    player: int
    outcome: Optional[float] = None
    game_id: int = 0
    is_full_search: bool = True
    turn_index: int = 0
    lookahead_values: List[float] = field(default_factory=list)
    # NEW:
    opp_policy_target: Dict[int, float] = field(default_factory=dict)
    regret_rank: float = 0.0
    regret_value: float = 0.0
    axis_label: int = -1   # -1 = unknown
    moves_left: float = 0.0
```

**Worker** (`worker.py`) — populate during game loop:

- `opp_policy_target`: between turns, after a player makes a move, the *next* position's `opp_policy_target` is THIS position's policy. Implement by buffering the previous policy and assigning to the next record after sampling.
- `axis_label`: derive from the dominant connection axis at game end. Use a simple heuristic: of the 3 hex axes (q, r, q+r), which had the longest contiguous run for the winner? Compute once at game end, label all positions with that integer in {0, 1, 2}. Stub returning -1 if game drew.
- `moves_left`: at game end, set `pos.moves_left = total_game_moves - pos.turn_index`.

**Targets** (`targets.py`) — extend `process_game_record`:

```python
def process_game_record(record, lookahead_horizons=None, lookahead_lambdas=None):
    record.assign_outcomes()

    # Lookahead (existing)
    lookahead_targets = {}
    if lookahead_horizons and lookahead_lambdas:
        for h, lam in zip(lookahead_horizons, lookahead_lambdas):
            lookahead_targets[h] = compute_ema_lookahead(record.positions, horizon=h, lambda_=lam)
    for i, pos in enumerate(record.positions):
        pos.lookahead_values = [float(lookahead_targets[h][i]) for h in lookahead_targets]

    # Regret targets (RGSC Eq.2): per-position cumulative MSE vs outcome
    T = len(record.positions)
    for t, pos in enumerate(record.positions):
        z = record.outcome if pos.player == 0 else -record.outcome
        s = sum((p.root_value - (record.outcome if p.player == 0 else -record.outcome)) ** 2
                for p in record.positions[t:]) / max(T - t, 1)
        pos.regret_value = float(s)
        pos.regret_rank = float(s)   # same scalar; rank loss handles ordering

    # Moves-left
    for i, pos in enumerate(record.positions):
        pos.moves_left = float(T - i)

    # Opp-policy: shift policy_target by 1 (current pos's opp_policy = previous pos's policy)
    prev = {}
    for pos in record.positions:
        pos.opp_policy_target = prev
        prev = pos.policy_target

    # Axis label: simple heuristic from final winning line if any
    record._compute_axis_labels()

    return record.positions
```

Add `_compute_axis_labels` to `GameRecord`:

```python
def _compute_axis_labels(self):
    if abs(self.outcome) < 0.5:
        for p in self.positions: p.axis_label = -1
        return
    # Heuristic: count player's stones along each axis from move history.
    # Axis 0: q-axis, Axis 1: r-axis, Axis 2: q+r diagonal
    winner = 0 if self.outcome > 0 else 1
    counts = [0, 0, 0]
    for p in self.positions:
        if p.player == winner:
            # Use turn index parity as a proxy — replace with real axis detection later if needed.
            counts[p.turn_index % 3] += 1
    label = max(range(3), key=lambda i: counts[i])
    for p in self.positions:
        p.axis_label = label
```

**Ring buffer** (`ring.py`) — store the new fields. Add storage arrays in `__init__`:

```python
self._opp_policies = np.zeros((capacity, max_policy_entries), dtype=np.uint16)
self._opp_policy_probs = np.zeros((capacity, max_policy_entries), dtype=np.float32)
self._opp_policy_counts = np.zeros(capacity, dtype=np.uint16)
self._regret_rank = np.zeros(capacity, dtype=np.float32)
self._regret_value = np.zeros(capacity, dtype=np.float32)
self._axis_label = np.full(capacity, -1, dtype=np.int8)
self._moves_left = np.zeros(capacity, dtype=np.float32)
```

In `_append_unlocked` and `append`, write all of these. In `__getitem__`, read them back into the `PositionRecord`. In `clear`, reset them.

**Sampler** (`sampler.py._sample_batch`) — extend the return tuple:

```python
opp_policies = np.zeros((self.batch_size, BOARD_AREA), dtype=np.float32)
regret_rank = np.zeros(self.batch_size, dtype=np.float32)
regret_value = np.zeros(self.batch_size, dtype=np.float32)
axis_labels = np.full(self.batch_size, -1, dtype=np.int64)
moves_left = np.zeros(self.batch_size, dtype=np.float32)

for i, rec in enumerate(records):
    ...  # existing per-record loop
    # New target population:
    for idx, prob in rec.opp_policy_target.items():
        if 0 <= idx < BOARD_AREA:
            opp_policies[i, idx] = prob
    regret_rank[i] = rec.regret_rank
    regret_value[i] = rec.regret_value
    axis_labels[i] = rec.axis_label
    moves_left[i] = rec.moves_left

# Return as a dict for clarity (BREAKING CHANGE for trainer)
return {
    "tensors": tensors,
    "policy": policies,
    "value": values,
    "lookahead": lookahead_arrays,
    "opp_policy": opp_policies,
    "regret_rank": regret_rank,
    "regret_value": regret_value,
    "axis": axis_labels,
    "moves_left": moves_left,
}
```

**Trainer** (`trainer.py._train_step`) — adapt to the dict format:

```python
def _train_step(self, batch, batch_idx):
    if isinstance(batch, dict):
        tensors = batch["tensors"].to(self.device, non_blocking=True)
        targets = {
            "policy": batch["policy"].to(self.device, non_blocking=True),
            "value":  batch["value"].to(self.device, non_blocking=True),
        }
        for key, lv in zip(self._lookahead_keys, batch.get("lookahead", [])):
            targets[key] = lv.to(self.device, non_blocking=True)
        for k in ("opp_policy", "regret_rank", "regret_value", "moves_left"):
            if k in batch:
                targets[k] = batch[k].to(self.device, non_blocking=True)
        if "axis" in batch:
            ax = batch["axis"].to(self.device, non_blocking=True)
            # axis_loss handles -1 by skipping; mask in compute_losses or filter here
            targets["axis"] = ax if (ax >= 0).any() else None
    else:
        # Backwards-compatible legacy 4-tuple path
        tensors, policies, values, lookahead_list = (*batch, [])[:4] if len(batch) < 4 else batch
        tensors = tensors.to(self.device, non_blocking=True)
        targets = {"policy": policies.to(self.device, non_blocking=True),
                   "value": values.to(self.device, non_blocking=True)}
        for key, lv in zip(self._lookahead_keys, lookahead_list):
            targets[key] = lv.to(self.device, non_blocking=True)

    # ... rest unchanged
```

**`axis_loss`**: extend to ignore `-1` labels via `ignore_index=-1`:

```python
def axis_loss(pred_logits, target_axis):
    if target_axis is None:
        return torch.tensor(0.0, device=pred_logits.device)
    return F.cross_entropy(pred_logits, target_axis, ignore_index=-1)
```

---

## M2 — EMA Skip Integer Buffers

**File:** `Python/src/hexorl/train/ema.py`

`BatchNorm.num_batches_tracked` is `torch.long`. `.mul_(1.0 - d)` raises `RuntimeError`. Skip integer buffers in `_init_shadow`, `update`, `apply_shadow`, `restore`:

```python
def _is_floating(buf):
    return buf is not None and buf.is_floating_point()

# _init_shadow
for name, buf in self.model.named_buffers():
    if _is_floating(buf):
        self._shadow[f"__buf__{name}"] = buf.data.clone().detach()

# update
for name, buf in self.model.named_buffers():
    key = f"__buf__{name}"
    if _is_floating(buf) and key in self._shadow:
        self._shadow[key].mul_(1.0 - d).add_(buf.data, alpha=d)

# apply_shadow / restore: same _is_floating guard
```

Also fix `effective_decay` (M6 below).

---

## M3 — Worker_id ↔ Slot_id Alignment

**File:** `Python/src/hexorl/selfplay/orchestrator.py:142-148, 117-135`

Current `_monitor_workers` swap-pop scrambles indices. The inference server addresses workers by `worker_id` which corresponds to a fixed shared-memory slot. After respawn, `self._workers[i]` no longer corresponds to `worker_id == i`.

Switch to a `dict[int, mp.Process]`:

```python
# __init__
self._workers: Dict[int, mp.Process] = {}

# _spawn_worker
def _spawn_worker(self, worker_id: int):
    worker = SelfPlayWorker(worker_id=worker_id, cfg=self.cfg, ...)
    p = mp.Process(target=worker.run, name=f"selfplay-worker-{worker_id}", daemon=False)
    p.start()
    self._workers[worker_id] = p
    logger.info(f"Worker {worker_id} started (pid={p.pid})")

# _monitor_workers
def _monitor_workers(self):
    for wid, p in list(self._workers.items()):
        if not p.is_alive():
            logger.warning(f"Worker {wid} died — respawning")
            self._spawn_worker(wid)   # overwrites slot with same wid

# stop
for p in self._workers.values():
    if p.is_alive():
        p.terminate(); p.join(timeout=2.0)
self._workers.clear()

# stats
"workers_alive": sum(1 for p in self._workers.values() if p.is_alive()),
"workers_total": len(self._workers),
```

Also remove the dead import `from hexorl.buffer.targets import process_game_record` (line 20).

---

## M4 — Eliminate Remaining Bare-Except Blocks

**File:** `worker.py`

Three blocks remain. Replace each with targeted handling:

**a) Root expansion network failure (lines 442-457):** Keep the fallback (network glitch shouldn't crash the worker) but log + bound retries:

```python
if client is not None:
    try:
        p, v = client.submit(tensor_3d.reshape(1, 13, 33, 33).astype(np.float32), 1)
    except (TimeoutError, ConnectionError, RuntimeError) as e:
        logger.warning(f"Worker {self.worker_id}: root submit failed: {e}; using uniform")
        p = np.ones(1089, dtype=np.float32) / 1089
        v = np.array([0.0], dtype=np.float32)
    engine.expand_root(p, v[0], offset_q, offset_r, legal_bytes)
else:
    engine.expand_root(np.ones(1089, dtype=np.float32) / 1089, 0.0,
                       offset_q, offset_r, legal_bytes)
```

**b) Dirichlet n_children (lines 467-476):** Just call directly — `root_child_priors()` after `expand_root` should always succeed:

```python
if self.dirichlet_alpha > 0:
    child_priors = engine.root_child_priors()
    n_children = len(child_priors) if not hasattr(child_priors, "shape") else child_priors.shape[0]
    if n_children > 0:
        noise = np.random.dirichlet([self.dirichlet_alpha] * n_children)
        engine.add_dirichlet_noise(noise.astype(np.float32), self.dirichlet_fraction)
```

**c) Sims loop (lines 484-507):** Catch only network-class errors; let engine bugs propagate:

```python
while not engine.done():
    batch_tensor, count = engine.select_leaves(self.batch_size)
    if count == 0:
        break
    batch_4d = batch_tensor if isinstance(batch_tensor, np.ndarray) else np.array(batch_tensor)
    if client is not None:
        try:
            p, v = client.submit(batch_4d.astype(np.float32), count)
        except (TimeoutError, ConnectionError, RuntimeError) as e:
            logger.warning(f"Worker {self.worker_id}: sim submit failed: {e}; ending sims")
            break
        engine.expand_and_backprop(p, v)
    else:
        engine.expand_and_backprop(np.ones(count * 1089, dtype=np.float32) / 1089,
                                   np.zeros(count, dtype=np.float32))
```

Note the `/1089` normalization fix on the no-client mock fallback.

---

## M5 — Wire Arena to Real Engine

**File:** `Python/src/hexorl/eval/arena.py`

`_play_mock_match` is currently a stub. Replace with a real game loop using `_engine.HexGame`:

```python
def _play_match(side_a_fn, side_b_fn, game_idx, a_is_black,
                sims=400, max_moves=200) -> MatchResult:
    try:
        import _engine
        game = _engine.HexGame()
    except ImportError:
        return _play_mock_match(side_a_fn, side_b_fn, game_idx, a_is_black, sims, max_moves)

    moves_played = 0
    move_history = []
    reason = "normal"
    winner = -1

    while not game.is_over and moves_played < max_moves:
        player = game.current_player
        is_side_a = (player == 0 and a_is_black) or (player == 1 and not a_is_black)
        current_fn = side_a_fn if is_side_a else side_b_fn

        try:
            move = current_fn(list(move_history), 100, player)
            if move is None or move[0] is None:
                winner = 1 - player   # current player resigned
                reason = "resign"
                break
            q, r = move
        except Exception as e:
            winner = 1 - player
            reason = f"crash:{e}"
            break

        try:
            game.place(q, r)
        except Exception as e:
            # Illegal move = loss for current player
            winner = 1 - player
            reason = f"illegal:{e}"
            break

        move_history.append((player, q, r))
        moves_played += 1

    if winner == -1:
        if game.is_over:
            w = game.winner
            winner = 0 if (w == 0 and a_is_black) or (w == 1 and not a_is_black) else 1
            reason = "engine_terminal"
        else:
            winner = 0 if moves_played % 2 == 0 else 1
            reason = "max_moves"

    return MatchResult(
        winner=winner,
        side_a_score=1.0 if winner == 0 else 0.0,
        side_b_score=1.0 if winner == 1 else 0.0,
        moves=moves_played,
        time_ms=0.0,
        opening_is_black=a_is_black,
        reason=reason,
    )
```

Update `run_arena` to call `_play_match` instead of `_play_mock_match`. Keep `_play_mock_match` as the import-fallback.

**Note for `side_a_fn` callers:** `classical_opponent_fn` in `classical.py` calls `opponent.reset()` then replays `move_history` — that's fine. For an MCTS-based side, write a wrapper that constructs an MCTSEngine, replays moves, runs `sims`, returns `sample_action(temp=0)`.

Add helper `mcts_opponent_fn(client, sims, c_puct=1.5, near_radius=8)` to `arena.py` (uses the inference client to evaluate; returns `(q, r)`). This lets you run model-vs-classical and model-vs-model arenas.

---

## M6 — `effective_decay` Consistency

**File:** `ema.py:137-140`

```python
@property
def effective_decay(self) -> float:
    return min(self.decay, 1.0 - 1.0 / (1.0 + max(self._num_updates, 1)))
```

---

## Cleanup Batch (Minor)

Apply all of these in one pass:

| # | File | Line | Change |
|---|---|---|---|
| 1 | `orchestrator.py` | 20 | Remove `from hexorl.buffer.targets import process_game_record` (dead import) |
| 2 | `records.py` | 200-251 | `from_game_data` re-splits move history. Pass `pos_histories` directly when caller already has them. Or just delete the redundant call path in worker. |
| 3 | `records.py` | 289-299 | `action_to_board_index` defaults `offset_q=16, offset_r=16` are inconsistent with Rust encoder. Make defaults required (no default) so the caller must pass actual offsets from `init_root()`. |
| 4 | `targets.py` | 84-94 | `boundaries.index(i)` is O(N) inside a loop. Replace with a precomputed `pos_to_bi` dict: `pos_to_bi = {b: bi for bi, b in enumerate(boundaries)}`. For non-boundary i, use `bisect.bisect_left(boundaries, i)`. |
| 5 | `worker.py` | 502-505 | Mock fallback policies should be normalized: `np.ones(count * 1089) / 1089` (handled in M4c). |
| 6 | `losses.py` | 229 | `pred.squeeze()` → `pred.squeeze(-1)` to avoid collapsing batch dim when B=1. |

---

## Verification Checklist

After all fixes applied, run these in order:

1. `cargo test -p hexgame-core && cargo test -p hexgame-py` — all green
2. `cargo build --release -p hexgame-py` — extension builds
3. **Encoder smoke test:** call `_engine.encode_compact_record(b"", 8)` → returns `(1, 13, 33, 33)` empty board (no error)
4. **Encoder smoke test:** call with 3-move history → returns `(4, 13, 33, 33)`; first frame all-zero; final frame has 3 stones
5. **Symmetry smoke test:** `_engine.apply_d6_symmetry(np.zeros((13,33,33), dtype=np.float32), 5)` → returns `(13,33,33)` no error
6. **Buffer roundtrip:** append 10 records with all new fields populated, sample 5, confirm all per-head targets non-zero
7. **Trainer smoke:** load `default.toml`, run 10 batches, confirm loss decreases for `policy`, `value`, `lookahead_*`, `regret_*`, `axis`, `moves_left`, `opp_policy` (no KeyError, no NaN)
8. **EMA smoke:** train 100 steps, call `ema.apply_shadow()` then `ema.restore()`; confirm model state unchanged after restore
9. **Server smoke:** start server + 4 workers, run 1 minute, confirm no `TimeoutError` from clients, no stale results, `n_batches > 0`
10. **Arena smoke:** `run_arena(classical_opponent_fn(100, 4), classical_opponent_fn(100, 4), num_games=4)` → completes, returns valid `ArenaStats` with non-mock `reason` values
11. **Orchestrator respawn:** kill a worker mid-run, confirm orchestrator respawns it with the **same** worker_id (check log)
12. **End-to-end:** run `Configs/default.toml` for 1 epoch (small `games_per_epoch=8`, `batches_per_epoch=20`); confirm checkpoint saves, EMA swap roundtrips, buffer fills

---

## Out of Scope (Future Work)

- True axis label detection from winning line geometry (current heuristic uses turn parity)
- True opp_policy from opponent's MCTS visits (current uses lagged self-policy)
- CUDA stream pipelining in inference server (still single stream)
- Zero-copy FFI for `select_leaves` / `expand_and_backprop` (still `.to_vec()`)
- Bayesian ELO with full likelihood (current is iterative MLE)
- Regret buffer integration into self-play restart logic (RGSC §3.3 PRB sampling)

---

## Summary

**3 critical bugs** (C1-C3) currently make training impossible.
**6 majors** (M1-M6) including missing target pipelines and broken arena.
**6 minors** (cleanup batch).

After this pass, V1 is a complete training pipeline: self-play → buffer → trainer → EMA → arena evaluation → ELO. Open items above are quality improvements, not correctness blockers.
