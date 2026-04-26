# Phase 4 Implementation Plan

**Source reviewed:** commit `d09b2d6` (Phase 4)  
**Reviewer:** Claude Sonnet  
**Date:** 2026-04-26  

This document gives exact, unambiguous before/after code for every fix required in the Phase 4 training stack. Items are ordered: **P0** (training produces wrong/garbage results) → **P1** (critical runtime crashes or data corruption) → **P2** (major correctness/performance) → **P3** (cleanup/minor).

All fixes must be applied to the files as they exist in git HEAD. The working tree currently has these files deleted — restore them with `git checkout HEAD -- Python/` before starting.

---

## Change Execution Order

Apply changes in this exact sequence to avoid dependency breaks:

1. P0-1: Fix `encode_compact_record` (Rust)  
2. P0-2: Fix policy target in `worker.py`  
3. P0-3a: Add lookahead storage to `RingBuffer`  
4. P0-3b: Update `ReplayDataset` to yield per-head targets  
5. P0-3c: Fix `Trainer._train_step` to consume per-head targets  
6. P0-3d: Fix `Trainer.train_epoch` to track all head losses  
7. P1-1: Fix inference server race condition  
8. P1-2: Enforce `max_batch_size` in server drain  
9. P1-3: Fix adaptive batching  
10. P1-4: Fix resignation timing  
11. P1-5: Lock `RingBuffer.__getitem__`  
12. P1-6: Remove bare `except` traps  
13. P2-1: Add buffers to EMA tracking  
14. P2-2: Fix EMA adaptive warmup  
15. P2-3: Fix `ReplayDataset` base class  
16. P2-4: Fix double-processing in orchestrator  
17. P2-5: Fix `mp.Event` threading mismatch  
18. P2-6: Fix `regret_rank_loss` scaling  
19. P3-1: Track all head losses in epoch stats  
20. P3-2: Delete stale root `src/` directory  
21. P3-3: Fix `mp.Event` in orchestrator  

---

## P0 — Training Produces Wrong Results

These bugs make self-play data completely wrong before any gradient is computed. Fix all of them before attempting a training run.

---

### P0-1 — `encode_compact_record` double-places the opening stone

**File:** `crates/hexgame-py/src/encode.rs`  
**Lines:** 34–45  

**Problem:** The first entry in any move history is `(player=0, q=0, r=0)` (the opening stone, which the game rules force to the origin). The current code calls `game.place(0, 0)` hardcoded AND then calls `game.place(q, r)` from the history bytes. For the first move, both calls are `(0, 0)` — the second call hits an already-occupied cell and returns `Err(AlreadyOccupied)` → `PyValueError` crash.

For mock-engine histories where the first (q, r) is non-origin, the encoder silently places a phantom stone at (0,0) and then treats the actual first coordinate as the second stone of the opening turn, encoding a two-stone board when the game had only one. Every tensor derived from a mock game is corrupted.

**Before:**
```rust
for chunk in history_bytes.chunks_exact(12) {
    let tensor = encoder::encode_board(&game, near_radius, false).tensor;
    positions.extend_from_slice(&tensor);

    let _player = i32::from_le_bytes(chunk[0..4].try_into().unwrap()) as u8;
    let q = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
    let r = i32::from_le_bytes(chunk[8..12].try_into().unwrap());

    if game.move_count() == 0 {
        game.place(0, 0).map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    game.place(q, r).map_err(|e| PyValueError::new_err(e.to_string()))?;
}
```

**After:**
```rust
for chunk in history_bytes.chunks_exact(12) {
    let tensor = encoder::encode_board(&game, near_radius, false).tensor;
    positions.extend_from_slice(&tensor);

    let q = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
    let r = i32::from_le_bytes(chunk[8..12].try_into().unwrap());

    game.place(q, r).map_err(|e| PyValueError::new_err(e.to_string()))?;
}
```

**What changed:** Delete the `let _player` line, delete the entire `if game.move_count() == 0 { ... }` block. The game's own validation already enforces that the first move must be at origin — no special-casing is needed.

---

### P0-2 — Policy target uses neural-net priors instead of MCTS visit counts

**File:** `Python/src/hexorl/selfplay/worker.py`  
**Lines:** 508–530  

**Problem:** `engine.get_results()` returns `(moves_q, moves_r, visits, root_value)`. The `visits` list is the MCTS search-improved policy. The code ignores it entirely and builds the policy target from `engine.root_child_priors()` — the raw neural-network output. This trains the network to predict its own prior, making MCTS self-play a no-op.

Additionally, `offset_q` and `offset_r` (the board coordinate offsets from `engine.init_root()`) are available in scope but not used for the coordinate-to-flat-index mapping.

**Before:**
```python
            try:
                moves_q, moves_r, visits, root_value = engine.get_results()
                priors = engine.root_child_priors()
                q_values = engine.root_child_q_values()
            except Exception:
                visits = [1] * 10
                priors = [0.1] * 10
                q_values = [0.0] * 10
                root_value = 0.0

            temp = get_temperature(move_idx, self.temperature_schedule)
            q, r = engine.sample_action(temp)
            if q is None:
                q, r = 0, 0

            if HAS_ENGINE:
                player = engine._game.current_player
            else:
                player = move_idx % 2

            record_history = bytes(move_history)

            policy = sparsify_policy(np.array(priors), top_k=20)
```

**After:**
```python
            moves_q, moves_r, visits, root_value = engine.get_results()
            priors = engine.root_child_priors()
            q_values = engine.root_child_q_values()

            temp = get_temperature(move_idx, self.temperature_schedule)
            q, r = engine.sample_action(temp)
            if q is None:
                q, r = 0, 0

            if HAS_ENGINE:
                player = engine._game.current_player
            else:
                player = move_idx % 2

            record_history = bytes(move_history)

            # Build visit distribution over the full board (MCTS-improved policy).
            # offset_q/offset_r come from init_root() above — still in scope.
            visit_arr = np.zeros(BOARD_AREA, dtype=np.float32)
            for q_coord, r_coord, v in zip(moves_q, moves_r, visits):
                flat_idx = action_to_board_index(q_coord, r_coord, offset_q, offset_r)
                visit_arr[flat_idx] = float(v)
            policy = sparsify_policy(visit_arr, top_k=20)
```

**What changed:** Remove the `try/except` around `get_results()` (see also P1-6). Replace `sparsify_policy(np.array(priors), ...)` with a visit-count array built by mapping each child's `(q, r)` coordinates to a flat board index using `action_to_board_index`.

**Note:** `BOARD_AREA` is already imported at the top of `worker.py` via `from hexorl.selfplay.records import ... BOARD_AREA`. `action_to_board_index` is also imported from the same module.

---

### P0-3a — Ring buffer must store per-head target arrays

**File:** `Python/src/hexorl/buffer/ring.py`  

**Problem:** `targets.py` computes EMA lookahead values and stores them in `pos.lookahead_values` (a list, one float per configured horizon). The ring buffer's `append` discards them — it only stores `record.to_value_target()` (a single scalar). Lookahead training targets never reach the DataLoader or trainer.

The buffer config has `lookahead_horizons: [4, 12, 36]` (3 horizons). The number of horizons must be known at buffer construction time to pre-allocate storage arrays.

**Change `RingBuffer.__init__`** — add a `num_lookahead` parameter and storage arrays:

**Before:**
```python
    def __init__(
        self,
        capacity: int,
        max_policy_entries: int = 20,
        recency_decay: float = 0.99,
    ):
        self.capacity = capacity
        self.max_policy_entries = max_policy_entries
        self.recency_decay = recency_decay

        # Storage arrays — struct of arrays
        self._histories: List[Optional[bytes]] = [None] * capacity
        self._policies = np.zeros((capacity, max_policy_entries), dtype=np.uint16)
        self._policy_probs = np.zeros((capacity, max_policy_entries), dtype=np.float32)
        self._policy_counts = np.zeros(capacity, dtype=np.uint16)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._game_ids = np.zeros(capacity, dtype=np.uint32)
        self._is_full = np.zeros(capacity, dtype=np.bool_)
        self._players = np.zeros(capacity, dtype=np.uint8)
```

**After:**
```python
    def __init__(
        self,
        capacity: int,
        max_policy_entries: int = 20,
        recency_decay: float = 0.99,
        num_lookahead: int = 0,
    ):
        self.capacity = capacity
        self.max_policy_entries = max_policy_entries
        self.recency_decay = recency_decay
        self.num_lookahead = num_lookahead

        # Storage arrays — struct of arrays
        self._histories: List[Optional[bytes]] = [None] * capacity
        self._policies = np.zeros((capacity, max_policy_entries), dtype=np.uint16)
        self._policy_probs = np.zeros((capacity, max_policy_entries), dtype=np.float32)
        self._policy_counts = np.zeros(capacity, dtype=np.uint16)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._game_ids = np.zeros(capacity, dtype=np.uint32)
        self._is_full = np.zeros(capacity, dtype=np.bool_)
        self._players = np.zeros(capacity, dtype=np.uint8)
        # Per-horizon lookahead value targets — shape (capacity, num_lookahead)
        if num_lookahead > 0:
            self._lookahead = np.zeros((capacity, num_lookahead), dtype=np.float32)
        else:
            self._lookahead = None
```

**Change `_append_unlocked`** — write lookahead values if present:

**Before:**
```python
    def _append_unlocked(self, record: PositionRecord):
        """Internal append — caller holds self._lock."""
        idx = self._head
        self._histories[idx] = record.move_history
        entries = list(record.policy_target.items())
        n = min(len(entries), self.max_policy_entries)
        self._policy_counts[idx] = n
        for j, (action_idx, prob) in enumerate(entries[:n]):
            self._policies[idx, j] = action_idx
            self._policy_probs[idx, j] = prob
        self._values[idx] = record.to_value_target()
        self._game_ids[idx] = record.game_id
        self._is_full[idx] = record.is_full_search
        self._players[idx] = record.player
        self._head = (self._head + 1) % self.capacity
        if self._size == self.capacity:
            self._tail = (self._tail + 1) % self.capacity
        else:
            self._size += 1
        self._max_game_id = max(self._max_game_id, record.game_id)
```

**After:**
```python
    def _append_unlocked(self, record: PositionRecord):
        """Internal append — caller holds self._lock."""
        idx = self._head
        self._histories[idx] = record.move_history
        entries = list(record.policy_target.items())
        n = min(len(entries), self.max_policy_entries)
        self._policy_counts[idx] = n
        for j, (action_idx, prob) in enumerate(entries[:n]):
            self._policies[idx, j] = action_idx
            self._policy_probs[idx, j] = prob
        self._values[idx] = record.to_value_target()
        self._game_ids[idx] = record.game_id
        self._is_full[idx] = record.is_full_search
        self._players[idx] = record.player
        if self._lookahead is not None:
            lv = record.lookahead_values
            k = min(len(lv), self.num_lookahead)
            self._lookahead[idx, :k] = lv[:k]
            if k < self.num_lookahead:
                self._lookahead[idx, k:] = self._values[idx]  # bootstrap with main value
        self._head = (self._head + 1) % self.capacity
        if self._size == self.capacity:
            self._tail = (self._tail + 1) % self.capacity
        else:
            self._size += 1
        self._max_game_id = max(self._max_game_id, record.game_id)
```

**Apply the same lookahead write block to the `append` method** (the public single-record path that does NOT call `_append_unlocked`). Find this block inside `append`:
```python
        self._values[idx] = record.to_value_target()
        self._game_ids[idx] = record.game_id
        self._is_full[idx] = record.is_full_search
        self._players[idx] = record.player
```
and insert immediately after `self._players[idx] = record.player`:
```python
        if self._lookahead is not None:
            lv = record.lookahead_values
            k = min(len(lv), self.num_lookahead)
            self._lookahead[idx, :k] = lv[:k]
            if k < self.num_lookahead:
                self._lookahead[idx, k:] = self._values[idx]
```

**Change `__getitem__`** — return lookahead array as part of `PositionRecord`:

Find the `return PositionRecord(...)` block at the end of `__getitem__` and add `lookahead_values`:

**Before:**
```python
        return PositionRecord(
            move_history=self._histories[idx],
            policy_target=policy,
            root_value=0.0,
            player=player,
            game_id=int(self._game_ids[idx]),
            is_full_search=bool(self._is_full[idx]),
            outcome=outcome,
        )
```

**After:**
```python
        lv: List[float] = []
        if self._lookahead is not None:
            lv = self._lookahead[idx].tolist()

        return PositionRecord(
            move_history=self._histories[idx],
            policy_target=policy,
            root_value=0.0,
            player=player,
            game_id=int(self._game_ids[idx]),
            is_full_search=bool(self._is_full[idx]),
            outcome=outcome,
            lookahead_values=lv,
        )
```

**Change `clear`** — reset the lookahead array:

At the bottom of `clear`, after `self._max_game_id = 0`, add:
```python
            if self._lookahead is not None:
                self._lookahead.fill(0.0)
```

**Change `SelfPlayOrchestrator.__init__`** (`orchestrator.py`) — pass `num_lookahead` when constructing the buffer:

**Before:**
```python
        self._buffer = RingBuffer(
            capacity=buffer_capacity,
            recency_decay=cfg.buffer.recency_decay,
        )
```

**After:**
```python
        self._buffer = RingBuffer(
            capacity=buffer_capacity,
            recency_decay=cfg.buffer.recency_decay,
            num_lookahead=len(cfg.buffer.lookahead_horizons),
        )
```

---

### P0-3b — `ReplayDataset` must yield per-head targets

**File:** `Python/src/hexorl/buffer/sampler.py`  

**Change `ReplayDataset.__init__`** — accept horizon list for target labelling:

**Before:**
```python
    def __init__(
        self,
        buffer: RingBuffer,
        batch_size: int = 256,
        recency_decay: float = 0.99,
        pcr_weight: float = 0.25,
        use_symmetry: bool = True,
        near_radius: int = 8,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.pcr_weight = pcr_weight
        self.use_symmetry = use_symmetry
        self.near_radius = near_radius

        self._rng = np.random.RandomState()
```

**After:**
```python
    def __init__(
        self,
        buffer: RingBuffer,
        batch_size: int = 256,
        recency_decay: float = 0.99,
        pcr_weight: float = 0.25,
        use_symmetry: bool = True,
        near_radius: int = 8,
        lookahead_horizons: Optional[List[int]] = None,
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.pcr_weight = pcr_weight
        self.use_symmetry = use_symmetry
        self.near_radius = near_radius
        self.lookahead_horizons = lookahead_horizons or []

        self._rng = np.random.RandomState()
```

**Change `_sample_batch`** — allocate and populate lookahead arrays, return them in the tuple:

**Before (bottom of _sample_batch):**
```python
        for i, rec in enumerate(records):
            if HAS_ENGINE and hasattr(_engine, 'encode_compact_record'):
                tensor = np.array(
                    _engine.encode_compact_record(rec.move_history, self.near_radius),
                    dtype=np.float32,
                )
                if tensor.ndim == 4:
                    tensor = tensor[0]
                tensors[i] = tensor
            else:
                decoded = _py_decode_compact_record(rec.move_history, self.near_radius)
                if decoded.ndim == 4:
                    tensors[i] = decoded[-1]
                else:
                    tensors[i] = decoded

            if self.use_symmetry:
                sym_idx = self._rng.randint(0, 12)
                if HAS_ENGINE and hasattr(_engine, 'apply_d6_symmetry'):
                    tensors[i] = np.array(
                        _engine.apply_d6_symmetry(tensors[i].tolist(), sym_idx),
                        dtype=np.float32,
                    )
                else:
                    tensors[i] = _py_apply_d6_symmetry(tensors[i], sym_idx)

            policies[i] = rec.to_dense_policy()
            values[i] = rec.to_value_target()

        return tensors, policies, values
```

**After:**
```python
        n_lookahead = len(self.lookahead_horizons)
        lookahead_arrays = [
            np.zeros(self.batch_size, dtype=np.float32) for _ in range(n_lookahead)
        ]

        for i, rec in enumerate(records):
            if HAS_ENGINE and hasattr(_engine, 'encode_compact_record'):
                tensor = np.array(
                    _engine.encode_compact_record(rec.move_history, self.near_radius),
                    dtype=np.float32,
                )
                if tensor.ndim == 4:
                    tensor = tensor[0]
                tensors[i] = tensor
            else:
                decoded = _py_decode_compact_record(rec.move_history, self.near_radius)
                if decoded.ndim == 4:
                    tensors[i] = decoded[-1]
                else:
                    tensors[i] = decoded

            if self.use_symmetry:
                sym_idx = self._rng.randint(0, 12)
                if HAS_ENGINE and hasattr(_engine, 'apply_d6_symmetry'):
                    tensors[i] = np.array(
                        _engine.apply_d6_symmetry(tensors[i].tolist(), sym_idx),
                        dtype=np.float32,
                    )
                else:
                    tensors[i] = _py_apply_d6_symmetry(tensors[i], sym_idx)

            policies[i] = rec.to_dense_policy()
            values[i] = rec.to_value_target()

            for h_idx in range(n_lookahead):
                if h_idx < len(rec.lookahead_values):
                    lookahead_arrays[h_idx][i] = rec.lookahead_values[h_idx]
                else:
                    lookahead_arrays[h_idx][i] = values[i]  # fallback

        return tensors, policies, values, lookahead_arrays
```

**Change `__iter__`** — the return type changes, no code change needed (Python duck types this).

---

### P0-3c — Trainer must consume per-head targets

**File:** `Python/src/hexorl/train/trainer.py`  

**Change `__init__`** — accept horizon names for target labelling:

Add after `self._n_bins = getattr(self.model, 'n_bins', 65)`:
```python
        # Lookahead horizon names derived from buffer config
        self._lookahead_keys: List[str] = [
            f"lookahead_{h}" for h in getattr(cfg.buffer, 'lookahead_horizons', [])
        ]
```

Add `from typing import List` to the imports if not already present (it is already in the file).

**Change `_train_step`** — unpack multi-head batch and build complete targets dict:

**Before:**
```python
    def _train_step(self, batch, batch_idx: int) -> Dict[str, float]:
        tensors, policies, values = batch
        tensors = tensors.to(self.device, non_blocking=True)
        policies = policies.to(self.device, non_blocking=True)
        values = values.to(self.device, non_blocking=True)

        targets = {"policy": policies, "value": values}
```

**After:**
```python
    def _train_step(self, batch, batch_idx: int) -> Dict[str, float]:
        # Batch is (tensors, policies, values[, lookahead_list])
        # lookahead_list is a list of per-horizon arrays when present.
        if len(batch) == 4:
            tensors, policies, values, lookahead_list = batch
        else:
            tensors, policies, values = batch
            lookahead_list = []

        tensors = tensors.to(self.device, non_blocking=True)
        policies = policies.to(self.device, non_blocking=True)
        values = values.to(self.device, non_blocking=True)

        targets: Dict[str, torch.Tensor] = {"policy": policies, "value": values}

        for key, lv_arr in zip(self._lookahead_keys, lookahead_list):
            targets[key] = lv_arr.to(self.device, non_blocking=True)
```

**Note:** `lookahead_list` from `ReplayDataset` is a Python list of numpy arrays. The DataLoader collate_fn will stack each array into a (B,) tensor. The `zip` above will pair each horizon key with its corresponding tensor. If the model's `heads` list contains `"lookahead_4"`, `"lookahead_12"`, etc. and the target keys match (they will if `cfg.buffer.lookahead_horizons = [4, 12, 36]` and the head names are `"lookahead_4"`, `"lookahead_12"`, `"lookahead_36"`), `compute_losses` will find all targets.

**Action required on config:** Ensure `cfg.model.heads` uses the same naming as `cfg.buffer.lookahead_horizons`. If `lookahead_horizons = [4, 12, 36]`, then model heads should be `["policy", "value", "lookahead_4", "lookahead_12", "lookahead_36", ...]`. Update `Configs/default.toml` and `Configs/production.toml` head lists accordingly.

---

### P0-3d — Track all configured head losses in epoch stats

**File:** `Python/src/hexorl/train/trainer.py`  

**Change `train_epoch`** — initialize `_epoch_losses` for all heads that have weights:

**Before:**
```python
        self._epoch_losses = {k: [] for k in ("total", "policy", "value")}
```

**After:**
```python
        tracked_keys = ["total"] + list(self.train_cfg.loss_weights.keys())
        self._epoch_losses = {k: [] for k in tracked_keys}
```

**Change the accumulation loop** in `train_epoch` — replace the `if k in self._epoch_losses` guard so all returned loss keys are accumulated:

**Before:**
```python
            for k, v in loss_dict.items():
                if k in self._epoch_losses:
                    self._epoch_losses[k].append(v)
```

**After:**
```python
            for k, v in loss_dict.items():
                if k not in self._epoch_losses:
                    self._epoch_losses[k] = []
                self._epoch_losses[k].append(v)
```

---

## P1 — Critical Runtime Bugs

---

### P1-1 — Inference server race condition: `req_ready` cleared after forward pass

**File:** `Python/src/hexorl/inference/server.py`  
**Lines:** 165–178  

**Problem:** The server clears `req_ready` *after* the forward pass completes. If a worker submits a new request during the forward pass, the server overwrites it with the previous batch's stale results and then clears the new `req_ready`. The worker receives wrong inference data with no error.

**Before:**
```python
                if total_count > 0:
                    policies, values = await loop.run_in_executor(
                        None, self._forward, batch_tensor
                    )

                    self._scatter_results(ready_workers, per_worker_counts, policies, values)

                    for worker_id in ready_workers:
                        slot = self._queue.get_slot(worker_id)
                        slot.req_ready.clear()
                        slot.res_ready.set()
```

**After:**
```python
                if total_count > 0:
                    # Clear req_ready BEFORE the forward pass so any new request
                    # from the worker during the forward will be visible as a fresh
                    # req_ready signal on the next drain cycle.
                    for worker_id in ready_workers:
                        self._queue.get_slot(worker_id).req_ready.clear()

                    policies, values = await loop.run_in_executor(
                        None, self._forward, batch_tensor
                    )

                    self._scatter_results(ready_workers, per_worker_counts, policies, values)

                    for worker_id in ready_workers:
                        self._queue.get_slot(worker_id).res_ready.set()
```

---

### P1-2 — No `max_batch_size` enforcement in server drain

**File:** `Python/src/hexorl/inference/server.py`  
**Method:** `_drain_ready_workers`  

**Problem:** `_drain_ready_workers` collects every ready worker regardless of cumulative position count. With 30 workers each submitting 8 leaves, the GPU receives 240 positions when `max_batch=128`, causing OOM.

**Before:**
```python
    def _drain_ready_workers(self) -> List[int]:
        """Collect worker IDs whose req_ready event is set.

        Returns list of ready worker IDs (may be empty).
        """
        ready = []
        for i in range(self.num_workers):
            slot = self._queue.get_slot(i)
            if slot.req_ready.is_set():
                count = int(slot.req_count[0])
                if count > 0:
                    ready.append(i)
                else:
                    slot.req_ready.clear()

        return ready
```

**After:**
```python
    def _drain_ready_workers(self, max_total: Optional[int] = None) -> List[int]:
        """Collect worker IDs whose req_ready event is set.

        Stops accumulating when the cumulative position count reaches max_total.
        Returns list of ready worker IDs (may be empty).
        """
        if max_total is None:
            max_total = self.max_batch

        ready = []
        total = 0
        for i in range(self.num_workers):
            if total >= max_total:
                break
            slot = self._queue.get_slot(i)
            if slot.req_ready.is_set():
                count = int(slot.req_count[0])
                if count > 0:
                    ready.append(i)
                    total += count
                else:
                    slot.req_ready.clear()

        return ready
```

Also add `from typing import Optional` to the imports at the top of `server.py` if not already present (it is already present).

---

### P1-3 — Adaptive batching inverted

**File:** `Python/src/hexorl/inference/server.py`  
**Method:** `_event_loop`  

**Problem:** The server processes immediately when any worker is ready, and only sleeps when *no* workers are ready. This means every worker that arrives slightly late gets processed in a separate 1-element batch, preventing GPU saturation.

The correct behaviour: wait `max_wait_us` microseconds after the *first* ready worker arrives before draining, to allow more workers to accumulate.

**Before:**
```python
        while not self._stop_event.is_set():
            ready_workers = self._drain_ready_workers()

            if ready_workers:
                batch_tensor, per_worker_counts, total_count = self._build_batch(ready_workers)

                if total_count > 0:
                    policies, values = await loop.run_in_executor(
                        None, self._forward, batch_tensor
                    )

                    self._scatter_results(ready_workers, per_worker_counts, policies, values)

                    for worker_id in ready_workers:
                        slot = self._queue.get_slot(worker_id)
                        slot.req_ready.clear()
                        slot.res_ready.set()

                    self.n_batches += 1
                    self.n_positions += total_count
            else:
                await asyncio.sleep(0.0)
                await asyncio.sleep(self.max_wait_us / 1_000_000.0)
```

**After:**
```python
        wait_s = self.max_wait_us / 1_000_000.0

        while not self._stop_event.is_set():
            # Poll until at least one worker is ready, then wait for more.
            if not self._any_worker_ready():
                await asyncio.sleep(wait_s)
                continue

            # At least one worker ready — wait max_wait_us for more to arrive.
            await asyncio.sleep(wait_s)

            ready_workers = self._drain_ready_workers(max_total=self.max_batch)
            if not ready_workers:
                continue

            batch_tensor, per_worker_counts, total_count = self._build_batch(ready_workers)

            if total_count > 0:
                # Clear req_ready before the forward pass (P1-1 fix).
                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).req_ready.clear()

                policies, values = await loop.run_in_executor(
                    None, self._forward, batch_tensor
                )

                self._scatter_results(ready_workers, per_worker_counts, policies, values)

                for worker_id in ready_workers:
                    self._queue.get_slot(worker_id).res_ready.set()

                self.n_batches += 1
                self.n_positions += total_count
```

Add the helper method `_any_worker_ready` to `InferenceServer`:

```python
    def _any_worker_ready(self) -> bool:
        """Return True if at least one worker slot has req_ready set."""
        for i in range(self.num_workers):
            if self._queue.get_slot(i).req_ready.is_set():
                return True
        return False
```

**Note:** If P1-1 fix is applied first (req_ready cleared before forward), remove the duplicate `slot.req_ready.clear()` call inside the `if total_count > 0:` block — it is already handled above.

---

### P1-4 — Resignation checked after `re_root` (wrong player perspective)

**File:** `Python/src/hexorl/selfplay/worker.py`  
**Lines:** 549–558  

**Problem:** `engine.should_resign(threshold)` is called after `engine.re_root(q, r, sims)`. `re_root` advances the game to the opponent's turn. The resign threshold is therefore evaluated from the *opponent's* perspective on the *next* position, not the position that was just searched.

**Before:**
```python
            try:
                engine.re_root(q, r, sims)
            except Exception:
                pass

            if engine.is_over:
                break

            if resign_enabled and engine.should_resign(self.resign_threshold):
                break
```

**After:**
```python
            # Check resignation BEFORE advancing the tree to the next position.
            if resign_enabled and engine.should_resign(self.resign_threshold):
                break

            engine.re_root(q, r, sims)

            if engine.is_over:
                break
```

**Note:** The `try/except` around `re_root` is removed here — see P1-6 for that fix.

---

### P1-5 — `RingBuffer.__getitem__` is not thread-safe

**File:** `Python/src/hexorl/buffer/ring.py`  
**Method:** `__getitem__`  

**Problem:** DataLoader worker threads call `__getitem__` concurrently while the record-collector thread calls `append`/`extend` under `self._lock`. Reads of `_histories`, `_policy_counts`, `_policies`, `_policy_probs`, `_values`, `_players`, and `_game_ids` can see torn writes.

**Before:**
```python
    def __getitem__(self, idx: int) -> Optional[PositionRecord]:
        """Retrieve a single position record by physical index."""
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"Index {idx} out of range [0, {self.capacity})")
        if self._histories[idx] is None:
            return None
```

**After:**
```python
    def __getitem__(self, idx: int) -> Optional[PositionRecord]:
        """Retrieve a single position record by physical index. Thread-safe."""
        if idx < 0 or idx >= self.capacity:
            raise IndexError(f"Index {idx} out of range [0, {self.capacity})")
        with self._lock:
            if self._histories[idx] is None:
                return None
```

The `with self._lock:` block must cover the entire body of `__getitem__` through the final `return PositionRecord(...)`. Indent all existing lines after `if self._histories[idx] is None: return None` by 4 spaces to place them inside the `with` block.

---

### P1-6 — Bare `except` traps mask fatal engine errors

**File:** `Python/src/hexorl/selfplay/worker.py`  

**Problem A — `get_results()` fallback fabricates fake data** (lines 508-516):

```python
            except Exception:
                visits = [1] * 10
                priors = [0.1] * 10
                q_values = [0.0] * 10
                root_value = 0.0
```

If `get_results()` raises (corrupted engine state), this fabricates uniform visits and inserts garbage into the game record silently.

**Fix:** Remove the `try/except` entirely. The P0-2 fix already removes the wrapping `try` block. Let exceptions propagate to the outer crash handler in `run()`, which logs and respawns the worker.

**Problem B — `re_root()` exception swallowed** (lines 549-553):

```python
            try:
                engine.re_root(q, r, sims)
            except Exception:
                pass
```

`re_root` can raise `PyValueError` (e.g., `MCTSError::ChildNotFound`). Swallowing it leaves the engine in an undefined state and the game loop continues with a stale tree.

**Fix:** Replace with a bare call — let it propagate:
```python
            engine.re_root(q, r, sims)
```

Exceptions will propagate to the `except Exception as e:` handler in `run()` that already increments `_crash_count` and sleeps before retrying.

---

## P2 — Major Correctness & Performance

---

### P2-1 — EMA does not track BatchNorm running statistics

**File:** `Python/src/hexorl/train/ema.py`  

**Problem:** `_init_shadow`, `update`, `apply_shadow`, and `restore` all iterate `model.named_parameters()`, which only returns trainable tensors (weight/bias). BatchNorm's `running_mean`, `running_var`, and `num_batches_tracked` are registered as **buffers** (`named_buffers()`), not parameters. After `apply_shadow()`, the model has EMA conv/linear weights but live-training BatchNorm statistics — inference is miscalibrated.

**Change `_init_shadow`:**

**Before:**
```python
    def _init_shadow(self):
        """Copy all model parameters into shadow storage."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()
```

**After:**
```python
    def _init_shadow(self):
        """Copy all model parameters and persistent buffers into shadow storage."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone().detach()
        # Also track BatchNorm running stats and other persistent buffers.
        for name, buf in self.model.named_buffers():
            if buf is not None:
                self._shadow[f"__buf__{name}"] = buf.data.clone().detach()
```

**Change `update`:**

**Before:**
```python
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._shadow[name].mul_(1.0 - d).add_(param.data, alpha=d)
```

**After:**
```python
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._shadow[name].mul_(1.0 - d).add_(param.data, alpha=d)
            # Update buffer shadows (BatchNorm stats etc.) with same decay.
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if buf is not None and key in self._shadow:
                    self._shadow[key].mul_(1.0 - d).add_(buf.data, alpha=d)
```

**Change `apply_shadow`:**

**Before:**
```python
    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._backup[name] = param.data.clone()
                    param.data.copy_(self._shadow[name])
```

**After:**
```python
    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self._shadow:
                    self._backup[name] = param.data.clone()
                    param.data.copy_(self._shadow[name])
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if buf is not None and key in self._shadow:
                    self._backup[key] = buf.data.clone()
                    buf.data.copy_(self._shadow[key])
```

**Change `restore`:**

**Before:**
```python
    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup.pop(name))
```

**After:**
```python
    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._backup:
                    param.data.copy_(self._backup.pop(name))
            for name, buf in self.model.named_buffers():
                key = f"__buf__{name}"
                if key in self._backup:
                    buf.data.copy_(self._backup.pop(key))
```

---

### P2-2 — EMA adaptive warmup is dead code

**File:** `Python/src/hexorl/train/ema.py`  
**Method:** `update`  

**Problem:** `self._num_updates` is incremented to 1 at line 57 *before* the `if self._num_updates <= 0` check at line 62. The condition is never true. The warmup branch never executes.

**Before:**
```python
    def update(self):
        """Update shadow parameters using polyak averaging."""
        self._num_updates += 1

        # Adaptive decay: decay = 1 - 1/(1 + num_updates) for first few steps,
        # then switches to fixed decay for stability
        if self._num_updates <= 0:
            d = min(self.decay, 1.0 - 1.0 / (1.0 + self._num_updates))
        else:
            d = self.decay
```

**After:**
```python
    def update(self):
        """Update shadow parameters using polyak averaging."""
        self._num_updates += 1

        # Adaptive warmup: ramp up from near-0 to self.decay over first ~1000 steps.
        # Once num_updates is large enough, 1 - 1/(1+n) ≈ self.decay and clamps there.
        d = min(self.decay, 1.0 - 1.0 / (1.0 + self._num_updates))
```

---

### P2-3 — `ReplayDataset` must inherit `IterableDataset`

**File:** `Python/src/hexorl/buffer/sampler.py`  

**Problem:** `ReplayDataset` does not inherit from `torch.utils.data.IterableDataset`. PyTorch's DataLoader uses duck-typing to detect map-style vs. iterable-style datasets. Without the inheritance, multi-worker DataLoader behaviour is undefined — workers may not share iterator state correctly, and the default collate_fn may attempt to add an extra batch dimension to already-batched outputs.

**Before:**
```python
class ReplayDataset:
    """Iterable dataset that samples from the ring buffer and decodes on-the-fly.

    Compatible with torch.utils.data.DataLoader.
    Runs in DataLoader worker threads.
    """
```

**After:**
```python
try:
    from torch.utils.data import IterableDataset as _IterableDataset
except ImportError:
    _IterableDataset = object  # type: ignore


class ReplayDataset(_IterableDataset):
    """Iterable dataset that samples from the ring buffer and decodes on-the-fly.

    Each DataLoader worker calls __iter__ independently. Since the buffer is
    shared, workers get different random samples automatically.
    Runs in DataLoader worker threads.
    """
```

Also update the `Trainer` to construct the DataLoader with `batch_size=None` (since `ReplayDataset` already yields batches) and `collate_fn=lambda x: x`:

In `trainer.py`, wherever the DataLoader is constructed (currently this is done externally and passed in as `dataloader`), document in the class docstring that the DataLoader **must** be constructed as:
```python
dataloader = DataLoader(
    dataset,
    batch_size=None,   # dataset yields pre-batched tuples
    num_workers=2,
    pin_memory=True,
    collate_fn=lambda x: x,
)
```

---

### P2-4 — Double-processing of game records in orchestrator

**File:** `Python/src/hexorl/selfplay/orchestrator.py`  
**Method:** `_ingest_game`  

**Problem:** Workers call `process_game_record(game_record, ...)` at `worker.py:376-380` before pushing to the queue. The orchestrator calls `process_game_record` again at `orchestrator.py:165`. EMA lookahead targets are recomputed from positions whose `outcome` and `root_value` are already set — producing different (incorrect) results the second time.

**Before:**
```python
    def _ingest_game(self, game_record):
        """Process and store one completed game record."""
        try:
            # Ensure targets are computed
            process_game_record(
                game_record,
                lookahead_horizons=self.cfg.buffer.lookahead_horizons,
                lookahead_lambdas=self.cfg.buffer.lookahead_lambdas,
            )

            # Push all positions into the ring buffer
            valid_positions = [p for p in game_record.positions
                               if p.move_history and len(p.move_history) > 0]
            self._buffer.extend(valid_positions)
```

**After:**
```python
    def _ingest_game(self, game_record):
        """Process and store one completed game record."""
        try:
            # Targets are already computed by the worker before pushing.
            # Do not reprocess — it overwrites correct EMA lookahead values.

            # Push all positions into the ring buffer
            valid_positions = [p for p in game_record.positions
                               if p.move_history and len(p.move_history) > 0]
            self._buffer.extend(valid_positions)
```

---

### P2-5 — `mp.Event` used as a threading event in orchestrator

**File:** `Python/src/hexorl/selfplay/orchestrator.py`  

**Problem:** `self._stop_event = mp.Event()` creates a multiprocessing event (backed by shared memory), but it is only ever checked by the collector *thread* (`_collect_records`) inside the same process. `mp.Event` is slower than `threading.Event` for intra-process use and has subtly different memory semantics on some platforms.

**Before:**
```python
        # Worker management
        self._workers: List[mp.Process] = []
        self._record_queue = mp.Queue(maxsize=5000)
        self._stop_event = mp.Event()
```

**After:**
```python
        import threading as _threading
        # Worker management
        self._workers: List[mp.Process] = []
        self._record_queue = mp.Queue(maxsize=5000)
        self._stop_event = _threading.Event()
```

Add `import threading` to the top-level imports of `orchestrator.py` alongside `import multiprocessing as mp` and remove the inline import above.

---

### P2-6 — `regret_rank_loss` can overflow with raw regret values

**File:** `Python/src/hexorl/train/losses.py`  
**Function:** `regret_rank_loss`  

**Problem:** RGSC Equation 7 is `L = -log(Σ_s softmax(φ(s)) · exp(R(s)))`. Raw regret values `R(s)` are mean-squared discrepancies, typically in `[0, 4]`. Adding them directly to `log_softmax_scores` (which are ≤ 0) produces values in `[-∞, ~4]`. `torch.logsumexp` of a batch where some values are ~4 and others are −∞ numerically explodes with large batch sizes.

**Before:**
```python
def regret_rank_loss(
    scores: torch.Tensor,
    regrets: torch.Tensor,
) -> torch.Tensor:
    log_softmax_scores = F.log_softmax(scores, dim=0)
    combined = log_softmax_scores + regrets
    loss = -torch.logsumexp(combined, dim=0)
    return loss
```

**After:**
```python
def regret_rank_loss(
    scores: torch.Tensor,
    regrets: torch.Tensor,
) -> torch.Tensor:
    """Exact RGSC ranking loss — Equation 7 from arXiv 2602.20809v1.

    Regrets are normalized to [0, 1] before use to prevent logsumexp overflow.
    """
    # Normalize regrets to [0, 1] so they are comparable to log-probabilities.
    r_min = regrets.min()
    r_max = regrets.max()
    r_range = (r_max - r_min).clamp(min=1e-6)
    regrets_norm = (regrets - r_min) / r_range

    log_softmax_scores = F.log_softmax(scores, dim=0)
    combined = log_softmax_scores + regrets_norm
    loss = -torch.logsumexp(combined, dim=0)
    return loss
```

---

## P3 — Cleanup

---

### P3-1 — Delete stale root `src/` directory

**Problem:** The workspace root `Cargo.toml` has no `[package]` section, so `src/` is never compiled. It contains the old monolithic implementation with **unfixed hard panics**:
- `src/mcts.rs:654` — `panic!("re_root: no child found...")`
- `src/mcts.rs:463` — `.expect("MCTS: illegal place...")`

Any IDE or developer jumping to definition can land on stale, unfixed code.

**Action:** Run:
```bash
git rm -r src/
git rm proptest-regressions/tests/threats.txt
git rm rust.md
```

These are all superseded by `crates/hexgame-core/`.

---

### P3-2 — `encode_compact_record` and `apply_d6_symmetry` should release the GIL

**File:** `crates/hexgame-py/src/encode.rs`  

Both `encode_compact_record` (which replays a full game and encodes every position) and `apply_d6_symmetry` (which transforms a dense spatial tensor) are pure computation with no Python callbacks. They hold the GIL for their entire duration, blocking the inference dispatcher thread if DataLoader workers are compiled into the same process.

**For `encode_compact_record`:**  
Move the computation loop into `py.allow_threads(|| { ... })`. Return the `Vec<f32>` from the closure and convert to `PyArray4` after re-acquiring the GIL.

**Before:**
```rust
fn encode_compact_record<'py>(
    py: Python<'py>,
    history_bytes: &[u8],
    near_radius: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    // ... validation ...
    let mut game = HexGameState::new();
    let mut positions = Vec::with_capacity(num_moves * TENSOR_SIZE);

    for chunk in history_bytes.chunks_exact(12) {
        // ... encode loop ...
    }

    let shape = (num_moves, NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize);
    let arr = numpy::ndarray::Array4::from_shape_vec(shape, positions)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray4::from_owned_array(py, arr))
}
```

**After:**
```rust
fn encode_compact_record<'py>(
    py: Python<'py>,
    history_bytes: &[u8],
    near_radius: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    if !history_bytes.len().is_multiple_of(12) {
        return Err(PyValueError::new_err(
            format!("history_bytes length {} is not a multiple of 12", history_bytes.len())
        ));
    }
    let num_moves = history_bytes.len() / 12;
    if num_moves == 0 {
        return Err(PyValueError::new_err("empty history"));
    }

    // Copy bytes so the closure owns them (history_bytes lifetime doesn't cross thread boundary).
    let bytes_owned: Vec<u8> = history_bytes.to_vec();

    let positions = py.allow_threads(move || -> Result<Vec<f32>, String> {
        let mut game = HexGameState::new();
        let mut positions = Vec::with_capacity(num_moves * TENSOR_SIZE);
        for chunk in bytes_owned.chunks_exact(12) {
            let tensor = encoder::encode_board(&game, near_radius, false).tensor;
            positions.extend_from_slice(&tensor);
            let q = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
            let r = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
            game.place(q, r).map_err(|e| e.to_string())?;
        }
        Ok(positions)
    }).map_err(|e| PyValueError::new_err(e))?;

    let shape = (num_moves, NUM_CHANNELS, BOARD_SIZE as usize, BOARD_SIZE as usize);
    let arr = numpy::ndarray::Array4::from_shape_vec(shape, positions)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray4::from_owned_array(py, arr))
}
```

Apply the same pattern to `apply_d6_symmetry` — copy the input array into a `Vec<f32>` before `allow_threads`, compute the transform inside, and build `PyArray3` after.

---

### P3-3 — `asyncio.get_event_loop()` is deprecated

**File:** `Python/src/hexorl/inference/server.py`  
**Line:** Inside `_event_loop`  

**Before:**
```python
        loop = asyncio.get_event_loop()
        ...
        policies, values = await loop.run_in_executor(
```

**After:**
```python
        loop = asyncio.get_running_loop()
        ...
        policies, values = await loop.run_in_executor(
```

---

### P3-4 — Fix deprecated `torch.cuda.amp.autocast` call in `network.py`

**File:** `Python/src/hexorl/model/network.py`  
**Method:** `forward_batch`  

**Before:**
```python
        if autocast and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = self.forward(x)
```

**After:**
```python
        if autocast and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.forward(x)
```

Also fix the same pattern in `server.py:_forward`:

**Before:**
```python
            if self.fp16 and self._device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = self._model(batch_tensor)
```

**After:**
```python
            if self.fp16 and self._device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self._model(batch_tensor)
```

---

## Verification Checklist

After applying all fixes, verify the following before starting a training run:

- [ ] `cargo test -p hexgame-core` passes (no regressions)
- [ ] `cargo test -p hexgame-py` passes
- [ ] Python smoke test: construct `RingBuffer(capacity=100, num_lookahead=3)`, append 10 `PositionRecord`s with `lookahead_values=[0.1, 0.2, 0.3]`, sample, verify `__getitem__` returns the correct lookahead list
- [ ] Python smoke test: construct `ReplayDataset` wrapping the above buffer, iterate one batch, verify the returned tuple has 4 elements and the 4th is a list of 3 arrays each of shape `(batch_size,)`
- [ ] Python smoke test: run `InferenceServer` + 2 `InferenceClient` workers, submit 10 batches, verify no timeout and no stale-result errors
- [ ] Python smoke test: run `SelfPlayWorker` for 5 mock games, inspect `GameRecord.positions[0].policy_target` — values should be non-uniform and sum to 1.0
- [ ] Python smoke test: run one `Trainer.train_epoch` with 10 batches, verify `_epoch_losses` contains keys for all configured heads
- [ ] `encode_compact_record` unit test: encode a 3-move history bytes (opener + 2 stones), verify output shape is `(3, 13, 33, 33)` and the first tensor is all-zeros (empty board)
