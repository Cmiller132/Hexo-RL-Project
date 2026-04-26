# 04 Training Pipeline

## Scope

This document describes the legacy training pipeline, identifies likely plateau risks, and compares it to the rewrite.

## Legacy Source Anchors

- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/loop.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/selfplay.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/trainer.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/buffer.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/regret_buffer.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/game/mcts.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/config.py`

## Legacy Epoch Flow

The main legacy training loop does roughly:

1. Build or load model.
2. Build optimizer/scheduler/scaler.
3. Load checkpoint and replay buffer.
4. Optionally bootstrap from classical data.
5. For each epoch:
   - Reload config from checkpoint `config.toml`.
   - Check pause/resume/stop signal.
   - Deep-copy and trace inference model.
   - Generate self-play games.
   - Optionally generate sparring games.
   - Compute/store samples and diagnostics.
   - Extend replay buffer.
   - Train on replay.
   - Evaluate/gate if configured.
   - Save checkpoint/replay buffer.
   - Write metrics/checkpoint/game records to DB.
   - Push dashboard metrics if embedded dashboard is active.

The design is feature-rich but tightly coupled.

## Self-Play

Legacy self-play uses Rust-backed MCTS through Python wrappers.

Features:

- Full-sim and low-sim playout cap randomization.
- Root Dirichlet noise.
- Root policy temperature.
- Gumbel root exploration option.
- Selector choice.
- Subtree reuse across two placements.
- Threat constraints.
- Regret candidate tree extraction for RGSC.
- Target entropy/policy surprise diagnostics.
- Truncated-game bootstrap options.

The self-play path produces dense samples containing:

- Compact move history.
- Policy target.
- Scalar value target.
- Axis/threat target.
- Regret labels.
- Opponent policy if available.
- Search metadata.
- PCR full/low flag.
- Policy surprise.

## Replay Buffer

Legacy `CompactReplayBuffer` stores:

- Move coordinate history per sample.
- Move count.
- Near radius.
- Dense policy target as `float16`.
- Dense opponent policy target as `float16`.
- Scalar value.
- Regret.
- Flags: has opponent policy, is full search, has regret.
- Epoch index.
- Policy surprise.

`HexDataset` reconstructs board tensors on demand by replaying move history through the Rust engine, then applies D6 augmentation.

Strengths:

- Avoids storing full `(13,33,33)` tensors.
- Keeps Rust encoder as source of truth.
- Supports recency and policy-surprise weighted sampling.

Weaknesses:

- Dense policy storage is memory-heavy compared with sparse top-K storage.
- Moves-left cannot be correctly derived from the stored snapshot alone.
- Ownership target is recomputed as stone occupancy, not final territory.

## Training Losses

Legacy `train_on_buffer()` trains:

- Value cross entropy over categorical bins.
- Policy cross entropy.
- Opponent policy cross entropy.
- Axis influence MSE.
- Regret rank loss.
- Regret value MSE.
- Ownership MSE.
- Moves-left MSE.
- Entropy regularization over legal support.

Policy loss is hard-masked to full-search samples. This helps avoid low-sim PCR policy targets dominating the gradient.

## Optimization

Legacy defaults vary by config, but the observed active path includes:

- SGD/Nesterov or configured optimizer.
- Optional AMP with BF16/FP16 handling.
- Gradient clipping at norm 1.0.
- Optional `torch.compile` path with fallback.
- Manual CUDA cleanup between training/inference phases.
- Scheduler stepping per epoch.

## Checkpointing

Legacy saves:

- `latest.pt`
- `epoch_N.pt`
- model state
- optimizer state
- scheduler state
- scaler state
- config snapshot
- total games
- target epochs
- run stamp
- optional prioritized regret buffer state
- separate `latest_buffer.pt`

Run stamp includes git, platform, torch, engine module hash, and source hashes.

## Evaluation And Gating

Legacy can evaluate against classical opponents and optionally gate updates.

Risk: on gate rejection, model weights are restored, but optimizer/scheduler state is not restored. This can create hidden drift if gating is enabled.

The active root config reportedly has `gate_games=0` and `eval_games=0`, so this path may often be disabled.

## Dashboard/DB Integration

At epoch end, the legacy loop writes:

- Training run records.
- Epoch metrics.
- Checkpoints.
- Game records.
- Analysis/import metadata.

It also pushes WebSocket metrics when an embedded `DashboardServer` is used. Standalone dashboard-spawned training does not necessarily get those pushes.

## Plateau Risks Found

### High Confidence

1. Sparring samples are generated but not inserted into replay. This means a whole adaptive-learning branch may produce logs/eval without changing training data.
2. Moves-left target is always zero because snapshot length is used as both current and total game length.
3. Ownership target is not meaningful ownership; it is only occupied stones from current perspective.
4. Gate rejection does not restore optimizer state.
5. Root neural value is ignored by legacy Rust `expand_root`; root value after search comes from visit Q instead.
6. Older docs and plateau notes are stale, making diagnosis harder.

### Medium Confidence

1. Low-sim PCR still contributes to value and auxiliary heads even when policy is masked.
2. RGSC relies on regret heads that are noisy early in training.
3. Small `near_radius=8` may limit strategic variety.
4. Config fields and compatibility shims make it easy for a setting to appear active but not affect the actual path.

## Rewrite Training Pipeline

Rewrite source anchors:

- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/epoch/pipeline.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/selfplay/worker.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/selfplay/orchestrator.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/inference/server.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/buffer/ring.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/buffer/sampler.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/train/trainer.py`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/train/losses.py`

Rewrite flow:

1. Optional synthetic bootstrap positions.
2. Optional self-play orchestrator.
3. Struct-of-arrays ring buffer.
4. Iterable replay sampler with D6 augmentation and regret-biased subset.
5. Trainer with configurable multi-head losses.
6. EMA update.
7. Checkpoint save.

The rewrite also introduces a central inference server:

- One process owns model/GPU.
- Workers write requests into shared memory.
- Server batches across workers.
- Results are scattered back to worker slots.

This is cleaner than legacy's per-loop model execution and aligns with the active system design.

## Rewrite Completeness Versus Legacy Training

| Area | Legacy | Rewrite Status |
|---|---:|---|
| Self-play loop | Yes | Implemented, but simpler and includes mock fallback. |
| Central inference server | No, legacy did per-process/wrapper inference | Implemented. |
| Compact replay | Yes | Implemented as sparse struct-of-arrays. |
| D6 augmentation | Yes | Implemented. |
| Multi-head training | Yes | Implemented. |
| Lookahead targets | No/limited | Implemented. |
| EMA | Not central | Implemented. |
| RGSC candidate extraction | Yes | Partial/simplified. |
| Gumbel/selector variants | Yes | Missing. |
| Sparring | Yes, but likely not training | Missing. |
| DB run persistence | Yes | Missing. |
| Checkpoint migration | Extensive | Missing/simple only. |
| Eval/gating | Yes | Eval smoke exists; full legacy gating missing. |
| Dashboard integration | Yes | Missing/pseudocode. |

## Rebuild Guidance

- Keep the rewrite inference-server topology.
- Keep sparse replay storage and computed moves-left targets.
- Avoid restoring sparring until its data path is explicitly tested end to end.
- Add regression tests for every target head, especially moves-left, value perspective, opponent policy, and lookahead horizons.
- Treat legacy RGSC, Gumbel, and selector variants as optional experiments, not core training requirements.
- Add structured metrics emission before rebuilding the dashboard; the dashboard should consume stable JSON/DB/WebSocket contracts rather than scraping logs.

