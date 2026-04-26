# 06 Rewrite Completeness And Gaps

## Scope

This document summarizes how complete the current rewrite is compared with legacy Hexagon and identifies what should be rebuilt, redesigned, or intentionally left behind.

## Current Rewrite Source Anchors

- System design: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Docs/SYSTEM_DESIGN.md`
- Model: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/model/network.py`
- Trainer: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/train/trainer.py`
- Losses: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/train/losses.py`
- Epoch pipeline: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/epoch/pipeline.py`
- Self-play: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/selfplay/`
- Inference server: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/inference/`
- Buffer: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/buffer/`
- Dashboard: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/dashboard/`
- Rust core: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/crates/hexgame-core/`
- PyO3: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/crates/hexgame-py/`

## Completion Summary

| Subsystem | Rewrite Status | Legacy Parity |
|---|---|---|
| Rust project structure | Strong | Better than legacy |
| Game rules | Strong | Near parity |
| Eval/threat internals | Strong | Cleaner than legacy, one Python-facing threat-mask gap to audit |
| Classical search | Strong | Near parity |
| Encoder | Strong | Near parity, cleaner canonical source |
| PyO3 bridge | Moderate | Simpler than legacy |
| MCTS | Moderate | Baseline parity, missing advanced exploration/pipeline features |
| Model | Moderate/strong | Cleaner, but axis semantics changed and legacy trunk features removed |
| Replay buffer | Moderate/strong | Cleaner sparse storage, lacks DB/history integration |
| Training loop | Moderate | Runnable smoke/epoch path, not full legacy production loop |
| Self-play orchestration | Moderate | New inference-server design, less mature than legacy |
| Eval/gating | Partial | Missing legacy gating and full evaluation workflow |
| Checkpoint migration | Minimal | Far below legacy, intentionally clean |
| Dashboard | Minimal | Not feature-complete |
| DB persistence | Missing | Not feature-complete |
| Model lab/play/arena UI | Missing | Not feature-complete |
| Corpus/opening analysis | Missing | Not feature-complete |

## Architecture Training Pipeline Comparison

### Legacy

Legacy pipeline is epoch-centric and monolithic. It is powerful but tangled:

- Self-play generation, sparring, replay insertion, training, checkpointing, DB writes, dashboard pushes, config reload, gating, and analysis live close together.
- Inference is driven from Python MCTS wrappers that call a traced model.
- Advanced MCTS and RGSC features are present but not all are reliable.
- Historical compatibility code is extensive.

### Rewrite

Rewrite pipeline is modular:

- Self-play workers are separate processes.
- One inference server owns model/GPU.
- Shared memory queues batch across workers.
- Ring buffer is a separate storage abstraction.
- Trainer consumes iterable replay batches.
- Epoch pipeline wires pieces conservatively.

This is closer to the intended robust architecture and less likely to hide model-plateau bugs behind accidental coupling.

## Model Architecture Comparison

Legacy:

- More specialized trunk: hex-masked convs, NBT, GroupNorm/FixScale, global pooling, RepVGG merge.
- Eight heads.
- Per-cell axis influence directly boosts policy.
- 16-bin value by active default.
- Ownership and moves-left heads exist, but targets are weak/broken.

Rewrite:

- Simpler BatchNorm gated residual CNN.
- Configurable head dictionary.
- 65-bin value and lookahead heads.
- Axis is 3-class classification, not per-cell influence.
- Drops ownership.
- Computes moves-left target in processed game record.

Assessment:

The rewrite model is simpler and cleaner. It should be easier to debug plateauing. The main feature to reconsider later is whether per-cell axis influence was useful enough to bring back in a tested form.

## Dashboard Comparison

Legacy:

- Full operational dashboard with charts, games, viewer, model lab, arena, training control, config editor, corpus/opening analysis.
- Dark dense visual style worth preserving.
- Implementation is brittle: one FastAPI monolith and one enormous static HTML/JS file.

Rewrite:

- Dashboard package exports only pseudocode.
- Design doc describes a read-only Rich TUI watching JSON stats.
- No web UI, no DB, no game browser, no replay viewer, no model lab, no arena, no config editor.

Assessment:

Dashboard is the biggest missing piece. A rebuild should not clone the implementation, but it should preserve most user-facing features.

## Features To Keep From Legacy

Keep or rebuild:

- Dense GitHub-dark visual style.
- Run KPIs and chart set.
- Game browser and replay viewer.
- Exact encoder debug view.
- Training config source-of-truth clarity.
- Training logs/status controls.
- Model Lab inference/compare workflow.
- Interactive play.
- Arena spectator and match setup.
- Corpus/opening explorer.
- Checkpoint/game import/indexing.
- SQLite or equivalent persistent history.
- Board renderer interactions: pan, zoom, fit, overlays, move list.

## Features To Redesign Before Keeping

Redesign:

- Dashboard backend route organization.
- Frontend component structure.
- WebSocket live update model.
- REST error status codes.
- Training process control.
- Model cache and GPU memory accounting.
- Config validation and JSON schema exposure.
- Analysis job scheduling.
- Arena persistence.
- Checkpoint migration boundaries.
- RGSC candidate extraction and regret labels.
- Gumbel/selector MCTS variants.

## Features To Drop Or Quarantine

Drop or quarantine unless specifically needed:

- Legacy monolithic dashboard HTML/JS.
- Global mutable dashboard process state as source of truth.
- Stale checkpoint compatibility shims in core training path.
- Broken moves-left target logic.
- Occupancy-only "ownership" target named as ownership.
- Sparring data path until an end-to-end test proves samples enter replay.
- CPU `_boost_with_axis()` dead path.
- Sparse legacy sample path that modern compact buffer rejects.
- Old docs as implementation truth.

## Recommended Next Build Order

1. Metrics/event emission from the rewrite training pipeline.
2. Persistent run/metric/game/checkpoint store.
3. Web dashboard backend with typed schemas.
4. Dashboard frontend shell matching legacy style.
5. Charts and run KPIs.
6. Game browser and replay viewer.
7. Exact encoder/debug endpoints.
8. Config editor and training control.
9. Model Lab.
10. Arena.
11. Corpus/opening analysis.
12. Optional advanced MCTS experiments behind flags.

## Suggested Dashboard Data Contracts

Minimum run metric:

```json
{
  "schema_version": 1,
  "run_id": "string",
  "epoch": 1,
  "global_step": 100,
  "elapsed_s": 12.3,
  "phase": "selfplay|train|eval|idle",
  "loss": {
    "total": 0.0,
    "policy": 0.0,
    "value": 0.0,
    "lookahead": {},
    "aux": {}
  },
  "selfplay": {
    "games_done": 0,
    "positions_done": 0,
    "workers_alive": 0,
    "workers_total": 0
  },
  "inference": {
    "device": "cuda",
    "fp16": true,
    "batches": 0,
    "positions": 0,
    "avg_forward_ms": 0.0
  },
  "buffer": {
    "size": 0,
    "capacity": 0,
    "full_search_pct": 0.0
  },
  "checkpoint": {
    "latest_path": "string"
  }
}
```

Minimum game record:

```json
{
  "schema_version": 1,
  "game_id": "string",
  "run_id": "string",
  "epoch": 1,
  "source": "selfplay|eval|arena|play",
  "winner": 0,
  "moves": [[0, 0, 0]],
  "positions": [],
  "analysis": null
}
```

## Overall Assessment

The rewrite is on the right path for the reason it was created: it removes large amounts of legacy coupling while preserving the core engine and training concepts. It is not done. It is currently a strong foundation with a partial production training stack and almost no dashboard parity.

The legacy dashboard should be treated as a product spec and visual reference, not as code to port. The highest-value next work is to establish clean metrics/game persistence and rebuild the dashboard around that contract.

