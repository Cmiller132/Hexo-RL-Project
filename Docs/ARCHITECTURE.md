# Hexo-RL Architecture

## Overview

Hexo-RL is a reinforcement learning training pipeline for the game **Hexo** (Connect 6 played on an infinite hexagonal grid). It follows the AlphaZero / KataGo architecture with RGSC (Regret-Guided Search Control).

## System Components

### Rust Engine (`crates/`)
- **hexgame-core** — Pure Rust game engine: board, rules, eval, threats, MCTS, search, encoder
- **hexgame-py** — PyO3 FFI bindings: PyHexGame, PyMCTSEngine, encode_compact_record, apply_d6_symmetry
- **hexgame-bench** — Criterion benchmarks: encode, MCTS, threats
- **hexgame-cli** — Standalone CLI for debugging

### Python Pipeline (`python/src/hexorl/`)
- **inference/** — GPU inference server with shared-memory batching
- **selfplay/** — Worker processes and orchestrator
- **buffer/** — Ring buffer, replay dataset, target computation, regret buffer
- **model/** — HexNet CNN (KataGo-style binned value heads, multi-head architecture)
- **train/** — Training loop, losses (including RGSC ranking loss), EMA tracking
- **eval/** — Arena, ELO computation, classical opponent
- **config/** — Pydantic configuration schema and TOML loader

## Data Flow

```
Self-play Workers → Inference Server (GPU)
     ↓
  Game Records → Ring Buffer (700 MB, 2M samples)
     ↓
  Ring Buffer → ReplayDataset → Trainer (AdamW + cosine LR)
     ↓
  Trained Model → EMA → Inference Server (hot-swap)
```

## Process Topology

- **Inference server** (1 process) — owns GPU, model weights
- **Self-play workers** (N processes, N≈24-30) — MCTS + inference client
- **Trainer** (1 process) — time-shares GPU with inference
- **Buffer process** (1 process) — owns ring buffer
- **Orchestrator** (1 process) — supervisor, restarts on crash

## Key Design Decisions

1. **Rust MCTS** — entire tree lives in Rust memory, Python only dispatches NN inference
2. **Shared-memory queues** — KataGo-style NNEvaluator pattern for GPU inference
3. **Compact records** — ~350 bytes/sample, decoded on-the-fly via Rust FFI
4. **Binned value head** — 65-bin discretization (KataGo-style) for better calibration
5. **RGSC** — Regret-Guided Search Control (§3 of paper) with ranking-based regret network
6. **Turn-boundary EMA** — Lookahead targets aligned to game turns, not individual placements
