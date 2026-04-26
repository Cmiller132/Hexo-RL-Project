# Legacy Hexagon Project Specs

Generated: 2026-04-26

Legacy project: `/Users/coltonmiller/Documents/GitHub/Hexagon`

Rewrite project: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project`

These documents capture the legacy Hexagon system as implemented, not as its older docs describe it. Several legacy docs are stale; the source code is the authority here.

## Document Set

- [01 Project Structure](01_PROJECT_STRUCTURE.md): repository layout, build system, Python package map, CLIs, data stores, and major feature inventory.
- [02 Engine And Rules](02_ENGINE_AND_RULES.md): Rust rules engine, search, MCTS, encoder, PyO3 bridge, and rewrite preservation/gaps.
- [03 Model Architecture](03_MODEL_ARCHITECTURE.md): legacy neural network, heads, shapes, inference contract, checkpoint migration, and rewrite comparison.
- [04 Training Pipeline](04_TRAINING_PIPELINE.md): epoch flow, self-play, replay, losses, optimization, checkpointing, eval/gating, and plateau risks.
- [05 Dashboard Spec](05_DASHBOARD_SPEC.md): visual style, tabs, backend routes, data flow, feature inventory, risks, and rebuild requirements.
- [06 Rewrite Completeness And Gaps](06_REWRITE_COMPLETENESS_AND_GAPS.md): current Hexo-RL completion status versus legacy, with recommended dashboard rebuild scope.

## Executive Summary

The rewrite is much cleaner and safer architecturally. It keeps the core rules, classical search, 13-channel encoder shape, baseline neural MCTS loop, compact replay concept, multi-head training intent, and a process-separated inference-server design. It also meaningfully improves Rust decomposition, undo correctness, eval/threat separation, and source-of-truth clarity.

The rewrite is not yet feature-complete versus legacy. The largest missing surface is the dashboard, which is currently only a Rich TUI pseudocode sketch. Other missing or simplified areas include legacy Gumbel Sequential Halving, variance-aware MCTS selectors, pipelined MCTS FFI, historical checkpoint/database migration tooling, full persisted run/game analysis database, model lab, interactive play, arena spectator UI, corpus/opening explorer, and several advanced target/debug endpoints.

Legacy was bloated, but it had many useful workflows. The dashboard rebuild should copy its dense GitHub-dark operational style and major tabs, while replacing the monolithic HTML/JS backend coupling with explicit typed APIs, persistent state boundaries, resilient WebSocket polling, local chart assets, and componentized board/model tooling.

## Most Important Legacy Bugs And Plateau Risks Found

- Sparring samples are generated but appear not to be inserted into replay; `sparring_samples` are collected in legacy `training/loop.py`, while replay insertion uses only `epoch_samples`.
- The legacy moves-left target is degenerate: the dataset computes `moves_left = move_counts - n_moves`, where both values are the same snapshot length, so the head is trained toward zero.
- The legacy ownership target is stone occupancy relative to current player, not true territory/control.
- Gating rejection reverts model weights but not optimizer/scheduler state.
- The legacy root neural value is passed into Rust root expansion but ignored by `expand_root`; search value comes from post-simulation root Q.
- Live dashboard updates are unreliable when the standalone dashboard launches training, because that path does not pass the embedded dashboard flag that pushes WebSocket metrics.
- Legacy docs and plateau notes are partly stale: live code has an eight-head model and 16-bin value head, while older docs mention fewer heads and a 2-bin value.

## How To Use These Specs

Use these as rebuild targets, not as instructions to preserve legacy implementation details. For each subsystem, distinguish:

- Keep: workflows, metrics, visual language, and proven core game/search contracts.
- Redesign: process boundaries, UI architecture, APIs, config validation, persistence, and monitoring.
- Drop or quarantine: stale compatibility shims, one-off diagnostic scripts, global mutable dashboard state, implicit checkpoint migration paths, and experimental code that was not wired into training.

