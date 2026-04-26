# 01 Project Structure

## Scope

This document maps the legacy Hexagon repository and compares it to the current Hexo-RL rewrite at a structural level.

## Top-Level Legacy Layout

Legacy root: `/Users/coltonmiller/Documents/GitHub/Hexagon`

Important top-level files and directories:

| Path | Purpose |
|---|---|
| `Cargo.toml` | Single Rust package, with Rust library and optional Python/PyO3 feature. |
| `src/` | Rust game engine, search, MCTS, eval, and PyO3 bridge in one crate. |
| `pyproject.toml` | Python package metadata and console scripts. |
| `python/hexgame/` | Main Python package: model, training, game wrappers, dashboard, CLI, integrations. |
| `configs/` | Model and training TOML presets. |
| `config.toml` | Active/root training config used by current legacy runs. |
| `tests/` | Pytest coverage, including training, dashboard config source-of-truth, MCTS, encoding, regret buffer. |
| `benchmarks/` | Python profiling and throughput scripts. |
| `docs/` | Design notes, old model docs, project structure, MCTS proposals, RGSC reviews. Several are stale. |
| `viewer.html` | Older standalone game viewer. |
| `PLATEAU_ANALYSIS_v73.md` | Plateau analysis notes. Useful historically, but stale in places. |

## Legacy Rust Structure

Legacy Rust uses one mixed crate:

| File | Role |
|---|---|
| `src/lib.rs` | Crate root and public exports. Documents game rules. |
| `src/core.rs` | Hex coordinate primitives, directions, distance. |
| `src/game.rs` | Monolithic rules engine, board state, win detection, legal moves, incremental eval data, threat helpers, axis/tactical target generation. |
| `src/eval.rs` | Classical evaluation helper surface. |
| `src/search.rs` | Alpha-beta classical player/search engine. |
| `src/mcts.rs` | Rust MCTS arena, selectors, Gumbel logic, subtree reuse, tree extraction. |
| `src/pybridge.rs` | PyO3 bindings for game, MCTS, encoding, classical self-play. |

The main structural issue is that `game.rs`, `mcts.rs`, and `pybridge.rs` each became large integration hubs. Many responsibilities are correct individually, but boundaries are blurry.

## Legacy Python Package Structure

Legacy Python root: `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame`

| Package | Purpose |
|---|---|
| `model/` | Network architecture, feature constants/transforms, checkpoint loading/migration. |
| `training/` | Config, epoch loop, self-play data generation, compact replay buffer, trainer, regret buffer, matchmaking. |
| `game/` | Python MCTS wrapper, players, arena, recorders, analysis, compact game decoding, corpus/opening analysis. |
| `ui/` | FastAPI dashboard backend and static dashboard HTML. |
| `cli/` | Train, evaluate, play, viewer, migrate, Sealbot, Axis CLIs. |
| `data/` | SQLite DB session and schema for runs, metrics, games, checkpoints, analyses, openings. |
| `integrations/` | External integrations, notably Sealbot. |

## Legacy Console Entry Points

Legacy console scripts are declared in `/Users/coltonmiller/Documents/GitHub/Hexagon/pyproject.toml`.

Important commands:

| Command | Module | Purpose |
|---|---|---|
| `hexgame-train` | `hexgame.cli.train` | Main training CLI. |
| `hexgame-evaluate` | `hexgame.cli.evaluate` | Standalone checkpoint/model evaluation. |
| `hexgame-dashboard` | `hexgame.ui.dashboard` | FastAPI dashboard. |
| `hexgame-play` | `hexgame.cli.play` | Older interactive play server. |
| `hexgame-viewer` | `hexgame.cli.viewer` | Older JSON game viewer. |
| `hexgame-migrate` | `hexgame.cli.migrate` | Import checkpoints/games into SQLite. |

## Legacy Data Stores

The dashboard and migration path use SQLite. By default, the DB is at:

`/Users/coltonmiller/Documents/GitHub/Hexagon/data/hexgame.db`

Core tables live in legacy `python/hexgame/data/schema.py`:

- `TrainingRun`
- `Checkpoint`
- `EpochMetrics`
- `GameRecord`
- `GameAnalysis`
- `CanonicalOpening`

The DB-backed dashboard is a major legacy feature not present in the rewrite.

## Current Rewrite Layout

Rewrite root: `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project`

The rewrite splits the project by capability:

| Path | Purpose |
|---|---|
| `Cargo.toml` | Cargo workspace root. |
| `crates/hexgame-core/` | Pure Rust core engine, eval, encoder, search, MCTS. |
| `crates/hexgame-py/` | PyO3 bindings split by engine/encoding/buffer concerns. |
| `crates/hexgame-bench/` | Criterion Rust benchmarks. |
| `crates/hexgame-cli/` | Standalone Rust CLI. |
| `Python/src/hexorl/` | New Python package: config, model, inference, self-play, buffer, train, eval, epoch, dashboard. |
| `Python/tests/` | Python smoke and pipeline tests. |
| `Configs/` | Pydantic-backed runtime configs. |
| `Docs/` | Active design and review docs. |
| `benches/` | Python/cross-language benchmark scripts. |

## Structural Comparison

| Area | Legacy Hexagon | Rewrite Hexo-RL |
|---|---|---|
| Rust build | Single mixed crate, PyO3 and core compiled together. | Workspace split into core, py bindings, bench, CLI. |
| Rust game state | Monolithic `game.rs`. | `board.rs` plus `eval/`, `threats.rs`, `encoder.rs`. |
| MCTS | Rich and experimental: Gumbel, selectors, pipeline APIs, tree extraction. | Simpler baseline MCTS with cleaner API and error handling. |
| Python training | One large epoch loop with many historical features. | Small modular pipeline with explicit inference server, ring buffer, trainer. |
| Dashboard | Full FastAPI plus 5.7k-line static HTML/JS app. | Pseudocode only. |
| Persistence | SQLite run/game/checkpoint/analysis DB plus checkpoint dirs. | Checkpoints and run dirs only; no DB-backed dashboard yet. |
| Config | Large dataclass/TOML config with compatibility shims. | Pydantic schema with smaller surface. |
| Tests | Broad but mixed with standalone diagnostics. | More focused smoke/property tests. |

## Legacy Feature Inventory

### Core/Engine

- Hex coordinate rules.
- Infinite board with practical 33x33 neural window.
- P0 opening at origin, then two-placement turns.
- Placement radius constraints.
- Win detection on three axes.
- Threat-constrained move generation.
- Classical alpha-beta search with iterative deepening, TT, PVS/LMR, quiescence, killer/history heuristics.
- Neural MCTS with PUCT, variance-aware selectors, Gumbel Sequential Halving, virtual loss, subtree reuse, pipelined leaf evaluation.
- 13-channel encoder and axis/tactical target helpers.

### Model/Training

- Multi-head `HexNet` with policy, value, axis influence, opponent policy, regret rank/value, ownership, moves-left.
- Compact replay buffer storing move histories and dense policy targets.
- D6 data augmentation.
- Self-play data generation with PCR, root noise, Gumbel/root exploration modes, subtree reuse, RGSC candidate scoring.
- Bootstrap classical data generation.
- Sparring/evaluation paths.
- Checkpoint migration for older architecture names/shapes.
- Replay persistence and checkpoint save/load.
- Optional dashboard embedding.

### Dashboard/UI

- Run selection, KPIs, charts.
- Game browser with filters.
- Game replay viewer with SVG hex board, overlays, move list, PRB markers.
- Analysis panel with badges, eval bar, rhythm strip, alternatives.
- Exact model-input and target debug views.
- Model Lab for checkpoint loading and inference comparison.
- Interactive play mode.
- Arena setup and live spectator.
- Training start/stop/log/import controls.
- Config editor.
- Corpus summary and opening explorer.

## Legacy Technical Debt Themes

- Several old docs describe earlier architectures, not live code.
- Dashboard backend, process control, model cache, arena, sessions, config, and API routes live in one module.
- Dashboard frontend is one huge HTML file with inline CSS and JS.
- Many old compatibility shims remain for previous configs/checkpoints.
- Some features are partially wired or misleading: sparring samples are not inserted into replay, moves-left target is always zero, ownership is not real ownership.
- Dashboard global state is in-memory and restart-fragile.
- REST endpoints often return HTTP 200 with an `error` payload instead of status codes.

