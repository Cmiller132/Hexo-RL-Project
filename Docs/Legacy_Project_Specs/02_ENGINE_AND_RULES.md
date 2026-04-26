# 02 Engine And Rules

## Scope

This document covers the legacy Rust engine and compares it to the current rewrite.

## Game Rules

Both projects implement the same game:

- Axial hex coordinates `(q, r)`.
- Player 0 opens at origin.
- After the first placement, turns contain two placements.
- Placement is constrained by a radius around existing stones.
- A player wins by forming six connected stones on one of the three principal axes.
- Neural encoding uses a 33x33 local board window.

Legacy source anchors:

- `/Users/coltonmiller/Documents/GitHub/Hexagon/src/lib.rs`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/src/game.rs`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/src/core.rs`

Rewrite source anchors:

- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/crates/hexgame-core/src/lib.rs`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/crates/hexgame-core/src/board.rs`
- `/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/crates/hexgame-core/src/core.rs`

## Legacy Engine Architecture

Legacy `HexGameState` is a single large owner of:

- Board map from coordinates to players.
- Move history.
- Current player and placements remaining.
- Winner/winning line.
- Zobrist hash.
- Candidate sets.
- Window eval and threat counters.
- Hot windows.
- Axis influence/tactical targets.
- Threat-constrained move helpers.

This made runtime behavior compact, but also made correctness auditing difficult because state restoration, candidate maintenance, eval deltas, and game legality were intertwined.

## Rewrite Engine Architecture

The rewrite breaks the same responsibilities into smaller modules:

| Rewrite Module | Responsibility |
|---|---|
| `core.rs` | Hex coordinates, directions, constants, turn representation. |
| `board.rs` | Board state, placement/unplacement, legal moves, win detection, candidates. |
| `eval/state.rs` | Incremental eval, threat counts, hot windows, delta stack. |
| `eval/grid.rs` | Finite 61x61x3 pattern window grid. |
| `eval/patterns.rs` | Pattern tables and checksum tests. |
| `eval/hot.rs` | Hot window tracking. |
| `threats.rs` | Threat status, block constraints, must-block logic. |
| `encoder.rs` | Canonical neural tensor encoding and feature extraction. |
| `search.rs` | Classical alpha-beta search. |
| `mcts.rs` | Baseline neural MCTS. |

This is a significant improvement over legacy for maintainability.

## Candidates And Legal Moves

Legacy:

- Keeps a search candidate set near existing stones.
- Validates placement radius by scanning stones in some paths.
- Threat constraints are embedded in `HexGameState`.

Rewrite:

- Keeps both search candidates and placement candidates.
- Radius-8 placement legality is faster and more explicit.
- Threat constraints are represented as data (`ThreatStatus`, `BlockConstraint`) and consumed by search.

Important gap: rewrite search uses pair-aware block constraints, but some encoder/Python exposure paths flatten constraints to single blocking cells. Legacy per-cell helper included cells participating in valid covering pairs. This should be audited before relying on Python-facing threat-constrained legal masks for training.

## Evaluation

Legacy:

- Pattern values and counts are mostly embedded in `game.rs`.
- Incremental eval and hot windows live on the game state.
- `eval.rs` provides additional classical evaluation helpers.

Rewrite:

- Pattern tables are isolated and checksum-tested.
- `EvalState` owns incremental updates and an undo delta stack.
- Debug invariant checks can recompute hot windows after unplace.

This is one of the rewrite's strongest improvements.

## Classical Search

Both projects share the same broad alpha-beta design:

- Search operates on whole turns, not individual placements.
- Iterative deepening.
- Aspiration windows.
- Principal variation search.
- Late move reductions.
- Transposition table.
- Killer moves and history heuristic.
- Tactical/quiescence extension.
- Candidate caps and move ordering.
- Instant-win detection.
- Optional noise for training variety.

Rewrite improvements:

- `Turn` canonicalization is moved into core primitives.
- Some illegal move paths return `Result` instead of being silently tolerated.
- Single-placement instant wins are represented directly even during a two-placement turn.

## MCTS

### Legacy MCTS

Legacy MCTS has the richer feature set:

- Arena allocation with node indices.
- PUCT, UCT-VP, and PUCT-V selectors.
- Welford variance tracking.
- Virtual loss.
- Root Dirichlet noise.
- Gumbel Sequential Halving.
- Subtree reuse via `re_root`.
- Threat invalidation/purge after re-root.
- Batched leaf selection.
- Pipelined leaf selection/expansion APIs.
- Tree node extraction for RGSC.
- Root child priors/Q/value diagnostics.

Key legacy files:

- `/Users/coltonmiller/Documents/GitHub/Hexagon/src/mcts.rs`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/game/mcts.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/src/pybridge.rs`

### Rewrite MCTS

Rewrite MCTS keeps:

- Arena nodes.
- PUCT baseline.
- Batched leaf selection.
- Virtual loss.
- Root expansion and backprop.
- Dirichlet root noise.
- Root result extraction.
- Subtree reuse.
- Temperature sampling.
- Resign check.
- Error-returning `re_root`.

Rewrite removes or has not yet reimplemented:

- Gumbel Sequential Halving.
- Variance-aware selectors.
- Welford variance tracking.
- Pipelined FFI methods.
- Full legacy MCTS diagnostic surface.

Given the rewrite goal of removing subtle plateau-causing complexity, this is a reasonable simplification. If strength stalls later, Gumbel and selector variants should come back behind small, tested interfaces rather than reintroducing the old monolith.

## Encoder

Both projects use the same 13-channel 33x33 tensor contract:

| Channel | Meaning |
|---|---|
| 0 | Own/current-player stones. |
| 1 | Opponent stones. |
| 2 | Empty mask. |
| 3 | Legal moves. |
| 4 | Turn phase / second-placement flag. |
| 5 | First stone of current turn. |
| 6 | Player color. |
| 7 | Own recency. |
| 8 | Opponent recency. |
| 9 | Opponent hot cells. |
| 10 | Own hot cells. |
| 11 | Distance from center. |
| 12 | Opponent last turn. |

Legacy had duplicated encoder code in Rust MCTS and PyO3 bridge paths. Rewrite makes `encoder.rs` canonical and exposes compact-record decoding and D6 symmetry through `hexgame-py`.

## PyO3 Bridge

Legacy bridge:

- One large `pybridge.rs`.
- Raw `PyBytes` for policy/value/tensor data.
- More knobs for selectors, Gumbel, and pipeline overlap.
- Exposes axis influence/tactical targets.

Rewrite bridge:

- Split into `engine.rs`, `encode.rs`, and a placeholder/stub buffer module.
- Uses typed NumPy arrays.
- Releases GIL around heavy Rust work.
- Smaller MCTS API surface.

## Idea Assessment

| Idea | Recommendation | Rationale |
|---|---|---|
| Core game rules | Adopt | Stable game definition; already cleaner in rewrite. |
| Explicit `Turn` primitive | Adopt | Reduces duplicate turn canonicalization and search edge cases. |
| Split board/eval/threat modules | Adopt | Makes subtle engine bugs easier to isolate. |
| Incremental eval with invariant tests | Adopt | Good performance idea, but must stay test-backed. |
| Pair-aware threat constraints | Adopt with audit | Important tactical idea; Python-facing legal masks need pair-aware semantics or clear approximation labels. |
| Canonical Rust encoder | Adopt | Prevents Python/Rust tensor drift. |
| 13-channel tensor layout | Adopt for continuity | Compatible with existing intuition and tests; changes should be deliberate. |
| Classical alpha-beta engine | Adopt as evaluator/baseline | Useful for bootstrap/eval/debug even if not core training target. |
| Baseline PUCT MCTS | Adopt | Necessary simple baseline for debugging learning. |
| Subtree reuse | Adopt after correctness tests | Useful speed idea, but subtle with threat invalidation and turn phases. |
| Dirichlet root noise | Adopt | Standard AlphaZero exploration mechanism. |
| Gumbel Sequential Halving | Investigate later | Potentially useful exploration idea, but should return only behind isolated tests and metrics. |
| Variance-aware selectors | Investigate later | Could help uncertainty handling, but adds complexity and another plateau suspect. |
| Pipelined MCTS FFI | Investigate after profiling | The rewrite inference server may make legacy pipeline overlap unnecessary. |
| Axis/tactical Rust target APIs | Investigate | Useful for dashboard/debug/training targets, but do not couple them back into the core loop prematurely. |
| Raw byte PyO3 APIs | Avoid unless measured | Typed NumPy bindings are clearer; raw bytes are only justified by profiling. |
| Monolithic game state owner | Avoid | The split rewrite design is easier to test and reason about. |

## Rebuild Guidance

- Keep rewrite engine boundaries.
- Treat `encoder.rs` as the canonical training/inference tensor source.
- Add tests before adding any legacy MCTS exploration idea.
- Keep Python-facing threat masks pair-aware, or explicitly document when they are only approximate.
- Do not add raw byte bridge APIs unless profiling proves NumPy array interop is a bottleneck.
