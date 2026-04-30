# Rust Invariants And Bounds Plan

Date: 2026-04-29

Purpose: define the invariants that must survive API narrowing and document the WindowKey/evaluation bounds section of the Rust review slice.

## Core Rule Invariants

The `rules` facade owns these invariants:

| Area | Required invariant | Verification path |
| --- | --- | --- |
| Opening | The first move is exactly one placement at `(0, 0)`. | Rust board tests and Python engine smoke. |
| Turn shape | After the opening, each turn has two placements unless the game ends. | Rust board tests and Python invariants. |
| Legal radius | A placement must be empty and within `PLACEMENT_RADIUS` of an existing stone. | Rust board tests, Python reference legal-move invariant. |
| Win detection | Winner is the first player with `WIN_LENGTH` contiguous stones on one hex axis. | Rust tests and Python axis-scan reference invariant. |
| Undo/history | `unplace` restores player, placement count, board occupancy, candidates, and history. | Rust board/eval tests plus MCTS state-restore tests. |

## Tactical Invariants

The `tactics` facade owns complete tactical classification:

| Status | Invariant |
| --- | --- |
| `Quiet` | No current-player immediate win and no opponent immediate threat window requiring a block. |
| `WinningTurns` | Contains every immediate winning single or pair available with the remaining placements. |
| `MustBlock` | `BlockConstraint::cells` and `pairs` describe all moves that cover every opponent threat window. |
| `Unblockable` | Opponent threats cannot be covered with the remaining placements. |

The accepted public behavior is complete tactical output, not a single winning move. Search, encoding masks, Python diagnostics, and future replay labels should consume `TacticalStatus`.

## WindowKey Bounds

`WindowKey` identifies a six-cell line segment by start coordinate and direction. Window-scanning code must keep these bounds explicit:

- Direction is one of the three hex axes in `HEX_DIRECTIONS`.
- A key represents exactly `WIN_LENGTH` cells.
- Full-board tactical scanning is sparse: it enumerates windows touching actual stones and does not depend on the bounded evaluation grid.
- Incremental evaluation may remain bounded for scoring, but tactical legality cannot silently inherit that bound.

The conservative tactical oracle search bound for review evidence is radius 3 around existing stones until a formal radius-2 proof is recorded. Production legal move generation remains governed by `PLACEMENT_RADIUS`.

## Evaluation Bounds

The incremental evaluation state is a scoring accelerator, not the source of tactical truth. Evaluation bounds are acceptable only when:

1. The scoring caller can tolerate bounded heuristic evaluation.
2. Tactical legality and training masks use full sparse tactical scanning.
3. Debug or test invariants can compare incremental hot windows/counts against a recompute on representative positions.
4. MCTS and classical search restore board state after temporary traversal.

## Review Fixtures

The next executable verification path is to promote these fixtures into tests as implementation work reaches each owner:

| Fixture | Owner | Expected proof |
| --- | --- | --- |
| Multi-win current-player position | `tactics` | `WinningTurns` includes every winning single/pair. |
| Two-axis crossing threat | `tactics` | Block cells and valid pairs cover all opponent windows. |
| Outside-eval-grid tactical win | `tactics` / `encoding` | Full sparse scan detects the win and mask includes it. |
| MCTS interrupted selection/retry | `mcts` | Pending leaves and game state restore deterministically. |
| Compact history round trip | `hexgame-py` protocol | Rust/Python history rows preserve player/q/r order. |
