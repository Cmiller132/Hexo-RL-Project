# Hexo — Game Guide

This document explains the game rules, core strategies, and how key properties
of Hexo shape the training pipeline.

Hexo is a new game most similar to Connect 6. It is played on an infinite
hexagonal grid, but it has no relation to Hex other than sharing the same
board geometry.

## Rules

**Board.** An infinite hexagonal grid using axial coordinates `(q, r)`. There
is no fixed board size, no edges, and no corners.

**Players.** Two players (0 and 1).

**Opening.** Player 0 places exactly one stone at the origin `(0, 0)`.

**Turns.** After the opening, every turn consists of **two** placements by the
same player. Player 1 goes next (2 stones), then Player 0 (2 stones), and so
on.

**Placement radius.** A stone may only be placed on an empty hex within
distance 8 of any existing stone. Hex distance is the cube-distance metric:
`max(|dq|, |dr|, |dq + dr|)`.

**Win condition.** The first player to form **6 in a row** along any of the
three hex axes wins. There is no draw — the board is infinite.

### Hex axes

Each hex has 6 neighbors. Lines are checked along three principal axes:

| Axis | Direction `(dq, dr)` | Informal name |
|------|----------------------|---------------|
| A    | `(1, 0)`             | East          |
| B    | `(0, 1)`             | Southeast     |
| C    | `(1, -1)`            | Northeast     |

Win detection scans each axis in both directions from the last placed stone.

## Strategy

### Windows and threats

The engine thinks in **6-cell windows**: every contiguous 6-cell segment along
each hex axis. A window is characterized by how many stones each player has in
it:

| Window state | Name | Meaning |
|---|---|---|
| 5 yours, 0 theirs, 1 empty | **5-window** | 1 placement to win |
| 4 yours, 0 theirs, 2 empty | **4-window** | 2 placements to win |
| 3 yours, 0 theirs, 3 empty | **3-window** | Developing — building material |
| Mixed (both players present) | **Dead window** | Can never become a win for either side |

A window becomes **hot** when it contains 4+ stones of one player and 0 of the
opponent. Hot windows drive all threat detection in the engine.

### Why 5-windows and 4-windows are equally dangerous

Because each turn gives you **two placements**, both are win-in-one-turn
threats:

- A **5-window** has 1 empty cell — fill it with one placement, use the other
  elsewhere.
- A **4-window** has 2 empty cells — fill both with your two placements for an
  immediate win.

### Stacking threats — the core win mechanism

**A single threat is manageable. You win by creating more independent threats
than 2 placements can cover.**

Each hot window has a set of empty "block cells" that the opponent could fill
to neutralize it. The opponent has exactly 2 placements per turn. The position
becomes **unblockable** when no pair of placements can hit every hot window.

| Situation | Can 2 placements block? |
|---|---|
| 1 threat | Yes — at most 2 block cells |
| 2 threats with overlapping block cells | Usually — one placement covers both |
| 2 threats with disjoint block cells | Depends on geometry |
| 3+ threats with no common block cells | **No — game is won** |

The real strategic endgame: accumulating non-overlapping hot windows across
different axes until no pair of blocking placements can cover them all.

### Two placements per turn

This mechanic fundamentally shapes every aspect of play:

- **Offensive**: you can create AND complete threats in a single turn (e.g.,
  place stones 4 and 5 of a line to jump from 3-window to 5-window). A
  4-window is an immediate win on your turn — not true in 1-placement games.
- **Defensive**: blocking consumes placements. Spending both on defense means
  zero development — the opponent builds freely while you tread water.
- **Block + counterattack**: the optimal defensive pattern uses 1 placement to
  block and 1 to develop, maintaining pressure while defending.
- **Forcing chains**: each threat you create forces the opponent to spend
  blocking resources. A sequence of forcing moves that culminates in 3+
  independent threats before the opponent can catch up produces an unblockable
  win.

### Axis forking

Every cell sits on exactly 3 lines (one per hex axis). A well-placed stone
contributes to windows on all three axes simultaneously. Strong play means
choosing cells at **axis intersections** where one placement extends or starts
lines on multiple axes.

Multi-axis development is the foundation of stacking threats. Building 3-windows
on 2+ axes before launching a forcing chain creates the geometric conditions
for unblockable forks. Single-axis development is weak — easily blocked with
1 placement because all block cells overlap.

### Game balance

The game is approximately **50/50 between the two players**. Player 0 opens
with 1 stone and Player 1 responds with 2, but this does not create a
meaningful structural advantage for either side. The model receives a
player-color channel (channel 6) so it can learn any subtle positional
differences between going first and second.

## Why this isn't Go

Many pipeline choices look unusual if you're familiar with AlphaZero or Go
engines. The differences fall out of fundamental game properties.

### Infinite board → dynamic bounding box

Go has a fixed 19×19 grid; policy output maps directly to board positions. This
game has no board boundary. The network operates on a **33×33 window centered
on the piece centroid**, recomputed every inference call. Coordinates are
relative, not absolute. The model never learns edge or corner patterns — there
are none.

### Hex grid → masked convolutions and D6 symmetry

A square grid has D4 symmetry (8 transforms). A hex grid has **D6 symmetry
(12 transforms: 6 rotations + 6 reflections)**. Augmentation uses precomputed
lookup tables for all 12 D6 transforms. The 3 hex axes **permute** under
rotation, so axis-related targets must be reindexed alongside spatial
coordinates.

Standard 3×3 convolutions include two corner positions `(dq=-1, dr=-1)` and
`(dq=+1, dr=+1)` that are hex-distance 2 (not neighbors). `HexConv2d` zeros
those kernel weights after each optimizer step so the network respects hex
topology.

### Two placements per turn → auto-regressive MCTS

In Go, one MCTS search produces one move. Here, each turn is two placements.
MCTS handles this **auto-regressively**: each tree action is a single
placement, but the second placement stays within the same player's subtree.

The classical alpha-beta engine takes the opposite approach: it treats the
**pair** as the atomic unit, with a pair-sum branching constraint to keep the
branching factor manageable.

### No captures, no territory → purely structural evaluation

Go has captures, ko, life-and-death, and territory scoring. This game is purely
additive: stones are never removed, and the only goal is forming a line. The
classical evaluator uses a **ternary window encoding** (each 6-cell window →
one of 729 base-3 patterns) with CMA-ES-optimized heuristic weights, maintained
incrementally in O(18) per placement.

### Auxiliary heads beyond policy + value

AlphaZero uses two heads (policy, value). HexNet uses **six**:

| Head | Purpose | Why it exists |
|------|---------|---------------|
| **Policy** | MCTS visit distribution target | Standard AlphaZero |
| **Value** | Game outcome prediction | Standard AlphaZero |
| **Axis influence** | Per-axis line pressure at each cell | Teaches spatial threat awareness along the 3 hex axes — no analog in Go |
| **Opponent policy** | Predict opponent's next move | Teaches the model to anticipate opponent responses |
| **Regret rank** | RGSC position ranking | Identifies high-learning-potential positions for restart |
| **Regret value** | RGSC absolute regret estimate | Estimates prediction error magnitude for prioritized replay |
