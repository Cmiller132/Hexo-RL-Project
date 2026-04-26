# Axis Lab Design Principles

The Axis Lab exists to design and inspect auxiliary axis-strength targets for Hexo-RL. Its purpose is not to choose moves directly. It should help the model understand board structure: which axes are becoming strong, where multi-axis pressure is forming, and where both players have simultaneous influence.

## Core Goal

The target should teach strategic line geometry, not immediate tactics.

Immediate wins, must-block cells, and forced tactical responses are already handled by the engine threat logic and the NN legal-mask restriction. Axis targets should therefore avoid duplicating "close this threat" or "block this threat" policy behavior. They should instead describe the structure that leads to future threats, especially multi-axis strength and fork potential.

## Perspective-Indexed Targets

The preferred training target is perspective-indexed:

```text
axis_strength: [own_axis_0, own_axis_1, own_axis_2,
                opp_axis_0, opp_axis_1, opp_axis_2]
```

This is better than absolute P0/P1 indexing for training because the same channel always means the same thing from the side-to-move perspective:

- `own_axis_*`: current player's structure.
- `opp_axis_*`: opponent's structure.

The dashboard can still render those channels with player colors by mapping current player to P0/P1.

## Avoid Subtractive Targets

Signed legacy-style targets collapse two facts into one value:

```text
net_axis = own_axis - opp_axis
```

That loses information. A cell can be strong for both players, and subtractive targets can incorrectly make that cell look neutral. The lab should preserve both strengths separately whenever possible:

```text
own_axis >= 0
opp_axis >= 0
```

Derived views are fine for visualization:

- `net = own - opp`
- `max = max(own, opp)`
- `both_strong = min(own, opp)`
- `pressure = own + opp`

But those derived views should not replace the raw dual-channel target.

## What Counts As Axis Strength

Axis strength should come from pure six-cell windows on each of the three Hex axes:

- Axis 0: `(1, 0)`
- Axis 1: `(0, 1)`
- Axis 2: `(1, -1)`

A pure window contains stones from only one player plus empty cells. A contested window contains both players and should usually contribute zero, because it is blocked as a line-building structure.

Strength should be nonlinear. Longer pure windows should matter much more:

```text
0 stones: 0.00
1 stone:  small
2 stones: small-medium
3 stones: developing
4 stones: hot-window pressure
5 stones: near-win pressure
6 stones: terminal/already complete
```

The exact weights are tunable. The legacy Hexagon weights were:

```text
[0.00, 0.02, 0.06, 0.15, 0.45, 1.00, 1.00]
```

These are a reasonable baseline, not sacred constants.

## Dense Field vs Legal-Cell Prototype

There are two distinct target families:

### Dense Axis Field

Dense field targets assign values across the full model crop:

```text
[6, 33, 33]
```

This is the cleanest auxiliary model target because it teaches board understanding everywhere in the encoded position, including occupied cells and empty cells.

### Legal-Cell Development

Legal-cell prototypes ask: "If this legal cell were occupied, how much axis strength would it build?"

This is useful for tuning and intuition, but it is closer to an action affordance than a dense board-understanding target. Treat it as experimental unless it proves useful.

## Multi-Axis Principle

A key goal is helping the model recognize positions that build strength on multiple axes at once. Single-axis extension is often easy; multi-axis pressure is where strategy becomes rich.

Useful derived diagnostics:

- Count how many axes have nonzero own strength.
- Count how many axes have nonzero opponent strength.
- Highlight cells where `own` is strong on two or more axes.
- Highlight cells where `opp` is strong on two or more axes.
- Highlight cells where both players have meaningful strength.

Do not confuse this with immediate tactical wins. Multi-axis pressure is about building future threats and forcing the opponent into difficult responses.

## Two-Placement Threat Logic

Hexo's turn structure changes the meaning of line strength. After the opening, a player normally places two stones per turn. Because of that, a pure 4-stone window and a pure 5-stone window are closer in tactical strength than they would be in a one-placement game.

In a one-placement game:

- A 5-window is an immediate one-move win.
- A 4-window is only a threat to become a 5-window.

In Hexo:

- A 5-window is still an immediate win with one placement.
- A 4-window with two empty cells can also be completed within the same turn if both empty cells are legal and available.

So the target should not treat 4-stone windows as merely "developing" in the ordinary Gomoku sense. They are often already forcing threats. This does not mean 4 and 5 should have identical values, because a 5-window wins with one stone and survives some edge cases better, but their values should be much closer than in a standard single-placement game.

This is one reason a purely move-ranking target is the wrong abstraction. The model needs to understand how many independent 4/5-window structures exist across axes, not just which single cell closes the closest line.

## Winning Against Strong Play

Against a good opponent, single threats are usually not enough. If the opponent can block the only relevant 4/5-window with their turn, the threat does not create lasting advantage. The practical path to winning is usually to create multiple simultaneous threats so the opponent cannot cover all of them.

Important patterns for the target to make visible:

- Double threats: one move or turn creates two independent hot windows.
- Cross-axis forks: one local region creates pressure on two or three axes.
- Shared-cell threats: multiple windows overlap on a key cell, creating leverage.
- Distributed threats: threats are far enough apart that a two-stone block cannot cover all of them.
- Tempo threats: a move creates a threat while also improving another latent axis, so every block concedes the next forcing step.

The target should therefore reward structure that creates future overload, not just immediate completion. A cell that builds two moderate axes may be strategically more important than a cell that merely extends one already-obvious line.

## Strategy Patterns To Consider

These are candidate strategic concepts the Axis Lab may eventually need to expose or measure:

- **Fork creation**: cells that increase own strength on at least two axes.
- **Threat overload**: positions where opponent must answer more independent windows than their placements can cover.
- **Reserve/latent axis strength**: a second axis that is not urgent yet, but becomes forcing after the opponent answers the primary threat.
- **Blocking efficiency**: whether opponent can block several threats with one cell or pair of cells.
- **Threat spacing**: threats that are geometrically separated may be harder to cover in one turn.
- **Shared pivot cells**: cells that participate in many pure windows and can become the center of a fork.
- **Contested tension**: cells or regions where both own and opponent strength are high; subtractive targets hide this.
- **Sacrificial pressure**: moves that allow one threat to be blocked but create a stronger follow-up threat elsewhere.
- **Anti-fork defense**: opponent cells that reduce your ability to create multiple threats, even if they are not immediate blocks.

The first-pass dense target does not need to encode all of these explicitly. But the Axis Lab should help inspect them so the formula can evolve toward the strategic shapes that actually win games.

## Dashboard Visualization

The Axis Lab should show raw target values, not policy ranks.

Preferred visualization:

- Blue values: P0 strength.
- Red values: P1 strength.
- Hover shows exact per-axis floats.
- Toggles can switch between own, opponent, net, max, and both-strong views.
- Labels should be signed or explicitly player-owned.

The UI should avoid normalized "probability mass" when the value being inspected is not a policy target.

## Training Guidance

First-pass training target recommendation:

```text
axis_head_channels = 6
target_layout = [own_axis_0, own_axis_1, own_axis_2,
                 opp_axis_0, opp_axis_1, opp_axis_2]
loss = SmoothL1 or MSE
```

Use modest loss weight at first. The axis target should regularize representation learning without overpowering policy/value learning.

Potential target transforms:

- Raw values first, if scale is stable.
- `log1p(value)` if high-threat windows dominate too much.
- Per-position clipping if rare extreme positions create unstable loss.

Avoid per-position normalization early, because absolute threat strength may be meaningful.

## Open Questions

- Should occupied cells receive target values, or only empty/legal cells?
- Should the final training target include a separate `both_strong` channel?
- Should 4/5-window values be clipped to avoid turning the head into a duplicate tactical detector?
- Should target weights be tuned independently for own and opponent channels?
- Should the dense field and legal-cell development prototype coexist, or should only dense field enter training?
- How close should 4-window and 5-window weights be, given two-placement turns?
- Should fork/multi-axis strength be a derived diagnostic only, or an explicit training channel?
- Should blocking efficiency be measured as part of opponent-strength targets?

The current stance: keep the training target dense and perspective-indexed, use Axis Lab prototypes for experimentation, and only integrate formulas after visual inspection shows the target behaves sensibly across normal game states.
