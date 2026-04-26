# 03 Model Architecture

## Scope

This document describes the live legacy neural network and compares it to the rewrite model.

## Legacy Source Anchors

- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/model/network.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/model/features.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/model/loading.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/training/trainer.py`
- `/Users/coltonmiller/Documents/GitHub/Hexagon/python/hexgame/game/mcts.py`

## Input Contract

Legacy `HexNet` consumes:

`float32 tensor (B, 13, 33, 33)`

Policy size:

`33 * 33 = 1089`

The input channels match the engine spec in [02 Engine And Rules](02_ENGINE_AND_RULES.md).

## Legacy Trunk

The legacy network is a configurable residual CNN:

- `HexConv2d(NUM_CHANNELS -> channels)`.
- Hex-masked 3x3 kernels with two invalid square-grid corners zeroed.
- Optional RepVGG-style 1x1 linear branch.
- GroupNorm or FixScale normalization.
- Mish or ReLU activation.
- Standard residual blocks or NBT nested bottleneck blocks.
- Periodic global pooling blocks using mean, max, and stddev board statistics.

Common live presets:

| Config | Shape |
|---|---|
| `configs/training_default.toml` | 10 NBT blocks, 256 channels, bottleneck 96, Mish, RepVGG, GroupNorm, about 4.9M trainable parameters. |
| root `config.toml` | Similar but 192 channels and FixScale, about 4.5M trainable parameters. |

## Legacy Heads

`forward()` returns eight tensors in a fixed tuple order:

| Index | Head | Shape | Meaning |
|---:|---|---|---|
| 0 | `policy` | `(B, 1089)` | Logits over 33x33 board cells. |
| 1 | `value` | `(B, num_value_bins)` | Categorical value distribution. Live default is 16 bins. |
| 2 | `axis_inf` | `(B, 3, 33, 33)` | Per-axis influence/threat target. |
| 3 | `opp_policy` | `(B, 1089)` | Opponent next-policy auxiliary logits. |
| 4 | `regret_rank` | `(B, 1)` | RGSC ranking score. |
| 5 | `regret_value` | `(B, 1)` | Absolute regret estimate. |
| 6 | `ownership` | `(B, 1, 33, 33)` | Current-player stone occupancy target, not true territory. |
| 7 | `moves_left` | `(B, 1)` | Intended remaining-moves target. Live target is broken. |

`inference_forward()` returns only:

- `policy`
- `value`
- `axis_inf`

That inference subset is traced for MCTS.

## Policy Head

The live policy head is simple:

- 1x1 conv to 32 filters.
- GroupNorm.
- Activation.
- 1x1 conv to 1 channel.
- Flatten to 1089 logits.

It is not threat-gated inside the model. Legal masking and target constraints happen outside the network.

## Value Head

The live value head is categorical:

- 1x1 conv.
- Global pool stats: mean, max, stddev.
- MLP to `num_value_bins`.

Training maps scalar targets from `[-1, 1]` into class bins using cross entropy. MCTS maps logits back to a scalar expectation over linearly spaced bins.

Legacy docs saying the model has a 2-bin value head are stale.

## Axis Influence Head

Axis influence predicts `(B, 3, 33, 33)`.

It is used in two ways:

- Auxiliary target during training via MSE.
- Policy-logit boost during MCTS and training policy loss.

This means axis predictions directly influence move priors, not merely diagnostics.

## Opponent Policy Head

Opponent policy predicts the next opponent policy distribution. It is trained when a valid next full-search opponent sample exists. Missing labels are masked.

## Regret Heads

Regret heads support RGSC:

- `regret_rank`: relative ranking score used to prioritize candidate states.
- `regret_value`: scalar regret estimate.

Training uses a ranking loss for `regret_rank` and MSE for `regret_value` in legacy. The rewrite changes `regret_value` to binned value loss.

## Ownership Head

The live target is not true ownership. It marks occupied cells:

- Current player's stones: `+1`.
- Opponent stones: `-1`.
- Empty cells: `0`.

This is more like occupancy-relative-to-perspective. It is a weak strategic signal and should not be blindly preserved as an "ownership" feature in a rebuild.

## Moves-Left Head

The legacy head exists, but the target is effectively always zero. The compact replay buffer stores only the snapshot move count. The dataset then calculates current length minus current length.

This is a likely training-noise source. A rebuild should store final game length or compute moves-left during game-record processing, as the rewrite does.

## Legacy Inference Path

Self-play:

1. Deep-copy the training model.
2. Optionally merge RepVGG branches with `prepare_for_inference()`.
3. Trace `inference_forward()`.
4. Run MCTS root/leaf calls using policy/value/axis.
5. Axis-boost policy logits before sending them to Rust MCTS.

Generic eval/play `NeuralPlayer` uses a narrower MCTS wrapper and does not expose all training-time exploration knobs.

## Checkpoint Loading And Migration

Legacy checkpoint loading is unusually broad because the architecture changed repeatedly. It handles:

- Directory resolution to `latest.pt`.
- Config normalization.
- Old key stripping, including old hex masks.
- Renames from older heads.
- BatchNorm-to-GroupNorm naming migration.
- Input channel count migration.
- Shape mismatch drops.
- RepVGG branch disablement if config and checkpoint disagree.
- Optimizer/scheduler skip when architecture changed.

This is useful as migration history, but should not define the new clean checkpoint format.

## Rewrite Model

Rewrite source:

`/Users/coltonmiller/Documents/GitHub/Hexo-RL-Project/Python/src/hexorl/model/network.py`

The rewrite model is smaller and cleaner:

- Input `(B, 13, 33, 33)`.
- Plain 3x3 input conv.
- Configurable number of `GatedResBlock`s.
- BatchNorm-based gated residual blocks.
- Configurable `ModuleDict` heads.
- Dict output keyed by head name.
- 65-bin binned value heads by default.

Supported rewrite heads:

- `policy`: `(B, 1089)`.
- `value`: `(B, 65)`.
- `lookahead_*`: `(B, 65)`.
- `opp_policy`: `(B, 1089)`.
- `axis`: `(B, 3)` classification, not per-cell influence.
- `regret_rank`: `(B, 1)`.
- `regret_value`: `(B, 65)`.
- `moves_left`: `(B, 1)` with softplus.

## Model Comparison

| Area | Legacy | Rewrite |
|---|---|---|
| Output style | Positional tuple. | Dict by head name. |
| Trunk | Hex-masked conv, GroupNorm/FixScale, NBT option, global pooling. | Simpler BatchNorm gated residual stack. |
| Value | 16 categorical bins by default. | 65 binned soft targets. |
| Axis | Per-cell `(3,33,33)` influence used for policy boost. | 3-class axis label. |
| Ownership | Present but weak/broken semantics. | Dropped. |
| Moves-left | Present but target broken. | Present with valid processed target. |
| Checkpoint migration | Extensive legacy migration. | Simple trainer checkpoint. |
| Inference heads | policy/value/axis. | policy/value only in server. |

## Rebuild Guidance

- Prefer the rewrite's dict output and config-driven heads.
- Do not copy the legacy positional output contract into new code.
- Reconsider whether axis should be per-cell influence, 3-class winner-axis classification, or both.
- If restoring axis-policy boosting, add isolated tests because it changes MCTS priors and training policy loss.
- Drop legacy ownership unless a true target can be generated.
- Keep binned value targets and lookahead heads from the rewrite; they address sparse terminal-value learning better than legacy.
- Keep checkpoint migration separate from the clean checkpoint format.

