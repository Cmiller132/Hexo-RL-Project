# Training Pipeline Guide

This document describes the current training pipeline and the near-term refactor direction.

## Quick Start

Build the Rust extension and run a small training/smoke job:

```bash
cargo build -p hexgame-py --release
maturin develop --features python
python -m hexorl.cli epoch configs/small_test.toml
```

Other useful CLI entry points:

```bash
python -m hexorl.cli smoke-train configs/small_test.toml
python -m hexorl.cli arena configs/small_test.toml
python -m hexorl.cli bench configs/small_test.toml
```

## Configuration

See `configs/` for example run configs.

Common sections:

- `[model]`: architecture, channels, blocks/layers, heads, sparse and pair strategy settings.
- `[selfplay]`: workers, games, simulations, temperature, Dirichlet noise, RGSC settings.
- `[inference]`: max batch, wait window, FP16, EMA inference behavior.
- `[buffer]`: capacity, lookahead horizons, replay feature storage.
- `[train]`: batch size, optimizer, LR schedule, loss weights, clipping.

## Current Epoch Structure

The epoch pipeline lives in:

```text
Python/src/hexorl/epoch/pipeline.py
```

A normal epoch can include:

1. Bootstrap/classical data loading or generation.
2. Optional self-play through `SelfPlayOrchestrator`.
3. Replay processing and sampling through `ReplayDataset`.
4. Training through `Trainer`.
5. Checkpointing and optional evaluation hooks.

## Current Training Data Flow

```text
GameRecord / replay records
-> buffer target processing
-> ReplayDataset
-> dense tensors, dense policy/value, lookahead arrays, aux_targets
-> Trainer._train_step
-> model forward
-> compute_losses
-> optimizer step
-> EMA update
```

Global graph training uses graph tensors inside `aux_targets`, including:

```text
token_features
token_type
token_qr
token_mask
legal_token_indices
legal_mask
relation_type
relation_bias
policy_target
opp_legal_qr
opp_legal_mask
pair_first_indices
pair_second_indices
pair_token_indices
pair_first_policy_target
pair_policy_target
pair_second_policy_target
```

Dense/sparse training uses dense board tensors plus optional candidate/pair candidate tensors.

## Current Model Families

The current code supports these broad paths:

- dense CNN / residual-style `HexNet`
- sparse/candidate policy heads on `HexNet`
- crop-compatible pair policy head
- global graph `GlobalHexGraphNet`
- global graph pair heads: `policy_pair_first`, `policy_pair_joint`, `policy_pair_second`
- auxiliary heads such as opponent policy, tactical, axis, regret, moves-left, lookahead

Model assembly currently lives in:

```text
Python/src/hexorl/model/network.py
Python/src/hexorl/model/global_graph.py
```

The planned architecture refactor will move architecture authority to:

```text
Python/src/hexorl/models/
```

## Current Loss Routing

Loss code currently lives in:

```text
Python/src/hexorl/train/losses.py
```

The current `compute_losses` function routes losses by raw output head names.

Common heads:

| Head | Target/mask source | Notes |
|---|---|---|
| `policy` | dense policy target | Dense 1089 policy. |
| `sparse_policy` | sparse candidate target + candidate mask | Requires sparse/candidate aux targets. |
| `pair_policy` | crop pair target + pair candidate mask | Crop-compatible auxiliary pair scorer. |
| `policy_place` | graph `policy_target` + `legal_mask` | Global graph legal-row policy. |
| `policy_pair_first` | `pair_first_policy_target` + `legal_mask` | First-placement pair first-action projection. |
| `policy_pair_joint` | `pair_policy_target` + pair row mask | First-placement unordered pair table. |
| `policy_pair_second` | `pair_second_policy_target` + pair row mask | Second-placement known-first pair table. |
| `value` | value target | Binned value loss. |
| `opp_policy` | opponent policy target | Auxiliary opponent policy. |
| `regret_rank` | regret ranking target | RGSC/regret auxiliary. |
| `regret_value` | regret value target | Binned regret magnitude. |

Important current limitation:

```text
Some losses skip when targets/masks are missing. The modular model refactor should replace this with explicit LossPlan validation.
```

## Pair Policy Training

Pair policy has multiple forms:

- `pair_policy`: crop/candidate pair scorer.
- `policy_pair_first`: graph legal-row first-action projection.
- `policy_pair_joint`: graph pair-row joint scorer for first-placement turns.
- `policy_pair_second`: graph known-first second-placement scorer.

Pair targets must preserve phase semantics:

```text
first-placement pair rows are unordered pairs of legal actions
second-placement pair rows are ordered by known first placement plus legal second action
```

The graph builder and replay sampler currently enforce many of these rules. The refactor will move the identity and validation into row/target contracts.

## Value And Lookahead Targets

Value heads train from game outcomes and processed replay targets.

Lookahead targets are configured through buffer lookahead horizons and are emitted as auxiliary targets when requested. They are perspective-adjusted by replay processing and consumed by configured lookahead heads.

## RGSC / Regret Training

Current status:

```text
regret auxiliary heads + regret-biased replay + experimental RGSC restart service
```

Implemented pieces include:

- `regret_rank` and `regret_value` heads/losses
- selected-action value fields in self-play records
- regret target assignment in buffer target processing
- regret-biased replay sampling
- `RGSCRestartService` wired into self-play when `selfplay.rgsc_beta > 0`
- RGSC metrics attached to game records and dashboard recorder output

Do not describe the current system as fully paper-complete RGSC until the tests and checklist in `Docs/RGSC_IMPLEMENTATION.md` pass.

## Current Trainer Behavior

The trainer currently handles:

- device placement
- optional channels-last on CUDA
- optional `torch.compile` on CUDA
- AMP/GradScaler on CUDA
- optimizer and scheduler setup
- EMA model updates
- gradient clipping
- graph-model input branching
- loss computation through `compute_losses`
- basic per-head metrics such as pair policy top-1 when available

Main implementation:

```text
Python/src/hexorl/train/trainer.py
```

## Planned Training Refactor

The modular model architecture plan changes training to:

```text
Replay projection
-> row/target contracts
-> TrainingAdapter
-> model forward
-> LossPlan validation
-> LossRegistry
-> metrics
```

Expected behavior changes:

- trainable heads must have required targets and masks
- missing required target/mask is an error, not a silent skip
- graph training should not accidentally consume dense policy targets
- target identity should be tied to row table contracts
- trainer should no longer branch broadly on model class or raw head names

See:

```text
Docs/refactor/MODEL_ARCHITECTURE_MODULARIZATION_PLAN.md
```

## Monitoring

Important metrics:

- self-play throughput: games/min, positions/sec
- inference throughput and latency: batch size, p50/p99 wait, GPU utilization
- replay health: buffer size, sample age, full-search/PCR mix, regret fraction
- training: total loss, per-head losses, gradient norm, learning rate
- policy quality: policy top-k/entropy where available
- value quality: value loss and calibration
- pair policy quality: pair top-1/top-k where available
- RGSC: restart attempts, successes, rejections, PRB size, PRB refreshes, tree-node insertions
- evaluation: arena win rate/ELO against baselines/checkpoints

## Current Caveats

- The top-level docs describe active code, but the model architecture refactor is still planned, not implemented.
- RGSC is experimental and should be reported conservatively.
- Shared-memory inference is performance-sensitive; protocol refactors must preserve or benchmark throughput.
- Global graph and pair-head behavior is powerful but still being moved toward stricter row/target contracts.
