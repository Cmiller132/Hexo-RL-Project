# System Design - Hexo-RL Training Pipeline

This is the active high-level design for the current repository. It describes what exists now and calls out the main refactor direction where behavior is intentionally changing.

## Executive Summary

Hexo-RL is a single-machine AlphaZero-style training system with these major pieces:

```text
Rust rules/MCTS engine
Python self-play workers
Python shared-memory inference server
Python replay buffer and sampler
Python PyTorch trainer
Python dashboard/recording utilities
```

The central runtime design is still:

```text
many CPU self-play workers -> one GPU inference server -> replay buffer -> trainer
```

Workers own game state and Rust MCTS engines. The inference server owns model inference and batches requests across workers. The trainer consumes replay batches and updates model weights. The buffer owns replay storage and sampling.

## Current Process Topology

```text
SelfPlayOrchestrator
  -> InferenceServer process
  -> N SelfPlayWorker processes
  -> completed GameRecord queue
  -> replay/buffer owner
  -> Trainer
```

Important current implementation files:

| Component | Code |
|---|---|
| Rust MCTS and rules | `crates/hexgame-core/` |
| PyO3 FFI | `crates/hexgame-py/src/engine.rs` |
| self-play worker | `Python/src/hexorl/selfplay/worker.py` |
| self-play orchestrator | `Python/src/hexorl/selfplay/orchestrator.py` |
| inference server/client | `Python/src/hexorl/inference/server.py`, `client.py`, `shm_queue.py` |
| replay buffer/sampler | `Python/src/hexorl/buffer/` |
| trainer | `Python/src/hexorl/train/trainer.py` |
| losses | `Python/src/hexorl/train/losses.py` |
| epoch pipeline | `Python/src/hexorl/epoch/pipeline.py` |
| dashboard recorder/app | `Python/src/hexorl/dashboard/` |

## Rust/Python Boundary

Rust owns:

- game rules
- legal move generation
- placement phase rules
- win detection
- tensor encoding helpers
- MCTS tree search
- dense/sparse/global root validation
- pair-prior legality checks
- classical search/bootstrap helpers

Python owns:

- model definitions
- inference batching
- training
- replay storage and sampling orchestration
- self-play process orchestration
- graph batch construction
- dashboard and metrics
- config validation

The active MCTS API is root/leaf expansion based, not the older `run_until_inference_needed` style design.

Root flow:

```text
engine.init_root()
-> Python inference
-> engine.expand_root(...) or expand_root_with_sparse_priors(...) or expand_root_with_global_priors(...)
-> optional root pair-prior application when pair_strategy enables it
```

Leaf flow:

```text
engine.select_leaves(batch_size)
-> Python inference
-> engine.expand_and_backprop(...)
```

Global graph leaf flow additionally uses `pending_leaf_metadata()` so Python can rebuild graph batches from compact histories and legal rows.

## Inference Design

Current inference uses shared memory slots per worker.

Worker side:

```text
InferenceClient.submit(...)
InferenceClient.submit_sparse(...)
InferenceClient.submit_graph(...)
```

Server side:

```text
InferenceServer batches ready workers
runs model forward
sanitizes logits/value outputs
writes response arrays and metadata back to worker slots
```

Global graph inference currently has explicit graph result arrays and pair-head arrays in shared memory.

Current limitation:

```text
Inference server/client still know too much about model head names and graph pair outputs.
```

Planned refactor:

```text
Inference protocol + architecture-selected adapters + shared memory as transport only
```

## Model Architecture Status

Current model authority lives under:

```text
Python/src/hexorl/model/
```

Important classes:

- `HexNet`
- `GlobalHexGraphNet`

Current model assembly:

```text
Python/src/hexorl/model/network.py::build_model_from_config
```

Current limitation:

- architecture membership and head behavior are duplicated across config, model assembly, training, inference, buffer, and self-play.
- trainer and inference still use model-class/head-name checks.

Planned refactor:

```text
Python/src/hexorl/models/ becomes the new architecture authority
ArchitectureSpec defines heads, trunks, targets, losses, adapters, providers, and pair capabilities
```

See:

```text
Docs/refactor/MODEL_ARCHITECTURE_MODULARIZATION_PLAN.md
```

## Replay And Training Design

Replay records store compact histories, policy targets, value targets, pair policy targets, regret fields, and auxiliary data. The sampler builds tensors and aux targets for dense, sparse, and global graph training.

Current training flow:

```text
ReplayDataset yields dense tensors, dense policy/value, lookahead arrays, aux_targets
Trainer chooses dense or graph input path
model forward
compute_losses routes by raw head name
gradient step
EMA update
```

Current limitations:

- graph training still branches on `GlobalHexGraphNet`
- losses are routed by raw head names
- some missing targets are skipped silently
- target ownership is split across buffer and graph modules

Planned refactor:

```text
Replay projection -> row/target contracts -> TrainingAdapter -> LossPlan -> LossRegistry
```

## Policy And Pair Behavior

Current policy paths:

- dense policy over 1089 board indices
- sparse/candidate policy over candidate rows
- global graph `policy_place` over all legal graph rows

Current pair paths:

- crop-compatible `pair_policy`
- global graph `policy_pair_first`
- global graph `policy_pair_joint`
- global graph `policy_pair_second`

Pair scoring is gated by `model.pair_strategy != "none"`, but self-play still directly checks pair-head outputs and applies pair priors.

Planned refactor:

```text
PolicyProvider maps model policy to legal rows
PairStrategy owns all pair row generation/scoring/blending
EngineAdapter owns Rust MCTS calls
```

## RGSC Status

Current status:

```text
regret auxiliary heads + regret-biased replay + experimental RGSC restart service
```

RGSC restart support is implemented through `RGSCRestartService` and self-play worker integration, but paper-complete RGSC still needs stronger tests and long-run evidence.

See:

```text
Docs/RGSC_IMPLEMENTATION.md
```

## Configuration

Config currently validates many architecture/head/pair relationships directly in `Python/src/hexorl/config/schema.py`.

Current useful behavior:

- type and range validation
- pair strategy local validation
- required sparse/pair config checks

Planned change:

```text
Config keeps syntax/range validation.
Architecture registry resolves supported heads, defaults, loss plans, adapters, providers, and pair capabilities.
```

## Dashboard And Metrics

Dashboard code currently supports:

- run metrics
- replay/debug inspection
- graph debug payloads
- pair target summaries
- RGSC metrics and PRB snapshots through recorder fields

The model architecture refactor should make dashboard debugging easier by exposing contract traces:

```text
architecture id
input contract
row table hash
output contract
adapter/provider/strategy id
```

## Performance Principles

The system is performance-sensitive in inference and self-play. Refactors must preserve this shape:

- resolve specs once, not in hot loops
- cache loss plans and adapter plans
- keep shared memory fixed-array transport unless an equivalent transport is benchmarked
- compute row hashes at row-table creation time
- avoid per-row Python loops in inference/self-play
- make telemetry levels configurable
- keep pair scoring disabled unless an explicit strategy requires it

See:

```text
Docs/refactor/PERFORMANCE_STRATEGY.md
```

## Current Known Technical Debt

- Model architecture behavior is fragmented and is scheduled for a clean rewrite.
- Training loss routing should move from raw head-name branches to resolved loss plans.
- Inference head decoding should move behind adapters.
- Self-play should stop directly consuming pair-head names.
- Row identity should become a first-class contract.
- RGSC should remain marked experimental until full acceptance evidence exists.

## Design Decisions For Current V1

- Rust remains canonical for rules, legal rows, and MCTS validation.
- Python remains canonical for model training/inference orchestration.
- Shared memory remains the inference transport for now.
- Global graph is a first-class model family, but pair-head runtime use requires explicit pair strategy.
- The modular model architecture refactor should follow the four-stage clean rewrite in `Docs/MODEL_ARCHITECTURE_MODULARIZATION_PLAN.md`, not a long compatibility-wrapper migration.
