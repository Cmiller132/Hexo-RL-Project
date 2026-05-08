# hexo-model-* Project Family

## Purpose

The `hexo-model-*` project family contains trainable model packages for Hexo-RL. Each package represents one model architecture family, including its model-specific data representation, inference surface, training targets, and evaluation diagnostics.

These projects are intended to make model experimentation explicit without mixing architecture decisions into the game engine, orchestration layer, or shared runner infrastructure. A model package should be easy to swap into a runner when it satisfies the expected contracts, but it should not become the owner of Hex game rules or global training policy.

## Family Boundaries

Each `hexo-model-*` package owns the parts of the system that are genuinely model-specific:

- Network architecture and model configuration.
- Model-specific preprocessing, feature extraction, graph building, or tensor construction.
- Inference adapters that expose the model through engine and runner contracts.
- Target and loss definitions for that model family.
- Optional search, player, or policy improvement logic when it is coupled to the model representation.
- Model-specific evaluation diagnostics, probes, calibration checks, and debugging views.

The family consumes contracts from the engine and runner projects. It may use shared utilities from `hexo-utils` for generic serialization, configuration, metrics, batching helpers, logging helpers, or numerical support. It should not own rule legality, terminal-state authority, canonical board semantics, replay ownership, run scheduling, or shared experiment orchestration.

The model family is therefore a consumer and implementer of contracts, not the source of game authority.

## Project Shape

The project family is expected to contain separate packages for distinct architecture families. Examples include:

- `hexo-model-dense-cnn` for dense convolutional models over board-like tensors.
- `hexo-model-resnet` for residual convolutional policies and value networks.
- `hexo-model-global-graph` for graph-based models that build global board or relationship representations.
- `hexo-model-v1` for the existing V1 model family and its pair-native assumptions.
- Future `hexo-model-*` packages for transformer, hybrid graph-convolutional, diffusion-style policy prior, or other experimental architectures.

The boundary should be architectural, not merely chronological. A new package is justified when the representation, preprocessing, inference contract implementation, target design, diagnostics, or search coupling is meaningfully different from an existing package.

## Shared Contract Consumption

Model packages should integrate through stable contracts supplied by the engine and runner layers. Those contracts define the shape of legal inputs, outputs, and runtime responsibilities. A model package may implement an adapter for those contracts, but it should not redefine the contracts locally.

Typical integration responsibilities include:

- Receiving canonical board or state payloads from engine-facing APIs.
- Transforming canonical state, legal rows, tactical payloads, and replay
  events into the model's chosen input representation.
- Producing policy, value, embedding, score, or auxiliary predictions in the expected contract form.
- Participating in runner-managed game execution, evaluation, and inference workflows.
- Exposing model-owned training adapters that separate training targets, losses, and checkpoint semantics from runner orchestration.
- Reporting diagnostics and metrics through shared observability surfaces.

This keeps the engine responsible for rule semantics, the runner responsible for execution lifecycle, and the model responsible for learned behavior.

## Architecture-Specific Ownership

Different model packages can own different preprocessing depth.

Dense CNN and ResNet packages may consume engine state snapshots and legal
action payloads, then construct their own board tensors, channel stacks, masks,
and target arrays. Their model-specific ownership is concentrated in tensor
layout, network architecture, loss wiring, inference adapters, and diagnostics.

Global Graph packages may own graph construction when that graph is part of the model representation. They may define node, edge, global feature, pooling, and message-passing inputs as model-specific preprocessing. The graph may encode model features, but it must not become an independent rules authority.

Future architectures may own custom tokenization, sequence construction, latent-state projection, sparse feature extraction, or representation-specific auxiliary objectives. Those choices belong in the model package when they are necessary to make the architecture meaningful and reproducible.

## State Intake And Input Construction

Every model package should define how it intakes the generic game context and
turns it into model-native inputs. The shared input boundary should be canonical
state plus rules-owned metadata; the model package chooses the representation.

Examples:

- Dense board models construct channel tensors, legal masks, and scalar
  features.
- ResNet-style models may share similar board tensors but own their exact
  channel semantics and augmentation choices.
- Global Graph models construct graph nodes, edges, global features, and
  batching metadata.
- V1 constructs legal-row and pair-row aligned features, learned proposal
  inputs, tactical target inputs, and pair-native search payloads.

This keeps the engine from becoming an encoder library and lets each model
explain exactly how canonical game state becomes trainable input.

## Training Data Ownership

The runner should emit game events, action history, player diagnostics, search
metadata, and optional model diagnostics. Model packages should decide how those
records become training examples.

Each model package should implement the shared training adapter contracts from
`hexo-utils` and own:

- replay filtering rules for that architecture,
- state-to-input conversion,
- target construction,
- loss masks and sample weights,
- augmentation policy,
- dataset validation,
- compact model-specific training records when generic replay is too broad,
- compatibility rules for older records.

The adapter boundary should be method-oriented rather than inheritance-heavy:
models provide functions for converting runner records into examples, collating
examples into batches, running model-specific training steps, and decoding
batched inference outputs. `hexo-utils` supplies the replay store, sampler,
worker pools, queueing, pinned-memory staging, GPU batch scheduling, and
telemetry around those methods.

Training data may be constructed eagerly during self-play when a model needs
metadata captured at decision time, or lazily during sampling when raw game
events are enough. The package should make that choice explicit:

- Eager construction is useful for search-heavy models that need candidate
  support, visit statistics, completed-Q, tactical labels, or trace metadata
  captured at move time.
- Lazy construction is useful when the model can recreate inputs and targets
  from canonical game history and terminal outcome.

Storage should be layered. The runner-owned record captures what happened in
the game. The model-owned record captures how that game position should train a
particular architecture. This allows multiple model packages to train from the
same games without forcing them to share target semantics.

## V1 And Pair-Native Search

`hexo-model-v1` is allowed to own custom pair-native search or player logic when that logic is intrinsic to the V1 representation and cannot be expressed cleanly as generic model inference plus shared search utilities.

This exception is about representation coupling, not rule ownership. V1 search may use pair-native model outputs, pair-specific heuristics, and representation-specific diagnostics, but legal move authority, terminal evaluation, and canonical game semantics still come from the engine contracts.

Simpler model packages should prefer shared player utilities, generic MCTS, or runner-provided evaluation loops when their outputs fit those abstractions. A package should only own custom search when the search behavior depends materially on the architecture's native representation or training targets.

## Evaluation And Diagnostics

Each model package should provide diagnostics that explain the behavior of that architecture family. These diagnostics may include policy calibration, value calibration, move-ranking quality, representation probes, graph statistics, auxiliary-head metrics, search agreement, rollout behavior, or architecture-specific failure analysis.

Shared benchmark orchestration belongs outside the model package, but the model package should expose the diagnostics needed by shared evaluation systems to compare architectures fairly.

## Non-Goals

The `hexo-model-*` family should not contain:

- Authoritative game rules or legality implementations.
- Global replay storage ownership.
- Runner lifecycle orchestration.
- Cross-model experiment scheduling.
- Shared infrastructure that belongs in `hexo-utils`, the engine, or the runner.
- Compatibility facades that preserve obsolete runtime paths.

Keeping these boundaries clear allows model packages to evolve quickly while preserving a single source of truth for rules, contracts, and execution lifecycle.
