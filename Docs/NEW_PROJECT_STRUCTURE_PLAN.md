# New Project Structure Plan

## Purpose

This document sketches a cleaner long-term project structure for Hexo-RL. The
goal is to separate the game/rules boundary from model-specific systems so that
new architectures can be developed, debugged, tested, and retired without
touching unrelated models.

The proposed structure is intentionally high level. It defines ownership and
interfaces rather than exact file moves or migration steps.

## Proposed Project Families

The target split is:

- `hexo-engine`: canonical game rules, state identity, legal actions, and
  tactical analysis.
- `hexo-runner`: game orchestration, arbitrary-player execution, evaluation,
  self-play, replay/event emission, and budget management.
- `hexo-utils`: shared reusable mechanisms such as generic MCTS utilities,
  schema helpers, metrics helpers, batching helpers, replay utilities, training
  adapter framework contracts, and game mutation test harnesses.
- `hexo-model-*`: independent model/player projects, such as Dense CNN,
  ResNet, Global Graph, V1 pair model, and future architectures.

The most important rule is that the engine owns rules authority, the runner owns
orchestration, utilities own reusable mechanisms, and model packages own their
model-specific thinking.

## Dependency Direction

The intended dependency shape is:

```text
hexo-engine
   ^
   |
hexo-utils  <--- optional reusable helpers, no rules authority
   ^
   |
hexo-model-*  ---> consume engine contracts and optional utils
   ^
   |
hexo-runner  ---> hosts players, asks for actions, applies through engine
```

More precisely:

- `hexo-engine` should be dependency-light and should not depend on model
  packages, runner packages, training code, dashboards, or experiment tooling.
- `hexo-utils` may depend on public engine contracts when useful, but should
  not become a second rules engine.
- `hexo-model-*` packages depend on engine contracts and may use utilities.
- `hexo-runner` depends on engine contracts, player interfaces, and whichever
  model/player packages it is configured to host.

This direction keeps model complexity from flowing back into the rules boundary.

## Engine Boundary

The engine should expose the canonical game state and all authoritative rule
operations:

- construct and copy game states,
- apply single-placement and full-turn actions,
- reject illegal moves,
- report terminal state and winner,
- enumerate legal actions or legal rows,
- expose stable state/action identity,
- expose tactical analysis useful to models and runners.

The engine should not know whether a player is a human, a CNN, a graph model, a
scripted bot, a remote service, or a V1 pair-search system.

## Runner Boundary

The runner should be the host for games and players. Its job is to:

- create games,
- call players for decisions,
- enforce time/simulation budgets,
- apply returned actions through the engine,
- emit replay/event records,
- run self-play, evaluation matches, tournaments, smoke tests, and benchmarks.

The runner should treat every player uniformly through a player contract. A
player may be a model, a human UI, a scripted policy, a remote service, a
classical searcher, or a hybrid system.

## Utils Boundary

`hexo-utils` should be broader than an MCTS package. It should hold reusable
mechanisms that are valuable across model packages and runner modes:

- generic tree-search and MCTS building blocks,
- reusable tree statistics and backup helpers,
- replay serialization helpers,
- training/replay adapter framework mechanics,
- schema/version helpers,
- telemetry and metrics helpers,
- batching and queue helpers,
- game-state mutation/test harness utilities,
- deterministic seeding and reproducibility helpers.

It must stay non-authoritative. Game legality belongs in `hexo-engine`; model
policy belongs in `hexo-model-*`.

## Model Package Boundary

Each model package should be able to evolve independently. A model package may
own:

- model architecture,
- model-specific preprocessing or graph construction,
- inference adapter,
- player implementation,
- optional search implementation,
- training targets and losses,
- model-specific replay interpretation and training adapters,
- architecture-specific diagnostics and scorecards.

For simple models, the package may import generic search utilities. For V1-like
systems, the model package can own custom pair-native search while still using
engine state, legal rows, tactical payloads, and replay contracts.

## Shared Player Contract

The central integration contract should be a player interface with roughly this
shape:

```text
observe game state + legal/tactical payload + budget
return chosen action or full-turn action + optional diagnostics
```

The exact API can be sync, async, local, remote, batched, or streaming, but the
semantic contract should stay stable:

- the runner asks for a move,
- the player returns a move,
- the engine validates and applies it,
- diagnostics are optional and never become rule authority.

## Replay And Training Boundary

Replay should be separated into layers:

- engine event log: what happened in the game,
- runner metadata: budgets, players, timings, seeds,
- model metadata: policy targets, candidate support, search traces, model
  diagnostics.

This prevents one model's training assumptions from leaking into all replay
consumers. A Dense CNN package and a V1 pair package should be able to read the
same game event log but attach different model metadata and target builders.

`hexo-utils` should provide the reusable training and replay mechanics:
sampling, shuffling, worker pools, pinned-memory staging, batching, queueing,
schema validation, and telemetry. Model packages should plug into those
mechanics through adapters that define how records become examples, how
examples become batches, and how batches produce losses or inference results.

## Resource And CPU Boundary

CPU parallelism should use centralized resource policy and decentralized
execution.

`hexo-utils` defines host profiles, thread-budget settings, queue limits, memory
guardrails, and telemetry helpers. `hexo-runner` applies those settings to
self-play, evaluation, inference services, and replay pipelines. `hexo-engine`
uses Rust-side parallelism for rules-owned game, search, and mutation-heavy
work. `hexo-model-*` packages use the shared profile for DataLoader workers,
adapter batch prep, and model-specific search.

The goal is not a universal CPU scheduler. The goal is to keep worker counts,
queue sizes, prefetch depth, Torch thread settings, Rust thread settings, and
GPU batching visible in one resource profile so split packages do not silently
oversubscribe the machine.

## Feasibility

This transition is feasible, but it should be staged. The main difficulty is
not moving files. The hard part is turning implicit shared assumptions into
explicit contracts:

- state and action identity,
- replay schemas,
- player interface,
- inference/batching expectations,
- model-specific target ownership,
- tactical payload semantics,
- evaluation scorecard boundaries.

The safest first milestone is to define and stabilize the contracts while the
current monorepo structure still exists. Physical package separation should come
after those contracts have tests and at least two model packages consuming them.

## Design Risks

### Accidental Shared Authority

The biggest risk is creating `hexo-utils` as a dumping ground that slowly
recreates model or rules authority. Utilities should provide mechanisms, not
decisions.

### Replay Coupling

Replay can easily become the hidden shared dependency. Keep engine events,
runner metadata, and model metadata distinct.

### Inference Coupling

The runner should not need to understand every tensor produced by every model.
Model packages should translate their own tensors into player decisions and
diagnostics.

### Too-Early Physical Split

Splitting into installable packages before contracts are stable can slow the
project down. Start with logical package boundaries, then make them physical.

### V1 As A Special Case

V1 is useful precisely because it stresses the abstraction. If V1 cannot live as
a model-owned player/search system under this structure, the structure is still
too runner-centric.

## Companion Documents

The component-specific high-level plans live under:

- `Docs/new_project_structure/HEXO_ENGINE.md`
- `Docs/new_project_structure/HEXO_RUNNER.md`
- `Docs/new_project_structure/HEXO_UTILS.md`
- `Docs/new_project_structure/HEXO_MODEL.md`
