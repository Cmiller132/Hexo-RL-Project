# hexo-engine

## Purpose

`hexo-engine` is the rules and game-state authority for Hexo-RL. It owns the canonical representation of a Hexo position, the legal transitions from that position, and the engine-level payloads needed by search, evaluation, self-play, debugging, and external runners.

The component should be usable by any host that needs to ask rules questions or advance a game, regardless of which player implementation is making decisions. A runner should be able to pair scripted players, classical search players, neural models, remote services, or human-controlled players without changing the engine itself.

## Boundaries

`hexo-engine` sits below model, training, experiment, and dashboard layers. It exposes stable game mechanics and state-derived facts; it does not decide how a model is built, trained, scheduled, compared, visualized, or deployed.

The engine boundary should provide:

- Authoritative game rules and terminal-state detection.
- Canonical state construction, mutation, validation, serialization, and replay.
- Legal action generation for the current state.
- State and action identity suitable for caching, reproducibility, deduplication, and diagnostics.
- Tactical and rules-derived analysis payloads that higher layers can consume without reimplementing rules.
- FFI and API surfaces that let Python and other hosts use the same canonical rule implementation.

The engine boundary should not provide:

- Model architecture code.
- Tensor construction, board tensor encoding, graph construction, or
  model-specific feature extraction.
- Training losses, target-building policy, optimizers, or gradient logic.
- Model-specific search policy or neural prior interpretation.
- Dashboard, plotting, or experiment-inspection UI.
- Experiment orchestration, sweep management, scheduling, or run lifecycle ownership.

The engine may use Rust-side parallelism for rules-owned batch operations such
as state mutation, legal row generation, tactical analysis, fixture generation,
and search-support utilities. It should expose compact batch APIs where useful,
while leaving model tensor construction and training batch preparation outside
the engine boundary.

## Rust Rules Engine

Rust should remain the canonical implementation of Hexo rules. The Rust side is responsible for enforcing legal placement, player-to-move behavior, placements remaining, undo/replay semantics, terminal detection, and invariant checks around malformed or stale inputs.

Higher-level systems may validate and inspect engine outputs, but they should not fork rules logic into parallel Python implementations for production behavior. Python and runner-facing APIs should call into the Rust-backed engine contract when they need canonical answers.

The Rust engine should be deterministic for the same initial state and action sequence. Where performance-sensitive caches or candidate structures are used, they should preserve the same observable rules behavior as a full canonical evaluation.

## Canonical Game State

The canonical game state is the source of truth for:

- Occupied cells and owners.
- Current player.
- Move history and replay order.
- Remaining placements or turn counters needed by the rules.
- Terminal status and winner, when known.
- Derived rules state needed to validate future actions.

State snapshots should be explicit enough for a runner to pause, resume, inspect, serialize, and replay games without relying on model or training objects. A state created from a serialized snapshot should produce the same legal actions, identity values, tactical facts, and terminal result as the original state.

The engine may expose compact rules-owned views such as sparse coordinates,
move history, legal rows, state hashes, or debugging summaries. It should not
construct neural tensors, board encodings, graph batches, or architecture-shaped
features. Those transforms belong in `hexo-model-*` when they are
architecture-specific, or in `hexo-utils` when they are generic reusable
helpers.

## Legal Action Generation

Legal action generation belongs in `hexo-engine`. It should return the valid actions for the current state according to the canonical rules, including placement limits, terminal-state restrictions, coordinate bounds, and any rules-defined constraints that are part of the game contract.

Legal actions should be represented in a stable, inspectable form that a generic runner can pass to any player. Model-specific policy heads may map these actions into logits or priors elsewhere, and tactical payloads may annotate why some actions are urgent or strategically important, but the engine remains responsible for saying which actions are legal.

For runner compatibility, legal action output should support both simple turn-taking use cases and batched/search use cases:

- A basic player can request the legal actions for a single state.
- A search implementation can request compact legal rows or metadata for many states.
- A debugging tool can compare a selected action against the legal set and get a clear rejection reason.

## Tactical Analysis Payloads

`hexo-engine` should expose tactical analysis that is derived directly from rules state. This includes high-level tactical facts such as threats, forced responses, immediate wins or losses, constrained legal sets, and other game-mechanics signals that must remain consistent with legal action generation.

These payloads should be descriptive rather than prescriptive. The engine can report tactical facts; it should not choose a model policy, tune search exploration, assign training loss weights, or decide how a player should trade off tactical and learned information.

Tactical payloads should be structured so they can be logged, inspected, validated, and passed through FFI without depending on a particular neural architecture.

## State And Action Identity

The engine should own stable state and action identity. Identity is needed for transposition tables, replay validation, cache keys, debug traces, reproducible search behavior, and cross-language consistency checks.

State identity should account for all rules-relevant information, including player to move and remaining turn or placement context. Action identity should distinguish actions unambiguously across coordinate systems, compact encodings, serialized rows, and FFI payloads.

Identity values are engine contracts, not model contracts. A neural model may attach embeddings, priors, or values to states and actions, but it should not redefine the canonical identity of a state or action.

## FFI And API Exposure

`hexo-engine` should expose a small, stable API that lets host layers create states, apply actions, query legal actions, retrieve tactical payloads, inspect identity values, return rules-owned compact state views, and validate serialized data.

The FFI/API boundary should make invalid inputs explicit. Stale legal rows, malformed serialized histories, illegal actions, mismatched batch metadata, non-finite numeric payloads, and incompatible contract versions should be rejected with actionable errors rather than silently accepted.

The API should be suitable for:

- Python training and evaluation code.
- Self-play and match runners.
- Search implementations that need repeated state expansion.
- Debugging and replay tools.
- Future non-Python hosts that need the same canonical rules behavior.

API outputs should carry enough metadata for a generic runner to coordinate players without knowing their internals. A runner should be able to ask the engine for the current state, current player, legal actions, terminal status, and any optional diagnostics, then hand that information to an arbitrary player implementation.

## Runner Compatibility

`hexo-engine` should assume the runner owns orchestration and player lifecycle. The engine advances states and exposes facts; the runner decides which player acts, how clocks or budgets are enforced, how matches are scheduled, and where results are recorded.

This separation allows the same engine to host:

- Deterministic rule-based players.
- Classical search players.
- Neural players with local inference.
- Remote or service-backed players.
- Human or interactive players.
- Mixed tournaments where each side uses a different implementation strategy.

The engine contract should therefore avoid assumptions about model architecture, tensor shape policy, training phase, search algorithm, device placement, or experiment context. It should provide canonical game mechanics in a form that any compliant player can consume.

## Non-Goals

The following responsibilities should live outside `hexo-engine`:

- Defining neural network architectures, encoders tied to a specific model family, or architecture registries.
- Constructing tensors, graph batches, channel stacks, token sequences, or
  architecture-specific board encodings.
- Computing training losses, value targets, policy targets, replay sampling strategy, or optimizer updates.
- Owning model-specific search policy, neural prior calibration, exploration schedules, or inference batching strategy.
- Managing dashboards, charts, inspection UIs, or experiment reports.
- Running experiment orchestration, sweeps, worker fleets, checkpoint policy, or run metadata stores.

Those layers can consume engine facts, legal actions, tactical payloads, and identity contracts, but they should remain separate clients of the engine rather than being embedded inside it.
