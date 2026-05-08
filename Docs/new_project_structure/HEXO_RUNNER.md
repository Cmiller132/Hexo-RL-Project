# hexo-runner

## Purpose

`hexo-runner` is the execution and orchestration layer for Hexo-RL. It owns the process of running games between arbitrary participants, collecting their decisions, advancing the authoritative game state through the engine, and emitting the resulting replay and event data.

The runner should answer operational questions:

- Which players are participating in this game or batch of games?
- What mode is being executed: self-play, evaluation, benchmark, match, interactive play, or scripted scenario?
- Whose turn is it, what observations and legal action context should they receive, and how long may they think?
- How are completed games recorded, summarized, and streamed to downstream consumers?
- How are budgets, cancellation, timeouts, and execution errors handled without corrupting game authority?

It should not answer rules, model, or training questions. The runner coordinates those systems through explicit package boundaries.

## Boundaries

`hexo-runner` sits above the rules engine and below user-facing tools, training pipelines, evaluation harnesses, and services that need games to be executed. It is a consumer of canonical game APIs, not a source of game truth.

The runner is responsible for:

- Creating and configuring game sessions from supplied player descriptors and execution options.
- Driving the turn loop until terminal state, cancellation, budget exhaustion, or a controlled failure.
- Asking the active player for an action through a uniform player interface.
- Passing chosen actions to the engine for validation and state transition.
- Capturing events, decisions, timing, errors, and terminal summaries.
- Supporting single games, batches, tournaments, evaluation sets, and self-play workloads.
- Applying execution policy such as per-move time limits, total game limits, retry behavior, and abort semantics.
- Integrating with model, utility, telemetry, persistence, and replay packages through stable interfaces.

The runner is not responsible for:

- Game rules authority, move legality, scoring, terminal detection, or state transition semantics.
- Model architecture internals, tensor layouts, checkpoint formats, or inference implementation details.
- Training losses, optimization, gradient computation, replay sampling policy, or dataset construction logic.
- Model-specific candidate selection, search policy, tree expansion, rollout logic, or action ranking internals.
- UI rendering or direct human input devices beyond delegating to a player adapter.
- Long-term storage schema ownership beyond emitting replay and event payloads to configured sinks.

## Player Orchestration

All participants should be represented through a common player contract. The runner should treat model players, human players, scripted players, and remote players uniformly: each receives a decision request and returns either a legal action response, a controlled refusal/error, or a timeout/cancellation result.

The player abstraction should describe capabilities without exposing implementation details. A model-backed player may call inference and search services; a human player may wait on an interactive frontend; a scripted player may read from a deterministic policy; a remote player may make a network request. From the runner's perspective, these are all decision providers with the same lifecycle:

1. Initialize for a session.
2. Receive game context and turn-local decision context.
3. Return an action outcome within the active budget.
4. Observe accepted moves and terminal summaries.
5. Close or release resources.

The runner should support heterogeneous matches where each seat can use a different player type. It should also support symmetric self-play where multiple seats share a model family, checkpoint, policy configuration, or remote service while still appearing as distinct participants in replay metadata.

Player identity should be separate from player implementation. A participant can have a display name, seed, team/seat assignment, version metadata, and evaluation labels without requiring the runner to know how that participant chooses actions.

## Game Loop

The game loop should be a small, explicit orchestration loop around the engine:

- Start from an engine-provided initial state or scenario state.
- Query the engine for the active player, legal action context, observations, and terminal status.
- Build a decision request for the active player.
- Enforce the configured time, step, and cancellation budgets while awaiting the decision.
- Submit the chosen action to the engine.
- Record emitted state transitions, decision metadata, timing, and any warnings or errors.
- Repeat until the engine reports a terminal result or execution policy stops the game.

The loop should not duplicate or second-guess rules logic. If a player returns an illegal action, the runner should handle it as an execution policy problem and use the engine's legality result as the source of truth. Possible policies include immediate forfeit, rejected move with structured error, configured fallback player, or controlled abort, but the legality decision itself belongs to the engine.

The runner should make determinism an explicit option. Seeds, player versions, model references, scenario inputs, and execution policy should be captured so replayed or audited runs can explain how a result was produced.

## Execution Modes

`hexo-runner` should provide shared orchestration primitives that can be composed into different run modes without creating separate rule paths.

Self-play execution should support generating games from one or more policy configurations, including mirrored or randomized seat assignments, seed schedules, and replay emission suitable for downstream training consumers. Self-play should still use the same player interface as every other mode.

Evaluation execution should support fixed model or player comparisons across scenario sets, seed ranges, opponent pools, and budget profiles. It should produce aggregate result records as well as per-game replay/event data.

Match execution should support direct player-versus-player games, including model-versus-model, human-versus-model, scripted-versus-model, remote-versus-local, and mixed participant configurations. It should expose enough metadata for match services or CLIs to display progress and outcomes.

Batch and tournament execution should coordinate many independent games, apply concurrency limits, aggregate results, and preserve per-game isolation. A failure in one game should be reported clearly without silently contaminating unrelated games.

Interactive execution should be possible through human or remote player adapters, but the runner should remain headless. User interfaces should consume runner events and submit decisions through adapters rather than embedding presentation logic inside the runner.

## Replay And Event Emission

The runner should emit structured events for observability, replay, debugging, and downstream data pipelines. Events should describe what happened at the orchestration boundary without requiring consumers to inspect internal player implementations.

Useful event categories include:

- Session lifecycle: game started, player initialized, game completed, game aborted.
- Turn lifecycle: turn started, decision requested, decision received, action accepted or rejected.
- Timing and budget: elapsed decision time, timeout, cancellation, queue delay, batch progress.
- Engine transition: action submitted, transition accepted, terminal result observed.
- Player diagnostics: controlled errors, unavailable player, remote failure, model service failure.
- Replay payloads: initial state reference, action sequence, observations or compact state references, terminal summary.

Replay output should be explicit about provenance: engine version, rules contract version, runner version, execution mode, player identities, seeds, scenario identifiers, and budget policy. The runner may format or route replay payloads, but it should not own semantic interpretation of game states beyond the contracts provided by the engine and replay packages.

Events should be suitable for multiple sinks, such as in-memory listeners, log streams, files, telemetry systems, training queues, or evaluation dashboards. Sinks should be pluggable so runner logic does not depend on a specific storage or observability backend.

## Time And Budget Management

The runner owns execution budgets because they govern orchestration rather than game rules. Budgets should be explicit, observable, and consistently enforced across player types.

Budget types should include:

- Per-decision time limits.
- Per-game wall-clock limits.
- Batch or tournament limits.
- Maximum turn counts or policy-defined stop conditions.
- Concurrency limits for local workers, model services, and remote players.
- Cancellation tokens for user-initiated or service-initiated shutdown.

Timeouts and cancellations should produce structured outcomes. The runner should record which budget was exceeded, which participant was active, what decision context was pending, and what policy was applied. It should not hide timeouts as ordinary moves or let late player responses mutate already-advanced games.

Budget enforcement should be independent of player implementation. A model player, human player, scripted player, and remote player should all experience the same orchestration contract, even if their internal latency profiles differ.

## Package Integration

`hexo-runner` should integrate with engine, utility, and model packages through narrow contracts.

The engine package should provide canonical game state, legal action context, transition application, terminal detection, scoring, and any compact state or replay primitives that are rules-owned. The runner should depend on these APIs and should not duplicate them.

Utility packages should provide shared concerns such as configuration loading, structured logging, seeding, serialization helpers, telemetry adapters, worker pools, and error/result types. The runner may compose these utilities but should avoid becoming a miscellaneous utility package itself.

Model packages should provide player adapters, inference clients, policy wrappers, or search providers that satisfy the common player contract. The runner should not import model architecture internals or know whether a model uses policy logits, value heads, candidate filters, MCTS, heuristics, or remote inference. Those details belong behind the player implementation.

Training packages should consume runner output rather than being embedded in the runner. The runner can produce self-play games, events, replay records, and model-supplied decision diagnostics, but model packages should decide how those records become training samples. Shared replay loaders, samplers, queues, and batch schedulers belong in `hexo-utils`; loss computation, target construction, optimization, curriculum logic, and checkpoint management belong to the model package.

The runner should therefore distinguish between:

- game/replay facts it owns, such as action history, players, seeds, timings,
  terminal result, and accepted engine transitions;
- model-attached diagnostics it transports, such as search traces, candidate
  support, policy targets, value estimates, or architecture-specific metadata;
- model-owned training datasets, which are constructed by the model package
  either during self-play or later during replay sampling.

## Design Principles

`hexo-runner` should be deterministic when configured, observable by default, and strict about ownership boundaries. It should make game execution easy to reproduce, inspect, and compare while keeping rules authority in the engine and decision intelligence inside player implementations.

The most important abstraction is not "model versus human" or "local versus remote." It is "participant asked for a decision under a contract." Once every participant flows through that contract, the same runner can support training self-play, benchmark evaluation, human matches, scripted tests, and service-hosted games without creating parallel orchestration paths.
