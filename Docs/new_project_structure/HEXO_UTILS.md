# hexo-utils

## Purpose

`hexo-utils` is the shared utility project for reusable mechanisms that are useful across Hexo model packages, training pipelines, evaluation tools, and Rust-facing test infrastructure. It replaces the narrower `hexo-mcts` concept with a broader home for common algorithms, data-handling helpers, instrumentation support, and contract utilities that do not belong to one model package.

The component should make repeated engineering patterns easier to reuse without becoming a second rules engine, model policy layer, or hidden runtime authority. It provides well-defined helpers that consumers can opt into when their semantics match the helper contract.

## Boundaries

`hexo-utils` owns generic mechanisms and helper contracts. It does not own game rules, legal move authority, tactical truth, reward definitions, policy choices, value calibration, or model-specific search behavior.

Authoritative game semantics remain at the Rust rules boundary and in the explicit contracts exposed from that boundary. Model packages remain responsible for their own policy decisions, architecture-specific behavior, and any specialized algorithms whose assumptions do not fit a shared utility.

Utilities should be deterministic where their contracts require determinism, explicit about schema and version expectations, and narrow enough that importing them does not implicitly change a model package's meaning.

## Shared Utility Areas

### Generic MCTS Utilities

`hexo-utils` can provide reusable Monte Carlo tree search mechanisms such as node bookkeeping, selection helpers, expansion scaffolding, visit and value accumulation, rollout coordination hooks, and common tree traversal patterns.

These helpers must stay generic. They may support search plumbing, but they must not encode Hex game authority, model-specific priors, hard-coded tactical policy, or assumptions that only one model generation can satisfy. Search consumers should supply legal action sources, evaluator calls, value interpretation, policy masking, and domain-specific stopping behavior through explicit inputs or adapters.

### Tree and Statistics Helpers

The project can centralize small, reusable helpers for tree inspection, aggregation, score normalization, visit distributions, pruning summaries, debugging snapshots, and statistical summaries used by search, evaluation, dashboards, and test reports.

These helpers should describe observed state and derived metrics. They should not decide which moves are legal, which move is strategically correct, or how a model should train against those measurements.

### Batching Helpers

`hexo-utils` can include batching and queue helpers for grouping model evaluations, replay transforms, simulation requests, and Rust bridge calls where consumers benefit from shared flow-control patterns.

These helpers may define common behavior around batch sizing, ordering, timeout handling, backpressure visibility, and result correlation. They should not choose model architecture, inference backend, tactical policy, or game-specific legality semantics for the caller.

### Training Adapter Framework

`hexo-utils` should define the shared framework that model packages plug into for replay ingestion, sampling, collation, batched inference, and training loops. This framework should behave like a small set of overridable contracts: each model package supplies adapters for model-specific encoding, target construction, batch collation, loss calculation, inference decoding, and checkpoint semantics.

The framework owns reusable mechanics such as replay indexing, sampling flow, CPU worker pools, pinned-memory staging, GPU batch scheduling, queue backpressure, deterministic seeding, schema checks, and telemetry. Model packages own the meaning of their examples and batches.

The intended data flow is:

1. Runner emits compact game records, decision events, search diagnostics, and terminal summaries.
2. A model adapter converts those records into model-specific replay examples when appropriate.
3. Shared replay and loader utilities sample, shard, shuffle, prefetch, and batch those examples.
4. The model adapter collates examples into tensors or graph batches and computes the model-specific loss.

This keeps training infrastructure reusable without forcing Dense CNN, ResNet, Global Graph, V1, or future models to share tensor layouts, targets, replay filtering rules, or search metadata semantics.

### Resource Profiles And Queue Limits

`hexo-utils` should provide shared resource-profile contracts rather than a
general CPU job scheduler. These contracts should describe host capabilities,
Torch thread settings, Rust thread settings, DataLoader worker counts, inference
batch limits, prefetch depth, queue limits, and memory guardrails.

Consumers should use standard execution tools that fit their workload:
Rust/Rayon for branchy game and search work, PyTorch DataLoader workers for
training replay pipelines, multiprocessing for self-play workers, and small
thread/process pools for background artifacts.

The utility layer centralizes limits, defaults, and telemetry names. It should
not own the work itself or route every CPU task through one generic queue.

### Replay Serialization Utilities

Shared replay utilities can cover stable serialization and deserialization support for replay records, compact payloads, round-trip validation, schema tags, compatibility checks, and inspection helpers.

The utilities should preserve and validate data according to declared contracts. They should not reinterpret rewards, rewrite action meaning, or silently adapt records across incompatible model generations.

### Metrics and Telemetry Helpers

The component can provide lightweight helpers for counters, timers, structured events, debug traces, utilization summaries, and common report fragments used across training, search, evaluation, and Rust bridge validation.

Telemetry utilities should make behavior observable without making control decisions. They can standardize names and payload shape where useful, but model packages and runtime owners remain responsible for deciding which metrics are required for their workflows and acceptance criteria.

### Rust Game Mutator and Test Harness Helpers

`hexo-utils` can host reusable Rust-side and Python-side helpers that exercise Rust-facing contracts in tests and search workloads, including state construction helpers, fast game mutators, mutator harnesses, negative-case builders, round-trip assertions, legal-row validation scaffolds, and compact-history inspection aids.

These helpers support verification and high-throughput exploration of the Rust
boundary. They do not replace the authoritative engine rules and should not
introduce alternate legal move logic that can drift from engine behavior.

Rust utilities are appropriate when Python would be too slow or too awkward for
the job, such as MCTS state mutation, cloning, child expansion, compact replay
mutation, search fixture construction, or stress tests that need to explore many
states quickly. The important boundary is authority, not language: shared Rust
helpers may be fast and reusable, but legality and terminal truth still come
from the engine contract.

### Schema and Version Helpers

The project can centralize utilities for schema identifiers, version comparison, compatibility gates, payload validation, contract examples, and migration-safe readers where shared structure is beneficial.

Version helpers should make compatibility explicit. They should fail clearly when a consumer requests an incompatible schema rather than silently falling back to legacy behavior or changing semantics in place.

### Non-Authoritative Tactical and Analysis Helpers

`hexo-utils` may include tactical or analysis helpers when they are explicitly non-authoritative, such as explainability aids, heuristic annotations, debug classifiers, visualization summaries, or test fixture generators.

These helpers can assist humans and tests, but they must not become the source of truth for legality, winning status, forced tactics, reward assignment, or model policy. Any tactical helper with game-facing implications must be clearly framed as advisory, derived, or fixture-oriented.

## Consumer Model

Model packages can import utilities from `hexo-utils` when the utility contract matches the package's assumptions. Shared helpers are appropriate when they reduce duplicated mechanics without weakening a model package's explicit ownership of its behavior.

Model packages can and should keep custom implementations when their semantics diverge from the shared contract. For example, V1 pair search may retain specialized logic if its pair-action assumptions, scoring flow, or compatibility requirements do not match a generic MCTS helper. In that case, `hexo-utils` can still provide lower-level support such as statistics formatting, serialization helpers, telemetry wrappers, or test harness utilities without absorbing the custom algorithm.

The import direction should remain simple: model packages depend on shared utilities, while `hexo-utils` does not depend on model packages. Shared utilities should avoid package-specific callbacks unless they are expressed as stable interfaces that multiple consumers can implement.

## Design Principles

- Reuse mechanisms, not authority.
- Keep contracts explicit and version-aware.
- Prefer deterministic helpers for tests, serialization, and replay handling.
- Keep policy, reward, and architecture choices inside model packages.
- Validate Rust-facing behavior against Rust-derived contracts instead of reimplementing rules.
- Make observability reusable without turning metrics into hidden control flow.
- Allow specialized model implementations to coexist with shared lower-level helpers.
- Keep the project high-cohesion by accepting only utilities with plausible cross-package reuse.

## Non-Goals

`hexo-utils` is not a rules engine, a replacement for Rust game logic, a universal MCTS implementation that every model must use, a model policy package, a model-specific trainer, or a compatibility layer for preserving obsolete runtime paths.

It should not become a dumping ground for code that merely lacks a current owner. Utilities belong here only when their contracts are stable enough to share and their behavior is independent of model-specific authority.
