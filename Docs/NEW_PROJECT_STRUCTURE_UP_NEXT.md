# New Project Structure Up Next

## Purpose

This document captures related design considerations that should be explored
after the initial project boundary, training/replay, CPU, GPU, and memory
planning passes.

The goal is not to add more structure for its own sake. The goal is to make
sure the modular split stays reproducible, debuggable, and practical once
models, runners, utilities, and engine code can evolve separately.

## Storage And I/O

Replay, checkpoints, scorecards, traces, and evaluation artifacts can become
major bottlenecks or sources of confusion. The structure should define compact
replay formats, append-friendly writes, checkpoint retention, artifact naming,
compression defaults, and when model-specific derived examples are cached
versus rebuilt.

This is especially important because model packages may own different replay
interpretations. Storage policy should prevent each model from producing large,
incompatible, hard-to-audit artifacts by default.

## Schema And Versioning

Every cross-package boundary should have explicit schema or contract versions:
engine state, action identity, legal rows, tactical payloads, runner events,
model replay examples, training adapters, and checkpoint metadata.

Versioning should fail clearly when records are incompatible. Silent fallback
or implicit adaptation across model generations should not be part of normal
runtime behavior.

## Determinism And Provenance

The runner should record enough provenance to reproduce or audit a run:
engine version, model package version, checkpoint hash, resource profile, seed
schedule, player config, inference precision, replay schema versions, search
budget, and model-owned filtering or target rules.

Model packages can contribute their own version/config block, but the runner
should preserve the full run-level record.

## Observability

The split should preserve cross-package debugging. Shared telemetry should cover
illegal action rejections, model decision latency, replay ingestion errors,
schema rejection counts, queue backpressure, GPU batch fill, sample construction
time, search summaries, and model-specific diagnostic payloads.

Performance telemetry is necessary, but correctness and data-quality telemetry
are just as important.

## Testing Boundaries

Each package should have clear test ownership:

- `hexo-engine`: rules, legality, tactical payloads, state identity, FFI.
- `hexo-runner`: game loop, player contract, budgets, cancellation, replay
  emission.
- `hexo-utils`: schemas, queues, batching, adapters, replay helpers, telemetry.
- `hexo-model-*`: encoding, targets, losses, inference adapters, checkpoints,
  and model-owned search.

Cross-package contract tests should be fewer and focused on composition rather
than duplicating every package's internal tests.

## Registry And Discovery

The runner and training tools need a simple way to discover model packages,
player factories, training adapters, checkpoint loaders, and evaluation hooks.

This should be explicit but lightweight. A small registry is enough if it keeps
the runner from hardcoding every model architecture.

## Failure And Backpressure Policy

The structure should define how common failures are surfaced and handled:
player timeout, unavailable inference service, full replay queue, schema
mismatch, checkpoint mismatch, GPU out-of-memory, DataLoader failure, Rust FFI
error, or model adapter rejection.

Failure policy belongs at the orchestration boundary, while the failing package
should provide precise structured errors.

## Packaging And Dependencies

Separate model projects should not force every environment to install every
optional dependency. Dense models, graph models, V1 search, dashboards, and
Rust extension tooling may have different dependency needs.

The package split should keep optional dependencies local to the packages that
need them.

## Evaluation Fairness

If models can own custom search, replay interpretation, or preprocessing,
evaluation needs strict fairness records: equal wall-clock or equal budget,
fixed openings, fixed opponent pools, fixed hardware/resource profile, inference
precision, search settings, and temperature/resignation rules.

Custom model behavior is allowed, but evaluation artifacts must make those
differences visible.

## API Ergonomics

The structure should still make it easy to add a new model. A reasonable path
should be:

1. Define the architecture.
2. Define state/input construction.
3. Define inference and training adapters.
4. Define a player factory.
5. Register the package.
6. Run self-play, evaluation, and training through shared tools.

If this path feels too heavy, model authors will bypass the structure.
