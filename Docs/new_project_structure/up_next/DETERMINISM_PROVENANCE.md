# Determinism And Provenance

## Purpose

The split should make it easy to answer why a run produced a result. The
current project already relies on seeds, runtime settings, inference behavior,
and replay metadata; these should become first-class run facts.

## Proposal

The runner should record a compact provenance block for each run:

- engine and runner versions;
- model package name, version, checkpoint reference, and player config;
- seed schedule and scenario set;
- resource profile, worker counts, inference precision, and batch policy;
- schema versions for replay, engine payloads, and model diagnostics;
- search or decision budgets;
- important model-owned training or target settings when replay is generated.

Model packages can add a model-specific provenance block, but the runner should
preserve it as metadata rather than interpreting it.

## Simplification Guardrails

Make deterministic execution an explicit mode, not a promise for every
performance run. Capture enough information to audit non-deterministic runs
without forcing every fast path into strict determinism.

Prefer a small, structured provenance record over storing full configs in many
places.
