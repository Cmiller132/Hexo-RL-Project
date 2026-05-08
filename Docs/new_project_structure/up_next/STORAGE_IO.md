# Storage And I/O

## Purpose

Storage should make runs reproducible without making every model package invent
its own file layout. The current project already produces replay records,
training artifacts, scorecards, traces, and profiles; the split should preserve
that usefulness while making ownership clearer.

## Proposal

Use layered storage:

- engine-owned game facts: compact state, action history, terminal result, and
  rules contract metadata;
- runner-owned run facts: players, seeds, budgets, timings, outcomes, and
  execution errors;
- model-owned facts: training examples, search traces, targets, calibration
  outputs, and architecture diagnostics;
- shared artifacts: scorecards, performance profiles, audit summaries, and
  lightweight debug bundles.

`hexo-utils` should provide common readers, writers, naming helpers, retention
policy helpers, compression defaults, and round-trip validation. Model packages
should decide whether their derived training records are stored eagerly or
rebuilt lazily from runner replay.

## Simplification Guardrails

Start with append-friendly local files and clear directory conventions. Avoid a
database, distributed artifact service, or complex caching layer until local
storage becomes a proven bottleneck.

Prefer one canonical game record plus optional model-specific sidecars over
large duplicated replay files per model.
