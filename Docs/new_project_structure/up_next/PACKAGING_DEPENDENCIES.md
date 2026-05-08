# Packaging And Dependencies

## Purpose

The split should let a simple runner or engine test run without installing
every model dependency. Dense board models, graph models, V1 search, dashboards,
and Rust tooling can have different needs.

## Proposal

Keep dependency direction simple:

- `hexo-engine` stays dependency-light and Rust/rules focused;
- `hexo-utils` depends only on broadly shared runtime, replay, telemetry, and
  adapter support;
- `hexo-runner` depends on engine contracts and player interfaces, then loads
  configured players;
- each `hexo-model-*` package owns its architecture-specific dependencies.

Optional features or extras can group common environments such as training,
evaluation, dashboards, graph models, or V1 pair search.

## Simplification Guardrails

Do not physically split packages before contracts are stable enough to test.
Start with logical boundaries and import discipline.

Avoid making `hexo-utils` depend on heavyweight model libraries unless they are
only imported behind optional training or GPU paths.
