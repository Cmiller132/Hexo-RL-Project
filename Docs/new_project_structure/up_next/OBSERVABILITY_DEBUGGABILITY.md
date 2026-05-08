# Observability And Debuggability

## Purpose

Splitting the project must not make failures harder to trace. The current
system already has runtime probes, replay debugging, search traces, and
performance profiles; the new structure should keep those signals consistent
across packages.

## Proposal

Define shared event and metric shapes for:

- game lifecycle and runner decisions;
- illegal action rejections and engine validation errors;
- player latency, timeout, and cancellation outcomes;
- queue depth, batch fill, GPU wait time, and CPU worker utilization;
- replay ingestion and schema rejection counts;
- model decision diagnostics and search summaries;
- training sample construction time and loss input validation.

`hexo-utils` should provide lightweight counters, timers, structured event
helpers, and report fragments. Packages should emit their own domain-specific
diagnostics through those helpers.

## Simplification Guardrails

Avoid a large telemetry framework at first. A consistent event shape, stable
metric names, and a few useful summary reports are enough.

Do not make telemetry control runtime decisions except for explicit budgets,
backpressure, and failure policy.
