# Phase 02 CI Routing Plan

## Tier 0

- Python compile checks for changed modules.
- Contract unit tests for candidate, pair, graph semantic, graph tensor projection, mutation safety, and corruption failures.
- Import-purity and banned-private-builder audits.

## Tier 1

- Focused runtime tests for replay sampler, self-play preparation helpers, dashboard fixtures, and graph model-input projection.
- Golden parity tests proving self-play, replay, training, evaluation debug, and dashboard fixtures consume the same builders.

## Tier 2

- Existing graph contract and training data pipeline tests touched by Phase 02.
- Performance smoke for candidate builder, pair builder with caps, graph semantic builder, graph tensorizer, and graph collator.

## Non-Closing Checks

Long-running full-suite or inference-server checks may be attempted and recorded, but Phase 02 cannot rely on skipped, flaky-only, manual-only, or timed-out checks for a hard exit gate.

