# Phase 02 Adversarial Review

## Findings And Resolutions

- Finding: `PairActionTableBuilder` initially treated raw first-action pair target mass as canonical unordered row mass, which made first-policy targets depend on sort order. Resolution: `PairActionTable.first_policy_target` now preserves raw first-action target projection and graph projection consumes it.
- Finding: graph truncation failures initially surfaced as generic pair cap errors. Resolution: `GraphSemanticBuilder` raises explicit `pair rows would be truncated` errors before calling the pair builder.
- Finding: second-placement replay records can contain first-placement-style pair targets from older product targets. Resolution: the pair builder accounts for that mass as missing when both target actions are legal but do not match `known_first`; true invalid known-first mismatches still fail.
- Finding: `action_contract/candidates.py` would have remained a runtime compatibility facade. Resolution: the file was deleted; tests use contract builders directly.

## Residual Risk

Full repository pytest was not run because Phase 02 evidence targets the owned candidate, pair, graph, and replay/training surfaces. Focused high-risk suites passed: Phase 02 contract tests, full graph contract tests, and full training data pipeline tests.

No skipped, xfailed, manual-only, or flaky-only check is claimed as phase-closing evidence.

