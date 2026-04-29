# Phase 05 — Search and Pair Strategy Isolation

## Purpose
Move pair-action consumption from implicit side effects to explicit search strategy policy.

## Target Modules
- `search/context.py`, `policy_provider.py`, `pair_strategy.py`, `priors.py`, `expansion.py`, `mcts_runner.py`, `engine_adapter.py`

## V2 Requirements
- Default pair strategy is `none` (including global graph families).
- Full pair enumeration allowed only with named strategy and hard caps.
- Pair scoring no longer implied by `pair_prior_mix`, `pair_head_present`, or architecture prefix.

## Parallel Subagent Work
- S1: `PairStrategySpec` schema + validation rules.
- S2: runtime context wiring for explicit strategy selection.
- S3: strategy implementations (none/top-k/two-stage/capped exhaustive diagnostic).
- S4: replay/targets plumbing for pair metadata.
- S5: regressions proving no implicit pair work.

## Mandatory Tests
- `strategy=none` must emit zero pair generation/scoring calls.
- Capped strategies enforce max pair rows and telemetry counters.
- Search priors tests ensure policy source traceability.
- MCTS integration tests validate strategy outputs consumed correctly.

## Exit Criteria
- Pair behavior entirely policy-driven and observable.
- No architecture/config side-effect toggles remain in search path.
