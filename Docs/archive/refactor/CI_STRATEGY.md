# CI Strategy

Date: 2026-04-29

CI must be deep enough to enforce the refactor and fast enough that developers still use it. The solution is tiered CI with explicit ownership, artifacts, timeouts, and promotion rules.

## CI Tiers

### Local Developer Gate

Fast commands used before handing work to review:

- formatting and lint checks for touched languages
- focused unit tests for changed packages
- focused import and architecture policy audits
- changed-area malformed-input tests
- quick no-hang smoke for touched runtime boundaries

### PR Required Gate

Every merge must pass deterministic checks that protect architecture and runtime safety:

- Rust fmt, workspace tests, release fast tests, and clippy
- Python contract, engine, model-registry, inference, search, replay, train, eval, dashboard, and tuning fast tests for touched areas
- Maturin extension build when Python/Rust boundary code changes
- dashboard build when dashboard/frontend code changes
- tuning recipe dry-run
- self-play no-pair default smoke
- inference protocol mismatch fail-fast smoke
- architecture policy audits: banned imports, direct `_engine` imports, direct Rust MCTS calls, architecture-string gates, pair-strategy bypasses, duplicate FFI decoders, legacy replay decode, old model/buffer/action-contract runtime imports
- manifest validation for phase artifacts and matrix rows touched by the PR

PR CI may use representative test shards, but it cannot skip a phase-closing invariant unless a deterministic replacement test covers that invariant in the PR gate.

### Merge Or Deep Gate

Run after merge or before phase closure when broad coverage is needed:

- broader Python test suites
- Rust release tests
- dashboard build and route smoke
- Maturin rebuild plus Python engine smoke
- mutation/corruption suites
- behavior debug-bundle generation
- Rust suspicion bundle
- self-play -> replay -> train -> eval smoke
- architecture policy audits over the full repo

### Scheduled Deep Gate

Run on a scheduled cadence and before final V2 closure:

- ignored Rust oracle tests
- fuzz/property tests
- full pytest suite
- long self-play smoke
- replay data-quality checks
- benchmark comparison for Rust, inference, MCTS, replay, training, and self-play
- GPU batching/utilization runs on configured GPU runners
- dashboard/autotune report checks

### Final V2 Closure Gate

Phase 09 cannot close until:

- all PR-required jobs are green on the final SHA
- latest scheduled/deep gate is green for the final SHA or rerun manually on the final SHA
- final end-to-end smoke and behavior debug bundle artifacts are archived
- public API drift, import graph, panic/unwrap inventory, and performance comparison artifacts are attached
- every V2 row is closed with implementation, test, CI, deletion/import, telemetry/debug, documentation, and performance evidence where relevant

## Required CI Metadata

Every required check must declare:

- tier: `local`, `pr_required`, `deep`, `scheduled`, or `final`
- owning phase and V2 row ids
- owner or owning subagent
- timeout
- command
- artifact paths
- runner profile requirements
- promotion rule for moving between tiers
- failure triage owner

## Artifact Retention

Recommended retention:

- PR artifacts: 14 days
- deep/nightly artifacts: 45 to 90 days
- phase exit artifacts: copied into `Docs/refactor/artifacts/phase_XX/`
- final V2 artifacts: retained with the release/conformance bundle

Phase artifact manifests must record the CI run id, git SHA, runner profile, command, config hash, and supersession note when a later artifact replaces an earlier one.

## Flaky Policy

Required gates may not silently skip flaky tests.

Allowed responses:

- fix the flake
- quarantine with owner, issue, expiry date, affected V2 rows, and continued scheduled execution
- replace the PR gate with a deterministic minimal invariant test and move broader stochastic coverage to scheduled CI

Retries are allowed only for infrastructure failures, and first-failure logs must remain attached.

No flaky or quarantined test may be the only proof for a phase-closing invariant.

## Performance CI Policy

Performance is enforced in stages:

- PR CI blocks obvious algorithmic regressions such as accidental pair scoring, uncapped pair enumeration, missing caps, unbounded waits, and invalid row counts.
- Deep CI records machine-normalized benchmark JSON.
- Scheduled CI compares against checked-in baselines once runner metadata and comparison tooling are stable.
- Final V2 closure requires benchmark artifacts, threshold ownership, and documented handling for accepted regressions.

Hard performance gates should not depend on noisy developer laptops. They belong on configured runners with stable host metadata.
