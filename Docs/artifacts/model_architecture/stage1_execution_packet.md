# Stage 1 Execution Packet

## Goal

Create the exact implementation blueprint and proof harness needed to replace
the fragmented model architecture logic with a contract-first authority under
`Python/src/hexorl/models/`, without preserving accidental legacy behavior.

## Success Criteria Checklist

- [x] Select `hexorl/models/` as the new architecture authority.
- [x] Inventory current architecture, head/loss, target, inference, and runtime
  behavior.
- [x] Inventory scattered architecture authority with exact duplicated runtime
  source locations and mismatched global graph id sets.
- [x] Attach a keep/replace/simplify/delete/move-behind-contract decision to
  every inventory row.
- [x] Classify existing model, graph, replay, training, inference, and self-play
  tests as `golden`, `rewrite`, or `delete` at per-test granularity.
- [x] Design row table, output, target, inference, and pair-strategy contracts
  before moving model code.
- [x] Identify golden rules that must survive the rewrite and the replacement
  tests needed for currently missed bug classes.
- [x] Classify silent loss skips and fallback aliases for removal or explicit
  retention.
- [x] Represent shared memory as a transport constraint, not as output
  semantics.
- [x] Specify pair strategies as executable runtime plans.
- [x] Decide pair target ordering, duplicate rows, zero target mass, missing
  weights/phases, and lookahead fallback behavior.
- [x] Lock config override behavior: specs own defaults; config may enable or
  disable supported optional heads; self-play policy and value outputs cannot
  be disabled.
- [x] Lock dynamic `lookahead_*` expansion from configured horizons.

## Constraints Applied

- No runtime wrappers were added.
- No second runtime path was introduced.
- No importable runtime modules were created in Stage 1.
- No model internals were physically split.
- Architecture prefixes are classified as legacy behavior and rejected as the
  future behavior mechanism.
- Contracts are limited to replay, rows, targets, outputs, losses, inference,
  and search boundaries.

## Required Evidence Files

- `Docs/artifacts/model_architecture/architecture_inventory.md`
- `Docs/artifacts/model_architecture/head_loss_inventory.md`
- `Docs/artifacts/model_architecture/target_inventory.md`
- `Docs/artifacts/model_architecture/inference_inventory.md`
- `Docs/artifacts/model_architecture/runtime_inventory.md`
- `Docs/artifacts/model_architecture/test_trust_audit.md`
- `Docs/artifacts/model_architecture/baseline_command_report.md`
- `Docs/artifacts/model_architecture/contract_design.md`
- `Docs/artifacts/model_architecture/design_examples.md`
- `Docs/artifacts/model_architecture/golden_test_plan.md`

## Local Refresh

The Stage 1 plan and artifacts were fast-forwarded from `origin/main` on
2026-05-06, then the local autotune/runtime work was reapplied and conflicts
were resolved. `Docs/artifacts/model_architecture/baseline_command_report.md`
was refreshed locally on this PC. The local harness has the Rust `_engine`
extension available and the focused model, graph, replay, inference, tactical,
engine, and config checks pass.

## Review Finding Closure

- The architecture inventory records exact duplicated authority sources in
  `global_graph.py`, config validation/default mutation, replay feature flags,
  epoch/process prefix gates, dashboard summaries, runtime estimates, and tests.
- The exact current global graph mismatch is recorded: `global_hybrid_action_0`
  and `global_graph768_champion` exist in model/config authority but are absent
  from `buffer/ring.py` replay feature authority.
- The test trust audit now classifies individual model, graph, replay,
  training, inference, smoke, engine, tactical, and self-play tests or named
  test groups, with rewrite/delete treatment tied to Stage 2-4 replacement
  tests.

## Stop Rule Results

No Stage 1 stop rule blocks Stage 2.

- Current trained heads can be mapped to target, mask, loss, and phase, with
  explicit removal decisions for silent skips and fallback aliases.
- Runtime-consumed policy and pair outputs can be tied to row tables.
- Shared-memory transport can carry required row identity by adding row hashes
  and protocol versions to metadata. This is a schema change for Stage 4.
- Pair strategy behavior can be separated from architecture capability.
- Self-play policy and value capabilities can be represented through resolved
  specs rather than model-class checks.

## Explicit Completeness Statement

Stage 1 closes design and inventory scope only. No skipped, deferred, flaky, or
manual-only implementation requirement is claimed complete for Stage 2, Stage 3,
or Stage 4.
