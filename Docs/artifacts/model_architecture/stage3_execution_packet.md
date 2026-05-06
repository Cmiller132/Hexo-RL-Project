# Stage 3 Execution Packet

## Goal

Move replay projection, target construction, training adapters, and loss
computation behind explicit contracts so trainable heads cannot silently bind
to the wrong target, mask, weight, phase, or row table.

## Success Criteria Checklist

- [x] Trainer uses adapter/loss-plan boundaries instead of raw head-name loss
  routing.
- [x] Dense, sparse, graph hybrid, and global graph batches train through the
  public `Trainer._train_step` flow and reach loss computation through
  prepared training batches.
- [x] Target construction carries row-table instances for dense board,
  candidate, legal, opponent legal, pair joint, and known-first pair rows.
- [x] Trainable heads fail loudly when required targets, masks, weights, or
  phases are missing.
- [x] Silent loss skips and fallback aliases are deleted for trainable heads.
- [x] Prepared graph batches carry explicit per-sample phase metadata:
  `placements_remaining`, `pair_first_unordered`,
  `pair_second_known_first`, and `pair_second_row_mask`.
- [x] `policy_pair_second` is gated by explicit known-first phase metadata and
  rejects positive target mass outside that phase.
- [x] Lookahead trainable heads require exact `lookahead_{horizon}` targets and
  no longer fall back to value targets.
- [x] Global graph training uses `policy_target`/`legal_mask` row contracts and
  cannot consume dense `policy` fields.
- [x] Architecture-specific target namespaces prevent dense, sparse, graph, and
  pair fields from being consumed accidentally.

## Runtime Consumers Changed

- `Python/src/hexorl/train/loss_plan.py` defines target contracts, loss-plan
  entries, row-table inference, target/mask/weight/phase validation,
  positive-mass validation, duplicate-active-row validation, and loss handler
  dispatch.
- `Python/src/hexorl/train/adapters.py` defines prepared dense and global graph
  training batches, including model inputs, target tensors, inferred row
  tables, graph phase metadata, and required sample weights. Adapter target
  assembly now rejects auxiliary targets that would overwrite prepared
  architecture targets.
- `Python/src/hexorl/train/losses.py` retains primitive loss functions only;
  `compute_losses` requires an explicit `LossPlan` and delegates all routing
  through that validated plan.
- `Python/src/hexorl/train/trainer.py` builds one resolved loss plan from the
  architecture spec and routes dense/global graph training through adapters.
- `Python/src/hexorl/buffer/sampler.py` now errors when configured lookahead
  targets are missing, masks crop/graph pair-head training when no positive
  pair target mass is represented, and emits graph target/phase metadata when
  graph batches are materialized.
- `Python/src/hexorl/graph/batch.py` preserves per-sample
  `placements_remaining_by_sample` through graph collation.
- `Python/src/hexorl/models/specs.py` gives dense and global opponent policy
  outputs distinct row-family target namespaces.

## Legacy Paths Deleted Or Quarantined

- The broad raw head-name loss switch was removed from
  `Python/src/hexorl/train/losses.py`.
- Global graph target and phase tensor assembly was removed from
  `Trainer`; global graph training now uses `prepare_global_graph_training_batch`.
- Lookahead value fallback in replay sampling was removed.
- Synthetic pair target fallback remains forbidden; incomplete first-placement
  graph pair rows are masked out of pair-head training instead of generating
  product targets.

## Required Evidence

- `Python/tests/test_model_architecture_stage3.py`
- Existing integration coverage in:
  - `Python/tests/test_training_data_pipeline.py`
  - `Python/tests/test_global_graph_contract.py`
  - `Python/tests/test_config_and_guardrails.py`
  - `Python/tests/test_production_smoke.py`
- Audit record:
  `Docs/artifacts/model_architecture/stage3_loss_plan_audit.md`

## Performance Smoke

Command:

```powershell
$env:PYTHONPATH='Python/src'; @'
# Inline CPU smoke profile for dense/global adapters plus loss-plan compute.
'@ | python -
```

Result:

```text
dense_adapter_plus_loss_plan_cpu: mean=1.345 ms p95=1.823 ms loops=100
global_deferred_adapter_plus_loss_plan_cpu: mean=0.685 ms p95=0.952 ms loops=30
```

This smoke measures the Stage 3 contract path itself: batch adapter preparation,
row-table inference/validation, and loss-plan compute. It does not claim GPU
throughput for full model forward/backward; that remains covered by training
benchmarks and autotune runs.

## Verification

Commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\train\loss_plan.py Python\src\hexorl\train\adapters.py Python\src\hexorl\train\losses.py Python\src\hexorl\train\trainer.py Python\src\hexorl\graph\batch.py Python\src\hexorl\buffer\sampler.py Python\src\hexorl\models\specs.py Python\tests\test_model_architecture_stage3.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage3.py Python/tests/test_training_data_pipeline.py Python/tests/test_global_graph_contract.py Python/tests/test_config_and_guardrails.py::test_compute_losses_raises_when_required_contract_target_is_missing Python/tests/test_production_smoke.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests
$env:PYTHONPATH='Python/src'; @'
# Resolve every registered architecture/static-output combination and verify
# each trainable resolved output has a loss-plan entry.
'@ | python -
Get-ChildItem -Path Python\src\hexorl -Recurse -File -Include *.py | Select-String -Pattern 'from hexorl\.model[\s\.]','import hexorl\.model[\s\.]','\bGlobalHexGraphNet\b','\bHexNet\b' | Where-Object { $_.Path -notmatch '\\model\\' -and $_.Path -notmatch '\\models\\recipes\\' }
cargo test --workspace
git diff --check
```

Results:

```text
py_compile passed
focused Stage 3 trainer/loss suite: 138 passed, 1 warning
full Python suite: 305 passed, 1 warning
resolved output coverage audit: all valid resolved trainable outputs have loss-plan entries
runtime legacy import audit: no matches
cargo workspace suite: 179 non-ignored tests/docs passed; 6 slow oracle tests remain ignored by the Rust test definitions
git diff --check passed with CRLF warnings only
```

## Stop Rule Results

- Dense, sparse, graph hybrid, and global graph training batches are covered by
  the adapter/loss-plan trainer path.
- Runtime-consumed and trainable row-backed outputs identify row tables before
  loss computation.
- `compute_losses` cannot execute without an explicit architecture-resolved
  loss plan.
- `policy_pair_second` cannot train outside `pair_second_known_first=True`
  when positive target mass exists.
- Global graph policy heads require graph `policy_target`/`legal_mask` targets
  and reject dense `policy` fields.
- Adapter target assembly fails if an auxiliary namespace attempts to overwrite
  a prepared trainable target.

## Explicit Completeness Statement

Stage 3 closes the training and replay cutover scope. No skipped, deferred,
flaky, or manual-only Stage 3 requirement is claimed complete. Inference
protocol, search providers, pair strategies, and final `hexorl/model/`
deletion remain Stage 4 scope.
