# Worker H Completion Packet - V1 Pair-Action Training/Eval Guardrails

## Goal

Complete V1 replay target, training loss, smoke-test, eval/autotune, CI/audit, and evidence artifacts needed to run `global_pair_biaffine_0:sampled_joint_pair_v1` alongside current models.

## Closed V1 Rows

- V1 target support: closed for Python graph training. V1 target arrays now distinguish admitted, explicit-negative, forced, terminal-equivalent, and unsampled rows through support type ids, explicit masks, and schema versioned target dictionaries.
- V1 pair posterior/loss wiring: closed for `global_pair_biaffine_0:sampled_joint_pair_v1`. The graph loss plan trains pruned pair posterior, softened pair targets, completed-Q regularization, unordered conditionals, explicit sampled-negative/ranking targets, tactical labels, and value.
- Unsampled legal pair safety: closed for deterministic target/loss gates. Unsampled legal rows are excluded from pair policy/ranking/Q/negative masks and rejected if they overlap supervised masks.
- Terminal-equivalent policy behavior: closed for the implemented target builder. Terminal-equivalent filler pairs are represented with forced terminal support and excluded from ordinary sampled-pair policy masks.
- Eval/autotune/scorecard schema gates: closed for deterministic schema and metadata enforcement. Actual equal-wall-clock strength evidence is not claimed.
- CI/audit gates: closed through focused tests and audit commands for banned V1 projection use, threat-filtered LEGAL activation, missing candidate metadata, unsampled-negative treatment, head-name-only activation, and current-model recipe stability.

## Runtime Consumers Changed

- `Python/src/hexorl/buffer/sampler.py`: builds and collates V1 pair targets from replay metadata, transforms pair keys under symmetry, attaches admitted graph pair rows, and masks legacy pair policy weights for V1 batches.
- `Python/src/hexorl/replay/training_batch.py`: passes graph `pair_features` through model inputs and target dictionaries.
- `Python/src/hexorl/train/loss_plan.py`: adds V1 target support validation and V1-specific loss handlers.
- `Python/src/hexorl/train/losses.py`: adds completed-Q, ranking, conditional, and masked BCE helpers used by the V1 loss plan.
- `Python/src/hexorl/train/trainer.py`: exposes V1 pair-row loader timing evidence in step results.
- `Python/src/hexorl/autotune/recipes.py`: adds V1 scorecard/baseline/metric metadata for `global_pair_biaffine_0:sampled_joint_pair_v1`.

## Files Changed

- `Python/src/hexorl/train/v1_pair_targets.py`
- `Python/src/hexorl/train/losses.py`
- `Python/src/hexorl/train/loss_plan.py`
- `Python/src/hexorl/replay/training_batch.py`
- `Python/src/hexorl/buffer/sampler.py`
- `Python/src/hexorl/train/trainer.py`
- `Python/src/hexorl/eval/v1_pair_scorecard.py`
- `Python/src/hexorl/autotune/recipes.py`
- `Python/tests/test_v1_pair_targets.py`
- `Python/tests/test_v1_pair_training_losses.py`
- `Python/tests/test_v1_pair_eval_guardrails.py`
- `Python/tests/test_v1_pair_ci_audit_gates.py`
- `Docs/refactor/artifacts/v1_pair_action/v1_pair_scorecard_schema_gate.json`
- `Docs/refactor/artifacts/v1_pair_action/worker_h_completion_packet.md`

## Legacy Paths Deleted Or Quarantined

- No files were deleted.
- V1 graph training quarantines legacy single-action/pair-projection behavior by requiring `v1_pair_schema_version`, explicit support masks, and V1 recipe strategy metadata before V1 losses activate.
- V1 runtime training/eval/autotune paths do not use `pair_logits_to_action_logits`.

## Tests And Commands Run

- `python -m pytest Python\tests\test_v1_pair_targets.py -q` -> exit 0.
- `python -m pytest Python\tests\test_v1_pair_training_losses.py -q` -> exit 0; 2 passed, 1 Triton/CUDA availability warning.
- `python -m pytest Python\tests\test_v1_pair_eval_guardrails.py -q` -> exit 0; 3 passed.
- `python -m pytest Python\tests\test_v1_pair_ci_audit_gates.py -q` -> exit 0; 4 passed.
- `python -m pytest Python\tests\test_v1_pair_targets.py Python\tests\test_v1_pair_training_losses.py Python\tests\test_v1_pair_eval_guardrails.py Python\tests\test_v1_pair_ci_audit_gates.py Python\tests\test_v1_pair_biaffine_model.py Python\tests\test_v1_pair_action_baselines.py Python\tests\test_optuna_config_surface.py -q` -> exit 0; 32 passed, 1 Triton/CUDA availability warning.
- `python -m pytest Python\tests\test_training_data_pipeline.py -q` -> exit 0; 93 passed, 1 Triton/CUDA availability warning.
- `python -m pytest Python\tests\test_global_graph_contract.py -q` -> exit 0; 50 passed, 1 skipped, 1 Triton/CUDA availability warning.
- `python -m py_compile Python\src\hexorl\train\v1_pair_targets.py Python\src\hexorl\train\loss_plan.py Python\src\hexorl\train\losses.py Python\src\hexorl\replay\training_batch.py Python\src\hexorl\buffer\sampler.py Python\src\hexorl\train\trainer.py Python\src\hexorl\eval\v1_pair_scorecard.py Python\src\hexorl\autotune\recipes.py` -> exit 0.
- `python -m json.tool Docs\refactor\artifacts\v1_pair_action\v1_pair_scorecard_schema_gate.json > $null` -> exit 0.
- Runtime V1 training/eval/autotune audit for `pair_logits_to_action_logits` -> exit 0; no references found.
- `sampled_joint_pair_v1` threat-filtered LEGAL activation audit -> exit 0; no activation references outside existing hard-fail guards.
- Unsampled-negative audit in V1 target builder -> exit 0; target builder excludes unsampled rows from negative/ranking support and has hard guards.
- `sampled_joint_pair_v1` strategy branch projection audit -> exit 0; explicit strategy branch has no pair-to-single projection.
- V1 autotune metadata audit -> exit 0; recipe exposes scorecard baseline and metric metadata.

## Artifacts Produced

- `Docs/refactor/artifacts/v1_pair_action/v1_pair_scorecard_schema_gate.json`
- `Docs/refactor/artifacts/v1_pair_action/worker_h_completion_packet.md`

## Performance And Utilization Evidence

- Deterministic scorecard schema now requires candidate generation p50/p95, pair scores/sec, inference latency p50/p95, neural calls per expanded full-turn node, queue/backpressure ratio, GPU utilization, candidate recall, tactical inclusion, target entropy, and equal-wall-clock arena fields before a V1 strength claim can pass.
- Training smoke exercises the V1 graph loss path and loader timing surface, including `graph_loader_v1_pair_rows_s`.
- No long-run arena strength, throughput, or hardware utilization result is claimed by Worker H.

## Contract Docs And Examples Added

- `Python/src/hexorl/eval/v1_pair_scorecard.py` documents the V1 scorecard contract and provides a deterministic template/validator.
- `Python/tests/test_v1_pair_training_losses.py` gives a smoke example for V1 target dictionaries and loss-plan activation.
- `Python/tests/test_v1_pair_eval_guardrails.py` gives schema-gate examples for scorecard/autotune evidence.
- `Python/tests/test_v1_pair_ci_audit_gates.py` gives deterministic audit-gate examples.

## Known Blockers

- Equal-wall-clock strength evaluation cannot be claimed from Worker H evidence. The current evidence is a deterministic schema gate only; a fair command-backed arena still needs fixed hardware, batching protocol, candidate budget, opponent checkpoints, confidence interval, stopping protocol, and a fair sequential DAG neural baseline run.

## Completion Statement

No skipped, deferred, flaky, or manual-only requirement is claimed complete.
