# Stage 4 Execution Packet

## Goal

Move inference and search runtime behavior to protocol, adapter, provider,
pair-strategy, and engine-adapter boundaries, then delete old scattered runtime
authority.

## Current Status

Stage 4 is complete. Inference, evaluation, self-play search, pair-prior
handling, engine validation, and model-family loading now flow through the
Stage 4 runtime boundaries, and the legacy `Python/src/hexorl/model/` package
has been moved out of the runtime tree.

## Checklist

- [x] Inference protocol metadata exists and treats shared memory as transport.
- [x] Server delegates dense/global output decoding to inference adapters.
- [x] Client delegates graph shared-memory response decoding to an adapter.
- [x] Graph inference responses carry output metadata, row-table hashes, and
  value decoder metadata.
- [x] Same-count reordered legal row tables fail row-identity validation.
- [x] Pair behavior config is validated by an explicit pair-strategy registry.
- [x] Self-play pair enablement comes from a `PairStrategy`, not head presence.
- [x] Engine legal-row/value validation is available through `EngineAdapter`.
- [x] Direct graph pair-head decode branches are removed from
  `inference/server.py` and `inference/client.py` hot logic.
- [x] Self-play no longer checks concrete global pair-head names directly.
- [x] Self-play root/leaf pair scoring should move fully into pair-strategy
  methods.
- [x] Evaluation should either use the provider boundary or receive a final
  runtime quarantine record.
- [x] `hexorl/model/` must be deleted or fully moved into `hexorl/models/`.
- [x] Final import audit must prove old model-class API is not runtime
  authority.
- [x] Final Stage 4 full-suite and performance evidence must be recorded.

## Runtime Consumers Changed

- `Python/src/hexorl/inference/protocol.py` defines inference protocol version,
  graph head flags, row-table metadata, value-decoder metadata, output
  metadata, and row-table identity validation.
- `Python/src/hexorl/inference/adapters.py` owns dense and global graph output
  decoding, finite/shape validation, value decoding, graph response metadata,
  and same-count row-table rejection.
- `Python/src/hexorl/inference/server.py` now calls inference adapters for
  dense and global graph output decode instead of interpreting graph output
  heads inline.
- `Python/src/hexorl/inference/client.py` now calls an adapter to decode graph
  shared-memory responses and attach semantic metadata.
- `Python/src/hexorl/search/pair_strategy.py` defines explicit pair strategy
  descriptors, required output contracts, max-pair/mix validation, graph pair
  output accessors, root/leaf graph pair chunk scoring, crop pair chunk
  scoring, pair-logit projection, pair-logit blending, and root pair-prior
  application.
- `Python/src/hexorl/search/engine_adapter.py` owns legal-row alignment,
  legal-subset validation, search-phase validation, batch-generation
  validation, legal-byte alignment, dense offset mapping, value range and
  perspective validation, and pair-phase validation before Rust MCTS
  consumption.
- `Python/src/hexorl/config/schema.py` validates pair strategy through the
  strategy registry.
- `Python/src/hexorl/selfplay/worker.py` delegates pair enablement and concrete
  graph pair-output access, root/leaf pair scoring, action-logit projection,
  pair blending, and root prior application to `PairStrategy`. It delegates
  global legal/value, dense offset, phase, and pair-row validation to
  `EngineAdapter`.
- `Python/src/hexorl/eval/model_provider.py` is the evaluation provider
  boundary. `Python/src/hexorl/eval/arena.py` loads checkpoints through it.
- `Python/src/hexorl/models/loading.py` is the runtime model loading boundary
  used by inference, training checkpoint restore, and evaluation.
- `Python/src/hexorl/models/families/` now contains `HexNet` and
  `GlobalHexGraphNet`; `Python/src/hexorl/model/` has been deleted.

## Evidence

Commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\inference\protocol.py Python\src\hexorl\inference\adapters.py Python\src\hexorl\inference\server.py Python\src\hexorl\inference\client.py Python\src\hexorl\search\__init__.py Python\src\hexorl\search\engine_adapter.py Python\src\hexorl\search\pair_strategy.py Python\src\hexorl\config\schema.py Python\src\hexorl\selfplay\worker.py Python\tests\test_model_architecture_stage4.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage4.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage2.py Python/tests/test_model_architecture_stage4.py Python/tests/test_global_graph_contract.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_inference_server.py Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_training_data_pipeline.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests
Select-String -Path Python\src\hexorl\inference\server.py,Python\src\hexorl\inference\client.py -Pattern 'if .*policy_pair','if .*policy_place','if .*opp_policy','head_flags &'
Select-String -Path Python\src\hexorl\selfplay\worker.py -Pattern 'policy_pair_joint','policy_pair_second','policy_pair_first'
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\inference\adapters.py Python\src\hexorl\inference\server.py Python\src\hexorl\inference\client.py Python\src\hexorl\search\engine_adapter.py Python\src\hexorl\search\pair_strategy.py Python\src\hexorl\selfplay\worker.py Python\src\hexorl\eval\arena.py Python\src\hexorl\eval\model_provider.py Python\src\hexorl\models\loading.py Python\src\hexorl\models\families\network.py Python\src\hexorl\models\families\global_graph.py Python\tests\test_model_architecture_stage4.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_inference_server.py Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_training_data_pipeline.py Python/tests/test_production_smoke.py Python/tests/test_dashboard_foundation.py
cargo test --workspace
Get-ChildItem -Path Python\src\hexorl,Python\tests -Recurse -File -Include *.py | Where-Object { $_.FullName -notmatch '\\__pycache__\\' } | Select-String -Pattern 'from hexorl\.model(?!s)','import hexorl\.model(?!s)'
Get-ChildItem -Path Python\src\hexorl\inference,Python\src\hexorl\train,Python\src\hexorl\eval,Python\src\hexorl\selfplay,Python\src\hexorl\config,Python\src\hexorl\buffer,Python\src\hexorl\dashboard -Recurse -File -Include *.py | Where-Object { $_.FullName -notmatch '\\__pycache__\\' } | Select-String -Pattern '\bHexNet\b','\bGlobalHexGraphNet\b','from_config','load_model_state'
Get-ChildItem -Path Python\src\hexorl\selfplay -Recurse -File -Include *.py | Where-Object { $_.FullName -notmatch '\\__pycache__\\' } | Select-String -Pattern '_score_graph_pair_chunks','_score_crop_pair_chunks','_graph_batch_with_pair_rows','_pair_logits_to_action_logits','_filter_pair_rows_for_root_children','_blend_action_logits','policy_pair_joint','policy_pair_second','policy_pair_first'
Test-Path Python\src\hexorl\model
```

Results:

```text
py_compile passed
Stage 4 focused tests: 11 passed
Stage 2/4/global output gate suite: 59 passed, 1 warning
affected integration suite: 162 passed, 1 warning
affected final integration suite: 182 passed, 1 warning
full Python suite: 317 passed, 1 warning
Rust workspace: 177 passed, 6 ignored, 2 doc-tests passed
inference hot-logic graph-head branch audit: no matches
self-play concrete global pair-head audit: no matches
legacy hexorl.model import audit: no matches
runtime old model-class API audit: no matches
old self-play pair helper/direct pair-head audit: no matches
explicit graph-native output head smoke: ['policy_place', 'value']
Python/src/hexorl/model exists: False
```

Performance smoke:

```text
graph_response_decode_pair_metadata rows legal=216 pair=256 iterations=300 mean_ms=0.0883 median_ms=0.0863 p95_ms=0.0970
```

## Stop Rule Notes

- Inference maps dense, global legal, pair-first, pair-joint, and known-first
  pair outputs to row contracts with value decoder metadata.
- Self-play consumes pair priors only through `PairStrategy`, including root and
  leaf pair scoring.
- Evaluation loads models only through `eval.model_provider`.
- Runtime model construction and checkpoint restore go through
  `models.loading`.
- `Python/src/hexorl/model/` is deleted.

## Full Verification Addendum

The 2026-05-06 completion pass fixed the prior full verification blockers:
legacy model implementation was moved under `hexorl.models.families`,
evaluation moved behind `eval.model_provider`, pair strategy now owns root/leaf
pair scoring and root prior application, inference adapters emit pair row-table
metadata, and final performance evidence has been recorded.

Additional evidence gathered during the pass:

```text
Stage 4 focused tests: 11 passed
full Python suite: 317 passed, 1 warning
Rust workspace: 177 passed, 6 ignored, 2 doc-tests passed
runtime import/deletion audits: clean
explicit graph-native output-head gate smoke: clean
```

## Explicit Completeness Statement

Stage 4 is complete. No skipped, deferred, flaky, quarantined, or manual-only
Stage 4 requirement is being claimed complete.
