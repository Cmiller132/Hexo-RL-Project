# Stage 4 Execution Packet

## Goal

Move inference and search runtime behavior to protocol, adapter, provider,
pair-strategy, and engine-adapter boundaries, then delete old scattered runtime
authority.

## Current Status

Stage 4 has started. This packet records the first completed cut: inference
protocol/adapters plus initial search pair-strategy and engine-adapter
boundaries. Stage 4 is not yet claimed complete because the final legacy model
deletion/move gate remains open.

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
- [ ] Self-play root/leaf pair scoring should move fully into pair-strategy
  methods.
- [ ] Evaluation should either use the provider boundary or receive a final
  runtime quarantine record.
- [ ] `hexorl/model/` must be deleted or fully moved into `hexorl/models/`.
- [ ] Final import audit must prove old model-class API is not runtime
  authority.
- [ ] Final Stage 4 full-suite and performance evidence must be recorded.

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
  descriptors, required output contracts, max-pair/mix validation, and graph
  pair output accessors.
- `Python/src/hexorl/search/engine_adapter.py` owns legal-row alignment,
  legal-subset validation, and value range validation before Rust MCTS
  consumption.
- `Python/src/hexorl/config/schema.py` validates pair strategy through the
  strategy registry.
- `Python/src/hexorl/selfplay/worker.py` delegates pair enablement and concrete
  graph pair-output access to `PairStrategy`, and delegates global legal/value
  validation to `EngineAdapter`.

## Evidence

Commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\inference\protocol.py Python\src\hexorl\inference\adapters.py Python\src\hexorl\inference\server.py Python\src\hexorl\inference\client.py Python\src\hexorl\search\__init__.py Python\src\hexorl\search\engine_adapter.py Python\src\hexorl\search\pair_strategy.py Python\src\hexorl\config\schema.py Python\src\hexorl\selfplay\worker.py Python\tests\test_model_architecture_stage4.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage4.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_inference_server.py Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_training_data_pipeline.py
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests
Select-String -Path Python\src\hexorl\inference\server.py,Python\src\hexorl\inference\client.py -Pattern 'if .*policy_pair','if .*policy_place','if .*opp_policy','head_flags &'
Select-String -Path Python\src\hexorl\selfplay\worker.py -Pattern 'policy_pair_joint','policy_pair_second','policy_pair_first'
```

Results:

```text
py_compile passed
Stage 4 focused tests: 6 passed
affected integration suite: 162 passed, 1 warning
full Python suite: 311 passed, 1 warning
inference hot-logic graph-head branch audit: no matches
self-play concrete global pair-head audit: no matches
```

## Stop Rule Notes

- Inference can map dense and global graph policy outputs to row contracts for
  the paths touched in this cut.
- Self-play no longer directly checks concrete global pair output names.
- Old direct runtime branches remain in scope for continued Phase 4 work:
  `hexorl/model/` has not yet been deleted or fully moved.

## Explicit Completeness Statement

This is a Phase 4 start packet, not a completion packet. No unchecked Stage 4
requirement is claimed complete.
