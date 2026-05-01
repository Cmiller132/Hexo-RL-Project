# Model-Owned Inference Contracts And Dynamic Arena Cutover - 2026-04-30

## Scope

Phase 04 inference semantics moved from inference-owned request kinds and fixed shared-memory fields to model-owned operation contracts and a typed dynamic shared-memory arena.

## Runtime Ownership

- `hexorl.models.inference_contracts` owns `ModelInferenceContract`, operation specs, head decoder specs, row mappings, and transport layout specs.
- Model family descriptors publish complete inference contracts through `inference_contract_factory`.
- `hexorl.inference.protocol` derives protocol manifests from model contracts and negotiates operation names, operation codes, layout hashes, capacities, and required heads.
- `hexorl.inference.shm_queue` owns only control words, doorbells, and request/response arena bytes with manifest-derived tensor views.
- `InferenceClient.evaluate(operation_name, payload)` is the only runtime client request API.
- Server scheduling batches by operation/layout compatibility and uses generic collation, execution, decoding, and scatter helpers.

## Deleted Legacy Paths

- Removed semantic `InferenceRequestKind`, `REQUEST_KIND_TO_CODE`, `REQUEST_CODE_TO_KIND`, and `default_protocol_manifest`.
- Removed fixed semantic shared-memory fields such as dense policy/value buffers, sparse/pair buffers, and graph-specific response buffers.
- Removed dense/sparse/pair/global semantic inference adapter modules.
- Removed client convenience calls for dense, sparse, pair, graph, and regret inference.

## Arena Layout Evidence

Estimated per-worker fixed-slot footprint before the cutover: `8,317,037` bytes.

| Family | Dynamic arena/control bytes | Estimated footprint reduction |
|---|---:|---:|
| `dense_cnn` | `976,194` | `88.26%` |
| `graph_hybrid` | `1,226,306` | `85.26%` |
| `global_xattn` | `1,847,106` | `77.79%` |

Dynamic means per-session manifest-derived tensor tables and arenas, not per-request OS shared-memory allocation.

## Verification

Focused command:

```bash
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q \
  Python/tests/inference \
  Python/tests/test_inference_server.py \
  Python/tests/search/test_policy_provider.py \
  Python/tests/search/test_pair_strategy.py \
  Python/tests/search/test_global_graph_pair_contracts.py \
  Python/tests/eval/test_phase08_eval_policy_provider.py \
  Python/tests/selfplay/test_rgsc_game_runner.py
```

Result: `53 passed`.

Compile command:

```bash
PYTHONPATH=Python/src python3 -m py_compile $(find Python/src/hexorl -name '*.py' -not -path '*/__pycache__/*')
```

Result: exit `0`.

## Completion Statement

No skipped, deferred, compatibility-shim, or manual-only requirement is claimed complete by this artifact. The implementation is a full runtime cutover; legacy request-kind and fixed semantic buffer paths are not kept as alternate runtime paths.
