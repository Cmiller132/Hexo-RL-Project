# Inference Transport Boundary Redesign

## Goal

Inference transport is now a contract-walking boundary. Model families declare operations, tensors, dynamic dimensions, batching modes, output heads, row mappings, and decoders in `hexorl/models/inference_contracts.py`. The inference package consumes those declarations mechanically.

North-star invariant: adding a new model operation should require edits only under `hexorl/models/`.

## Modules

- `hexorl/models/inference_contracts.py`: the single model inference spec, `ModelInferenceContract`.
- `hexorl/inference/protocol.py`: protocol manifest, handshake, request and response envelopes.
- `hexorl/inference/control.py`: generic control block fields and dynamic-dim table.
- `hexorl/inference/arena.py`: shared-memory arenas allocated from the union of declared tensors.
- `hexorl/inference/evaluator.py`: `Evaluator` protocol.
- `hexorl/inference/local.py`: in-process `LocalEvaluator`.
- `hexorl/inference/client/api.py`: shared-memory `RemoteEvaluator`.
- `hexorl/inference/client/transport.py`: generic request packing and response reading.
- `hexorl/inference/server/collation.py`: generic batching by `TensorSpec.batching`.
- `hexorl/inference/server/execution.py`: `model.forward(**inputs)` plus contract-driven decoders.
- `hexorl/inference/server/scatter.py`: generic response scatter by symbolic dims.
- `hexorl/search/policy_provider.py`: domain-to-flat-payload translation.

## Contract Pattern

Each `TensorSpec` declares:

- `name`
- `dtype`
- `shape`, with integer constants and symbolic dimension names such as `B`, `K`, `T`, and `L`
- `batching`: `stack_over_b`, `pad_and_stack`, or `singleton`

The control block no longer has semantic count slots. It stores protocol, contract, layout, opcode, status, generation, deadline, enqueue time, and a generic dynamic-dimension table:

```text
n_dims, (dim_name_hash, value), ...
```

Transport writes dimension values inferred from payload array shapes. Collation and scatter read them by dimension name.

## Evaluator Protocol

```python
class Evaluator(Protocol):
    manifest: InferenceProtocolManifest
    def evaluate(self, op: str, payload: dict[str, ndarray]) -> InferenceResponse: ...
    def close(self) -> None: ...
```

`RemoteEvaluator` uses the shared-memory arena. `LocalEvaluator` runs the same operation contract in-process. Policy providers accept the protocol and do not depend on a concrete client class.

## Adding An Operation

To add a new operation:

1. Edit `hexorl/models/inference_contracts.py`.
2. Add input `TensorSpec` declarations.
3. Add output head specs and decoders.
4. Add an `InferenceOperationSpec` with `required_inputs`, `output_heads`, and `required_heads`.

No inference package file should change. The CI grep gate checks that semantic operation, head, tensor, layout, and fixed request-count slot names do not appear under `hexorl/inference`.

## Verification

```text
PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/inference Python/tests/test_inference_server.py Python/tests/search Python/tests/eval Python/tests/selfplay/test_rgsc_game_runner.py Python/tests/models/test_phase03_model_registry.py
94 passed in 2.95s
```

```text
! grep -rE 'crop_batch|graph_batch|policy_place|candidate_indices|graph_token|REQ_(CANDIDATE|PAIR|TOKEN|LEGAL|OPP|GRAPH)' Python/src/hexorl/inference
pass
```
