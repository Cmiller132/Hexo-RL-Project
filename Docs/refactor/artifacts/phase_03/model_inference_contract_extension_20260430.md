# Model Inference Contract Extension - 2026-04-30

## Scope

Model family descriptors now expose `inference_contract_factory` in addition to build/train/checkpoint/policy facets. The contract declares model-owned inference operations, head decoder semantics, row mappings, capacities, and transport layouts.

## Runtime Consumers Changed

- `InferenceServer` derives its protocol manifest from the selected model family contract.
- `InferenceClient` negotiates and submits operation names declared by the model contract.
- `PolicyProvider`, pair scoring, eval, and RGSC scoring choose model operations instead of calling client convenience methods.

## Extension Proof

`Python/tests/inference/test_model_owned_contracts.py` verifies:

- every registered built-in family emits a complete inference contract;
- contract hashes change when head semantics change;
- a fake family can register an operation contract without modifying inference runtime internals;
- handshake rejects unsupported operations, missing heads, and contract/layout mismatches before enqueue.

## Completion Statement

No skipped, deferred, compatibility-shim, or manual-only requirement is claimed complete. The model registry is the semantic owner for inference operations; inference remains the transport/runtime owner.
