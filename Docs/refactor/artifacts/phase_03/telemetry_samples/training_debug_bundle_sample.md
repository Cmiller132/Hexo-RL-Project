# Phase 03 Training Debug Bundle Sample

The focused adapter test produces a single-position debug bundle with:

```text
trace_id=phase03-single-position
owner=train_adapter
spec_kind=dense_cnn
input_keys=["crop_tensor"]
target_keys=["policy", "value"]
mask_keys=[]
output_keys=["policy", "value"]
loss_keys=["policy", "value"]
tensor_hashes={...shape/dtype/sum identity hashes...}
```

The bundle reconstructs the path:

```text
replay-style batch -> TrainAdapter.project_batch -> model inputs -> targets -> model outputs -> compute_losses inputs
```

Mutation evidence:

```text
test_train_adapter_rejects_mutated_contract_after_projection
exit=0 as part of Phase 03 focused tests
Projected tensors are cloned onto the target device; mutating the source batch after projection does not mutate validated targets.
```
