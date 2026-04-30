# Phase 03 Registry And Checkpoint Contract Examples

## Registered Families

```text
dense_cnn: CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD
restnet: CROP_INPUT, DENSE_PLACE_POLICY, REGRET_HEAD
graph_hybrid: CROP_INPUT, DENSE_PLACE_POLICY, SPARSE_PLACE_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD
global_xattn: GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD
global_line_window: GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD
global_relation_graph: GLOBAL_GRAPH_INPUT, GLOBAL_PLACE_POLICY, PAIR_FIRST_POLICY, PAIR_SECOND_POLICY, JOINT_PAIR_POLICY, REGRET_HEAD
```

## Descriptor Facets

Every built-in descriptor supplies:

```text
ModelBuilder
TrainAdapterFactory
InferenceAdapterFactory
PolicyProviderFactory
LossPlanProvider
RecipeProvider
TuneSpaceProvider
CheckpointManifestProvider
```

## Inference Manifest Shape

```yaml
protocol_version: 1
request_kind: crop | global_graph
model_family: dense_cnn
model_spec_version: 1
input_contract: crop_tensor_v1
output_contract: dense_place_value_v1
action_contract: legal_action_table_v1
max_legal_rows: 1089
required_heads:
  - value
  - policy
capabilities:
  - CROP_INPUT
  - DENSE_PLACE_POLICY
```

## Checkpoint Manifest Shape

```yaml
checkpoint_schema_version: 1
model_family: dense_cnn
model_spec_version: 1
model_spec:
  kind: dense_cnn
  version: 1
input_contract: crop_tensor_v1
output_contract: dense_place_value_v1
action_contract: legal_action_table_v1
inference_protocol:
  protocol_version: 1
heads:
  - policy
  - value
pair_strategy_used: none
created_by:
  git_sha: unknown
  command: Trainer.save_checkpoint
  config_hash: unknown
```
