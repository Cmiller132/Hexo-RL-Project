# Contract Design Notes

These are Stage 1 design artifacts, examples, and pseudocode only. No importable
runtime modules are created in Stage 1.

## Semantic Phases

```text
search_any
first_placement_joint
second_placement_known_first
opponent_turn_start
completed_game
truncated_or_untrusted_value
diagnostic_only
```

Phase-sensitive losses and providers must receive explicit phase metadata. They
may not infer phase from zero target mass, row count, previous player, or head
name.

## Row Table Contracts

Every row table has:

```text
row_family
schema_version
rows
mask
identity_hash
source
phase
ordering_rule
payload_schema
```

Required row families:

- `dense_board`: fixed board indices `0..1088`, crop offset scoped.
- `candidate`: candidate `(q,r)`, dense action index, feature schema, mask.
- `legal`: all current Rust-legal global `(q,r)` rows.
- `opponent_legal`: independent future/opponent legal `(q,r)` rows.
- `pair_joint`: unordered canonical pair rows over legal rows.
- `known_first_pair`: ordered `(known_first, legal_second)` rows.
- `graph_token`: graph token rows with token type, qr, feature schema, and
  relation schema.

Identity hash inputs:

```text
row_family
schema_version
phase
ordered row payload bytes
mask bytes
feature schema version when applicable
relation schema version when applicable
```

Same-count row tables with different order, mask, payload, phase, or feature
schema must hash differently and be rejected.

## Output Contracts

Every model output is described by:

```text
output_name
kind: policy | pair_policy | value | auxiliary | diagnostic
prediction_key
row_family or state row
row_identity_hash requirement
mask_semantics
value_decoder when applicable
range and perspective when applicable
trainable
runtime_consumed
required_for_selfplay
optional
diagnostic_only
```

Value decoder contract:

```text
decoder = binned_expected_value
n_bins = 65
range = [-1.0, 1.0]
perspective = current_player
clamp_non_finite = true at adapter boundary
```

## Target Contracts

Every trainable target is described by:

```text
target_name
source_fields
row_family
row_identity_hash
mask_name
weight_name
phase_name
normalization_rule
missing_behavior
zero_mass_behavior
duplicate_behavior
invalid_row_behavior
loss_name
```

Default behavior:

- missing target, mask, weight, or phase: hard error;
- row identity mismatch: hard error;
- zero target mass for trainable row: hard error;
- duplicate rows: hard error;
- duplicate target entries for a valid row: sum then normalize only if allowed
  by target contract.

## Inference Protocol

Request protocol:

```text
protocol_version
architecture_id
adapter_id
requested_outputs
input_contract_id
row_tables
row_hashes
value_decoder_id
pair_phase
transport_schema_version
```

Response protocol:

```text
protocol_version
adapter_id
returned_outputs
row_hashes
value_decoder_id
pair_phase
diagnostics
transport_shape_summary
```

Adapters validate protocol facts before any logits reach MCTS or dashboard
consumers.

## Architecture Spec Contract

Spec resolution produces:

```text
architecture_id
family_id
recipe_id
required_inputs
optional_inputs
default_outputs
supported_optional_outputs
forbidden_outputs
dynamic_output_families
selfplay_required_outputs
training_adapter_id
inference_adapter_id
policy_provider_id
value_provider_id
pair_capabilities
loss_plan_id
```

Config override rules:

- enable supported optional output: allowed if loss/target dependencies resolve;
- disable optional output: allowed;
- disable self-play required output: hard error;
- enable unknown output: hard error;
- enable pair strategy without required pair capability: hard error;
- enable pair outputs without pair strategy: allowed only as trainable or
  diagnostic outputs, not runtime pair influence.

## Pair Strategy Contract

Executable pair strategy plan:

```text
strategy_id
phase_support
required_outputs
row_builder
request_plan
score_plan
mcts_application
telemetry
fallback
hard_errors
```

`none` is executable and explicit: it requests no pair rows and applies no pair
priors.
