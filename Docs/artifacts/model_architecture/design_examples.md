# Design Examples

These examples are intentionally pseudocode/design notes for Stage 1.

## Row Table Definition

```python
RowTableDefinition(
    family="legal",
    schema_version=1,
    payload_schema=("q:int32", "r:int32"),
    ordering_rule="rust_legal_order",
    mask_semantics="true means active row",
)
```

## Row Table Instance

```python
legal_rows = RowTableInstance(
    definition="legal:v1",
    rows=np.asarray([(0, 0), (1, 0), (0, 1)], dtype=np.int32),
    mask=np.asarray([True, True, True]),
    phase="search_any",
    source="rust_mcts_root",
)
```

The identity hash includes the ordered rows, mask, phase, and schema version.
The same three rows in a different order are not interchangeable.

## Output Contract

```python
OutputContract(
    name="policy_place",
    prediction_key="policy_place",
    kind="policy",
    row_family="legal",
    trainable=True,
    runtime_consumed=True,
    required_for_selfplay=True,
    mask_semantics="inactive logits must be ignored or set to -80 before CE",
)
```

## Value Decoder Contract

```python
ValueDecoderContract(
    name="binned_expected_value_65",
    logits_key="value",
    n_bins=65,
    centers=(-1.0, 1.0),
    output_range=(-1.0, 1.0),
    perspective="current_player",
)
```

## Pair Output Specs

```python
PairOutputSpec(
    name="policy_pair_joint",
    row_family="pair_joint",
    phase="first_placement_joint",
    ordering="unordered_canonical",
    runtime_consumed_by=("root_pair_mcts", "full_pair_mcts", "pair_joint_marginal_blend"),
    trainable=True,
)
```

```python
PairOutputSpec(
    name="policy_pair_second",
    row_family="known_first_pair",
    phase="second_placement_known_first",
    ordering="ordered_known_first_then_second",
    runtime_consumed_by=("full_pair_mcts", "pair_second_conditional_blend"),
    trainable=True,
)
```

## Target Contract Example

```python
TargetContract(
    name="pair_second_policy_target",
    row_family="known_first_pair",
    phase="second_placement_known_first",
    mask="pair_second_mask",
    weight="pair_policy_weight",
    normalization="positive_mass_over_active_rows",
    missing_behavior="hard_error",
    zero_mass_behavior="hard_error_when_trainable",
    duplicate_behavior="hard_error_duplicate_coordinates",
)
```

## Loss Plan Entry

```python
LossPlanEntry(
    output="policy_place",
    target="policy_target",
    row_table="legal",
    mask="legal_mask",
    weight="policy_weight",
    phase="search_any",
    loss="masked_cross_entropy",
    missing_behavior="hard_error",
)
```

## Inference Request Example

```python
InferenceRequest(
    protocol_version=1,
    architecture_id="global_pair_twostage_0",
    adapter_id="global_graph:v1",
    requested_outputs=("policy_place", "value", "policy_pair_joint"),
    row_hashes={
        "legal": "sha256:...",
        "pair_joint": "sha256:...",
        "graph_token": "sha256:...",
    },
    value_decoder_id="binned_expected_value_65",
    pair_phase="first_placement_joint",
)
```

## Pair Strategy Plan Example

```python
PairStrategyPlan(
    strategy_id="full_pair_mcts",
    phase_support=("first_placement_joint", "second_placement_known_first"),
    required_outputs={
        "first_placement_joint": ("policy_pair_joint",),
        "second_placement_known_first": ("policy_pair_second",),
    },
    mcts_application="apply_pair_priors_when_pair_prior_mix_gt_zero",
    telemetry=("root_pair_candidate_count", "root_pair_count", "leaf_pair_count"),
)
```
