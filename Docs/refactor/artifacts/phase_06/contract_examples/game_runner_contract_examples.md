# GameRunner Contract Examples

Construction:

```python
runner = GameRunner(
    policy_provider=provider,
    pair_strategy=pair_strategy,
    engine_adapter_factory=engine_factory,
    record_writer=writer,
    telemetry_sink=telemetry,
    contract_builders=SelfPlayContractBuilders(),
    runtime_spec=runtime_spec,
    runner_config=runner_config,
    model_spec=model_spec,
)
```

Execution:

```python
result = runner.run_game(GameRunRequest(run_id="run", game_id=1, game_index=0, seed=7))
assert result.ok
```

Validation failure example:

```python
SearchEvaluation(
    context=context,
    legal_row_ids=np.asarray([0]),
    legal_dense_indices=context.legal_table.dense_indices[:1],
    row_priors=np.ones(1, dtype=np.float32),
    prior_source=np.ones(1, dtype=np.uint8),
    value=0.0,
    policy_provider="fake",
    model_family="dense_cnn",
    model_spec_version="v2",
    inference_protocol="fake",
)
# raises ContractValidationError because prior length does not match legal rows
```
