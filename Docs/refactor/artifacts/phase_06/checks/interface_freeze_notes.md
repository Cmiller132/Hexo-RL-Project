# Interface Freeze Notes

Frozen public contracts:

- `GameRunRequest`
- `GameRunResult`
- `RuntimeResourceSpec`
- `GameRunnerConfig`
- `SelfPlayContractBuilders`
- `SelfPlayRecordWriter`
- `RecordWriteResult`
- `ContractTrace`
- `SelfPlayDebugBundle`
- `SelfPlayMutationGuard`
- `SelfPlayTelemetrySink`

Frozen runtime ownership:

- `SelfPlayWorker` owns process signals, IPC client connection, cancellation observation, crash accounting, and forwarding `GameRunRequest` to `GameRunner`.
- `GameRunner` owns per-game orchestration and composes explicit provider/adapter/builder/writer/telemetry dependencies.
- `record_writer.py` owns replay record validation and bounded output queue backpressure.
- `telemetry.py` owns event and debug payload schemas.

Approved implementation latitude:

- `GameRunner` receives an `engine_adapter_factory` rather than one long-lived adapter because the adapter lifecycle is per game and depends on seed, simulation count, and restored game state.
- Builder internals remain behind `SelfPlayContractBuilders`; the runner does not import worker lifecycle state.
