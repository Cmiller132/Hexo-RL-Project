# Phase 06 - GameRunner + SelfPlayWorker Cleanup

## Purpose

Move game execution out of `SelfPlayWorker` and into a narrow `GameRunner` that depends on explicit interfaces from the new architecture. After this phase, `SelfPlayWorker` is only a process, lifecycle, and IPC shell. It must not know model architecture details, run the game loop, assemble replay records, chunk graph/candidate/pair work, or wire MCTS priors directly.

Source of truth: `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

## Target Modules

- `Python/src/hexorl/selfplay/game_runner.py`
- `Python/src/hexorl/selfplay/worker.py`
- `Python/src/hexorl/selfplay/orchestrator.py`
- `Python/src/hexorl/selfplay/records.py`
- `Python/src/hexorl/selfplay/record_writer.py`
- `Python/src/hexorl/selfplay/telemetry.py`
- `Python/src/hexorl/selfplay/rgsc.py`
- `Python/src/hexorl/search/policy_provider.py`
- `Python/src/hexorl/search/pair_strategy.py`
- `Python/src/hexorl/search/engine_adapter.py`

## Required End State

`GameRunner` owns per-game execution and receives every behavioral dependency explicitly:

```python
class GameRunner:
    def __init__(
        self,
        *,
        policy_provider: PolicyProvider,
        pair_strategy: PairStrategy,
        engine_adapter: EngineAdapter,
        record_writer: SelfPlayRecordWriter,
        telemetry_sink: SelfPlayTelemetrySink,
        contract_builders: SelfPlayContractBuilders,
    ) -> None: ...

    def run_game(self, request: GameRunRequest) -> GameRunResult: ...
```

Required dependency contracts:

- `PolicyProvider`: evaluates policy/value outputs from typed position/search contracts.
- `PairStrategy`: decides whether pair policy is used; default strategy is `none`.
- `EngineAdapter`: owns Rust replay/legal/MCTS calls and hides engine-specific call details.
- `SelfPlayRecordWriter`: writes replay records; the runner does not write files directly.
- `SelfPlayTelemetrySink`: emits structured events, summaries, traces, and stall diagnostics.
- `SelfPlayContractBuilders`: bundles the canonical builders needed by the runner, including history/legal/tactical/candidate/pair/graph builders as applicable.

`SelfPlayWorker` must be reduced to:

- process startup/shutdown
- IPC connection management
- worker registration and heartbeat scheduling
- loading immutable runtime dependencies produced by orchestrator/factory code
- forwarding run requests to `GameRunner`
- forwarding lifecycle and IPC failures to telemetry
- clean cancellation, timeout, and child-process teardown

## Prohibited In Worker

`SelfPlayWorker` must not contain:

- architecture string checks, including `startswith("global_")`
- model-family capability checks
- game-loop details such as move selection, terminal-state handling, or turn transitions
- replay record assembly
- legal row construction, compact history parsing, or D6 transforms
- candidate construction or candidate chunking
- pair table construction, pair enumeration, or pair chunking
- graph semantic construction, tensorization, or graph batching
- MCTS prior wiring or direct prior blending
- inference request shape decisions
- pair enablement from `pair_prior_mix`, head presence, model family, or architecture name
- checkpoint cleanup or model-state compatibility handling

Any such logic must live in `contracts/`, `engine/`, `graph/`, `inference/`, `search/`, `models/`, `replay/`, or `selfplay/game_runner.py` as assigned by the V2 ownership map.

## GameRunner Responsibilities

`GameRunner` may:

- initialize a game from `GameRunRequest`
- request legal/replay state through `EngineAdapter`
- build canonical position/search contracts through `SelfPlayContractBuilders`
- request policy/value priors through `PolicyProvider`
- request pair behavior only through `PairStrategy`
- invoke search through `EngineAdapter`
- select and apply moves according to search results
- send completed positions/games to `SelfPlayRecordWriter`
- emit structured telemetry through `SelfPlayTelemetrySink`
- return `GameRunResult` with counts, timings, hashes, warnings, and failure status

`GameRunner` may not infer behavior from model-family names or architecture strings. Dense, graph hybrid, and global graph runs must use the same runner interface.

## Self-Play Logging Requirements

Self-play logs must make these failure classes distinguishable:

- worker alive but waiting on IPC or inference
- worker stuck before inference
- Rust replay/legal generation slow or failing
- tactical, candidate, pair, graph semantic, or tensorization construction slow
- pair scoring accidentally enabled or above budget
- IPC request packed but not answered
- model forward slow or non-finite
- MCTS expansion/backprop slow
- record writing slow or failing
- legal rows disagree between engine and contract
- priors missing, masked out, non-finite, or mapped to wrong rows

Required event types:

- `selfplay_worker_heartbeat`
- `selfplay_phase_transition`
- `selfplay_no_progress`
- `selfplay_game_summary`
- `policy_eval_timing`
- `pair_strategy_summary`
- `contract_validation_failure`
- `inference_protocol_mismatch`

Required heartbeat fields:

- worker id, process id, run id, game id
- current phase and move index
- positions completed since last heartbeat
- last successful inference request id
- last engine operation
- legal, candidate, pair, token, and relation counts
- active model family, recipe id, policy provider, and pair strategy
- pair rows possible and pair rows scored
- recent timing summary
- warning count and last warning
- no-progress duration

Required no-progress fields:

- phase, elapsed time, last completed position, last IPC send/receive
- last engine operation and duration
- last policy request id and wait duration
- last record writer operation and duration
- queue depth or transport state when available
- suggested next subsystem to inspect

Required game summary fields:

- game id, seed, winner/result, move count, terminal reason
- positions written and records skipped
- total engine/search/inference/contract/record-writer time
- average legal/candidate/pair/token/relation counts
- pair strategy used and total pair rows scored
- validation failures and warning summary

Required policy timing fields:

- trace id, history hash, move index, phase
- request kind, provider name, model family, protocol version
- IPC pack/wait, queue wait, collate, model forward, scatter, decode
- prior source, masked count, non-finite count, legal-row coverage

Required pair summary fields:

- pair strategy name and enabled/disabled reason
- phase, total possible pairs, selected rows, scored rows
- cap values and cap-hit status
- chunk count and chunk forward time
- first/second/joint head usage when applicable

## ContractTrace Requirements

Self-play telemetry must propagate `ContractTrace` from the V2 architecture:

```python
@dataclass(frozen=True)
class ContractTrace:
    trace_id: str
    history_hash: str
    model_family: str
    phase: str
    legal_count: int
    candidate_count: int
    pair_rows_total: int
    pair_rows_scored: int
    graph_token_count: int
    graph_relation_count: int
    timings_ms: dict[str, float]
    warnings: tuple[str, ...]
```

Required spans:

- `history_parse_ms`
- `engine_replay_ms`
- `legal_table_ms`
- `tactical_oracle_ms`
- `candidate_build_ms`
- `pair_table_build_ms`
- `graph_token_build_ms`
- `graph_relation_build_ms`
- `graph_tensorize_ms`
- `ipc_pack_ms`
- `ipc_wait_ms`
- `queue_wait_ms`
- `collate_ms`
- `model_forward_ms`
- `scatter_ms`
- `decode_ms`
- `pair_chunk_count`
- `pair_chunk_forward_ms`

## Concrete Work

1. Add `GameRunRequest`, `GameRunResult`, and any narrow per-game state objects needed by `GameRunner`.
2. Add `SelfPlayContractBuilders` as an explicit dependency bundle instead of ad hoc worker calls.
3. Move the game loop from `SelfPlayWorker` to `GameRunner`.
4. Move replay record creation and validation to `records.py` and `record_writer.py`.
5. Move worker telemetry shape decisions to `telemetry.py`; worker only emits lifecycle/IPC events.
6. Replace worker direct inference/search/MCTS calls with `PolicyProvider`, `PairStrategy`, and `EngineAdapter`.
7. Remove worker architecture gates and pair-policy side effects.
8. Ensure dense, graph hybrid, and global graph self-play use the same `GameRunner` interface.
9. Keep RGSC restart/service logic lifecycle-oriented; it must not rebuild contracts or records privately.

## Mandatory Tests

- `Python/tests/selfplay/test_game_runner_interface.py`
  - dense, graph hybrid, and global graph fixtures run through the same `GameRunner` constructor shape
  - fake `PolicyProvider`, `PairStrategy`, `EngineAdapter`, `SelfPlayRecordWriter`, `SelfPlayTelemetrySink`, and `SelfPlayContractBuilders` can drive a deterministic game

- `Python/tests/selfplay/test_worker_lifecycle_only.py`
  - worker starts, heartbeats, accepts a run request, forwards it to `GameRunner`, reports result, and shuts down
  - worker cancellation/timeouts cleanly tear down IPC resources
  - worker does not assemble replay records or perform search calls in lifecycle tests

- `Python/tests/selfplay/test_selfplay_logging.py`
  - heartbeat, no-progress, game summary, policy timing, and pair summary events contain required fields
  - `ContractTrace` fields and required spans are present for a traced position
  - stall diagnosis identifies IPC wait, engine wait, record writer wait, and pair-budget issues separately

- `Python/tests/selfplay/test_record_writer.py`
  - replay records are assembled and validated outside `SelfPlayWorker`
  - record writer failures are surfaced through `GameRunResult` and telemetry

- `Python/tests/selfplay/test_no_worker_architecture_logic.py`
  - no architecture string checks in `worker.py`
  - no graph/candidate/pair chunking helpers in `worker.py`
  - no direct MCTS prior wiring in `worker.py`

- `Python/tests/search/test_pair_strategy_selfplay_integration.py`
  - default pair strategy scores zero pair rows
  - pair scoring occurs only through `PairStrategy`
  - pair caps and cap-hit telemetry are emitted when a diagnostic strategy is enabled

- RGSC lifecycle tests
  - restart/service continuity remains covered
  - RGSC does not own game-loop, replay assembly, or contract-building details

## Import Audits

Run and satisfy these audits exactly:

```text
rg "architecture|startswith\\(\"global_|pair_prior_mix|pair_head|GlobalHexGraphNet|build_model_from_config" Python/src/hexorl/selfplay/worker.py
rg "Candidate|PairAction|PAIR_ACTION|graph_token|graph_relation|chunk|MCTS|prior" Python/src/hexorl/selfplay/worker.py
rg "Replay|record|writer|json|np.save|open\\(" Python/src/hexorl/selfplay/worker.py
rg "hexorl\\.selfplay\\.worker" Python/src/hexorl/search Python/src/hexorl/contracts Python/src/hexorl/graph Python/src/hexorl/replay
```

Expected results:

- the first three commands return no production worker logic matches, except comments/docstrings only if unavoidable and explicitly justified
- `search/`, `contracts/`, `graph/`, and `replay/` do not import `SelfPlayWorker`
- `worker.py` may import `GameRunner`, lifecycle config, IPC helpers, and telemetry lifecycle types
- `game_runner.py` may import `contracts`, `engine`, `search`, `selfplay.records`, `selfplay.record_writer`, and `selfplay.telemetry`

## Artifacts

This phase must leave behind:

- `GameRunner` interface documentation in or near `selfplay/game_runner.py`
- `GameRunRequest` and `GameRunResult` typed contracts
- `SelfPlayContractBuilders` dependency bundle
- `SelfPlayRecordWriter` implementation or interface in `record_writer.py`
- structured self-play telemetry event definitions
- import-audit output captured in the phase PR or implementation notes
- tests listed above, with deterministic fixtures or fakes

## Hard Gates

- `SelfPlayWorker` contains no architecture checks.
- `SelfPlayWorker` contains no game-loop details.
- `SelfPlayWorker` contains no replay assembly.
- `SelfPlayWorker` contains no legal/history/D6/candidate/pair/graph construction.
- `SelfPlayWorker` contains no candidate, pair, or graph chunking.
- `SelfPlayWorker` contains no direct MCTS prior wiring.
- Pair scoring is impossible unless `PairStrategy` explicitly enables it.
- Default pair strategy reports zero pair rows scored.
- Dense, graph hybrid, and global graph self-play run through the same `GameRunner` interface.
- Heartbeat, no-progress, game summary, policy timing, pair summary, and `ContractTrace` telemetry are emitted and tested.
- Import audits pass.
- Relevant test suites pass:

```text
pytest Python/tests/selfplay Python/tests/search/test_pair_strategy_selfplay_integration.py
pytest Python/tests/inference Python/tests/replay
```

## Exit Criteria

- `SelfPlayWorker` is lifecycle/IPC only.
- `GameRunner` owns game execution through explicit V2 interfaces.
- Replay records are assembled and written outside the worker.
- Self-play stalls and slow phases are diagnosable from structured logs.
- No old worker-owned architecture, pair, graph, candidate, replay, or MCTS wiring remains.
