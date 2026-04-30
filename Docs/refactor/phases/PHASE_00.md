# Phase 00 - Baseline Freeze, Guardrails, And Evidence

## Purpose
Freeze the post-Rust, pre-Python-foundation state, capture enough evidence to compare every later phase against it, and install the smallest immediate safety rails required by V2 before broader architecture work begins.

This phase must exceed the V2 Phase 0 outline. It is not just setup. It creates the audit trail, runtime transcripts, structured diagnostics, pair-policy guard, legacy inventory, and hard exit gates that keep the breaking refactor controlled.

## Source Of Truth
This phase implements and sharpens the Phase 0 requirements from `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`.

V2 rules that Phase 00 must enforce immediately:

- No subsystem should infer behavior from an architecture string.
- No pair scoring happens unless `PairStrategy` explicitly enables it.
- `global_xattn` default pair strategy is `none`.
- No full `A * (A - 1) / 2` pair scoring without a diagnostic strategy and cap.
- Self-play logs must make stalls, slow phases, pair scoring, inference waits, and contract mismatches diagnosable.
- Every runtime sweep must have a no-progress watchdog.
- Deprecated aliases and stale compatibility shims must be deleted by the owning phase, not preserved indefinitely.
- The current runtime must not be treated as a trusted oracle; later phases require independent or cross-validated verification, corruption tests, and mutation guards.

## In-Scope Repository Context
Baseline and inventory must cover at least:

- Python hotspots: `Python/src/hexorl/selfplay/worker.py`, `Python/src/hexorl/model/network.py`, `Python/src/hexorl/model/global_graph.py`, `Python/src/hexorl/config/schema.py`.
- Existing package families: `inference/`, `selfplay/`, `graph/`, `train/`, `eval/`, `dashboard/`, `tuning/`.
- Rust rule source: `crates/hexgame-core/`, `crates/hexgame-py/`.
- Current config, recipe, checkpoint, replay, run, dashboard, and autotune entrypoints.

Rust-specific baseline note:

- The starting Rust boundary already includes the Phase 2 hardening work from `Docs/refactor/rust_review/`.
- Baseline evidence must record the current Rust commit and test status separately from older pre-hardening assumptions.
- The inventory should classify Rust as the production rules source, not as an unquestioned oracle. Any suspected Rust failure must be narrowed with invariant hooks, FFI malformed-input tests, stale-token tests, and independent fixture checks rather than hidden Python fallback paths.

## Artifact Directory Requirements
Create and use only:

```text
Docs/refactor/artifacts/phase_00/
```

Required subdirectories:

```text
baseline/
commands/
config_hashes/
git/
inventory/
logs/
traces/
watchdog/
checks/
exit_gates/
```

Required manifest:

```text
Docs/refactor/artifacts/phase_00/MANIFEST.md
```

`MANIFEST.md` must list every artifact path, creation command, timestamp, git SHA, config hash when applicable, and owner. If an artifact cannot be produced, the manifest must record the exact reason and the exit gate that blocks progress.

No later phase may treat an undocumented local run, checkpoint, replay file, transcript, or benchmark result as baseline evidence.

## Required Deliverables
1. Baseline freeze report covering functional behavior, smoke performance, CI timings, config values, and known instability.
2. Git tag and archive manifest for the last pre-Python-foundation cutover commit, including the completed Rust refactor SHA it contains.
3. Command transcripts for all mandatory checks and smoke commands.
4. Config hashes for every baseline command that depends on config, recipe, checkpoint, or runtime flags.
5. Immediate `global_xattn` guard proving `pair_strategy=none` unless explicitly overridden by a named strategy.
6. Accidental pair scoring test that fails if pair rows are scored when pair strategy is absent or `none`.
7. Structured logging samples for self-play, inference policy evaluation, pair strategy summary, graph request summary, autotune trial lifecycle, scheduler decision, and no-progress events.
8. Trace samples containing the required timing spans from V2's `ContractTrace`.
9. No-progress watchdog smoke showing a stalled self-play or runtime sweep emits a diagnosable event and exits or aborts predictably.
10. Architecture-string dependency inventory.
11. Deletion and legacy inventory with owning phase for every fallback, duplicate helper, deprecated alias, and stale runtime path found.
12. Verification inventory listing golden positions, negative/corrupt cases, D6 variants, mutation-risk payloads, and independent oracle options for each boundary.
13. Hard exit gate report signed by the orchestrator before Phase 01 starts.

## Baseline Freeze
Freeze the baseline before any Python/project architecture cutover edits.

Required actions:

- Record current branch, git SHA, dirty status, submodule status if any, and relevant environment information.
- Tag the last pre-Python-foundation cutover commit with an explicit Phase 00 tag name and record the completed Rust refactor SHA it contains.
- Produce an archive manifest for important checkpoints, replay data, run outputs, dashboard artifacts, tuning results, configs, and fixtures.
- Record whether each archived artifact is copied, linked, intentionally skipped, or unavailable.
- Record restore instructions for every archived artifact class.

Required files:

```text
Docs/refactor/artifacts/phase_00/git/git_state.txt
Docs/refactor/artifacts/phase_00/git/tag.txt
Docs/refactor/artifacts/phase_00/git/archive_manifest.md
Docs/refactor/artifacts/phase_00/baseline/baseline_freeze.md
```

The freeze is invalid if the tag cannot be mapped to the command transcripts and config hashes used for the baseline.

## Baseline Command Transcripts And Config Hashes
Every baseline command must have:

- exact command line
- working directory
- git SHA
- start and end timestamp
- exit code
- stdout/stderr transcript or summarized log with path to full transcript
- config, recipe, checkpoint, and runtime flag hashes where applicable
- machine/runtime notes that affect performance interpretation

Required files:

```text
Docs/refactor/artifacts/phase_00/commands/COMMAND_INDEX.md
Docs/refactor/artifacts/phase_00/config_hashes/CONFIG_HASH_INDEX.md
```

Hash all config-bearing inputs with a stable digest and record the hash beside the command that used it. If a command has no config input, record `config_hash: none`.

## Immediate Pair-Strategy Guard
Install the smallest non-invasive guard that makes accidental pair scoring impossible for `global_xattn` by default.

Minimum behavior:

- `global_xattn` resolves to `pair_strategy=none` unless the config or recipe explicitly names another valid pair strategy.
- `pair_prior_mix`, pair-capable heads, architecture names, or checkpoint head presence must not enable pair scoring.
- Any attempt to score pair rows with `pair_strategy=none` must fail fast in tests or emit a hard runtime error before expensive scoring.
- Logs must report `pair_strategy`, `pair_rows_possible`, and `pair_rows_scored`.
- Default `global_xattn` self-play must report zero pair rows scored.

This is a temporary guard until the later `PairStrategy` owner exists, but it must be strict enough to prevent the V2 performance trap immediately.

## Accidental Pair Scoring Test
Add or update a focused test that proves pair scoring is opt-in.

Required cases:

- `global_xattn` with default settings scores zero pair rows.
- `global_xattn` with `pair_strategy=none` scores zero pair rows even when pair-capable heads exist.
- Nonzero `pair_prior_mix` alone does not score pair rows.
- Any full pair enumeration path requires an explicit diagnostic strategy and cap.

The test should be narrow and fast. It should fail on accidental scoring, architecture-name side effects, or head-presence side effects.

## Structured Logging And Trace Samples
Add non-invasive structured logging where needed to produce sample evidence. Do not build the final observability stack in this phase.

Required event samples:

```text
selfplay_worker_heartbeat
selfplay_phase_transition
selfplay_no_progress
selfplay_game_summary
policy_eval_timing
pair_strategy_summary
graph_request_summary
autotune_recipe_validation
autotune_trial_lifecycle
autotune_scheduler_decision
runtime_sweep_no_progress
inference_protocol_mismatch
contract_validation_failure
```

Required trace fields:

```text
trace_id
history_hash
model_family
phase
legal_count
candidate_count
pair_rows_total
pair_rows_scored
graph_token_count
graph_relation_count
timings_ms
warnings
```

Required timing spans when available:

```text
history_parse_ms
engine_replay_ms
legal_table_ms
tactical_oracle_ms
candidate_build_ms
pair_table_build_ms
graph_token_build_ms
graph_relation_build_ms
graph_tensorize_ms
ipc_pack_ms
ipc_wait_ms
queue_wait_ms
collate_ms
model_forward_ms
scatter_ms
decode_ms
pair_chunk_count
pair_chunk_forward_ms
```

Sample files must live under:

```text
Docs/refactor/artifacts/phase_00/logs/
Docs/refactor/artifacts/phase_00/traces/
```

Samples may be small, but they must be real outputs from commands recorded in `COMMAND_INDEX.md`.

## Verification Inventory
Create the verification plan before Phase 01 starts. This inventory exists because the current runtime is not trusted enough to be the sole oracle for the refactor.

Required inventory:

- golden positions with hand-audited compact histories, board state, side to move, legal moves, terminal status, and expected failure class
- D6 variants for each golden position, including inverse and composition expectations
- negative/corrupt cases for malformed history, illegal move order, duplicate occupancy, stale known-first placement, bad legal rows, bad masks, bad hashes, bad schema versions, non-finite model outputs, and replay corruption
- mutation-risk payloads, including NumPy arrays, Torch tensors, cached views, graph batches, replay records, inference transport buffers, search evaluations, and MCTS inputs
- independent or cross-validated oracle options for each boundary, such as Rust replay, contract validation, hand-audited fixtures, inverse D6 checks, replay round-trips, schema/hash checks, and owner-specific corruption tests
- planned single-position debug bundle sections for engine, contracts, D6, candidates, pairs, graph, targets, model inputs, model outputs, policy priors, MCTS, replay, dashboard, and autotune reports

Required file:

```text
Docs/refactor/artifacts/phase_00/checks/verification_inventory.md
```

The inventory must explicitly say where old-runtime comparison is allowed only as a weak signal and where it is forbidden as the sole proof.

## No-Progress Watchdog Smoke
Prove that a no-progress condition is observable and actionable.

Required smoke:

- Run a short self-play or runtime sweep command with a deliberately tiny watchdog threshold or controlled stall.
- Emit a structured `selfplay_no_progress` or `runtime_sweep_no_progress` event.
- Include last successful phase, last inference request if applicable, last engine operation if applicable, progress counters, elapsed time, pair strategy, pair rows scored, and suggested next diagnostic area.
- Exit, abort, or mark the trial failed predictably.

Required file:

```text
Docs/refactor/artifacts/phase_00/watchdog/no_progress_smoke.md
```

## Deletion And Legacy Inventory
Inventory legacy paths now so later phases delete them deliberately.

Required inventories:

- Python legal fallbacks outside explicit fixtures.
- Private compact-history parsers.
- Private D6 helpers.
- Python or Rust code that duplicates `crates/hexgame-py/src/protocol.rs` legal/history/pair byte decoding.
- Any Python call path that can reach old Rust MCTS panic wrappers, stringly errors, or tokenless root/batch APIs.
- Architecture-string gates and `startswith("global_")` behavior.
- Pair enablement based on `pair_prior_mix`, pair head presence, or architecture names.
- Candidate construction duplicated across worker, sampler, dashboard, graph, or training.
- Pair mini-contracts and parallel crop/global pair row builders.
- Graph code that combines semantic construction, tensorization, and collation ownership.
- Checkpoint prefix cleanup, deprecated aliases, and model factory switches.
- Inference server dispatch by architecture string.
- Trainer model-class branches.
- Dashboard private model-input reconstruction.
- Autotune scripts that mutate raw config fields or depend on family internals.
- Old replay/buffer runtime paths and stale conversion shims.

Each entry must include:

- file path
- symbol or pattern
- current behavior
- risk
- owning phase
- expected deletion or replacement
- blocking tests or evidence required before deletion

Required files:

```text
Docs/refactor/artifacts/phase_00/inventory/architecture_string_inventory.md
Docs/refactor/artifacts/phase_00/inventory/deletion_legacy_inventory.md
Docs/refactor/artifacts/phase_00/inventory/pair_policy_inventory.md
Docs/refactor/artifacts/phase_00/inventory/rust_python_boundary_inventory.md
```

## Mandatory Checks
Run and transcript these exact checks unless the manifest records a blocker:

```text
cargo fmt --all -- --check
cargo test --workspace
cargo test --workspace --release
cargo clippy --workspace --release -- -D warnings
maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python
pytest Python/tests/test_engine_smoke.py Python/tests/test_engine_invariants.py Python/tests/test_inference_server.py -q
pytest -q Python/tests
python -m hexorl.cli --help
```

Also run and transcript the team's current baseline smoke commands for:

```text
self-play
inference
training
autotune or runtime sweep dry-run
dashboard build if dashboard dependencies are present
```

Also create the verification inventory and transcript any lightweight checks used to build it. If a golden position or corruption case cannot be created in Phase 00, the inventory must assign it to the exact owner phase and explain why.

The command index must mark every check as one of:

```text
passed
failed-blocking
failed-known-baseline
skipped-blocking
not-applicable
```

No failed or skipped mandatory check may be waved through without an explicit hard exit gate decision.

## Parallel Subagent Split
- S1 Contracts/Schema: enumerate implicit data shapes, config hashes, recipe inputs, and architecture-string dependencies.
- S2 Engine/Runtime: map Rust/Python ownership boundaries, legal fallback paths, and no-progress watchdog behavior.
- S3 Models/Search: inventory model family checks, `global_xattn` defaults, pair-policy coupling points, and accidental pair scoring tests.
- S4 Data/Train/Eval: baseline replay-to-sampler-to-trainer path, training smoke, evaluation assumptions, and deletion owners.
- S5 Quality/Obs/Docs: command transcripts, structured logs, trace samples, verification inventory, artifact manifest, and exit gate report.

## Hard Exit Gates
Phase 00 is complete only when every gate below is satisfied or explicitly marked blocking.

Hard gates:

- Baseline git tag exists and maps to recorded git SHA.
- Archive manifest exists and covers important checkpoints, replay data, runs, configs, fixtures, and tuning outputs.
- `MANIFEST.md`, `COMMAND_INDEX.md`, and `CONFIG_HASH_INDEX.md` exist and are internally consistent.
- Exact mandatory checks are transcripted.
- Baseline self-play, inference, and training smokes are transcripted or blocked with reasons.
- `global_xattn` default reports `pair_strategy=none`.
- Default `global_xattn` scores zero pair rows.
- Pair scoring requires an explicit strategy in tests.
- Nonzero `pair_prior_mix` alone cannot enable pair scoring.
- Full pair enumeration is impossible without explicit diagnostic strategy and cap.
- Structured log samples exist for self-play progress, pair summary, policy timing, autotune lifecycle, scheduler decision, and no-progress.
- Trace sample includes pair row counts, graph counts, timings, and warnings.
- No-progress watchdog smoke emits an actionable event and exits, aborts, or marks failure predictably.
- Verification inventory exists and treats old-runtime comparison as insufficient by itself.
- Golden positions, negative/corrupt cases, D6 variants, mutation-risk payloads, and independent oracle options are mapped to owner phases.
- Architecture-string inventory exists.
- Deletion and legacy inventory exists with owning phases.
- Rust/Python rule boundary inventory exists.
- Rust/Python rule boundary inventory includes the completed Rust Phase 2 state, remaining suspicion points, protocol owners, MCTS token lifecycle, structured error gaps, and crash-containment requirements for panic/abort behavior.
- All known blockers have owners and next actions.

## Exit Criteria
The orchestrator may open Phase 01 only after publishing:

```text
Docs/refactor/artifacts/phase_00/exit_gates/PHASE_00_EXIT_REPORT.md
```

The exit report must state:

- baseline tag and SHA
- mandatory check status
- smoke command status
- pair guard status
- accidental pair scoring test status
- structured logging and trace sample status
- watchdog smoke status
- inventory completion status
- unresolved blockers
- explicit go/no-go decision for Phase 01

If any hard gate is incomplete, Phase 01 must not begin except for narrowly scoped fixes needed to complete Phase 00.
