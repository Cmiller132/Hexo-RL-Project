# Phase 00 Scope And CI Freeze

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`
- Frozen scope: V2 rows `V2-000` through `V2-006` only.
- Phase 01 and later implementation was not started.
- Public interface changed in Phase 00: `ModelConfig.pair_strategy` and `ModelConfig.pair_strategy_max_pairs` are explicit guard fields; allowed strategies are `none` and `diagnostic_full_pair`.
- Runtime guard changed in Phase 00: `SelfPlayWorker` enables pair scoring only when `pair_strategy != none`; pair scoring helpers reject non-positive caps.
- Fixture/artifact plan: use existing verification inventory as the Phase 01+ fixture source map; no later phase may treat current runtime output as sole oracle.

## Agent Assignment Freeze

| Agent | Goal | Success criteria | Constraints | Required evidence | Stop rules |
|---|---|---|---|---|---|
| S1 Contracts/Schema | Inventory implicit data shapes, config inputs, architecture strings, verification risks | Inventory and verification map with owners | Docs/artifacts only | `inventory/architecture_string_inventory.md`, `checks/verification_inventory.md` | Stop before implementation beyond inventory. |
| S2 Engine/Runtime | Map Rust/Python boundary and watchdog status | Boundary inventory plus watchdog evidence | Non-overlapping artifacts | `inventory/rust_python_boundary_inventory.md`, `watchdog/no_progress_smoke.md` | Stop if real runtime watchdog needs later-phase owner. |
| S3 Models/Search | Map pair-policy coupling and accidental pair scoring surface | Pair inventory and guard/test recommendations | Non-overlapping artifact | `inventory/pair_policy_inventory.md` | Stop before long self-play or code edits. |
| S4 Data/Train/Eval | Map replay/train/eval/dashboard/autotune legacy paths | Deletion-owner inventory | Non-overlapping artifact | `inventory/deletion_legacy_inventory.md`, `git/archive_manifest.md` | Stop before deleting owner-phase paths. |
| S5 Quality/Obs/Docs | Seed command/hash/manifest/exit templates | Templates for orchestrator to fill | Non-overlapping control-plane artifacts | S5 template files | Stop if real evidence missing. |
| Orchestrator | Integrate Phase 00 guard, run evidence, reconcile artifacts | Rows V2-000..V2-006 complete with evidence | Phase 00 only | Command transcripts, logs, traces, audits, performance, exit report | Stop if any hard gate remains blocking. |

## CI Routing Plan

- `pr_required`: Rust fmt/test/clippy, maturin extension build, focused engine/inference pytest, self-play no-pair smoke, inference smoke, dashboard build, watchdog expected abort.
- `deep`: Rust release tests, full Python tests, training smoke, autotune/runtime dry-run.
- `scheduled`: Rust MCTS/tactical/encoding benches captured locally now and promoted to stable-runner comparison in Phase 09.
- `artifact_only`: inventories, verification map, structured sample logs, trace sample, deletion manifest, contract/config example docs.
