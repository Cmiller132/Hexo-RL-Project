# V1 Memory Efficiency Profile

Run id: `optuna_mcts500_v1fresh_20260508_01`  
Snapshot time: `2026-05-08 07:45 America/New_York`  
Workspace: `D:\Hexo\Hexo-RL-Project`

## Current Status

Phase 1 is running under the compact V1 storage/runtime path with the intended fresh comparison candidates:

- `global_xattn_0:none`
- `global_graph768_champion:none`
- `global_pair_biaffine_0:sampled_joint_pair_v1`

Active command:

```powershell
D:\Hexo\Hexo-RL-Project\.venv\Scripts\python.exe -u scripts\run_phase1_optuna_scout.py --runs-root runs --run-id optuna_mcts500_v1fresh_20260508_01 --production --candidate-plan global_xattn_0:none --candidate-plan global_graph768_champion:none --candidate-plan global_pair_biaffine_0:sampled_joint_pair_v1 --max-game-moves 500 --phase1-mcts-simulations 512
```

The V1 architecture/search/training identity remains unchanged from `Docs\HEXORL_V1_ARCHITECTURE_PROPOSAL.md`. The memory work is storage/runtime efficiency only.

## Implemented Memory Changes

- Replaced per-position dashboard move history storage with schema v2.
- Removed `move_history_b64` from new position rows.
- Store final game history once in `games.final_history_b64`.
- Reconstruct replay board states from final game history plus `turn_index`.
- Rotated old schema-v1 dashboard artifacts to `dashboard_schema_v1_archive_20260508_073019`.
- Added strict compact V1 metadata schema v2.
- Store V1 training metadata as compact typed binary payloads instead of JSON object lists.
- Compress cold V1 metadata in the replay buffer with zlib and lazily decode sampled records.
- Added replay and runtime memory telemetry fields.
- Added V1 worker memory probe script.
- Updated the Codex heartbeat to monitor compact schema health, RAM headroom, dashboards, and stale orphan Python processes.

## Current Memory Snapshot

System memory:

| Metric | Value |
| --- | ---: |
| Total RAM | 31.16 GB |
| Free RAM | 20.87 GB |
| Used RAM | 10.29 GB |

GPU:

| Metric | Value |
| --- | ---: |
| VRAM used | 5014 MB |
| VRAM total | 12282 MB |
| GPU utilization | 30% |

Active Python processes:

| Role | PID | Working Set MB | Private MB | Notes |
| --- | ---: | ---: | ---: | --- |
| Phase 1 controller child | 24632 | 106.1 | 2379.7 | Owns active `run_phase1_optuna_scout.py` child process |
| Inference/self-play child | 68468 | 333.1 | 2780.0 | Multiprocessing child |
| Self-play worker child | 45004 | 389.9 | 1664.1 | Multiprocessing child |
| Self-play worker child | 81508 | 319.7 | 1596.0 | Multiprocessing child |
| Optuna dashboard | 72828 | 153.7 | 1657.3 | `optuna_dashboard` on port 8080 |
| Normal dashboard | 32632 | 145.6 | 1539.2 | Uvicorn dashboard on port 8765 |
| LAN proxy | 52676 | 5.9 | 22.4 | Proxy on port 8766 |
| Dashboard mirror child | 9188 | 9.2 | 12.0 | Suite mirror helper |

The active Phase 1 tree has modest resident usage compared with the pre-cleanup state. The largest measured problem was an orphaned stale `python -` process using about 24 GB private memory; it was not part of the active Phase 1 command or dashboard stack and was stopped. Free RAM increased from about 2 GB to about 22 GB afterward.

## Biggest Current Memory Users

The active memory users are now:

1. Windows multiprocessing child processes for self-play/inference.
2. Python virtual address reservations reported as private memory.
3. Dashboard service processes, mostly baseline Python/import overhead.
4. GPU model/inference state, about 5 GB VRAM.

The dashboard replay DB is no longer the dominant memory/storage issue. The new active compact V1 dashboard DB was about 0.117 MB at the snapshot and had schema v2 position columns:

```text
position_id, game_id, turn_index, player, root_value, policy_json, debug_json
```

For comparison, the archived pre-refactor V1 dashboard DB was about 800.9 MB.

## Efficiency Assessment

The clean storage refactor reduced dashboard/replay growth substantially. The remaining RAM cost is mostly process topology, not model size or replay metadata.

Most promising next reductions:

- Add automatic stale-process cleanup/reporting for orphaned high-memory Python processes.
- Avoid importing training/model modules in dashboard-only processes where possible.
- Investigate a thread/Rust worker-pool path for V1 self-play to reduce Windows `spawn` duplication.
- Re-run worker-count probes after the stale process cleanup so the result is not distorted by low free RAM.
- Consider raising workers only in a new lineage-safe run or after an explicit runtime-only migration plan, because changing `selfplay.num_workers` in this active run changes the config hash guard.

Not recommended:

- Reducing V1 pair budget, MCTS simulations, graph token budget, candidate identity, or model heads. Those would weaken or change the experiment instead of improving runtime efficiency.
- Lowering replay capacity as the primary fix. The compact metadata path addresses the memory pressure without discarding Phase 1 evidence.

## Operational Notes

- Keep free system RAM above 4 GB, preferably above 8 GB.
- Treat long-lived `python -` processes with no run/dashboard/mirror command line and high private memory as stale orphan suspects.
- Preserve Optuna DB, checkpoints, events, scorecards, and raw lineage.
- Only dashboard-derived replay artifacts may be regenerated during schema cleanup.
- Use Hexo scorecards and hard gates as production authority; Optuna values are secondary.

## Verification Performed

- Confirmed Phase 1 process tree is alive.
- Confirmed Optuna dashboard responds on `http://127.0.0.1:8080/`.
- Confirmed normal dashboard health responds on `http://127.0.0.1:8765/api/health`.
- Captured current system RAM, GPU memory/utilization, and Python process memory.
- Confirmed active V1 dashboard schema v2 position columns do not include `move_history_b64`.
