# WSL Phase 3 Autotune Handoff

This note is for a new agent taking over the Hexo Phase 2/3 autotune run on the Windows host.

## Host And Paths

- Windows workspace: `D:\Hexo\Hexo-RL-Project`
- WSL distro: `Ubuntu-24.04`
- WSL repo copy used for training: `/root/Hexo-RL-Project-ext4`
- Active run root: `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4`
- Dashboard port in WSL: `8765`
- Local dashboard URL: `http://127.0.0.1:8765`
- LAN dashboard proxy URL: `http://192.168.68.62:8766`

The training run is intentionally on the WSL ext4 filesystem, not `/mnt/d`, to avoid Windows filesystem overhead.

## Active Run Intent

The current run is the mixed-family Phase 2/3 autotune comparison with fair high-sim dense CNN/ResTNet coverage:

- `max_game_moves=384`
- full-search MCTS sims: `512,800,1200,1600`
- low/PCR sims: `128,192,256,384`
- runtime sweep workers: `2,3,4,5,6`
- runtime sweep states: `768`
- ASHA resources: `2,5,10`
- champion minimum epochs: `20`
- calibration throughput gate: `0.35`

This was restarted because earlier 80/192 move caps could starve the value head, and earlier low-memory guards unfairly downshifted CNN/ResTNet sims.

## Starting Services

Use the Windows helper script from PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_wsl_phase3_services.ps1
```

That script installs and starts a WSL keepalive wrapper. The wrapper supervises:

- `scripts/run_phase3_48h_autotune.py`
- `python -m hexorl.cli dashboard`
- `scripts/monitor_phase_autotune.sh`

It restarts any of those if their PID file is stale or their process disappears.

To install the WSL wrapper without starting it:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_wsl_phase3_services.ps1 -NoStartProcess
```

## Why The Keepalive Exists

Launching WSL services with ad hoc `nohup ... &` from a short-lived `wsl.exe` command was unreliable on this host. The dashboard and supervisor appeared healthy for a short time, then the WSL distro unmounted and all PIDs became stale. `dmesg` showed WSL shutdown/unmount messages rather than Python tracebacks.

The keepalive script is started as a Windows-owned hidden `wsl.exe` process, so the WSL distro stays pinned and the child services have a live owner.

## Key Logs

All paths below are inside WSL:

- Keepalive/service lifecycle:
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/service_keepalive.log`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/service_keepalive.pid`
- Supervisor launch details:
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/launcher.log`
- Supervisor/autotune:
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/supervisor.log`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/supervisor.pid`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/events.jsonl`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/state.json`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/runtime_sweep_results.jsonl`
- Dashboard:
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/dashboard.log`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/dashboard.pid`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/dashboard_suite.sqlite3`
- Shell monitor:
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/monitor.log`
  - `/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4/monitor.pid`
- Human progress notes:
  - `D:\Hexo\Hexo-RL-Project\Docs\OVERNIGHT_PHASE3_AUTOTUNE_MONITOR_20260428.md`

## Health Checks

From Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/suite/status
Invoke-RestMethod http://192.168.68.62:8766/api/suite/status
Test-NetConnection 127.0.0.1 -Port 8765
```

From WSL:

```bash
RUN=/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4
ps -eo pid,ppid,stat,etime,%cpu,%mem,args | grep -E 'hexo_phase3_keepalive|run_phase3_48h|hexorl.cli dashboard|monitor_phase' | grep -v grep
tail -n 80 "$RUN/service_keepalive.log"
tail -n 80 "$RUN/launcher.log"
tail -n 80 "$RUN/supervisor.log"
tail -n 80 "$RUN/dashboard.log"
nvidia-smi
free -h
```

Expected healthy signals:

- Dashboard API responds on `127.0.0.1:8765`.
- `service_keepalive.log` has periodic snapshots with a listener on `0.0.0.0:8765`.
- `supervisor.log` shows active workers and progress lines.
- `nvidia-smi` shows RTX 4070 Ti activity during self-play/inference.
- Memory should not be saturating swap. If RAM approaches the WSL limit and swap fills, reduce worker pressure before lowering game length.

## Current Known Risks

- High-sim 384-move dense CNN/ResTNet trials can consume significant RAM because each Python worker has MCTS/replay/process overhead. This is expected to be heavy, but it should not automatically disqualify these families.
- On this 24 GB WSL / 12 GB CUDA host, 6-worker dense high-search probes can push WSL near reset territory. The current tuner caps low-memory WSL high-search non-graph runtime sweeps at 5 workers, records memory telemetry for every candidate, rejects old cache entries without memory telemetry, excludes memory-unsafe candidates from selection, and skips same-or-larger worker counts after an unsafe candidate.
- If the dashboard appears to "crash" at the same time all WSL PIDs go stale, treat WSL memory pressure as the first suspect, not the dashboard server. Check `service_keepalive.log`, `free -h`, and the latest `runtime_sweep_result.memory` block before just restarting.
- Earlier orchestration incorrectly pruned or underpowered CNN/ResTNet relative to graph. Current code keeps slower valid families in the comparison and scores speed rather than dropping them solely for moderate throughput differences.
- If the dashboard is down but WSL processes are still alive, inspect `dashboard.log` first. If all PIDs are stale and `dmesg` shows WSL unmount/shutdown, restart with `scripts\launch_wsl_phase3_services.ps1`.
- Avoid ad hoc PowerShell commands containing unescaped Bash variables like `$RUN`; PowerShell will expand them first and can corrupt paths/arguments. Prefer scripts or literal paths.

## Files Changed For This Setup

- `scripts/launch_wsl_phase3_services.ps1`
  - Windows-side service launcher and keepalive installer.
- `scripts/launch_phase3_48h_autotune.sh`
  - Default `max_game_moves=384`, `runtime_sweep_states=768`, plus `launcher.log` diagnostics.
- `Python/src/hexorl/cli.py`
  - Dashboard startup/shutdown/exception logging.
- `scripts/run_phase3_48h_autotune.py`
  - Fairer high-sim/full-length mixed-family autotune behavior, plus memory-aware runtime sweeps for WSL.
- `Python/src/hexorl/dashboard/app.py`
  - More complete suite status, stage, and score reporting.
- `Python/dashboard_frontend/src/main.tsx`
  - Dashboard auto-refresh and richer suite stats.
- `Python/dashboard_frontend/src/styles.css`
  - Scrollable tables for recent events/games and suite tables.

## Useful Recovery Commands

Restart only the logged service wrapper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_wsl_phase3_services.ps1
```

Check dashboard quickly:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/suite/status |
  Select-Object current_stage,current_trial_id,current_model,total_games,total_positions,last_event_name
```

Sync changed dashboard/autotune files from Windows workspace to WSL repo:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc "cd /root/Hexo-RL-Project-ext4 && \
  cp /mnt/d/Hexo/Hexo-RL-Project/scripts/launch_phase3_48h_autotune.sh scripts/launch_phase3_48h_autotune.sh && \
  cp /mnt/d/Hexo/Hexo-RL-Project/scripts/run_phase3_48h_autotune.py scripts/run_phase3_48h_autotune.py && \
  cp /mnt/d/Hexo/Hexo-RL-Project/Python/src/hexorl/cli.py Python/src/hexorl/cli.py && \
  cp /mnt/d/Hexo/Hexo-RL-Project/Python/src/hexorl/dashboard/app.py Python/src/hexorl/dashboard/app.py && \
  chmod +x scripts/launch_phase3_48h_autotune.sh && \
  . .venv-wsl/bin/activate && python -m py_compile Python/src/hexorl/cli.py Python/src/hexorl/dashboard/app.py scripts/run_phase3_48h_autotune.py"
```

Use that sync command only after intentional source edits on the Windows workspace.
