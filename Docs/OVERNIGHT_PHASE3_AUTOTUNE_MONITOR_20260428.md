# Overnight Phase 2/3 Autotune Monitor - 2026-04-28

## Current Status

- Active WSL ext4 run root: `/root/hexo_runs/phase2_phase3_autotune_overnight_20260428_ext4`
- Windows symlink: `runs/phase2_phase3_autotune_overnight_20260428_ext4_active`
- Stable search shape as of ~10:25 EDT: graph-only long search with 3 workers, PBT population auto-capped to 3 on low-RAM WSL, and non-calibration `max_game_moves <= 80`.
- WSL2 RAM was raised via `C:\Users\epicm\.wslconfig` from the default ~15 GiB to `24GB` memory plus `12GB` swap; verified inside WSL as ~23.46 GiB total.
- Fresh restarted run began at ~10:26 EDT. Dense calibration is active and progressing with GPU around `72%`, ~3.2 GiB GPU memory used, and ~14 GiB WSL RAM still available after the first few minutes.
- At ~10:33 EDT, dense calibration completed in `190.8s`, ResTNet calibration completed in `162.5s`, graph calibration was running, no warnings were present, and swap remained unused.
- At ~10:39 EDT, graph calibration completed in `85.7s` for 1024 positions. Dense and ResTNet were quarantined by graph-relative throughput gates and their runtime state was released. ASHA started graph-only with six graph trials.
- At ~10:50 EDT, graph ASHA trials were completing under budget: `asha_00` at `358.0s` / ~`1355 positions/min`, `asha_01` at `324.5s` / ~`995 positions/min`. No warnings and WSL swap still unused.
- At ~11:05 EDT, the suite-aware dashboard was launched at `http://127.0.0.1:8765` against `/root/hexo_runs/phase2_phase3_autotune_overnight_20260428_ext4`. It aggregated `19` trial databases, `2678+` saved games, and ranked best checkpoints from the live run root.
- At ~11:18 EDT, the run was restarted with the full pair-policy implementation synced into the WSL ext4 repo and `--max-game-moves 192`. The first post-fix epoch completed as `6` games / `1152` positions, confirming the active cap is now `192` rather than the old `80`/`128` clamp.
- Learning-curve policy update: 1-4 epoch scores are treated as stability/throughput screens only, not proof of strategic strength. ASHA should now use `2,5,10` epoch rungs, promote broadly, and promising/champion candidates should reach `20+` epochs before final trust.
- The earlier 3-worker / 96-move guard got through ASHA, but PBT later wedged when it reused calibration-era trials and reached move `82+` / `114+` under full WSL memory pressure.
- Current intervention: restart from a clean WSL state with PBT restricted to eligible/unquarantined families and pruned/quarantined trial runtime released immediately.
- Stable capped graph throughput before the PBT stall was roughly `950-1870 self-play positions/min` across graph variants; PBT graph `256 token / 1 layer` hit `~1873 positions/min` before the next population member stalled.

## Interventions

1. Moved the active run to the WSL ext4 repo copy to avoid `/mnt/d` DrvFS instability.
2. Disabled Codex app heartbeat automation and used terminal/session monitoring instead.
3. Added a calibration throughput gate so slow latency-sensitive families are excluded from long ASHA/PBT.
4. Added a 12 GB / 16-core host guard blocking combined ResTNet+sparse policy, after it repeatedly stalled with 30s inference timeouts.
5. Added a 12 GB graph-capable host guard blocking non-graph sparse controls, after sparse CNN also stalled in long ASHA despite passing short calibration.
6. Removed graph sparse-prior stage 1 from the overnight ASHA ladder on this host after stage 1 measured much slower than stage 0.
7. Added dense long-search worker caps, then escalated to graph-relative quarantine for dense CNN because dense long ASHA still timed out even at 4 workers.
8. Added hard pruning for ASHA/PBT epochs exceeding `1.20x` the target epoch budget.
9. Tightened the graph-relative calibration gate to quarantine dense CNN from long search when graph is much faster; dense calibration remains recorded as a baseline.
10. Removed graph sparse-prior stage 1 from the overnight ASHA ladder on this 12 GB / 16-core host after repeated measurements showed it was safe but too slow.
11. Added a non-calibration long-trial cap of `max_game_moves <= 96` on 12 GB CUDA hosts after PBT hit repeated inference timeouts around moves `117-118`.
12. Added a global 12 GB CUDA worker cap of 3 workers / small batches after WSL reported only ~15 GiB RAM and the kernel OOM-killed Python workers.
13. Added host memory detection to the runtime profile so WSL RAM pressure is visible in manifests and reports.
14. Added a low-RAM WSL guard that auto-caps PBT population from `8` to `3` on this 12 GB CUDA / ~15 GiB WSL host.
15. Fixed PBT seeding so it only uses currently eligible, unquarantined families; previously it carried dense CNN and ResTNet calibration trials into long PBT even after graph-relative quarantine.
16. Fixed per-epoch move caps so calibration-created configs are re-capped when reused in ASHA/PBT/champion stages; this prevents 128-move calibration configs from leaking into long PBT.
17. Added runtime release for pruned/quarantined trials so old trainers and replay buffers do not accumulate across ASHA rungs and PBT.
18. Tightened the low-RAM non-calibration game cap to `80` moves after the latest stall showed repeated timeouts at move `82` and WSL memory+swap saturation.
19. Added `C:\Users\epicm\.wslconfig` with `memory=24GB`, `swap=12GB`, and `processors=32`, then restarted WSL and verified the new memory limit.
20. Fixed dashboard visibility for autotune suites: games and checkpoints were already being saved per trial in `trials/<trial>/dashboard.sqlite3`, but the dashboard only read one default DB. Added suite aggregation endpoints, a Suite tab, best-model ranking, recent saved games, and run-aware replay loading.
21. Synced the full Phase 2 pair-policy path into the WSL ext4 repo before PBT could sample `graph_tactical`: pair targets, pair replay tensors, `PairPolicyHead`, pair loss, trainer metrics, and tactical candidate recall gates are now present in the active codebase.
22. Raised the autotune game length to `192` and removed the low-memory override that forced non-calibration runs to `96`/`80`. Also changed the config application path so `--max-game-moves` is the actual requested length instead of `min(base_config, requested)`.
23. Changed the tuning schedule to avoid overvaluing early learning: ASHA resources are `2,5,10`, ASHA promotion keeps the top half, PBT mutates after `5` epochs, and champion selection now prefers candidates with at least `20` epochs.
24. Added a per-config startup runtime sweep before a trial's first real epoch. It runs short self-play probes, tests worker counts from `2,3,4,5` by default, scores by positions/min with GPU snapshots, caches the best tuple by model/static recipe, and applies the selected `num_workers`, `batch_size_per_worker`, `max_batch_size`, and `max_wait_us` automatically.
25. Loosened the calibration throughput quarantine from graph-relative `0.85x` to configurable `0.35x` by default, so current CNN and ResTNet stay in the comparison unless they are more than roughly `3x` slower or fail a hard safety gate.
26. Expanded the dashboard suite view with current model/activity, positions/sec, readable timestamps, model/config inspection, architecture summaries, runtime sweep details, and checkpoint config metadata.

## Performance Findings

- Graph stage 0 is the only family that is both fast and stable on this machine overnight.
- Dense CNN short calibration is stable at roughly `545-565 positions/min`, but long ASHA with high simulation counts repeatedly times out in late moves.
- ResTNet short calibration is stable but slower, roughly `350-400 positions/min`, and is consistently below the graph-relative throughput gate.
- After WSL RAM expansion, fresh calibration measured dense at ~`322 positions/min`, ResTNet at ~`378 positions/min`, and graph at ~`717 positions/min`; graph remained the clear long-run candidate.
- Sparse CNN short calibration can pass candidate recall, but long ASHA stalls with inference timeouts and poor GPU utilization.
- Graph sparse-prior stage 1 can be made safe enough to measure at `256 tokens / 1 layer`, but it is CPU-bound and scored poorly versus graph stage 0.
- Best graph stage-0 ASHA trials are currently in the `1000-1800 positions/min` range depending on token budget and simulation count.
- Graph-only PBT is currently the first configuration that has run for an extended period with no timeout warnings.
- Late-game positions are the remaining expensive edge: timeouts repeatedly appeared after move ~110, then again at move ~82 under full memory pressure, so overnight optimization now avoids spending most wall time there.
- The limiting resource is not just GPU: WSL memory pressure can kill Python workers. The run now prioritizes stable 3-worker graph throughput over unstable 4-worker peaks.
- The best practical overnight profile on this WSL setup is graph stage-0, 3 self-play workers, small inference batches, and shorter long-search games. This leaves some raw GPU headroom but avoids CPU/RAM/inference-server collapse.
- Worker count is no longer only heuristic. New trials should emit `runtime_sweep_start`, `runtime_sweep_result`, and `runtime_sweep_selected` events, plus `runtime_sweep_cache.json`, so different model sizes can select different CPU/GPU feeding settings.

## Next Watch Items

- Confirm the restarted run logs `pbt_population=3` and no blocked-family PBT population members.
- Confirm no self-play warnings above move `80` in non-calibration stages.
- Watch for non-finite losses after long graph training epochs.
- Watch GPU utilization and memory during graph `512 token / 2 layer` configs; these are stable so far but slower than `256 token / 1 layer`.
- Watch whether 192-move games reintroduce late-game inference timeout warnings; if they do, prefer reducing workers/batch/wait before lowering the game length again.
- Watch for the first `graph_tactical` trial to confirm `loss_pair_policy`, `pair_policy_top1_acc`, and decisive candidate recall metrics are emitted.
- Treat sub-10-epoch results as screening signals only. Use 10 epochs for early architecture comparison, then require 20+ epochs on promising models before trusting strategy/strength conclusions.
- If graph-only PBT remains stable for an hour at 192 moves, let the 48h run continue without further narrowing.
- Confirm runtime sweeps choose a stable worker count for graph 256/384/512 token configs; if `5` workers wins but warnings return in late-game positions, narrow the worker list or increase the probe length before reducing max game length.
- The next clean run should be treated as a mixed-family comparison again: graph may still win, but CNN and ResTNet should no longer be dropped just for a moderate calibration speed gap.
