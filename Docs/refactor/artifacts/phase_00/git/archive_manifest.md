# Phase 00 Archive Manifest

Date: 2026-04-29

Scope: inventory-only archive manifest for checkpoints, replay data, runs,
dashboard artifacts, tuning results, configs, fixtures, and restore instructions.

Constraint decision: no artifact bytes were copied. Existing local files are recorded
as linked in place where present. Large artifacts remain under their current paths.

Status values:

| Status | Meaning in this manifest |
|---|---|
| copied | A duplicate was produced under `Docs/refactor/artifacts/phase_00/`. |
| linked-in-place | Existing workspace artifact was inspected and is referenced by path only. |
| skipped | Not copied because Phase 00 S4 owns inventory only or because copying would violate the request constraints. |
| unavailable | No matching artifact was present, or the local path cannot be resolved/read as an artifact source. |

## Inspection Summary

| Artifact area | Current workspace evidence | Archive status | Notes |
|---|---|---|---|
| Run tree | `runs/` contains 62 directories, 355 files, and 11,952,443,683 bytes total. | linked-in-place; skipped copying | Large local baseline tree. Restore from the same relative `runs/` layout. |
| Checkpoints | 76 checkpoint files matched `runs/**/*.pt` or `runs/**/*.pth`. Representative paths include `runs/ablations_priority_20260427_v2/baseline_128x16_noise025/epoch_0001.pt`, `runs/restnet_sparse_stage0_epoch10_stable_20260428/epoch_0010.pt`, and `runs/phase3_autotune_smoke/trials/cal_best_current_33/epoch_0001_ema.pt`. | linked-in-place; skipped copying | Checkpoints are large. No strict manifest copy was created by this S4 pass. |
| Replay data | 42 `dashboard.sqlite3` files under `runs/` contain dashboard/game/position/replay metadata. No standalone `.npz`, `.npy`, `.pkl`, `.parquet`, `.db`, or `.sqlite` replay snapshot files were found by the inspection commands. | linked-in-place for SQLite-backed replay metadata; unavailable for standalone replay snapshots | Current `RingBuffer` runtime replay appears in-memory or embedded in run/dashboard artifacts rather than as a separate durable replay archive. |
| Dashboard artifacts | 42 `dashboard.sqlite3` files under `runs/`; root dashboard process logs `dashboard-lan-proxy.*.txt`, `dashboard-mixed.*.txt`, `dashboard-start.*.txt`, and `monitor-mixed.*.txt`; several run-local dashboard logs and pid files. No image artifacts matched `runs/**/*.{png,jpg,jpeg,webp,gif,svg}`. | linked-in-place; skipped copying; image artifacts unavailable | SQLite files are the primary dashboard baseline artifacts. |
| Tuning results | Under `runs/`: `events.jsonl` 46, `summary.jsonl` 19, `scores.jsonl` 12, `trial.json` 20, `manifest.json` 9, `PHASE3_AUTOTUNE_REPORT.md` 8, `state.json` 2, `static_search_space.json` 2, `final_selection.json` 1, `suite_manifest.json` 4, `suite_summary.jsonl` 3. | linked-in-place; skipped copying | `runtime_sweep_results.jsonl`, `runtime_sweep_cache.json`, and `bohb_sampler.json` were not present in the counted run artifacts. |
| Config files | Six tracked config files in `Configs/`: `default.toml`, `default_config.toml`, `production.toml`, `reproducible.toml`, `small_test.toml`, `wsl_speed_probe.toml`. Run-local config-bearing artifacts include seven `config.resolved.json`, seven `variant.json`, and twenty `trial.json` files. | linked-in-place; skipped copying | Future config hashes should reference both tracked configs and run-local resolved configs. |
| Fixtures and regression seeds | Tracked fixture-like files found: `crates/hexgame-core/proptest-regressions/tests/threats.txt` and `crates/hexgame-core/tests/encoder.proptest-regressions`. Test fixtures are otherwise inline in `Python/tests/*.py`. | linked-in-place for tracked regression files and inline tests; unavailable for standalone golden fixture bundle | Phase 00 verification inventory must create/point to explicit golden and corruption fixtures separately. |
| Phase 00 artifacts | `Docs/refactor/artifacts/phase_00/` subdirectories exist. Before this S4 pass, no files were present under `Docs/refactor/artifacts/phase_00/git/` or `Docs/refactor/artifacts/phase_00/inventory/`. | linked-in-place for directories; unavailable for missing prior files | This manifest and `inventory/deletion_legacy_inventory.md` were created by S4 inventory work. |
| Ext4 active pointer | `runs/phase2_phase3_autotune_overnight_20260428_ext4_active` is a reparse-point file with no resolved `Target` in PowerShell output. | unavailable | Treat as an unresolved external/WSL pointer. Restore requires the external target path from WSL/run handoff notes, not this Windows workspace alone. |

## Checkpoint Inventory

Status: linked-in-place; skipped copying.

Observed classes:

- Ablation checkpoints under `runs/ablations_priority_20260427*`.
- RestNet sparse checkpoints under `runs/restnet_sparse_stage0_epoch10_*`.
- Phase 3 autotune smoke/debug checkpoints under `runs/phase3_*`.
- Speed-probe checkpoints under `runs/train_speed_*` and `runs/wsl_speed_probe_auto_opt`.

Restore instructions:

1. Restore the repository and preserve the relative `runs/` tree.
2. Use checkpoint paths as recorded in run `LATEST.json`, `summary.jsonl`, `trial.json`, dashboard SQLite rows, or the paths listed by `Get-ChildItem -Recurse -File runs -Include *.pt,*.pth`.
3. For current runtime restore, use existing loader entrypoints only as baseline reproduction. For V2 closure, convert through the Phase 03 offline migration/`CheckpointManager` path and require strict manifest validation.

Unavailable checkpoint classes:

- Strict V2 `CheckpointManager` manifests are not present yet.
- External WSL/ext4 checkpoints referenced through the unresolved reparse point are unavailable from this workspace inspection.

## Replay Data Inventory

Status: linked-in-place for SQLite-backed run data; unavailable for standalone old-buffer snapshots.

Observed classes:

- `dashboard.sqlite3` files under run directories contain replay/game/position records used by dashboard and reports.
- `events.jsonl`, `summary.jsonl`, and `scores.jsonl` provide replay/training/tuning telemetry around generated positions.

Skipped or unavailable:

- No standalone durable `RingBuffer` snapshot file was found.
- No `*.npz`, `*.npy`, `*.pkl`, or `*.parquet` replay archive was found in `runs/`.
- In-memory buffer state from prior processes cannot be archived by this manifest.

Restore instructions:

1. Restore `runs/**/dashboard.sqlite3` beside the matching run metadata.
2. Re-index or open through existing dashboard baseline paths to inspect game/position rows.
3. Do not treat old SQLite/replay data as V2 replay truth. Phase 07 must migrate or reject old records explicitly through a validated replay codec.

## Runs Inventory

Status: linked-in-place; skipped copying.

Observed run roots include:

- `runs/ablations_priority_20260427`
- `runs/ablations_priority_20260427_v2`
- `runs/ablations_priority_20260427_v3_post_opt`
- `runs/phase2_phase3_autotune_overnight_20260428`
- `runs/phase3_48h_autotune_20260428`
- `runs/phase3_autotune_smoke`
- `runs/restnet_sparse_stage0_epoch10_stable_20260428`
- `runs/wsl_speed_probe*`

Restore instructions:

1. Restore the entire run root when reproducing dashboards, tuning reports, or checkpoint provenance.
2. Keep `events.jsonl`, `summary.jsonl`, `LATEST.json`, `trial.json`, `config.resolved.json`, checkpoint files, and `dashboard.sqlite3` together.
3. Treat pid files and live-process markers as historical only; do not reuse them to restart processes.

Unavailable run classes:

- The unresolved ext4 active pointer cannot be restored from Windows workspace contents alone.

## Dashboard Artifact Inventory

Status: linked-in-place; skipped copying.

Observed classes:

- Dashboard SQLite stores: 42 `dashboard.sqlite3` files.
- Root dashboard logs: `dashboard-lan-proxy.err.txt`, `dashboard-lan-proxy.out.txt`, `dashboard-mixed.err.txt`, `dashboard-mixed.out.txt`, `dashboard-start.err.txt`, `dashboard-start.out.txt`.
- Run-local dashboard logs and pid files under several `runs/wsl_speed_probe*` and `runs/restnet_sparse_stage0_epoch10_*` roots.

Skipped or unavailable:

- No rendered dashboard images/screenshots were found under `runs/`.
- Root `monitor-mixed.*.txt` files exist but are zero bytes.

Restore instructions:

1. Restore `dashboard.sqlite3` with its sibling run metadata.
2. Restore dashboard log files only for baseline diagnostics.
3. Regenerate screenshots/images from restored dashboard routes if visual evidence is needed later.

## Tuning Results Inventory

Status: linked-in-place; skipped copying.

Observed classes and counts:

| Artifact name | Count |
|---|---:|
| `events.jsonl` | 46 |
| `summary.jsonl` | 19 |
| `scores.jsonl` | 12 |
| `trial.json` | 20 |
| `manifest.json` | 9 |
| `PHASE3_AUTOTUNE_REPORT.md` | 8 |
| `state.json` | 2 |
| `static_search_space.json` | 2 |
| `final_selection.json` | 1 |
| `suite_manifest.json` | 4 |
| `suite_summary.jsonl` | 3 |

Unavailable classes:

- `runtime_sweep_results.jsonl`
- `runtime_sweep_cache.json`
- `bohb_sampler.json`

Restore instructions:

1. Restore a tuning root as a complete directory.
2. Read `manifest.json`, `state.json`, and `static_search_space.json` first.
3. Rehydrate trial evidence from each `trial.json`, `events.jsonl`, `scores.jsonl`, and checkpoint path.
4. For V2 Phase 08, do not mutate these raw configs in place; convert evidence into typed `ModelRecipe`, `RuntimeSpec`, and tuning manifests.

## Config Inventory

Status: linked-in-place; skipped copying.

Tracked configs:

- `Configs/default.toml`
- `Configs/default_config.toml`
- `Configs/production.toml`
- `Configs/reproducible.toml`
- `Configs/small_test.toml`
- `Configs/wsl_speed_probe.toml`

Run-local config-bearing files:

- `config.resolved.json`: 7
- `variant.json`: 7
- `trial.json`: 20

Restore instructions:

1. Restore tracked `Configs/*.toml` from git or this workspace.
2. Restore run-local `config.resolved.json`, `variant.json`, and `trial.json` with their run roots.
3. Hash config-bearing inputs before any Phase 00 baseline command claims reproducibility.

Unavailable config classes:

- No Phase 00 config hash files existed before this S4 pass.
- External WSL resolved configs behind the unresolved reparse point are unavailable.

## Fixtures Inventory

Status: linked-in-place for tracked regression files and inline test fixtures; unavailable for standalone V2 golden fixture bundle.

Observed tracked fixture/regression files:

- `crates/hexgame-core/proptest-regressions/tests/threats.txt`
- `crates/hexgame-core/tests/encoder.proptest-regressions`
- Inline Python fixtures in `Python/tests/*.py`

Restore instructions:

1. Restore tracked fixtures from git.
2. Preserve test files with their inline fixture helpers.
3. Phase 00 verification inventory must separately name golden positions, negative/corruption cases, D6 variants, and mutation-risk payloads before later phases rely on them.

Unavailable fixture classes:

- No standalone golden-position bundle.
- No standalone replay corruption fixture bundle.
- No standalone dashboard/debug-bundle fixture directory.

## Copy/Link Decisions

| Artifact class | Copied | Linked in place | Skipped | Unavailable |
|---|---:|---:|---:|---:|
| Checkpoints | 0 | 76 | yes | strict V2 manifests; unresolved ext4 pointer |
| Replay data | 0 | 42 SQLite stores plus JSONL telemetry | yes | standalone old-buffer snapshots; standalone V2 replay records |
| Runs | 0 | 62 run directories | yes | unresolved ext4 active target |
| Dashboard artifacts | 0 | 42 SQLite stores plus logs | yes | rendered dashboard images/screenshots |
| Tuning results | 0 | counted JSON/JSONL/MD reports above | yes | runtime sweep cache/results and BOHB sampler files |
| Configs | 0 | 6 tracked TOML plus run-local JSON configs | yes | Phase 00 config hashes |
| Fixtures | 0 | tracked regression files and inline test fixtures | yes | standalone V2 golden/corruption/debug fixtures |

## Inspection Commands

All commands ran in `D:\Hexo\Hexo-RL-Project`.

| Command | Exit status | Purpose |
|---|---:|---|
| `git status --short` | 0 | Confirm starting dirty state. |
| `Get-ChildItem -Recurse -File runs ... | Measure-Object` variants | 0 | Count run files, checkpoint files, SQLite/replay/dashboard files, tuning result files, and total bytes. |
| `Get-ChildItem -Recurse -File Configs ...` | 0 | List tracked config inputs. |
| `Get-ChildItem -Recurse -File Python/tests,crates/...` | 0 | Find fixture/regression files. |
| `Get-Item runs/phase2_phase3_autotune_overnight_20260428_ext4_active ...` | 0 | Inspect unresolved reparse point. |
| `Get-ChildItem -Recurse -File runs -Include *.png,*.jpg,*.jpeg,*.webp,*.gif,*.svg \| Measure-Object` | 0 | Check for dashboard image artifacts. |

## Stop-Rule Assessment

Artifact copying is not necessary for this Phase 00 S4 assignment. The manifest records inventory status only, so the "do not copy large artifacts" stop rule is satisfied.
