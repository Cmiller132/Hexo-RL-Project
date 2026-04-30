# Phase 01 CI Routing Plan And Required Commands

- Created: `2026-04-30T03:35:54Z`
- Git SHA: `4055227a880a3f6995bc2d18e30f61a11b4a7ef4`
- Scope: command plan only. Commands listed here are required for Phase 01 closure after implementation; they were not run in this setup pass unless also listed in `COMMAND_INDEX.md`.

## CI Routing

| Tier | Rows | Required checks | Closure rule |
|---|---|---|---|
| local | V2-010,V2-015,V2-016 | Focused contract unit tests, import-purity test, mutation tests | Supports development only; cannot close rows without PR/deep evidence. |
| pr_required | V2-010,V2-011,V2-012,V2-013,V2-014,V2-015,V2-016 | Rust fmt/test, maturin build, focused contracts/engine pytest, import audits | All must pass with transcripts before any Phase 01 row can move beyond planned/in_progress. |
| deep | V2-012,V2-013,V2-014,V2-016 | Broader Python tests touching dashboard/sampler/graph/self-play/tactical/replay/bootstrap consumers; Rust/Python parity suites | Required before claiming runtime consumption, deletion proof, and semantic parity. |
| scheduled | V2-015,V2-016 | Hot-path view/mutation benchmark, legal table throughput, history replay throughput, D6 transform throughput | Required if implementation changes hot paths or cached views. |
| artifact_only | V2-010,V2-011,V2-012,V2-013,V2-014,V2-015,V2-016 | Manifest, checklist, command index, audit plans, debug payload example | Never sufficient by itself to close implementation rows. |

## Required Command List

Build and Rust checks:

```powershell
cargo fmt --all -- --check
cargo test --workspace
cargo test --workspace --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
maturin develop --manifest-path crates/hexgame-py/Cargo.toml --features python
```

Focused Python checks:

```powershell
$env:PYTHONPATH="Python/src"
python -m pytest Python/tests/contracts Python/tests/engine -q
python -m pytest Python/tests/contracts/test_import_purity.py Python/tests/engine/test_rust_boundary.py -q
python -m pytest Python/tests/contracts/test_history.py Python/tests/engine/test_history_parity.py -q
python -m pytest Python/tests/contracts/test_legal.py Python/tests/engine/test_legal_provider.py -q
python -m pytest Python/tests/contracts/test_symmetry.py Python/tests/engine/test_d6_parity.py -q
python -m pytest Python/tests/contracts/test_mutation_safety.py Python/tests/contracts/test_schema_source_hash.py -q
```

Consumer verification checks:

```powershell
$env:PYTHONPATH="Python/src"
python -m pytest Python/tests/test_engine_smoke.py Python/tests/test_engine_invariants.py Python/tests/test_tactical_oracle.py -q
python -m pytest Python/tests/test_training_data_pipeline.py Python/tests/test_dashboard_replay_debug.py Python/tests/test_dashboard_foundation.py -q
```

Import audit checks:

```powershell
git grep -n "_engine" -- Python/src/hexorl Python/tests crates Docs/RUST_API.md Docs/refactor/rust_review
git grep -n -E "source=`"fallback`"|source='fallback'|python_fallback|fallback_source" -- Python/src/hexorl Python/tests crates
git grep -n -E "ThreatStatus|threat_status" -- Python/src/hexorl Python/tests crates
git grep -n -E "decode_compact|compact_history|encode_compact|from_compact|history_bytes" -- Python/src/hexorl Python/tests crates
git grep -n -E "legal_moves_for_stones|legal_rows|legal_bytes|build_legal|python legal|fallback legal" -- Python/src/hexorl Python/tests crates
git grep -n -E "d6|symmetry|transform_qr|rotate|reflect|axis_map" -- Python/src/hexorl Python/tests crates
git grep -n -E "protocol.rs|row_width|from_le_bytes|struct.unpack|np.frombuffer|legal_bytes|pair_bytes|history_bytes" -- Python/src/hexorl crates
```

PowerShell fallback pattern when `rg` is unavailable and a `git grep` expression needs filtering:

```powershell
Get-ChildItem -Path Python/src/hexorl,Python/tests,crates -Recurse -File |
  Select-String -Pattern "_engine" |
  Where-Object { $_.Path -notmatch "Python\\src\\hexorl\\engine" }
```

## Required Artifacts After Implementation

- Command transcripts under `Docs/refactor/artifacts/phase_01/commands/`.
- Import audit outputs under `Docs/refactor/artifacts/phase_01/import_audits/`.
- Single-position debug payload under `Docs/refactor/artifacts/phase_01/telemetry_samples/` or `fixtures_or_references/`.
- Deletion manifest under `Docs/refactor/artifacts/phase_01/deletion_manifest/`.
- Phase exit report under `Docs/refactor/artifacts/phase_01/exit_gates/`.

## Non-Claim Statement

This CI routing plan is not evidence that the commands pass. Passing transcripts must be added after implementation before any row can be claimed complete.
