# Phase 03 Pre-Implementation Checklist

- Created: `2026-04-30T01:01:35-04:00`
- Git SHA: `5638e8bc6b20b2dc27821602d4fa1f5adac9b4f8`
- Branch: `codex/phase-03-model-registry-specs`
- Source docs: `Docs/refactor/phases/PHASE_03.md`, `Docs/MODULAR_HEXO_ARCHITECTURE_REDESIGN_V2_20260429.md`, `Docs/refactor/V2_REQUIREMENTS_MATRIX.md`
- Scope: implement and close Phase 03 rows `V2-030` through `V2-035`.

## Goal

Replace runtime architecture-string model wiring with `hexorl.models` registry/spec/capability ownership, route training through `TrainAdapter`, and make checkpoint save/load/inspect strict through `CheckpointManager`.

## Success Criteria

- [ ] `V2-030`: runtime imports use `hexorl.models`, not `hexorl.model`; `Python/src/hexorl/model/` is deleted or absent from runtime.
- [ ] `V2-031`: every registered family exposes complete descriptor facets: model builder, train adapter factory, inference adapter manifest, policy provider, loss plan, default recipe, tune space, and checkpoint manifest.
- [ ] `V2-032`: trainer resolves a `TrainAdapter` from the registry and contains no architecture branches, model-class branches, or output-key behavior inference.
- [ ] `V2-033`: pair target training validation rejects invalid opening, first-placement, second-placement known-first, stale legal table, corrupt masks, and non-finite cases.
- [ ] `V2-034`: `CheckpointManager` owns strict save/load/inspect and runtime duplicate checkpoint cleanup is removed.
- [ ] `V2-035`: training debug bundle proves replay -> contracts -> tensors -> targets -> outputs -> loss inputs, with mutation and corruption guards.
- [ ] Import/deletion audits pass for `hexorl.model`, architecture gates, checkpoint cleanup, and pair consumption bypasses.
- [ ] Required artifacts are produced under `Docs/refactor/artifacts/phase_03/`.

## Constraints

- Do not add a `hexorl.model` compatibility facade or alias module.
- Any migration support for old checkpoints or architecture names must live outside runtime.
- Runtime family selection must go through registry/spec validation, not architecture-name heuristics.
- Capabilities describe model outputs; they do not decide MCTS pair consumption.
- Pair consumption remains Phase 05-owned; Phase 03 may validate pair training targets only.
- Checkpoint loading is strict by default; no silent `_orig_mod` stripping, prefix cleanup, partial load, or old-name remapping in runtime.
- `rg` is preferred, but local `rg.exe` access has failed with access denied in this environment; use `git grep` and record that fallback.

## Required Evidence

- Interface freeze notes in `checks/interface_freeze_notes.md`.
- Fixture/artifact plan in `fixtures_or_references/phase03_fixture_plan.md`.
- CI routing plan in `commands/ci_routing_plan.md`.
- Command transcripts with exit codes in `commands/` and `test_output/`.
- Import/deletion audits in `import_audits/`.
- Deletion manifest in `deletion_manifest/phase03_deletion_manifest.md`.
- Registered family capability list and fake-family extension proof.
- Checkpoint manifest round-trip and inspect-without-weights proof.
- Training debug bundle and performance profile.
- Adversarial review, completion packet, evidence reconciliation, and exit gate report.

## Stop Rules

- Stop before coding if deleting `hexorl/model` would remove the only runtime model path without an already-frozen `hexorl.models` replacement.
- Stop if a required Phase 03 invariant can only pass by retaining a runtime compatibility shim.
- Stop if exact phase tests would need to be skipped, xfailed, flaky-only, or manual-only.
- Stop if checkpoint compatibility requires silent cleanup in runtime instead of offline migration.
- Stop if train adapter performance evidence cannot be produced for touched hot paths.

No stop rule is active at scope freeze. Current audits show broad legacy usage, but every usage has a Phase 03 replacement target.

## Matrix Row Checklist

| Row | Owner | Required close evidence |
|---|---|---|
| `V2-030` | `Python/src/hexorl/models/` | registry/spec/capability tests, no `hexorl.model` runtime imports, deletion manifest |
| `V2-031` | `models/registry.py`, `models/families/*` | descriptor matrix, fake-family extension proof |
| `V2-032` | `train/adapters.py`, `train/trainer.py` | trainer no-branch audit, one-batch family smoke |
| `V2-033` | `train/adapters.py`, losses | pair target validation tests for phase/known-first/D6/corruption |
| `V2-034` | `models/checkpoint.py` | manifest round-trip, inspect without weights, strict rejection tests |
| `V2-035` | `train/adapters.py`, debug artifacts | single-position training debug bundle, mutation/corruption tests |
