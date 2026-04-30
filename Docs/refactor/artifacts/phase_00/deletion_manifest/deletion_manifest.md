# Phase 00 Deletion Manifest

- Created: `2026-04-30T03:31:49Z`
- Git SHA: `9d7a24ca196e2c3343d34cbd6721ec96bb195d96`

| Item | Phase 00 action | Owner phase for deletion/replacement | Evidence |
|---|---|---|---|
| Implicit pair scoring from architecture, head presence, or `pair_prior_mix` | Guarded in runtime by explicit `pair_strategy` and positive cap | Phase 05 final `PairStrategy` owner | `Python/src/hexorl/config/schema.py`, `Python/src/hexorl/selfplay/worker.py`, `Python/tests/test_config_and_guardrails.py`, full pytest transcript |
| Architecture-string runtime gates | Inventoried, not deleted in Phase 00 | Phases 03, 04, 05, 06, 08, 09 | `inventory/architecture_string_inventory.md`, `import_audits/architecture_string_audit.txt` |
| Python legal/history/D6 fallbacks | Inventoried, not deleted in Phase 00 | Phases 01, 02, 07, 08, 09 | `inventory/deletion_legacy_inventory.md`, `import_audits/legacy_runtime_path_audit.txt` |
| Direct `_engine` runtime imports | Inventoried, not deleted in Phase 00 | Phases 01, 05, 08, 09 | `inventory/rust_python_boundary_inventory.md`, `import_audits/rust_boundary_direct_engine_audit.txt` |
| Old replay/buffer runtime path | Inventoried, not deleted in Phase 00 | Phase 07 | `inventory/deletion_legacy_inventory.md` |

No Phase 00-owned legacy runtime deletion was left incomplete. Later-phase deletions are intentionally not claimed complete here.
