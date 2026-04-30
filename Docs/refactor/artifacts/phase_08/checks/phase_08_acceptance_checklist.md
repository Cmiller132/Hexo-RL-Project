# Phase 08 Acceptance Checklist

- [x] `V2-080`: eval player construction uses `PolicyProvider` for every registered family.
- [x] `V2-081`: dashboard routes dispatch through `ContractInspector` and focused read-only inspector services.
- [x] `V2-082`: dashboard payloads surface hash/source/version/trace, checkpoint, protocol, family, and recipe facts.
- [x] `V2-083`: autotune owns typed `ModelRecipe` and `family_spaces`; raw config mutation entrypoints are deleted.
- [x] `V2-084`: scheduler decisions, dry-run validation, watchdog aborts, and likely subsystem owners are logged as structured dicts.
- [x] `V2-085`: dashboard debug bundles and autotune poor-learning reports localize model, targets, engine, D6, policy, MCTS, replay, and runtime failures.
- [x] `V2-086`: `RuntimeSpec` separates host-utilization knobs from model semantics and scores throughput/utilization/stability/stalls.
