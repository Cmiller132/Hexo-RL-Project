# Phase 03 — Model Registry and Family Adapters
## Phase Intent
Complete this phase with production-ready behavior only; partial or scaffold-only delivery is not allowed.
## Scope
- Create Python/src/hexorl/models/registry.py specs.py capabilities.py and family modules.
- Refactor legacy model/network.py into adapter shim only.

## Parallel Subagent Split (5-way)
- **S1 Contracts/Schema:** define interfaces, validation, and versioning constraints for this phase.
- **S2 Engine/Runtime:** runtime integration and cutover mechanics.
- **S3 Models/Search:** model/search-facing adaptation and capability compliance.
- **S4 Data/Train/Eval:** downstream data-path compatibility and regression checks.
- **S5 Quality/Obs/Docs:** test suites, telemetry assertions, artifact curation, and docs updates.

## Orchestrator Gate Reviews
1. **Design Gate:** contracts/interfaces approved before branch merges.
2. **Integration Gate:** all consumers migrated within phase scope; no hybrid hidden paths.
3. **Evidence Gate:** required tests pass with stored artifacts.
4. **Strictness Gate:** no TODO/FIXME, no spec gaps, no feature-incomplete behavior.
5. **Rollback Gate:** rollback tag exists and recovery smoke is verified.

## Mandatory Checks
- ModelSpec validation suite
- Checkpoint compatibility load/save tests
- Capability gates for head/policy availability
- CI jobs touching changed paths must be green on two consecutive runs.

## Completion Criteria
- Phase deliverables merged and operational.
- All mandatory checks pass with logs under `Docs/refactor/artifacts/phase_03/`.
- Orchestrator signs a phase-close note confirming no half-implementation remains.
