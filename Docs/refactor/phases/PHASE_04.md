# Phase 04 — Inference Protocol and Adapterization
## Phase Intent
Complete this phase with production-ready behavior only; partial or scaffold-only delivery is not allowed.
## Scope
- Create inference/protocol.py + adapters for dense/sparse/global/pair payloads.
- Unify client/server message contracts and transport lifecycle semantics.

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
- Protocol compatibility tests
- Inference throughput and latency smoke vs baseline
- Serialization/deserialization fuzz tests
- CI jobs touching changed paths must be green on two consecutive runs.

## Completion Criteria
- Phase deliverables merged and operational.
- All mandatory checks pass with logs under `Docs/refactor/artifacts/phase_04/`.
- Orchestrator signs a phase-close note confirming no half-implementation remains.
