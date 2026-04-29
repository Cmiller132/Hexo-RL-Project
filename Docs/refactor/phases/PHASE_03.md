# Phase 03 — Model Registry and Family Adapters

## Purpose
Eliminate architecture-string behavior inference by introducing a capability-driven model family registry.

## Target Structure
Create `Python/src/hexorl/models/`:
- `registry.py`, `specs.py`, `capabilities.py`, `checkpoint.py`, `factory.py`
- `families/` modules (`dense_cnn`, `restnet`, `graph_hybrid`, `global_xattn`, `global_line_window`, `global_relation_graph`)
- `heads/` and `trunks/` split per V2 ownership

## Key V2 Rules
- Model family declares capabilities and adapters.
- `ModelSpec` is discriminated by `kind` (not name heuristics).
- Non-model subsystems must not switch behavior using architecture names.
- Checkpoint cleanup ownership moved into `models/checkpoint.py`.

## Parallel Subagent Work
- S1: spec/capability schemas and validation.
- S2: runtime factory cutover behind adapter shim.
- S3: family implementations and defaults/tune space interfaces.
- S4: train/eval adapter integration with new registry APIs.
- S5: tests for capability enforcement and migration docs.

## Mandatory Tests
- Spec validation matrix across families.
- Build-model smoke tests for each family.
- Checkpoint load/save compatibility for migrated models.
- Assertions that architecture strings are not consumed outside `models/`.

## Exit Criteria
- Registry is authoritative for model assembly and capabilities.
- Legacy `model/network.py` only thin compatibility adapter pending deletion.
