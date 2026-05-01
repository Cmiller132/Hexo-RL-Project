# Model Construction Modularization Evidence

Date: 2026-05-01

## Goal

Replace crop/global model monolith dispatch with descriptor-driven trunk, head, and composer composition while preserving the registry surface.

## Success Criteria

- Crop and global families build through `CropModel` / `GlobalModel` composers.
- Crop CNN, crop transformer, crop graph-hybrid, and the three global graph variants are distinct trunk classes.
- Heads are allocated from one `HEAD_REGISTRY` and the allocated set equals the descriptor head tuple.
- Forward inputs are `CropInputs` / `GraphInputs` dataclasses.
- Inference contracts derive output head specs from `HEAD_REGISTRY`.
- Aliases live on descriptors; `MODEL_KIND_ALIASES` is removed.
- Grep gates reject `family_kind` dispatch and `lookahead_` prefix head dispatch in `models/`.

## Constraints

- No runtime compatibility facade for the old `HexNet` or `GlobalHexGraphNet` monoliths.
- Existing registry and descriptor entry points remain the public construction surface.
- Optional row-mapped heads may emit empty tensors when their optional row inputs are absent; this keeps declared-head allocation deterministic without inventing rows.

## Required Evidence

- `PYTHONPATH=Python/src ./.venv/bin/python -m pytest -q Python/tests/models Python/tests/inference Python/tests/test_inference_server.py Python/tests/search Python/tests/eval Python/tests/selfplay Python/tests/train`
  - Exit status: 0
  - Result: 129 passed in 5.54s
- `! grep -rE 'family_kind\s*[=!]=|family_kind\s+in\s' Python/src/hexorl/models/trunks Python/src/hexorl/models/composers`
  - Exit status: 0
- `! grep -rE 'name\.startswith\("lookahead_"\)' Python/src/hexorl/models`
  - Exit status: 0

## Completion Packet

- closed V2 rows: V2-030, V2-031 architecture-hardening evidence only; no new phase closure claimed.
- runtime consumers changed: model factory, train adapter, local/server inference execution, policy provider payload names, checkpoint manifest head source.
- files changed: `Python/src/hexorl/models/**`, `Python/src/hexorl/train/adapters.py`, `Python/src/hexorl/inference/local.py`, `Python/src/hexorl/inference/server/execution.py`, `Python/src/hexorl/search/policy_provider.py`, `Python/src/hexorl/search/pair_strategy.py`, `Python/src/hexorl/config/schema.py`, `Python/tests/models/test_phase03_model_registry.py`.
- legacy paths deleted or quarantined: old `HexNet` and `GlobalHexGraphNet` monolith implementations removed from runtime modules.
- tests and commands run with exit status: listed above.
- artifacts produced: this evidence note.
- performance/utilization evidence for hot paths: not a hot-path numerical change; existing requested runtime suites passed. No benchmark closure claimed.
- contract examples/docs added where relevant: model tests now cover declared-head allocation, fake trunk/family registration, typed params rejection, and contract/head agreement.
- known blockers: none for the requested verification bundle.
- skipped/deferred/manual-only claim: no skipped, deferred, or manual-only requirement is claimed complete.

## Stop Rules

No stop rule triggered.
