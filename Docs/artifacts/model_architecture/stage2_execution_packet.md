# Stage 2 Execution Packet

## Goal

Implement `Python/src/hexorl/models/` as the single architecture authority and
route model assembly, architecture membership, output resolution, and runtime
feature decisions through registered specs.

## Success Criteria Checklist

- [x] `hexorl/models/` is the architecture authority.
- [x] `build_model_from_config` delegates to `hexorl.models.assembly`.
- [x] Model construction uses registered architecture metadata and attaches
  resolved metadata to built modules.
- [x] Architecture specs own default outputs, supported optional outputs,
  dynamic output families, self-play required outputs, adapter ids, provider
  ids, and pair capabilities.
- [x] Config can enable/disable supported optional outputs but cannot disable
  self-play policy or value outputs.
- [x] `lookahead_*` expands to concrete `lookahead_{horizon}` heads during
  spec resolution.
- [x] Current self-play architectures resolve search policy and search value
  capabilities.
- [x] Current architecture ids resolve through the registry; the deprecated
  `graph` alias has an explicit deletion decision.
- [x] Output contracts, row-table definitions, the `RowTableInstance` contract,
  and value decoder contracts are available from `hexorl.models.contracts`;
  assembled models carry resolved output contracts, row-table definitions, and
  value decoder metadata.
- [x] Old architecture-name lists are removed as runtime authority from config,
  buffer, epoch, inference, self-play, eval, dashboard summaries, and runtime
  estimates.
- [x] Retained legacy PyTorch implementations are imported only by
  `hexorl.models.recipes`.

## Runtime Consumers Changed

- `Config` validates architecture ids through `hexorl.models.registry` and
  rewrites resolved heads from the spec.
- `RingBuffer` replay feature flags use registry capability metadata.
- `BufferProcess`, epoch orchestration, self-play, inference, eval, dashboard,
  trainer, and runtime memory estimates now query `hexorl.models`.
- Phase 3 autotune orchestration validates its explicit global-graph scout
  scope against `hexorl.models.registry` and delegates global-graph membership
  checks to the registry.
- Legacy `hexorl.model.network.build_model_from_config` is a delegating entry
  point for old callers, not the assembly authority.

## Legacy Quarantine

Retained owner: `Python/src/hexorl/models/recipes/legacy.py`.

Allowed legacy imports:

- `hexorl.model.network.HexNet`
- `hexorl.model.network.load_model_state`
- `hexorl.model.global_graph.GlobalHexGraphNet`

Deletion gate: Stage 4 must move or delete retained implementation code so no
runtime code depends on `Python/src/hexorl/model/`. Until then, legacy code is
implementation-only and cannot own architecture membership, config behavior,
runtime feature flags, or self-play/search behavior.

## Required Evidence

- `Python/tests/test_model_architecture_stage2.py`
- `Docs/artifacts/model_architecture/stage2_import_audit.md`

## Stop Rule Results

- All current supported architecture ids resolve through `hexorl.models`.
- `graph` is explicitly classified as a deprecated config alias targeting
  `graph_hybrid_0`, with a Stage 4 deletion gate.
- Config required-output protection rejects missing dense `policy`, dense
  `value`, global `policy_place`, and global `value`.
- Direct global-architecture prefix checks and incomplete global architecture
  sets were removed from runtime and orchestration authority paths.
- No direct `hexorl.model` imports remain outside retained implementation code
  and `hexorl.models.recipes`.

## Gate Review Addendum

Additional review on 2026-05-06 found that
`scripts/run_phase3_48h_autotune.py` still carried a duplicated global graph
membership set. That was not acceptable for Stage 3 readiness because the
autotune supervisor is the Stage 3 runtime entry point. The duplicate set was
removed; the script now imports `global_graph_architecture_ids` and
`is_global_graph_architecture` from `hexorl.models.registry`.

The explicit `GLOBAL_GRAPH_SCOUT_FAMILIES` tuple remains only as the Stage 3
experiment scope for the four pre-champion candidates. Import-time validation
fails if any scout family is not present in the registry, so this tuple cannot
become separate membership authority.

The Stage 2 unit coverage was also tightened so every registered architecture
must expose search policy and value provider ids, and the `RowTableInstance`
contract must produce phase-sensitive identity hashes.

## Verification

Commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage2.py Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_inference_server.py Python/tests/test_training_data_pipeline.py Python/tests/test_dashboard_foundation.py Python/tests/test_phase3_autotune.py
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\models\__init__.py Python\src\hexorl\models\contracts.py Python\src\hexorl\models\specs.py Python\src\hexorl\models\registry.py Python\src\hexorl\models\assembly.py Python\src\hexorl\models\recipes\legacy.py scripts\run_phase3_48h_autotune.py
git diff --check
```

Results:

```text
226 passed, 1 warning
py_compile passed
git diff --check passed with CRLF warnings only
```

## Explicit Completeness Statement

Stage 2 closes architecture authority and assembly cutover scope. No skipped,
deferred, flaky, or manual-only Stage 2 requirement is claimed complete.
Training/replay loss-plan cutover remains Stage 3. Inference protocol/search
provider/pair-strategy deletion remains Stage 4.
