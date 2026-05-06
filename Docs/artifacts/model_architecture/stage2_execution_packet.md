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
  and value decoder contracts are available from `hexorl.contracts`;
  assembled models carry resolved output contracts, row-table definitions, and
  value decoder metadata.
- [x] Old architecture-name lists are removed as runtime authority from config,
  buffer, epoch, inference, self-play, eval, dashboard summaries, and runtime
  estimates.
- [x] PyTorch family implementations are built only through
  `hexorl.models.recipes`.

## Runtime Consumers Changed

- `Config` validates architecture ids through `hexorl.models.registry`;
  resolved heads stay in the architecture metadata path instead of being
  materialized back into config.
- `RingBuffer` replay feature flags use registry capability metadata.
- `BufferProcess`, epoch orchestration, self-play, inference, eval, dashboard,
  trainer, and runtime memory estimates now query `hexorl.models`.
- Phase 3 autotune orchestration validates its explicit global-graph scout
  scope against `hexorl.models.registry` and delegates global-graph membership
  checks to the registry.
- Stage 4 removed the legacy `hexorl.model` runtime package. Model family
  implementations now live under `hexorl.models.families`, and runtime callers
  use `hexorl.models.assembly` or `hexorl.models.loading`.

## Legacy Quarantine

Retained owner: closed by Stage 4.

Formerly allowed legacy imports:

- `hexorl.model.network.HexNet`
- `hexorl.model.network.load_model_state`
- `hexorl.model.global_graph.GlobalHexGraphNet`

Deletion gate: closed. `Python/src/hexorl/model/` no longer exists. `HexNet`
and `GlobalHexGraphNet` moved to `Python/src/hexorl/models/families/`, and
runtime code no longer imports `hexorl.model`, `HexNet`, `GlobalHexGraphNet`,
`from_config`, or `load_model_state` as runtime authority.

## Required Evidence

- `Python/tests/test_model_architecture_stage2.py`
- `Docs/artifacts/model_architecture/stage2_import_audit.md`

## Stop Rule Results

- All current supported architecture ids resolve through `hexorl.models`.
- `graph` is deleted as a runtime architecture alias. Configs must use
  `graph_hybrid_0` explicitly.
- Config required-output protection rejects missing dense `policy`, dense
  `value`, global `policy_place`, and global `value`.
- Direct global-architecture prefix checks and incomplete global architecture
  sets were removed from runtime and orchestration authority paths.
- No direct `hexorl.model` imports remain in runtime or test code. The retained
  implementation quarantine has been removed by Stage 4.

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

## Full Verification Addendum

The 2026-05-06 full verification pass found remaining config/runtime
architecture-id conditionals for attention-head divisibility, attention-position
support, display summaries, and training-memory estimates. Those checks were
moved behind registered spec capabilities:

- `ArchitectureSpec.requires_attention_head_divisibility`
- `ArchitectureSpec.supports_attention_positions`
- existing graph/global graph capability flags

`Config`, `architecture_display_summary`, `trial_model_summary`, and runtime
training-memory estimation now consume those spec fields instead of concrete
architecture-name branches. The retained explicit family names in
`scripts/run_phase3_48h_autotune.py` are experiment scope and recipe choices,
not membership authority; global graph membership still validates against
`hexorl.models.registry`.

## Verification

Commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage2.py Python/tests/test_config_and_guardrails.py Python/tests/test_global_graph_contract.py Python/tests/test_inference_server.py Python/tests/test_training_data_pipeline.py Python/tests/test_dashboard_foundation.py Python/tests/test_phase3_autotune.py
$env:PYTHONPATH='Python/src'; python -m py_compile Python\src\hexorl\contracts\__init__.py Python\src\hexorl\models\__init__.py Python\src\hexorl\models\specs.py Python\src\hexorl\models\registry.py Python\src\hexorl\models\assembly.py Python\src\hexorl\models\recipes\family.py scripts\run_phase3_48h_autotune.py
git diff --check
```

Results:

```text
226 passed, 1 warning
py_compile passed
git diff --check passed with CRLF warnings only
```

Additional 2026-05-06 commands:

```powershell
$env:PYTHONPATH='Python/src'; python -m pytest -q Python/tests/test_model_architecture_stage2.py Python/tests/test_config_and_guardrails.py
Get-ChildItem -Path Python/src/hexorl -Recurse -File -Include *.py |
  Select-String -Pattern 'architecture\.startswith','startswith\("global_','GlobalHexGraphNet\.ARCHITECTURES','GLOBAL_GRAPH_ARCHITECTURES','spec\.architecture_id in','spec\.architecture_id ==','arch == "cnn"','arch == "restnet"','arch == "graph_hybrid_0"'
```

Results:

```text
45 passed, 1 warning
architecture-name authority audit: no matches
```

## Cleanup Verification Addendum

The follow-up cleanup pass moved contracts to `hexorl.contracts`, moved replay
training-batch conversion to `hexorl.replay`, deleted the `graph` runtime alias,
removed permanent `legacy` recipe naming, and stopped materializing resolved
heads/loss defaults back into config.

Current local evidence from the cleanup pass:

```text
py_compile for touched model/config/replay/train/inference/runtime/test files: passed
Config graph alias smoke: hard ValidationError with deleted-alias decision
global graph default smoke: cfg.model.heads stayed ['policy', 'value']; resolved outputs were policy_place/value/lookahead_4/12/36
deleted path audit: Python/src/hexorl/model exists: False
stale import/name audits for removed contract, replay-adapter, and recipe paths: clean
git diff --check: passed
pytest rerun: blocked in this shell because the active Python interpreter has no pytest module
```

## Explicit Completeness Statement

Stage 2 closes architecture authority and assembly cutover scope. No skipped,
deferred, flaky, or manual-only Stage 2 requirement is claimed complete.
Training/replay loss-plan cutover remains Stage 3. Inference protocol/search
provider/pair-strategy deletion remains Stage 4.
