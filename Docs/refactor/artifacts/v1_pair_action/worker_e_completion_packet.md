# Worker E Completion Packet: V1 Pair Biaffine Model And Contracts

## Closed V1 Rows

- V1-4 partial config contract: registered explicit `sampled_joint_pair_v1` strategy identifier and prevented head-name-only activation.
- V1-5 model contract: registered `global_pair_biaffine_0` and implemented bounded symmetric low-rank biaffine `pair_joint_logits` over supplied pair rows.
- V1-1 partial validation contract: V1 flagship configs require `selfplay.legal_row_mode="full_rust_legal"`, `selfplay.tactical_mode="proposal_and_label"`, and `selfplay.constrain_threats=false`.

## Runtime Consumers Changed

- Model registry/spec resolution now exposes V1 output contracts:
  `cell_marginal_logits`, `pair_completion_logits`, `pair_proposal_score`,
  `pair_joint_logits`, `value`, `terminal_tactical_v1`.
- Config/autotune recipe materialization now accepts
  `global_pair_biaffine_0:sampled_joint_pair_v1` side by side with existing
  `none` baselines.
- Graph batches can carry externally admitted V1 pair rows by LEGAL-token
  reference plus bounded pair features without adding pair-action tokens.
- Inference adapters decode V1 output names and row metadata while retaining
  legacy graph output names.

## Files Changed

- `Python/src/hexorl/autotune/recipes.py`
- `Python/src/hexorl/config/schema.py`
- `Python/src/hexorl/graph/batch.py`
- `Python/src/hexorl/inference/adapters.py`
- `Python/src/hexorl/inference/protocol.py`
- `Python/src/hexorl/inference/server.py`
- `Python/src/hexorl/models/families/global_graph.py`
- `Python/src/hexorl/models/registry.py`
- `Python/src/hexorl/models/specs.py`
- `Python/src/hexorl/search/pair_strategy.py`
- `Python/tests/test_v1_pair_biaffine_model.py`
- Existing contract tests updated for the new registered architecture:
  `Python/tests/test_global_graph_contract.py`,
  `Python/tests/test_model_architecture_stage2.py`,
  `Python/tests/test_optuna_config_surface.py`

## Legacy Paths Deleted Or Quarantined

- No legacy paths were deleted in Worker E scope.
- Existing baselines remain materialized unchanged:
  `global_xattn_0:none` and `global_graph768_champion:none`.
- All-pair graph attention tokens remain rejected by
  `materialize_pair_context_tokens`.

## Tests And Commands Run

- `python -m pytest Python\tests\test_v1_pair_biaffine_model.py Python\tests\test_v1_pair_action_baselines.py -q`
  - Exit status: 0
  - Result: 7 passed
- `python -m pytest Python\tests\test_model_architecture_stage2.py Python\tests\test_optuna_config_surface.py Python\tests\test_model_architecture_stage4.py -q`
  - Exit status: 0
  - Result: 34 passed
- `python -m pytest Python\tests\test_global_graph_contract.py -q`
  - Exit status: 0
  - Result: 50 passed, 1 skipped
- `python -m pytest Python\tests\test_config_and_guardrails.py -q`
  - Exit status: 0
  - Result: 36 passed
- `python -m pytest Python\tests\test_v1_pair_biaffine_model.py Python\tests\test_v1_pair_action_baselines.py Python\tests\test_model_architecture_stage2.py Python\tests\test_optuna_config_surface.py Python\tests\test_model_architecture_stage4.py Python\tests\test_config_and_guardrails.py Python\tests\test_global_graph_contract.py -q`
  - Exit status: 0
  - Result: 127 passed, 1 skipped, 1 warning

## Side-By-Side Materialization Evidence

Command exit status: 0

```text
global_xattn_0__none__v1 architecture=global_xattn_0 pair_strategy=none heads=['policy_place', 'value'] legal_row_mode=legacy tactical_mode=legacy constrain_threats=True max_pairs=0
global_graph768_champion__none__v1 architecture=global_graph768_champion pair_strategy=none heads=['policy_place', 'value'] legal_row_mode=legacy tactical_mode=legacy constrain_threats=True max_pairs=0
global_pair_biaffine_0__sampled_joint_pair_v1__v1 architecture=global_pair_biaffine_0 pair_strategy=sampled_joint_pair_v1 heads=['cell_marginal_logits', 'pair_completion_logits', 'pair_proposal_score', 'pair_joint_logits', 'value', 'terminal_tactical_v1'] legal_row_mode=full_rust_legal tactical_mode=proposal_and_label constrain_threats=False max_pairs=256
```

## Audit Evidence

- Registration audit command exit status: 0.
  - `global_pair_biaffine_0` appears in model registry/spec/config/autotune/model tests.
  - `sampled_joint_pair_v1` appears in pair strategy/config/autotune/model tests.
  - Existing `global_xattn_0` and `global_graph768_champion` registrations remain present.
- All-pair token audit command exit status: 0.
  - `Python/src/hexorl/graph/batch.py` still raises
    `PAIR_ACTION context tokens were removed from the minimal global graph schema`.

## Performance / Utilization Evidence For Hot Paths

- Model implementation scores only supplied pair rows with one bounded tensor pass;
  it does not materialize all pair actions as graph attention tokens.
- CPU timing smoke command exit status: 0.

```text
bounded_v1_forward_cpu pair_rows=256 tokens=259 iterations=20 mean_ms=1.640 finite_pair_joint=True
```

- Worker E did not integrate search runtime hot paths or run GPU throughput profiles.

## Contract Docs / Examples Added

- This completion packet documents V1 output/config/inference contracts and
  side-by-side materialization evidence.
- Focused tests in `Python/tests/test_v1_pair_biaffine_model.py` serve as
  executable contract examples.

## Known Blockers

- None in Worker E scope.

## No Deferred Claim Statement

No skipped, deferred, flaky, or manual-only requirement is being claimed complete.
Search runtime integration, selector/scorer internals, Rust row APIs, self-play
worker consumption, replay core, and target-stack integration remain outside
Worker E ownership.
