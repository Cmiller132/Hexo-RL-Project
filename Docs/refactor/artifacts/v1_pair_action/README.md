# V1 Pair-Action Acceptance And Inventory

Worker scope: docs/artifacts and focused baseline/config tests only. No Rust,
model runtime, search runtime, or behavior changes are claimed here.

Primary V1 side-by-side target:

```text
global_pair_biaffine_0:sampled_joint_pair_v1
```

Protected current baselines:

```text
global_xattn_0:none
global_graph768_champion:none
```

## Acceptance Matrix

| Row | Requirement | Worker A baseline status | Evidence owner for closure |
|---|---|---|---|
| V1-0 Inventory | Current pair heads, search modes, legal filters, replay targets, and banned paths are inventoried. | This artifact lists current audit terms, known runtime locations, and banned V1 paths. | Worker A for inventory; runtime workers for closure proof after changes. |
| V1-1 Legal rows | Flagship neural self-play and training must use full Rust legal rows; threat filters must not become semantic `LEGAL` sources. | Baseline audit notes current `constrain_threats` entry points only. No runtime change. | Runtime legal-row owner. |
| V1-2 Pair identity | Pair rows must be canonical unordered actions over the same start-of-turn legal table, with D6-stable identity and negative tests. | Not implemented here. | Pair contract/runtime owner. |
| V1-3 Candidate selector | `pair_candidate_selector_v1` must own quotas, canonicalization, deduplication, tactical support, metadata, and selector telemetry. | Not implemented here. | Candidate selector owner. |
| V1-4 Pair MCTS | `sampled_joint_pair_v1` must own pair prior construction, root Gumbel admission, progressive widening, pair expansion/backup, correction, telemetry, and targets. | V1 config identifier is protected by a conditional strict xfail until registered. | Pair-search owner. |
| V1-5 Biaffine model | `global_pair_biaffine_0` must provide bounded symmetric biaffine `pair_joint_logits` without all-pair input tokens. | V1 architecture identifier is protected by a conditional strict xfail until registered. | Model owner. |
| V1-6 Targets | Replay and target builders must distinguish admitted, explicit negative, forced, diagnostic, and unsampled rows; unsampled legal pairs are never negatives. | Not implemented here. | Replay/training owner. |
| V1-7 Curriculum | Training must avoid support collapse and produce candidate/tactical recall, entropy, and throughput evidence. | Not implemented here. | Training/curriculum owner. |
| V1-8 Cutover | Final acceptance requires fair equal-wall-clock evidence, D6 reports, performance profiles, and deletion/import proof for obsolete paths. | Not implemented here. | Cutover owner. |

## Current Baseline Protection

Focused tests added in `Python/tests/test_v1_pair_action_baselines.py` protect
the two existing `none` baselines:

- `global_xattn_0:none` must materialize as `global_xattn_0__none__v1`,
  keep `model.pair_strategy == "none"`, keep zero pair rows, and keep the
  global legal `policy_place`/`value` head bundle.
- `global_graph768_champion:none` must materialize as
  `global_graph768_champion__none__v1`, keep `model.pair_strategy == "none"`,
  keep graph token budget `768`, graph layers `6`, and retain conservative
  fp32/microbatch runtime settings.
- `global_pair_biaffine_0:sampled_joint_pair_v1` is asserted as the expected
  future candidate ID `global_pair_biaffine_0__sampled_joint_pair_v1__v1`.
  The test is conditionally strict-xfailed while either identifier is absent
  from config/model registries.

## Banned V1 Paths

These paths cannot be claimed as the V1 pair-action implementation:

- Treating `root_pair_mcts` or `full_pair_mcts` as final V1 pair-action MCTS
  unless they truly expand, apply, and back up unordered pair macro-actions.
- Calling `pair_logits_to_action_logits` and using the projected single-cell
  logits as final move-choice authority for V1.
- Triggering V1 behavior merely from pair head names instead of the explicit
  `sampled_joint_pair_v1` strategy.
- Using `constrain_threats` or threat-constrained rows as the flagship neural
  self-play/training `LEGAL` source.
- Hardcoding normal two-placement hot-window actions instead of searching them.
- Generating V1 candidates by legal-row first-N order.
- Materializing all pair actions as graph input tokens.
- Training unsampled legal pairs as negative pair examples.
- Introducing ANN/MIPS retrieval before blockwise-exact retrieval has recall,
  D6, diagonal-mask, deduplication, legal-row-order, and chunking proof.
- Keeping permanent parallel old/new runtime paths after V1 cutover.

## Audit Command Notes

`rg` is the preferred command in the project instructions. In this environment,
`rg.exe` returned `Access is denied`, so Worker A used `git grep -n` fallbacks.

Primary audits to run:

```powershell
rg -n "constrain_threats|threat_constrained_moves" Python/src Python/tests crates scripts Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
rg -n "root_pair_mcts|full_pair_mcts" Python/src Python/tests Configs scripts Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
rg -n "pair_logits_to_action_logits" Python/src Python/tests Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
rg -n "global_pair_biaffine_0|sampled_joint_pair_v1|global_xattn_0:none|global_graph768_champion:none" Python/src Python/tests Configs Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
git status --short
```

Fallback commands used by Worker A:

```powershell
git grep -n "constrain_threats" -- Python/src Python/tests crates scripts Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
git grep -n "root_pair_mcts\|full_pair_mcts" -- Python/src Python/tests Configs scripts Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
git grep -n "pair_logits_to_action_logits" -- Python/src Python/tests Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
git grep -n "global_pair_biaffine_0\|sampled_joint_pair_v1\|global_xattn_0:none\|global_graph768_champion:none" -- Python/src Python/tests Configs Docs/HEXORL_V1_ARCHITECTURE_PROPOSAL.md Docs/FINAL_HEXORL_IMPLEMENTATION_PLAN.md
```

Current inventory highlights:

- `global_xattn_0:none` and `global_graph768_champion:none` are registered in
  `Python/src/hexorl/config/schema.py` and protected by existing optuna tests.
- `global_pair_biaffine_0` and `sampled_joint_pair_v1` currently appear only
  in the proposal/plan docs, not in Python config/runtime registries.
- `root_pair_mcts` and `full_pair_mcts` are current config/runtime pair modes
  and must be treated as prior-blend baselines until a runtime owner proves
  pair macro-action expansion, application, and backup semantics.
- `pair_logits_to_action_logits` is present in `Python/src/hexorl/search/pair_strategy.py`
  and consumed by `Python/src/hexorl/selfplay/worker.py`; this is banned as V1
  final action authority.
- `constrain_threats` remains present across Python, Rust, scripts, and tests;
  V1 legal-row workers must classify or quarantine each runtime use before
  claiming flagship full-legal row closure.

## Subagent Completion Packet Template

```text
closed V1 rows:
runtime consumers changed:
files changed:
legacy paths deleted or quarantined:
tests and commands run with exit status:
artifacts produced:
performance/utilization evidence for hot paths:
contract examples/docs added where relevant:
known blockers, if any:
explicit statement that no skipped/deferred/manual-only requirement is claimed complete:
```

## Worker F Pair-Search Foundation Evidence

Scope implemented:

- Separate Rust `V1PairSearchEngine` under `crates/hexgame-core/src/v1_pair_search.rs`;
  existing `MCTSEngine` behavior and legacy pair-prior projection APIs are not
  changed.
- Root V1 pair admission validates canonical unordered pairs against the
  start-of-turn V1 legal row table, applies typed proposal correction, emits
  Gumbel admission/allocation telemetry, and selects a pair action directly.
- Opening and one-placement terminal-win states are explicit single-action
  exceptions with `hardcoded_reason` telemetry.
- Selected normal pair actions require root token, legal table hash, and
  `pair_key` before apply. Pair apply uses the Rust engine and rolls back on a
  second-placement failure.
- Interior full-turn reservoir API caches one candidate reservoir and one
  scoring pass per node key; progressive widening reveals cached rows without
  rescoring or refilling.
- PyO3 exposes `PyV1PairSearchEngine`/`V1PairSearchEngine`,
  V1 correction-mode constants, selected-action apply, replay telemetry, and
  interior reservoir/widening calls.

Focused evidence commands:

```powershell
cargo test -p hexgame-core v1_pair_search -- --nocapture
# exit 0; 5 passed

cargo test -p hexgame-core v1::tests::pair_rows -- --nocapture
# exit 0; 2 passed

cargo test -p hexgame-py --no-default-features
# exit 0

.venv\Scripts\python.exe -m maturin develop --manifest-path crates\hexgame-py\Cargo.toml --quiet
# exit 0

.venv\Scripts\python.exe -m pytest Python\tests\test_v1_pair_search_ffi.py -q
# exit 0; 3 passed

.venv\Scripts\python.exe -m pytest Python\tests\test_engine_smoke.py Python\tests\test_v1_pair_search_ffi.py -q
# exit 0; 21 passed
```

Projection audit:

```powershell
Select-String -Path crates\hexgame-core\src\v1_pair_search.rs,Python\tests\test_v1_pair_search_ffi.py `
  -Pattern 'apply_root_pair_priors|apply_root_pair_first_priors|apply_root_pair_second_priors|pair_logits_to_action_logits|root_pair_mcts|full_pair_mcts'
# exit 0; no banned V1 projection references in new V1 pair-search files

git diff -- crates/hexgame-core/src/v1_pair_search.rs crates/hexgame-py/src/engine.rs Python/tests/test_v1_pair_search_ffi.py `
  | Select-String -Pattern '^\+.*(apply_root_pair_priors|apply_root_pair_first_priors|apply_root_pair_second_priors|pair_logits_to_action_logits|root_pair_mcts|full_pair_mcts)'
# exit 0; no banned projection calls added by V1 pair-search diff
```

Performance/telemetry evidence:

- `v1_interior_reservoir_scores_once_and_widens_from_cache` asserts one
  reservoir build and one scoring pass for the expanded full-turn node, and
  duplicate node-key expansion is rejected.
- `test_v1_pair_search_selects_and_applies_canonical_pair` asserts replay
  telemetry includes `neural_calls_per_expanded_full_turn_node == 1`,
  `reservoir_refill_events == 0`, and simulation allocation equals visit
  counts.
- No equal-wall-clock strength or throughput budget is claimed complete by this
  Worker F foundation packet.
