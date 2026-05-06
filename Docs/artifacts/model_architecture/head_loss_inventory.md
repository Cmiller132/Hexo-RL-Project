# Head And Loss Inventory

The current trainer calls `compute_losses(predictions, targets, loss_weights)`
and routes by raw head name. Stage 3 must replace this with resolved loss plan
entries. Trainable outputs default to hard errors when required data is absent.

## Current Head Table

| Head name | Prediction key | Current target keys | Current mask keys | Current weight keys | Current loss | Current skip/fallback behavior | Required input contracts | Semantic phase | Runtime consumers | Decision | New metadata |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `policy` | `policy` | `policy` | none | `policy_weight` | dense soft CE | silently skipped if target missing; entropy may use it | dense board rows `0..1088` | root/leaf policy | inference, self-play, trainer | keep, strict | trainable/runtime policy over dense board rows |
| `sparse_policy` | `sparse_policy` | `sparse_policy_target` | `candidate_mask` | `sparse_policy_weight` or `policy_weight` | masked CE | skipped if target or mask missing; zero loss if no valid rows | candidate row table with qr, board index, feature schema | root/leaf candidate policy | sparse inference, MCTS sparse priors | keep, strict | trainable/runtime candidate policy with row hash |
| `pair_policy` | `pair_policy` | `pair_policy_target` | `pair_candidate_mask` | `pair_policy_weight` or `policy_weight` | masked CE | skipped if target or mask missing | crop candidate rows plus pair row table of candidate row indices | pair diagnostic/runtime only by strategy | crop pair strategy, trainer | keep only as crop pair output; strict | pair output over candidate-pair rows |
| `policy_place` | `policy_place` | `policy_target` | `legal_mask` | `policy_weight` | graph masked CE | skipped if target or mask missing | global legal row table | root/leaf graph policy | global graph inference/MCTS/trainer | keep, strict | runtime policy over global legal rows |
| `legal_token_quality` | `legal_token_quality` | `legal_token_quality_target` fallback `policy_target` | `legal_mask` | `policy_weight` | masked CE | target fallback to policy target; skipped if neither | global legal row table | diagnostic legal quality | trainer only | simplify; diagnostic unless explicitly trained | optional diagnostic or trainable with its own target, no fallback |
| `policy_pair_first` | `policy_pair_first` | `pair_first_policy_target` fallback `policy_target` | `legal_mask` | `pair_policy_weight` or `policy_weight` | graph masked CE | falls back to policy target; skipped if target/mask missing | global legal row table, first-placement phase | first-placement pair marginal | pair strategy and trainer | keep, strict; remove fallback | trainable/runtime pair-first over legal rows |
| `policy_pair_joint` | `policy_pair_joint` | `pair_policy_target` | inferred from pair indices or pair token indices | `pair_policy_weight` or `policy_weight` | graph masked CE | skipped if target or inferred mask absent | global unordered pair row table | first-placement joint pair | pair strategy and trainer | keep, strict | trainable/runtime unordered joint pair rows |
| `policy_pair_second` | `policy_pair_second` | `pair_second_policy_target` | inferred from pair indices or pair token indices | `pair_policy_weight` or `policy_weight` | graph masked CE | skipped if target missing or zero target mass | known-first pair row table | second-placement known-first | pair strategy and trainer | keep, strict phase gate | trainable/runtime known-first second rows; zero mass is hard/explicit optional |
| `opp_policy` | `opp_policy` | `opp_policy_target`, fallback `opp_policy`, fallback `policy` | `opp_legal_mask` for graph rows | `opp_policy_weight` | dense or graph CE | falls back through aliases; zero loss when empty | opponent dense or opponent legal row table | next opponent turn start | trainer, optional inference output | keep, remove aliases | trainable opponent policy with independent row identity |
| `value` | `value` | `value` | none | `value_weight` | binned value CE | skipped if missing; non-finite masked; zero if all invalid | state value target | all phases | inference, MCTS, trainer | keep, strict | runtime value with decoder/range/perspective |
| `lookahead_*` | concrete `lookahead_h` | same concrete key | none | none today | binned value CE | skipped when target missing; sampler falls back to value | configured horizon value target | horizon turn boundary | trainer | keep, strict; delete fallback | trainable family expanded from configured horizons |
| `regret_rank` | `regret_rank` | `regret_rank` | none | `regret_weight` | batch ranking loss | skipped if missing; zero if all weight zero | regret suffix target | completed game/replay | trainer, optional inference diagnostics | keep | trainable auxiliary with replay-only weighting |
| `regret_value` | `regret_value` | `regret_value` | none | `regret_weight` | binned regret CE | skipped if missing; non-finite masked | regret scalar target | completed game/replay | trainer | keep | trainable auxiliary |
| `axis` | `axis` | `axis` | implicit valid target `>=0` | none | CE | returns zero if target missing or invalid | dominant-axis label | terminal-derived auxiliary | trainer | keep as optional auxiliary | trainable optional with valid-label mask |
| `axis_delta_norm` | `axis_delta_norm` | `axis_delta_norm` | none | none | MSE | skipped if target missing | six-plane axis map | replay auxiliary | trainer/dashboard experiments | keep as optional auxiliary | trainable optional dense map |
| `moves_left` | `moves_left` | `moves_left` | none | `moves_left_weight` | MSE | skipped if missing; zero if all weight zero | normalized moves-left scalar | completed/truncated game | trainer | keep | trainable auxiliary |
| `tactical` | `tactical` | `tactical_target` | none | `policy_weight` | BCE | skipped if target missing | tactical oracle labels | state tactical | graph trainer | keep, strict when enabled | trainable auxiliary with tactical source contract |
| `entropy` | synthetic loss entry | none | none | none | negative entropy | uses `policy` else `policy_place` if present | policy output row table | regularizer | trainer | keep as loss-plan entry | non-output regularizer tied to one policy contract |

## Removal Decisions

- Remove broad raw head-name switches from trainer/runtime loss routing.
- Remove target aliases:
  - `legal_token_quality_target` may not fallback to `policy_target`.
  - `policy_pair_first` may not fallback to `policy_target`.
  - `opp_policy` may not fallback through `opp_policy` or `policy`.
- Remove silent skips for trainable outputs. Optional outputs are either:
  - not in the resolved trainable loss plan, or
  - hard-error when target, mask, weight, row identity, or phase is missing.
- Remove lookahead fallback from missing lookahead targets to value targets.
- Keep zero-weight loss behavior only when a contract explicitly marks the row
  batch as non-trainable for a known reason such as truncated value, low-PCR
  policy filtering, regret replay-only, or critical candidate overflow.

## Loss Plan Shape For Stage 3

Each loss plan row must name:

```text
prediction contract
target contract
row contract
mask contract
weight contract
phase contract
loss function
optional-data policy
telemetry key
```

The only allowed optional-data policy for a trainable head is `hard_error` unless
the architecture spec marks the output diagnostic/non-trainable.
