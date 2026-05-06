# Target Inventory

Targets are training contracts, not model internals. Stage 1 classifies current
target behavior and locks the stricter target metadata needed for Stage 3.

## Current Target Table

| Target name | Source record fields | Row table | Mask | Phase | Normalization | Invalid input behavior | Duplicate behavior | Zero-mass behavior | Required weight/phase behavior | Negative tests | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense `policy` | `PositionRecord.policy_target`, `policy_target_v2` projected by `dense_policy_from_v2` | dense board rows `0..1088` | none | any search position | normalized dense distribution | invalid/outside crop mass ignored or projected from v2 | duplicate action sums by dict/projection | current dense target can become all zeros only in malformed data | `policy_weight` gates low-PCR/full-search | add missing zero-mass hard error when trainable | keep, strict row contract |
| `sparse_policy_target` | `policy_target_v2` plus candidate builder | candidate qr/action rows | `candidate_mask` | any search position | represented target mass renormalized; missing mass reported | candidate builder fails unsupported modes; overflow zeros weights | active duplicate candidate row fails in pair builder, not candidate builder | no valid rows returns zero loss today | `sparse_policy_weight`; overflow sets zero | existing sparse tests rewrite/golden after engine issue | keep, hard error unless row intentionally non-trainable |
| graph `policy_target` | `policy_target_v2` in `build_graph_batch_from_history` | global legal rows | `legal_mask` | any search position | positive legal mass normalized over legal rows | illegal target rows raise `ValueError` | duplicate qr masses sum | if no target mass, current loss zero/skips depending head | `policy_weight` | existing graph tests intended golden but blocked by missing Rust engine | keep, hard error on zero target mass for trainable policy |
| `pair_first_policy_target` | projection from pair target in `_pair_first_target_for_legal` | global legal rows | `legal_mask` | first placement only | target mass over first coordinate normalized | illegal pairs handled by pair target builder | duplicate first coordinate sums | zero for non-first-placement or absent pair target | `pair_policy_weight`, explicit phase | add test that first marginal includes both unordered cells if chosen | keep with marginal-over-both-cells decision below |
| `pair_policy_target` | `pair_policy_target_v2` and graph/candidate pair builders | unordered joint pair rows for first placement; ordered known-first rows for second placement | pair row mask | first-placement joint or second-placement known-first | represented mass normalized | illegal pair rows raise; missing complete first-placement graph target raises in replay | duplicate coordinates fail; duplicate pair rows sum by canonical key | zero currently skips pair-second or returns zero | `pair_policy_weight`, explicit pair phase | existing duplicate/illegal tests intended golden | keep, split into `pair_joint_target` and `pair_second_target` contracts |
| `pair_second_policy_target` | copy of pair target only when `placements_remaining == 1` | known-first pair rows | pair row mask | second placement only | normalized over legal second rows | wrong first action raises | duplicate second target sums | zero on first-placement today; loss skipped | `pair_policy_weight`, `known_first` phase | existing known-first tests intended golden | keep, strict phase; zero mass hard error when enabled in known-first phase |
| `opp_policy_target` | next full-search opponent turn target | opponent legal rows for graph or dense board rows for crop | `opp_legal_mask` for graph | next opponent turn start | normalized over independent opponent legal table | graph builder rejects target without independent legal rows | duplicates sum | empty target weight zero | `opp_policy_weight` required | existing opponent policy tests classify golden/rewrite | keep, remove target aliases |
| `value` | outcome/root perspective through `to_value_target` | state row | none | any | scalar in `[-1,1]` binned by loss | non-finite masked in loss today | n/a | all invalid returns zero loss | `value_weight` required | keep tests for truncated value weight | keep |
| `tactical_target` | `scan_tactical_oracle_from_history` | state row | none | any | 4 multi-label logits | currently hard requires Rust tactical oracle unless fallback enabled | n/a | quiet label set when no tactic | policy/search weight today | existing tactical tests golden; graph tests blocked if engine missing | keep, explicit oracle source contract |
| `regret_rank` | suffix selected-action value regret | state row | none | completed game/replay | raw regret scale | missing selected value zeroes weight | n/a | zero if all weights zero | `regret_weight` required | existing regret tests golden | keep |
| `regret_value` | same regret scalar | state row | none | completed game/replay | binned `[0,4]` | non-finite masked | n/a | zero if all weights zero | `regret_weight` required | existing regret tests golden | keep |
| `moves_left` | `PositionRecord.moves_left` | state row | none | completed game | `log1p(moves_left)/log1p(max_game_turns)` | no validation beyond numeric | n/a | zero if all weights zero | `moves_left_weight` required | add bounds test in Stage 3 | keep |
| `axis` | dominant winner axis | state row | valid label `>=0` | terminal-derived | class id `0..2`, `-1` invalid | invalid labels skipped | n/a | all invalid returns zero | optional phase/source | existing axis tests golden | keep optional |
| `axis_delta_norm` | axis policy prototype map | dense six-plane board map | none | replay auxiliary | MSE target | zeros if prototype unavailable | n/a | zero map allowed only if diagnostic disabled | explicit include flag | existing axis delta tests golden | keep optional |
| `lookahead_*` | `lookahead_values` in ring/sampler | state row | none | configured turn-boundary horizon | scalar binned by value loss | sampler currently falls back to value if missing | n/a | missing silently trains value-like target | weight by concrete horizon | add missing target negative test | keep family, delete fallback |

## Locked Semantic Decisions

### Pair Target Ordering

First-placement `policy_pair_joint` rows are unordered canonical pairs. The
target row identity is canonical `{cell_a, cell_b}` and the model output must be
symmetric for reversed row materialization.

`policy_pair_first` trains on a marginal over both cells in an unordered pair,
not only the syntactic `first` coordinate. The current implementation only adds
mass to `first`; Stage 3 must replace it or mark `policy_pair_first` diagnostic
until the marginal target is implemented. The selected Stage 1 decision is:
replace with marginal-over-both-cells and keep it trainable.

Second-placement `policy_pair_second` rows are ordered and conditional:
`known_first -> legal_second`. Any target whose first coordinate does not match
the known first placement is invalid.

### Duplicate Rows

- Duplicate candidate rows are invalid when active in a pair target row table.
- Duplicate legal rows are invalid at row-table construction.
- Duplicate target entries for the same valid row may be summed before
  normalization only when the row table identity is unchanged.
- Duplicate coordinates inside a pair are invalid.

### Zero Target Mass

Zero target mass is not an implicit skip. A trainable output with zero target
mass fails unless the resolved contract marks the sample/head non-trainable for
a named reason:

- truncated/no terminal value for value/regret;
- low-PCR policy excluded by `train_policy_on_full_search_only`;
- critical candidate overflow;
- optional diagnostic output not in the loss plan.

### Missing Weights And Phases

Policy, sparse policy, pair policy, opponent policy, value, regret, moves-left,
and phase-sensitive pair targets require explicit weights. Pair targets require
explicit `pair_phase`: `none`, `first_placement_joint`, or
`second_placement_known_first`.

### Lookahead Fallback

Synthetic fallback from missing lookahead targets to `value` is deleted. A
configured `lookahead_h` head requires a configured horizon target `h`, a loss
weight, and a target tensor for every trainable sample.
