# Runtime Inventory

Runtime consumers currently infer behavior from architecture strings, model
classes, head names, and shared-memory flags. Stage 4 must replace these with
resolved providers and executable pair strategies.

## Consumption Paths

| Consumer | Required policy outputs | Required pair outputs | Row expectations | MCTS API used | Engine alignment assumptions | Value assumptions | Telemetry emitted | New provider or strategy | Old branch to delete |
|---|---|---|---|---|---|---|---|---|---|
| dense self-play root | `policy` | none | dense board row index maps through root offset/legal bytes | `expand_root` | legal bytes and crop offset identify legal dense logits | scalar in current-player perspective, `[-1,1]` | prior source dense/default, root stats | dense policy provider | direct dense branch in worker |
| sparse self-play root/leaf | `policy`, `sparse_policy` | optional crop `pair_policy` only by strategy | candidate qr rows align with legal bytes; candidate row count <= 512 | `expand_root_with_sparse_priors`, `expand_and_backprop_with_sparse_sources` | candidate rows contain legal/tactical actions; fallback dense priors exist | same as dense | sparse candidate recall, sparse source fractions | candidate policy provider | sparse-prior-stage branch checks |
| graph self-play root | `policy_place` | none by default | graph legal qr must match Rust legal bytes exactly, order may be realigned after set validation | `expand_root_with_global_priors` | graph legal rows are all Rust legal rows; duplicate/missing/extra rows fail | same scalar decode | global prior source metadata | global legal-row provider | `GlobalHexGraphNet.is_global_graph_architecture` branch |
| graph pair first strategy | `policy_place`, `policy_pair_first` | legal-row pair-first logits | same legal table as root | `apply_root_pair_first_priors` | pair-first logits cover every root child | same as graph | root pair source fraction | `diagnostic_full_pair` or future pair-first strategy | raw `policy_pair_first` key check |
| graph pair joint strategy | `policy_pair_joint` chunks | pair qr rows `(q1,r1,q2,r2)` | unordered pair rows over current legal rows | `apply_root_pair_priors` or blend to action logits | pair rows must match legal-row table and strategy cap | n/a | root pair candidate count/source | executable pair strategy | `_score_graph_pair_chunks` direct call from worker |
| graph pair second strategy | `policy_pair_second` chunks | known-first pair rows | first is previous stone token, second is legal row | `apply_root_pair_second_priors` | known-first matches current turn phase | n/a | second pair source | executable pair strategy with phase validation | direct second-placement branch |
| crop pair strategy | `pair_policy` chunks | candidate pair logits | candidate row table plus pair row table | `apply_root_pair_priors`, `apply_root_pair_second_priors`, or blended action logits | candidate rows must include known first and legal seconds | n/a | pair source counters | crop pair strategy | `_score_crop_pair_chunks` direct call |
| trainer | all trainable outputs | trainable pair outputs | target row tables match output row tables | n/a | graph batches cannot consume dense policy fields | value target current-player perspective | per-head losses | training adapter plus loss plan | `isinstance(GlobalHexGraphNet)` and raw loss switch |
| inference server | runtime requested outputs | pair outputs only when requested | transport rows validated by adapter | n/a | response row identity must match request | value decoder declared by output contract | batching/timing stats | inference protocol adapter | model-class graph mode and head flags |
| dashboard/eval model players | dense policy/value, optional pair diagnostics | optional pair logits | checkpoint/config-derived outputs | local model call | model dtype and legal mapping | scalar decode | dashboard debug payloads | local inference provider | ad hoc model cache/head assumptions |

## Pair Strategy Decisions

`none`

- Does not request pair rows.
- Ignores pair outputs even if the architecture can produce them.
- Fails if config asks for pair influence with no strategy.

`diagnostic_full_pair`

- Requires explicit `pair_strategy_max_pairs > 0` and `pair_prior_mix > 0`.
- Requests pair chunks only at root positions where the strategy is configured
  to observe or apply pair behavior.
- Records pair logits and source counters.
- May blend into action priors only through the strategy, never by head
  presence.

Planned variants for later stages:

- `pair_first_blend`: uses `policy_pair_first` only over legal rows.
- `pair_joint_marginal_blend`: scores unordered first-placement pairs and
  marginalizes to legal action priors.
- `pair_second_conditional_blend`: scores known-first second-placement rows.

Each pair strategy must declare:

```text
required architecture capabilities
requested model outputs
row table builders
pair phase
MCTS API calls
fallback behavior
telemetry keys
hard error conditions
```

No strategy may be selected implicitly by head presence.
