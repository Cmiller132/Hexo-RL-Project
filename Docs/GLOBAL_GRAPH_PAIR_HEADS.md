# Global Graph Pair Heads

This document describes the current global graph pair-head contract used by
`Python/src/hexorl/model/global_graph.py`, graph batching, replay sampling, and
training losses.

## Purpose

Hexo can place two stones on a turn after the opening. The global graph model
therefore exposes pair-aware policy heads in addition to the normal
single-placement `policy_place` head:

- `policy_pair_first`: scores the first placement row among current legal rows.
- `policy_pair_joint`: scores a canonical unordered first-placement pair row.
- `policy_pair_second`: scores the second placement row when the first
  placement is already known.

These heads are model outputs only. MCTS pair influence must still be enabled by
an explicit pair strategy; the presence of a pair head is not permission for
search to consume pair priors.

## Graph Rows

Pair rows are represented as references to existing `LEGAL` token rows:

- `pair_first_indices`: token index for the first row.
- `pair_second_indices`: token index for the second row.
- `pair_token_indices`: `-1` for non-materialized pair rows.

The default path uses non-materialized pair rows. This keeps pair actions out of
the attention token sequence and avoids making the relation matrix scale with
the number of pair candidates. Materialized pair context tokens are still
available behind `materialize_pair_context_tokens=True` for diagnostics, but
they are not the hot path.

## Head Semantics

### `policy_pair_first`

`policy_pair_first` emits one logit per legal current-position action. It can be
used to learn which stone should be chosen first before second-placement
conditioning is applied.

For `global_pair_twostage_0`, the legal vector is refined with the global state
before scoring:

```text
first_refined = legal_vector + MLP([legal_vector, state])
first_logit = Linear(first_refined)
```

Other global graph trunks score the legal vector directly.

### `policy_pair_joint`

`policy_pair_joint` emits one logit per canonical first-placement pair row. A
first-placement pair is unordered, so the joint scorer uses symmetric features:

```text
joint_features = [
  state,
  first_vector + second_vector,
  abs(first_vector - second_vector),
  first_vector * second_vector,
]
```

This makes the joint score invariant to first/second row order for unordered
pairs. Invalid rows are masked to `-80.0`.

### `policy_pair_second`

`policy_pair_second` emits one logit per known-first second-placement row. This
head is intentionally ordered and conditional:

```text
second_features = [
  state,
  first_condition_vector,
  second_condition_vector,
  abs(first_condition_vector - second_condition_vector),
]
```

For `global_pair_twostage_0`, `second_condition_vector` is refined with both the
candidate second row and the known first row:

```text
second_refined = second_vector + MLP([second_vector, first_refined, state])
```

The second head must not train on unordered first-placement rows.

## Training Targets

Graph batches carry two separate pair targets:

- `pair_policy_target`: the canonical joint pair target.
- `pair_second_policy_target`: the known-first second-placement target.

`pair_second_policy_target` is populated only when `placements_remaining == 1`.
For first-placement rows (`placements_remaining == 2`) it is all zeros, so
`policy_pair_second` loss is skipped. This prevents the conditional second head
from learning unordered first-placement pair tables.

Loss ownership:

- `policy_pair_joint` uses `pair_policy_target`.
- `policy_pair_second` uses `pair_second_policy_target`.
- `policy_pair_first` uses `pair_first_policy_target`.

## Output Head Gating

Config-built global graph models only compute requested heads. Important aliases:

- `policy` requests `policy_place`.
- `pair_policy` requests `policy_pair_first`, `policy_pair_joint`, and
  `policy_pair_second`.

Direct `GlobalHexGraphNet(...)` construction with `output_heads=None` preserves
the legacy all-head behavior for low-level tests.

## Performance Notes

The model pair-head hot path is now dominated by pair row count, not graph
construction. On the current benchmark position:

- graph construction with no pair rows: about 10 ms
- graph construction with 4096 non-materialized pair refs: about 14 ms
- graph construction with 256 materialized pair tokens: about 26 ms

CUDA forward time on an RTX 4070 Ti with 23,220 pair rows was about 4.0 ms for
`global_xattn_0`, 5.0 ms for `global_line_window_0`, 5.2 ms for
`global_graph_full_0`, and 6.1 ms for `global_pair_twostage_0`.

The pair-two-stage model is the most pair-specific but also the most expensive.
Use it as the main pair-head quality scout; use `global_xattn_0`,
`global_line_window_0`, and plain ResTNet as controls.

## Quality Expectations

The current pair head is sound for the implemented contract:

- unordered joint pairs are symmetric
- known-first second rows are conditional and not trained on unordered rows
- optional heads are gated
- pair rows no longer inflate the graph attention sequence by default

The architecture is still an MLP scorer over selected pair rows. A likely
stronger follow-up is a pointer or biaffine scorer that builds a full legal-row
interaction matrix:

```text
score(i, j) = h_i^T U h_j + w^T [h_i, h_j, state]
```

That style can score all legal pairs with batched matrix operations and gives a
cleaner conditional second-placement distribution.

Relevant references:

- Vinyals, Fortunato, and Jaitly, "Pointer Networks", 2015.
- Dozat and Manning, "Deep Biaffine Attention for Neural Dependency Parsing",
  2016.
- Lee et al., "Set Transformer", 2019.
- Battaglia et al., "Interaction Networks", 2016.
