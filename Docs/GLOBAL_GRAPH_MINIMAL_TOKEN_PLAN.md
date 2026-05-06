# Minimal Global Graph Token Plan

## Intent

The global graph input should start from the smallest useful object set for
Hexo search, then add complexity only when an ablation proves the value. The
previous graph schema mixed core objects, tactical annotations, aggregate
summaries, and optional pair objects in one large token vocabulary and one
48-slot shared feature vector. That made the contract harder to reason about
and increased the chance that redundant tokens consumed attention budget
without improving policy quality.

The minimal schema treats tokens as real objects the model should reason over:

- `STATE`: one global readout token for value and auxiliary state heads.
- `TURN`: current player and placements-remaining context.
- `STONE`: an occupied cell from compact history.
- `LEGAL`: one legal action row and the policy-output anchor.
- `WINDOW6`: an active unblocked six-cell segment, the rule-level threat
  object for Hexo's six-in-a-row win condition.

Annotations such as hot cells, cover membership, line summaries, connected
components, and materialized pair actions are removed from the starting runtime
schema. Pair policy still exists, but pair rows are represented by references
to existing `LEGAL` or known-first `STONE` tokens rather than by pair tokens in
the attention sequence.

## Implementation Changes

- Bumped the graph schema to v3.
- Bumped the relation schema to v2 because the relation vocabulary was narrowed.
- Reduced `GRAPH_FEATURE_DIM` from 48 to 12.
- Replaced the shared catch-all feature layout with:

```text
0  placements_remaining_norm
1  current_player_norm
2  owner_relative
3  move_age_norm
4  nearest_own_distance_norm
5  nearest_opponent_distance_norm
6  window_owner_relative
7  window_stone_count_norm
8  window_empty_count_norm
9  window_axis_norm
10 legal_window_count_norm
11 reserved_zero
```

- Removed runtime token creation for:
  - `PLAYER`
  - `HOT_CELL`
  - `LINE`
  - `COVER_SET`
  - `COMPONENT`
  - `PAIR_ACTION`
- Kept pair rows as non-materialized references:

```text
pair_first_indices[row]  -> first LEGAL token, or known-first STONE token
pair_second_indices[row] -> second LEGAL token
pair_token_indices[row]  -> always -1
```

- Rejected `materialize_pair_context_tokens=True` because `PAIR_ACTION` tokens
  would reintroduce an O(A^2) input-token path.
- Kept relation support for:
  - distance and direction buckets
  - same axis and same line
  - stone/legal membership in `WINDOW6`
  - same-window membership
  - recent move and age-order relations
  - first/second pair row references
- Updated `global_line_window_0` to gate legal actions from `WINDOW6` tokens
  only.

## Why `PAIR_ACTION` Was Removed

`PAIR_ACTION` represented one possible two-placement action as an input token,
for example "place at A and B together." It carried pair facts such as pair
distance, whether the pair touched winning/blocking cells, and pair membership.

The actual pair policy output did not require this token. Pair heads already
score rows from references to the first and second token vectors. Keeping pair
actions as input tokens risks turning all possible pairs into attention context:

```text
legal rows = A
unordered pair rows = A * (A - 1) / 2
```

That can explode token count, relation matrix size, memory use, and attention
cost. The minimal schema therefore removes `PAIR_ACTION` from runtime and keeps
pair reasoning in the pair heads.

## Success Criteria

- Legal policy rows still point to `LEGAL` tokens.
- Pair heads still score first/second token references.
- Threat-constrained legal-row filtering remains owned by the engine/legal
  table, not by extra tactical input tokens.
- `WINDOW6` carries enough rule-level tactical information to expose
  one-placement wins/blocks, two-placement wins/blocks, and developing
  three-stone material.
- Runtime graph batches contain only the five minimal token types.
- Materialized pair tokens are rejected deterministically.

## Follow-Up Ablations

Run the minimal schema as the baseline before adding back complexity:

1. Add `COVER_SET` only if the model struggles with multi-threat coverage or
   pair blocking after enough training signal.
2. Add component summaries only if connected-group reasoning is weak.
3. Add line summaries only if `WINDOW6` plus same-line relations underperform.
4. Keep `PAIR_ACTION` out of the main path unless testing a small bounded
   candidate-pair experiment.

## Performance Expectation

The meaningful performance improvement should come from reducing token count
and relation work, not mainly from shrinking scalar features. Attention and
relation memory scale with token count, while the feature projection is a much
smaller part of the cost. The smaller 12-slot feature vector should still help
debuggability and reduce projection noise.
