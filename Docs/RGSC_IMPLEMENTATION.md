# RGSC / Regret Adaptation Status

Source: `Docs/2602.20809v1.txt` and the RGSC paper
"Regret-Guided Search Control for Efficient Learning in AlphaZero."

This document is intentionally strict: the current workspace should not call
RGSC "complete" unless the full search-control loop is implemented, wired,
tested, and observable. Regret heads alone are useful, but they are not full
RGSC.

## Current Status

```text
regret_rank head: implemented as an auxiliary scalar head
regret_value head: implemented as a binned auxiliary head
selected_action_value recording: implemented in current self-play records
regret-biased replay sampling: implemented through RingBuffer regret_fraction
PrioritizedRegretBuffer class: exists, but is not the active self-play restart path
full RGSC search control: not complete
```

The active training system currently uses regret mostly as an auxiliary target
and replay-sampling bias. That is a good intermediate diagnostic, but it is not
paper-complete RGSC.

## Paper Contract

A no-compromise RGSC adaptation for Hexo requires all of these pieces:

```text
1. Compute selected-action regret from completed trajectories.
2. Train regret_rank to score high-learning-potential states.
3. Train regret_value to estimate regret magnitude.
4. Maintain a prioritized regret buffer of restart states.
5. Sample from that buffer during self-play with RGSC probability beta.
6. Reconstruct the exact sampled game state and resume MCTS from it.
7. Periodically re-evaluate buffer entries and update regret by EMA.
8. Persist PRB entries, update history, and sampling events.
9. Report RGSC restart rate, value of sampled states, staleness, and win/loss
   quality separately from normal opening self-play.
```

Hexo-specific additions:

- restart states must preserve the one-stone opening and later two-placement
  turn structure;
- restart cannot begin from an illegal mid-turn state unless the engine can
  exactly restore whose first/second placement is pending;
- the regret target must use the selected child's value for the action actually
  played, not only the root value;
- opponent policy and lookahead labels generated from restarted games must
  still respect full-search opponent turns and perspective conversion;
- D6 augmentation must transform restarted histories, sparse policies, pair
  policies, and regret labels together.

## Equation Mapping And Current Gaps

### Equation 2: Trajectory Regret

Paper form:

```text
R(s_t) = (1 / (T - t)) * sum_{i=t..T} (V_selected(s_i) - z)^2
```

Current code:

```text
Python/src/hexorl/buffer/regret_buffer.py::compute_regret()
Python/src/hexorl/buffer/targets.py::_assign_auxiliary_targets()
```

Current gap:

- `selected_action_value` is recorded in self-play and the active target path
  uses a suffix average, but missing selected values can still fall back to
  root value in some helper paths.
- Truncated/max-move games can have pseudo-outcome `0.0`. Those records should
  keep policy/structure data, but regret heads should receive zero weight
  because Hexo has no real draw outcome.
- A finished production path should treat missing selected-action value in new
  records as invalid or set regret weight to zero. Silent fallback should be
  limited to explicitly marked legacy/research records.

Required tests:

```text
test_rgsc_regret_uses_selected_child_value
test_rgsc_missing_selected_value_zeroes_regret_weight
test_rgsc_regret_suffix_average_matches_hand_case
test_rgsc_regret_zero_weight_for_truncated_games
```

### Equation 3: Restart Distribution

Paper idea:

```text
rho(s | S) is induced by the regret-rank network over candidate states
```

Current code:

```text
Python/src/hexorl/buffer/regret_buffer.py::PrioritizedRegretBuffer.sample()
Python/src/hexorl/buffer/ring.py::sample_regret_indices()
```

Current gap:

- `PrioritizedRegretBuffer.sample()` exists, but the active self-play worker is
  not sampling PRB restart states and resuming games from them.
- `RingBuffer.sample_regret_indices()` biases training batches toward
  high-regret replay rows. That is prioritized replay, not RGSC search control.

Required implementation:

- Add an owned RGSC service used by self-play orchestration.
- At the start of a self-play game, with probability `rgsc_beta`, sample a PRB
  entry and restore the `HexGame` from its compact move history.
- Validate restored game legality, current player, and whether the position is
  at a turn boundary or mid-turn.
- Log every restart attempt, success, rejection, and resulting terminal reason.

Required tests:

```text
test_rgsc_restart_samples_from_prb_when_beta_one
test_rgsc_restart_restores_current_player_and_turn_phase
test_rgsc_restart_rejects_illegal_or_stale_history
```

### Equation 7: Ranking Loss

Current code:

```text
Python/src/hexorl/train/losses.py::regret_rank_loss()
```

Current status:

- The loss shape follows the paper-style batch distribution objective.
- The full adaptation is incomplete until the scores drive PRB candidate
  selection and restart-state ranking, not only an auxiliary loss.

Required implementation:

- During PRB insertion, score candidate states with the current/EMA regret-rank
  network.
- Candidate states must include both played trajectory states and eligible MCTS
  tree states, because RGSC search control is meant to restart from high-regret
  learning opportunities discovered during search, not only from positions that
  happened to be played.
- Select or refresh PRB entries using rank score plus observed regret according
  to the paper's intended control loop.
- Persist rank score, observed regret, EMA regret, model checkpoint used for
  scoring, and staleness.

Required tests:

```text
test_regret_rank_scores_drive_prb_selection
test_rgsc_scores_tree_node_candidates_for_prb
test_prb_entry_persists_rank_score_and_observed_regret
test_rgsc_restart_distribution_changes_when_rank_scores_change
```

### Equation 13: EMA Regret Update

Current code:

```text
Python/src/hexorl/buffer/regret_buffer.py::PrioritizedRegretBuffer.update_regret()
```

Current gap:

- The method exists, but no active loop proves entries are re-evaluated after
  replay, updated by EMA, and kept/dropped based on the refreshed value.

Required implementation:

- After a PRB-sampled restart game finishes, recompute regret for the sampled
  entry and update it with the configured EMA alpha.
- Track `rgsc_staleness`, `rgsc_refresh_count`, `rgsc_ema_delta`, and
  `rgsc_eviction_reason`.

Required tests:

```text
test_prb_ema_update_after_restart_game
test_prb_eviction_prefers_lower_ema_regret
test_rgsc_metrics_report_staleness_and_refresh_count
```

## Full Implementation Checklist

The regret adaptation is complete only when all of this is true:

```text
selected_action_value is mandatory in new production records
regret targets have explicit weights and no silent root-value fallback
regret_rank and regret_value heads are trained when configured
PRB is owned by self-play/orchestration, not just defined in a helper file
PRB restart sampling is active and configurable
restart states restore Hexo two-placement turn phase exactly
played trajectory and MCTS tree candidate states can enter PRB
PRB entries are refreshed with EMA regret updates
rank scores are used for state selection, not only logged
RGSC metrics are persisted to SQLite/JSONL/dashboard
truncated games do not train regret heads against pseudo-draw outcomes
D6 tests cover regret labels and restarted histories
end-to-end test proves beta=1 starts from PRB states
```

Until then, describe the system as:

```text
regret auxiliary heads + regret-biased replay sampling
```

not:

```text
full RGSC search control
```
