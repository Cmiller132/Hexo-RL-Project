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
PrioritizedRegretBuffer class: owned by the self-play RGSC restart service
full RGSC search control: implemented for active self-play workers
```

The active training system now has the paper RGSC loop: completed trajectories
produce selected-action regret targets, the active regret heads score trajectory
and MCTS-tree candidate states, one rank-selected candidate is admitted to PRB
after non-restart games, PRB openings are sampled by stored regret, and sampled
openings are refreshed by EMA after replay.

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
Python/src/hexorl/selfplay/regret_buffer.py::compute_regret()
Python/src/hexorl/replay/codec.py::_assign_auxiliary_targets()
Python/src/hexorl/selfplay/game_runner.py::_attach_rgsc_ranked_candidates()
```

Current status:

- Production RGSC candidate scoring uses `selected_action_value`; missing
  selected values make trajectory regret invalid for candidate admission.
- Replay targets use the same suffix-average formula and set regret weight to
  zero for truncated/no-outcome games.
- The legacy `compute_regret(..., allow_root_value_fallback=True)` escape hatch
  remains explicit and is not used for active production RGSC scoring.

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
Python/src/hexorl/selfplay/regret_buffer.py::PrioritizedRegretBuffer.sample()
Python/src/hexorl/selfplay/rgsc.py::RGSCRestartService.maybe_restart()
```

Current status:

- `RGSCRestartService` is constructed by the self-play runner and samples PRB
  openings with probability `rgsc_beta`.
- PRB sampling follows the paper's regret-temperature distribution over stored
  EMA regret values.
- Restart restore validates compact history, current player, terminal status,
  move cap, and Hexo's one-placement/two-placement turn phase.
- Restart attempts, successes, rejections, EMA deltas, staleness, insertions,
  and PRB snapshots are recorded through self-play metrics/dashboard payloads.

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
Python/src/hexorl/selfplay/game_runner.py::_attach_rgsc_ranked_candidates()
Python/src/hexorl/inference/client.py::evaluate_regret_heads()
```

Current status:

- The loss shape follows the paper-style batch distribution objective.
- The self-play runner scores played trajectory histories and extracted MCTS
  tree-node histories with the active regret-rank head.
- The highest rank-scored candidate is selected for PRB admission after
  non-restart games.
- If the selected candidate is on the played trajectory, its stored regret is
  the true Equation 2 trajectory regret; if it is tree-only, its stored regret
  is the active regret-value head estimate.
- PRB entries preserve rank score, observed regret, EMA regret, source, sample
  and update steps, refresh count, and dashboard snapshot fields.

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
Python/src/hexorl/selfplay/regret_buffer.py::PrioritizedRegretBuffer.update_regret()
Python/src/hexorl/selfplay/rgsc.py::RGSCRestartService.observe_game()
```

Current status:

- After a PRB-sampled restart game finishes, the starting position's recomputed
  trajectory regret refreshes the sampled entry using the configured EMA alpha.
- Restart games update the sampled opening; non-restart games admit the
  rank-selected candidate.
- Metrics track PRB refreshes, EMA delta, staleness, restart success/rejection,
  and tree-node candidate insertions.

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
