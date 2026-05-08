# RGSC / Regret Adaptation Status

Source reference: `Docs/archive/2602.20809v1.txt` and the RGSC paper, "Regret-Guided Search Control for Efficient Learning in AlphaZero."

This document describes the current implementation honestly. Hexo currently has regret heads, regret targets, regret-biased replay, and an active restart service, but it should still be described as an in-progress RGSC adaptation unless every paper-level search-control requirement is validated end to end.

## Current Status

```text
regret_rank head: implemented as an auxiliary scalar head
regret_value head: implemented as a binned auxiliary head
selected_action_value recording: implemented in self-play records
regret target assignment: implemented in buffer target processing
regret-biased replay sampling: implemented in RingBuffer sampling
RGSCRestartService: implemented and wired into SelfPlayWorker
PRB restart attempts: active when selfplay.rgsc_beta > 0
PRB insertion from completed games: implemented
PRB insertion from MCTS tree node candidates: implemented when extraction/scoring is available
PRB EMA refresh after restart games: implemented in service path
full paper-complete RGSC validation: not complete
```

Use this wording for current status:

```text
regret auxiliary heads + regret-biased replay + experimental RGSC restart service
```

Do not call it fully paper-complete RGSC until the acceptance checklist at the end of this document is satisfied by tests, metrics, and long-run evidence.

## Main Code Paths

| Concern | Code |
|---|---|
| RGSC service | `Python/src/hexorl/selfplay/rgsc.py::RGSCRestartService` |
| Worker integration | `Python/src/hexorl/selfplay/worker.py` |
| Self-play record fields | `Python/src/hexorl/selfplay/records.py` |
| Regret target processing | `Python/src/hexorl/buffer/targets.py` |
| Replay regret sampling | `Python/src/hexorl/buffer/ring.py` |
| Regret losses | `Python/src/hexorl/train/losses.py` |
| Dashboard metrics capture | `Python/src/hexorl/dashboard/recorder.py` |

## Configuration

RGSC-related config fields currently live under `[selfplay]`:

```toml
rgsc_beta = 0.0
rgsc_prb_capacity = 100
rgsc_prb_temperature = 0.1
rgsc_prb_ema_alpha = 0.5
```

Behavior:

- `rgsc_beta = 0.0` disables restart sampling.
- Nonzero `rgsc_beta` allows the worker to attempt PRB restarts.
- PRB capacity, sampling temperature, and EMA alpha are validated by config schema.

## Current Worker Flow

At worker startup:

```text
SelfPlayWorker creates RGSCRestartService from selfplay config.
```

At game start:

```text
rgsc.maybe_restart() may sample a PRB entry.
If accepted, the worker restores a HexGame from compact move history and starts MCTS from that state.
If rejected or disabled, the worker starts from a fresh HexGame.
```

During/after game:

```text
self-play records selected_action_value, root value, policies, pair targets, and RGSC metadata.
if enabled, tree node histories can be extracted and scored for PRB candidate insertion.
rgsc.observe_game() inserts or refreshes PRB entries after completed games.
rgsc metrics are attached to GameRecord and aggregated by the orchestrator.
```

Recorded metrics include:

```text
rgsc_prb_size
rgsc_restart_attempts
rgsc_restart_successes
rgsc_restart_rejections
rgsc_prb_insertions
rgsc_prb_refreshes
rgsc_tree_node_insertions
rgsc_last_ema_delta
rgsc_last_staleness
```

## What Is Already Good

The current implementation is beyond "just a regret head." It has:

- restart service wiring in self-play
- PRB entry storage and sampling
- restart attempt/success/rejection accounting
- EMA refresh fields
- tree node candidate insertion hook
- record-level RGSC metadata
- dashboard recorder support for RGSC metrics and PRB snapshots

## Remaining Risks And Gaps

### 1. Paper-complete restart quality still needs proof

The service can restart from PRB entries, but production confidence requires tests and long-run metrics proving:

- restored histories preserve current player and placement phase
- illegal/stale histories are rejected
- beta=1 reliably attempts PRB starts when the PRB has entries
- restarted games produce valid replay records and targets

### 2. Regret targets need strict missing-value behavior

The intended target should use selected-action value for the action actually played. Missing selected-action values in new production records should not silently become normal root-value regret.

Required behavior:

```text
new record missing selected_action_value -> regret target weight is zero or record is rejected for regret training
legacy/research fallback -> explicitly marked
truncated/max-move pseudo-outcomes -> regret heads receive zero weight unless a real terminal outcome exists
```

### 3. PRB scoring and refresh need stronger evidence

The service tracks refreshes and EMA deltas, but paper-level RGSC requires proof that rank scores and observed regret actually control PRB selection and refresh behavior.

Required evidence:

- rank scores affect PRB selection
- observed regret affects insertion/refresh
- stale entries are refreshed or evicted
- restart distribution changes when scores change

### 4. Tree-node candidate flow needs end-to-end validation

Worker code can extract tree node histories and insert candidates, but the paper expects high-regret opportunities discovered during search to influence restart distribution. This needs explicit tests and metrics.

## Required Tests Before Calling RGSC Complete

```text
test_rgsc_restart_samples_from_prb_when_beta_one
test_rgsc_restart_restores_current_player_and_turn_phase
test_rgsc_restart_rejects_illegal_or_stale_history
test_rgsc_regret_uses_selected_child_value
test_rgsc_missing_selected_value_zeroes_regret_weight
test_rgsc_regret_suffix_average_matches_hand_case
test_rgsc_regret_zero_weight_for_truncated_games
test_regret_rank_scores_drive_prb_selection
test_rgsc_scores_tree_node_candidates_for_prb
test_prb_entry_persists_rank_score_and_observed_regret
test_rgsc_restart_distribution_changes_when_rank_scores_change
test_prb_ema_update_after_restart_game
test_prb_eviction_prefers_lower_ema_regret
test_rgsc_metrics_report_staleness_and_refresh_count
test_rgsc_restart_game_produces_valid_replay_targets
```

## Completion Checklist

RGSC is complete only when all of this is true:

```text
selected_action_value is mandatory or explicitly zero-weighted for regret training
regret targets have explicit weights and no silent production fallback
regret_rank and regret_value heads train when configured
PRB restart sampling is active and configurable
restart states restore Hexo two-placement turn phase exactly
played trajectory and MCTS tree candidate states can enter PRB
PRB entries are refreshed with EMA regret updates
rank scores and observed regret affect state selection
RGSC metrics are persisted to SQLite/JSONL/dashboard
truncated games do not train regret heads against pseudo-draw outcomes
D6 tests cover regret labels and restarted histories
end-to-end test proves beta=1 starts from PRB states
long-run smoke shows restart games remain legal and useful
```

Until then, call it experimental RGSC restart support, not final paper-complete RGSC.
