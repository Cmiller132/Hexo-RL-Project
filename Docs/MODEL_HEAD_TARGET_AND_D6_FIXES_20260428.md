# Model Head Target And D6 Fixes - 2026-04-28

This document records the investigation into the seven model-head findings from the review pass, plus one additional inference test failure found during the broader test sweep. The issues below were confirmed against the current codebase and fixed in this pass.

## Verification Summary

| Area | Confirmed? | Fixed? | Regression coverage |
|---|---:|---:|---|
| Hexo lookahead turn boundaries | Yes | Yes | `test_hexo_turn_boundaries_follow_player_runs` |
| Lookahead value perspective | Yes | Yes | `test_lookahead_flips_future_player_perspective`, `test_mid_turn_lookahead_targets_next_turn_start` |
| Opponent policy target semantics | Yes | Yes | `test_opponent_policy_uses_next_full_search_opponent_turn_start` |
| Regret target semantics | Yes | Yes | `test_regret_uses_selected_action_value_and_raw_scale` |
| Sparse/pair D6 augmentation | Yes | Yes | `test_sparse_sampler_keeps_d6_enabled_and_transforms_candidates`, `test_sparse_d6_batch_trains_for_all_model_architectures` |
| Opponent policy loss weighting | Yes | Yes | `test_opp_policy_loss_uses_opponent_policy_weight`, `test_opp_policy_loss_skips_empty_targets` |
| Production axis head loss | Yes | Yes | Config load check plus production loss-weight update |
| Sparse head config contract | Yes | Yes | `test_sparse_policy_head_enables_sparse_data_contract` |
| Shared-memory sparse buffer names | Yes | Yes | Full `Python/tests` inference server suite |

Commands run:

```bash
.venv/bin/python -m pytest Python/tests/test_training_data_pipeline.py Python/tests/test_config_and_guardrails.py -q
.venv/bin/python -m pytest Python/tests -q
git diff --check
```

Result:

```text
60 passed
81 passed
git diff --check clean
```

## 1. Lookahead Turn Boundaries

### What Was Happening

`Python/src/hexorl/buffer/targets.py::_turn_boundary_indices()` treated every even placement index as a new turn boundary. That is wrong for Hexo.

Hexo turn structure is:

- Opening: player 0 places one stone.
- After opening: players take two placements per turn.

So the player sequence is normally:

```text
0, 1, 1, 0, 0, 1, 1, ...
```

The true turn-start boundaries are:

```text
0, 1, 3, 5, ...
```

not:

```text
0, 2, 4, 6, ...
```

### Why It Matters

Lookahead heads are trained on future values at turn horizons. If the boundary indices are wrong, the model learns a future value target from the wrong phase of the turn. This is especially bad after the opening, because every two-placement turn is shifted.

### Code Fix

Changed `_turn_boundary_indices()` to detect player-run starts:

```python
return [
    i
    for i, pos in enumerate(positions)
    if i == 0 or pos.player != positions[i - 1].player
]
```

For non-boundary positions inside a two-placement turn, `compute_ema_lookahead()` now uses `bisect_right(boundaries, i) + horizon - 1` so horizon 1 points to the next turn start.

### Tests

`test_hexo_turn_boundaries_follow_player_runs()` verifies `[0, 1, 1, 0, 0] -> [0, 1, 3]`.

`test_mid_turn_lookahead_targets_next_turn_start()` verifies the second placement of a turn looks to the next player-run start.

## 2. Lookahead Value Perspective

### What Was Happening

`PositionRecord.root_value` is from the current player's perspective. The old lookahead EMA blended a future position's `root_value` directly into the current position target. If the future position belonged to the opponent, the sign was wrong.

Example:

```text
P0 position root target looks ahead to P1 position root_value = +0.6
From P0's perspective, that future value is -0.6
```

### Why It Matters

Binned lookahead heads are supposed to teach short/mid/long-term value from the current position's perspective. Mixing future values without sign conversion turns some winning futures into losing labels and vice versa.

### Code Fix

`compute_ema_lookahead()` now flips the future EMA value when the target position's player differs:

```python
future = result[j]
if positions[j].player != positions[i].player:
    future = -future
result[i] = (1.0 - lambda_) * mcts_values[i] + lambda_ * future
```

### Tests

`test_lookahead_flips_future_player_perspective()` verifies horizon-1 lookahead from P0 to P1 flips sign.

## 3. Opponent Policy Target

### What Was Happening

`_assign_auxiliary_targets()` copied `positions[i + 1]` as the opponent policy target. In Hexo, `i + 1` can be the same player's second placement, not the opponent's next turn.

### Desired Behavior

Opponent policy should be trained from the next full-search opponent turn start.

The source turn's PCR status does not matter. A low-PCR source position can train `opp_policy` if the future opponent turn target was full-search.

### Code Fix

Added `_next_full_search_opponent_turn_start()`:

```python
def _next_full_search_opponent_turn_start(positions, i):
    player = positions[i].player
    for j in range(i + 1, len(positions)):
        if positions[j].player == player:
            continue
        if j > 0 and positions[j - 1].player == positions[j].player:
            continue
        if positions[j].is_full_search:
            return j
    return None
```

The target assignment now copies only that future full-search opponent turn. It also sets `PositionRecord.opp_policy_weight = 1.0` when such a target exists and `0.0` otherwise.

`ReplayDataset` emits `opp_policy_weight`, and `opp_policy_loss()` skips empty target rows.

### Tests

`test_opponent_policy_uses_next_full_search_opponent_turn_start()` covers low-PCR source positions using later full-search opponent targets.

`test_opp_policy_loss_uses_opponent_policy_weight()` and `test_opp_policy_loss_skips_empty_targets()` cover loss behavior.

## 4. Regret Rank And Regret Value

### What Was Happening

The previous regret label used `root_value` and then squashed `regret_value` into `[-1, 1]`:

```python
regret = mean((p.root_value - final_outcome_perspective) ** 2)
pos.regret_value = clamp(2.0 * regret - 1.0, -1.0, 1.0)
```

That is not the RGSC definition. The paper defines trajectory regret using the MCTS value of the selected action, averaged over the suffix of the trajectory:

```text
R(s_t) = average_{i=t..T} (V_selected(s_i) - z)^2
```

### Why It Matters

The selected action value is a sharper target than root value for regret. Root value can be high even if the sampled action was bad, or low even if the selected action was the important mistake. Squashing to `[-1, 1]` also destroys the actual nonnegative regret scale.

### Code Fix

Added `selected_action_value` to `PositionRecord`, compact record serialization, ring buffer storage, and self-play recording.

In self-play, after sampling `(q, r)`, the worker maps the selected move back to `engine.root_child_q_values()` and stores that child Q:

```python
selected_action_value = root_value
for child_q, child_r, child_value in zip(moves_q, moves_r, q_values):
    if int(child_q) == q and int(child_r) == r:
        selected_action_value = float(child_value)
        break
```

Regret now uses `selected_action_value` when present and keeps raw nonnegative regret:

```python
selected = p.selected_action_value if p.selected_action_value is not None else p.root_value
regret = mean((selected - perspective_outcome) ** 2)
pos.regret_rank = regret
pos.regret_value = regret
```

`regret_rank_loss()` now uses raw regret as the additive bias in the RGSC rank objective instead of batch min-max normalizing it.

`regret_value_loss()` now bins regret over `[0, 4]`, the natural squared-error range for values in `[-1, 1]`.

### Tests

`test_regret_uses_selected_action_value_and_raw_scale()` verifies selected-action value is used and raw regret can reach `4.0`.

## 5. D6 Augmentation For Sparse, Pair, RestNet, And Graph Models

### What Was Happening

D6 augmentation was explicitly disabled whenever sparse policy was enabled:

```python
self.use_symmetry = bool(use_symmetry) and not self.include_sparse_policy
```

`run_epoch()` also passed `use_symmetry=not cfg.model.sparse_policy`.

This affected:

- sparse policy;
- pair policy;
- graph_hybrid_0 models using sparse action heads;
- RestNet models using sparse action heads.

The guard existed because sparse global candidates were not transformed. If the guard were removed without more work, the tensor and dense policy could be transformed while `candidate_qr`, `candidate_indices`, `policy_target_v2`, and `pair_policy_target_v2` stayed in the original coordinate system.

### Why Direct Tensor Rotation Was Not Enough

The Rust encoder uses a dynamic crop offset. A 33x33 axial crop is not always centered at `(0, 0)`, and rotating an already-cropped axial rectangle can clip or misalign cells. The reliable path is to transform the compact move history and re-run the canonical encoder.

### Code Fix

Added D6 helpers in `Python/src/hexorl/buffer/sampler.py`:

- `_transform_history_bytes()`
- `_transform_policy_v2()`
- `_transform_pair_policy_v2()`

The sampler now:

1. Samples a D6 symmetry.
2. Transforms the compact move history.
3. Re-encodes the transformed history through the normal encoder path.
4. Transforms global policy v2 targets.
5. Transforms opponent policy v2 targets.
6. Transforms pair-policy v2 targets.
7. Builds sparse candidates from transformed legal moves and transformed targets.
8. Keeps D6 enabled for sparse and pair models.

`run_epoch()` and `tiny_smoke_run()` now pass `use_symmetry=True` regardless of sparse policy.

### Pair Candidate Padding Fix

`build_pair_candidate_batch()` used to treat padded `(0, 0)` candidate rows as real cells. It now accepts `candidate_mask` and only maps real candidate rows, while preserving the original candidate row indices used by `PairPolicyHead`.

### Tests

`test_sparse_sampler_keeps_d6_enabled_and_transforms_candidates()` verifies sparse candidates and sparse targets transform together.

`test_pair_candidate_builder_ignores_padded_candidate_rows()` verifies padded candidate rows cannot create fake pair targets.

`test_sparse_d6_batch_trains_for_all_model_architectures()` runs a finite forward/loss pass for `cnn`, `restnet`, and `graph_hybrid_0` with sparse D6 data.

## 6. Opponent Policy Loss Weighting

### What Was Happening

`policy`, `sparse_policy`, and `pair_policy` honored sample weights, but `opp_policy` did not.

### Correct Contract

`opp_policy` should use `opp_policy_weight`, not the source position's `policy_weight`.

This matches the desired PCR behavior:

- source turn low-PCR + future opponent full-PCR -> train opponent policy;
- source turn full-PCR + future opponent low-PCR -> do not train opponent policy.

### Code Fix

`opp_policy_loss()` now accepts an optional weight tensor and skips empty targets. `compute_losses()` passes `targets.get("opp_policy_weight")`.

## 7. Axis Head Configured But Untrained In Production

### What Was Happening

`Configs/production.toml` included the `axis` head but omitted an `axis` loss weight.

### Code Fix

Added:

```toml
axis = 0.05
```

to the production `loss_weights`.

## 8. Sparse Policy Head Config Contract

### What Was Happening

A config could list `"sparse_policy"` in `model.heads` without setting `model.sparse_policy = true`. The model could build the head, but the data path would not emit candidate tensors and the trainer would not train it.

### Code Fix

Config validation now treats either `"sparse_policy"` or `"pair_policy"` in `model.heads` as enabling the sparse data contract:

```python
if ("sparse_policy" in self.model.heads or "pair_policy" in self.model.heads) and not self.model.sparse_policy:
    self.model.sparse_policy = True
```

The sparse default loss weight is then added as before.

## 9. Shared-Memory Sparse Buffer Names

### What Was Happening

The full Python test suite exposed a macOS POSIX shared-memory failure:

```text
OSError: [Errno 63] File name too long: '/hexorl_req_candidate_features_0'
```

The new sparse-candidate shared-memory segment names were too long for this platform.

### Code Fix

`Python/src/hexorl/inference/shm_queue.py::_shm_name()` now uses short stable aliases:

```text
req_candidate_features -> hx_qcf_0
req_candidate_indices  -> hx_qci_0
res_sparse_logits      -> hx_rsl_0
```

Server and clients share the same mapping, so the protocol is unchanged except for shorter segment names.

## Remaining Notes

Update 2026-04-30: the RGSC path now includes the prioritized-regret-buffer
restart loop. Self-play scores played trajectory histories and extracted MCTS
tree histories with the active regret heads, admits the rank-selected candidate
after non-restart games, samples PRB openings by stored EMA regret, and refreshes
sampled openings by EMA after replay.

The D6 path now uses transformed history re-encoding, which is the safest route for sparse/graph_hybrid_0 models. If performance ever becomes a bottleneck, the optimization should be a tested Rust batch transform/re-encode helper, not a return to rotating already-cropped tensors.
