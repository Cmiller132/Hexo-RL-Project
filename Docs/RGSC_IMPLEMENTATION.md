# RGSC Implementation Details

Exact mapping from the RGSC paper (arXiv:2602.20809v1) to our codebase.

## Equation 2 — Regret Definition

```
R(st) = (1/(T-t)) * Σ_{i=t}^T (V_selected(si) - z)^2
```

**Implementation:** `python/src/hexorl/buffer/regret_buffer.py:compute_regret()`

Computes average squared discrepancy between MCTS root value (V_selected) and game outcome (z) for each position in a completed game trajectory. Accumulated from position t to terminal T.

## Equation 3 — Restart Distribution

```
ρ(s|S) = exp(φ(s)) / Σ_s' exp(φ(s'))
```

**Implementation:** The regret ranking head `RegretRankHead` outputs unnormalized scores φ(s). The softmax is implicit in the ranking loss — not computed explicitly.

## Equation 7 — Ranking Loss

```
L_rank = -log( Σ_s exp( log(softmax(φ(s))) + R(s) ) )
```

**Implementation:** `python/src/hexorl/train/losses.py:regret_rank_loss()` (lines 77-81)

```python
log_softmax_scores = F.log_softmax(scores, dim=0)  # over batch
combined = log_softmax_scores + regrets              # additive bias
loss = -torch.logsumexp(combined, dim=0)             # -log Σ exp
```

This is an **exact match** to the paper. The batch dimension (dim=0) is used for the softmax and sum — the loss optimizes the distribution over all samples in the batch.

## Equation 13 — EMA Regret Update

```
R_new(si) = (1-α) × R_old(si) + α × R(si)
```

**Implementation:** `python/src/hexorl/buffer/regret_buffer.py:PrioritizedRegretBuffer.update_regret()`

α = 0.5 (from RGSC hyperparameter sweep, §D.4 of paper).

## Prioritized Regret Buffer (§3.3)

| Paper Spec | Our Implementation |
|-----------|-------------------|
| Capacity K | `capacity=100` (default) |
| Insert only if regret > min in buffer | `add()` method checks `is_full` and compares |
| Sampling: softmax over `R^(1/τ)` | `sample()` with `sampling_temperature=0.1` |
| Buffer rate β=0.5 | Configurable via `cfg.buffer.regret_fraction` |
| Select one state per game (highest rank) | `add()` called once per game with best state |

## Algorithm 1 (Appendix B)

The full RGSC algorithm is implemented across these modules:
- **Line 3-10 (buffer sampling):** `PrioritizedRegretBuffer.sample()`
- **Line 13-23 (self-play):** `selfplay/worker.py:SelfPlayWorker._play_one_game()`
- **Line 14 (MCTS):** `hexgame-core/src/mcts.rs:MCTSEngine`
- **Line 16-17 (regret network):** `model/network.py:RegretRankHead`
- **Line 24-30 (trajectory regret):** `regret_buffer.py:compute_regret()`
- **Line 31-35 (buffer update):** `PrioritizedRegretBuffer.update_regret()`
