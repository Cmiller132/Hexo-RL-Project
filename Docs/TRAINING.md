# Training Pipeline Guide

## Quick Start

```bash
# Build Rust extension
cargo build -p hexgame-py --release
maturin develop --features python

# Run training with small test config
python -m hexorl.cli epoch configs/small_test.toml
```

## Configuration

See `configs/` directory for example config files:
- `small_test.toml` — 32ch, 4 blocks, 100 batches — for smoke testing
- `default.toml` — 128ch, 16 blocks, 2000 batches — production
- `production.toml` — 128ch, 16 blocks, multi-head — full training
- `reproducible.toml` — deterministic mode for debugging

## Epoch Structure

Each epoch consists of:
1. **Self-play** — N workers play M games each, pushing positions to ring buffer
2. **Training** — K batches drawn from buffer with recency weighting, model updated
3. **Evaluation** — Arena matches against reference model, ELO computed
4. **Checkpoint** — Model, optimizer, EMA, and config saved

## Loss Functions

| Head | Loss | Weight (default) |
|------|------|------------------|
| policy | Cross-entropy vs MCTS visits | 1.0 |
| value | Soft XE (65 bins) vs outcome | 1.5 |
| lookahead_6 | Soft XE vs EMA MCTS values | 0.15 |
| lookahead_12 | Soft XE vs EMA MCTS values | 0.15 |
| lookahead_36 | Soft XE vs EMA MCTS values | 0.10 |
| regret_rank | RGSC ranking loss (Eq 7) | 0.10 |
| regret_value | Soft XE vs computed regret | 0.10 |
| opp_policy | Cross-entropy (auxiliary) | 0.15 |
| axis | Cross-entropy (3-class) | 0.05 |
| entropy | Negative entropy (regularizer) | 0.01 |

## Value Target Computation

Value targets use KataGo-style EMA lookahead at turn-boundary horizons:
- `V_target(t) = (1-λ) * V_MCTS(t) + λ * V_target(t + horizon)`
- λ values: 0.75 (short ~4 turns), 0.90 (mid ~10 turns), 0.97 (long ~36 turns)
- Turn boundaries detected by `placements_remaining == 2`
- Targets reframed to source player's perspective

## RGSC (Regret-Guided Search Control)

For each self-play game:
1. Evaluate all MCTS search nodes + trajectory states through regret network
2. Select highest-regret state
3. Insert/update in Prioritized Regret Buffer (PRB, capacity K=100)
4. With probability β=0.5, next game starts from PRB-sampled state
5. PRB uses EMA regret updates: `R_new = 0.5*R_old + 0.5*R_observed`
6. Sampling: softmax over `R^(1/τ)` with τ=0.1

## Monitoring

Key metrics to track:
- `games_per_min` — self-play throughput
- `samples_per_min` — training data generation rate
- `loss_total` — should decrease monotonically
- `loss_policy`, `loss_value` — per-head convergence
- `elo_diff` — arena ELO vs previous checkpoint
- `gpu_utilization` — should be >80% at batch=64
