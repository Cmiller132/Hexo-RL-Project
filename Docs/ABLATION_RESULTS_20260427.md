# Hexo-RL Long Ablation Results

Suite root: `runs/ablations_priority_20260427_v2`

This document is generated from the suite JSONL summaries while the run is in progress. Treat partially completed ablations as early signals, not final conclusions.

## Current Leaders

- Lowest latest loss: `model_64x8` at epoch 12 with loss `5.4717`.
- Fastest latest self-play: `model_64x8` at `33.08` games/min.

## Latest Metrics

| ablation | epoch | loss | top1 | train_bps | games_min | pos_min | buffer | elapsed_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_128x16_noise025 | 12 | 6.6487 | 0.1136 | 3.19 | 12.66 | 1619.4 | 100000 | 10.66 |
| model_64x8 | 12 | 5.4717 | 0.1219 | 2.32 | 33.08 | 4222.3 | 100000 | 4.6 |
| model_96x12 | 12 | 5.8112 | 0.106 | 2.7 | 21.21 | 2708.9 | 100000 | 6.77 |

## Interpretation Notes

- Compare ablations primarily after they reach the same epoch count; early epochs are dominated by bootstrap and replay composition.
- Throughput should be judged alongside loss and evaluation, because faster search settings may produce weaker targets.
- Model-size variants are expected to change both training speed and MCTS/inference throughput, so wall-clock progress matters as much as per-epoch loss.

## Improvement Ideas To Revisit

- Add sparse policy transfer from inference to MCTS so workers receive priors only for legal moves rather than full 1089-logit vectors.
- Add optional bucketed inference batches for CUDA graph or compile-friendly static shapes, then ablate padding cost versus compile speedup.
- Keep train compile only if the multi-epoch ablation shows the warmup cost amortizes cleanly.
- Add checkpoint-vs-checkpoint arenas between ablations once several variants complete, not only model-vs-classical smoke eval.
