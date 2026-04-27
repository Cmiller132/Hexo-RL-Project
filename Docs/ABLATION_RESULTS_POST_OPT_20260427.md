# Hexo-RL Long Ablation Results

Suite root: `runs/ablations_priority_20260427_v3_post_opt`

This document is generated from the suite JSONL summaries while the run is in progress. Treat partially completed ablations as early signals, not final conclusions.

## Current Leaders

- Lowest latest loss: `baseline_128x16_noise025` at epoch 1 with loss `76.9154`.
- Fastest latest self-play: `baseline_128x16_noise025` at `11.51` games/min.

## Latest Metrics

| ablation | epoch | loss | top1 | train_bps | games_min | pos_min | buffer | elapsed_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_128x16_noise025 | 1 | 76.9154 | 0.0064 | 3.1 | 11.51 | 1472.8 | 22656 | 12.09 |

## Interpretation Notes

- Compare ablations primarily after they reach the same epoch count; early epochs are dominated by bootstrap and replay composition.
- Throughput should be judged alongside loss and evaluation, because faster search settings may produce weaker targets.
- Model-size variants are expected to change both training speed and MCTS/inference throughput, so wall-clock progress matters as much as per-epoch loss.

## Improvement Ideas To Revisit

- Add sparse policy transfer from inference to MCTS so workers receive priors only for legal moves rather than full 1089-logit vectors.
- Add optional bucketed inference batches for CUDA graph or compile-friendly static shapes, then ablate padding cost versus compile speedup.
- Keep train compile only if the multi-epoch ablation shows the warmup cost amortizes cleanly.
- Add checkpoint-vs-checkpoint arenas between ablations once several variants complete, not only model-vs-classical smoke eval.
