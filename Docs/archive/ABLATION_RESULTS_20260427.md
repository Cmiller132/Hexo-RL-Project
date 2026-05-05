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
| model_160x20 | 1 | 177.5752 | 0.0065 | 2.32 | 7.2 | 922.2 | 22528 | 18.9 |
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

## Optimization Decision Log

The live suite was stopped after `model_160x20` epoch 1 so later measurements would not mix old-code and new-code variants.

- Implemented now: FP16-safe arena/dashboard model inference, arena reason-count reporting, radius-8 legal-move fast path, legal-move buffer reuse in encoding, MCTS pending-buffer capacity reuse, and Python `astype(copy=False)` request submits.
- Deferred: sparse legal-prior transfer. It is invasive and likely saves only low single-digit end-to-end time unless profiling proves policy response copies dominate.
- Deferred: bucketed/static inference and CUDA graphs. Worth revisiting after batch histograms and persistent-server work; otherwise compile/graph warmup is paid too often.
- Deferred: persistent self-play workers. Existing `update_weights()` is useful, but full worker pause/resume plus shared-memory namespacing is invasive for roughly 5% on short ablations and less on production epochs.
- Deferred: automatic startup sweep as default. Best shape is an opt-in cached sweep over workers, leaf batch, and wait time after heuristic autotune.
- Fixed measurement correctness: prior model-vs-classical eval numbers are invalid because CUDA FP16 models received FP32 eval inputs and crashes were scored as losses.
