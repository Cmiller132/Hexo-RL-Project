# Hexo-RL Long Ablation Results

Suite root: `runs/ablations_priority_20260427_v2`

This document is generated from the suite JSONL summaries while the run is in progress. Treat partially completed ablations as early signals, not final conclusions.

## Current Status

No completed epoch summaries have been written yet.


## Interpretation Notes

- Compare ablations primarily after they reach the same epoch count; early epochs are dominated by bootstrap and replay composition.
- Throughput should be judged alongside loss and evaluation, because faster search settings may produce weaker targets.
- Model-size variants are expected to change both training speed and MCTS/inference throughput, so wall-clock progress matters as much as per-epoch loss.

## Improvement Ideas To Revisit

- Add sparse policy transfer from inference to MCTS so workers receive priors only for legal moves rather than full 1089-logit vectors.
- Add optional bucketed inference batches for CUDA graph or compile-friendly static shapes, then ablate padding cost versus compile speedup.
- Keep train compile only if the multi-epoch ablation shows the warmup cost amortizes cleanly.
- Add checkpoint-vs-checkpoint arenas between ablations once several variants complete, not only model-vs-classical smoke eval.
