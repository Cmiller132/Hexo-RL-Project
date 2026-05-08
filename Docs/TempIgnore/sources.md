# Sources

Date: 2026-04-28

This document tracks the external papers behind the newer architecture,
training, and optimization ideas in Hexo-RL. It is intentionally short: use it
as a map from project feature to source paper.

## ResTNet

**Paper:** Bridging Local and Global Knowledge via Transformer in Board Games
**Authors:** Yan-Ru Ju, Tai-Lin Wu, Chung-Chin Shih, Ti-Rong Wu
**Venue:** IJCAI 2025
**Links:** [arXiv](https://arxiv.org/abs/2410.05347), [IJCAI proceedings](https://www.ijcai.org/proceedings/2025/828)

**Used for:**

- `restnet` architecture family.
- Interleaving residual CNN blocks with spatial Transformer blocks.
- Testing whether global attention inside the fixed `33x33` crop improves the
  crop baseline before moving to a true global graph model.

## Regret / RGSC

**Paper:** Regret-Guided Search Control for Efficient Learning in AlphaZero
**Authors:** Yun-Jui Tsai, Wei-Yu Chen, Yan-Ru Ju, Yu-Hung Chang, Ti-Rong Wu
**Venue:** ICLR 2026
**Links:** [arXiv](https://arxiv.org/abs/2602.20809), [OpenReview](https://openreview.net/forum?id=Eoiu5iJD71)

**Used for:**

- `regret_rank` head.
- `regret_value` head.
- Prioritized regret buffer ideas.
- Restart/search-control training from high-regret states.

Local copy:

```text
Docs/2602.20809v1.txt
Docs/RGSC_IMPLEMENTATION.md
```

## PB2

**Paper:** Provably Efficient Online Hyperparameter Optimization with Population-Based Bandits
**Authors:** Jack Parker-Holder, Vu Nguyen, Stephen Roberts
**Venue:** NeurIPS 2020
**Links:** [arXiv](https://arxiv.org/abs/2002.02518), [code](https://github.com/jparkerholder/PB2)

**Used for:**

- Phase 3 dynamic schedule tuning.
- Population-based online hyperparameter optimization.
- The planned real PB2 scheduler path, distinct from plain PBT.

## ASHA

**Paper:** Massively Parallel Hyperparameter Tuning
**Authors:** Liam Li, Kevin Jamieson, Afshin Rostamizadeh, Ekaterina Gonina, Moritz Hardt, Ben Recht, Ameet Talwalkar
**Links:** [OpenReview](https://openreview.net/forum?id=S1MAriC5F7), [ML@CMU blog](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/)

**Used for:**

- Phase 3 ASHA-style static narrowing.
- Early stopping / promotion rungs for architecture and config sweeps.

## BOHB

**Paper:** BOHB: Robust and Efficient Hyperparameter Optimization at Scale
**Authors:** Stefan Falkner, Aaron Klein, Frank Hutter
**Venue:** ICML 2018
**Links:** [PMLR](https://proceedings.mlr.press/v80/falkner18a.html), [arXiv](https://arxiv.org/abs/1807.01774)

**Used for:**

- Phase 3 BOHB-style static finalist narrowing.
- Combining bandit-style resource allocation with Bayesian/config-model
  guidance.

## Mish

**Paper:** Mish: A Self Regularized Non-Monotonic Activation Function
**Author:** Diganta Misra
**Link:** [arXiv](https://arxiv.org/abs/1908.08681)

**Used for:**

- Legacy Hexagon model inspiration.
- Potential activation-function ablations if specialized trunk variants are
  reintroduced.

Current rewrite note:

- The active rewrite mainly uses ReLU, SiLU, and GELU in current model paths.
- Mish is a source for legacy/possible ablation work, not a current core
  dependency.

## RepVGG

**Paper:** RepVGG: Making VGG-style ConvNets Great Again
**Authors:** Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, Jian Sun
**Venue:** CVPR 2021
**Links:** [arXiv](https://arxiv.org/abs/2101.03697), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)

**Used for:**

- Legacy Hexagon RepVGG-style branch merge idea.
- Potential inference-time reparameterization of convolution branches.

Current rewrite note:

- RepVGG branch merging is not part of the current clean training path.
- If reintroduced, it should be implemented as a complete, tested model variant
  with checkpoint/load/inference parity.
