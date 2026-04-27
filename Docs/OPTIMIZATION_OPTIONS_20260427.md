# Hexo-RL Optimization Options - 2026-04-27

This is a menu of realistic speed paths after profiling showed the 128x16 run is primarily model-forward bound.

## Highest-Impact Options

### 1. Replace Or Slim The Gated Residual Trunk

Current hotspot:

- `Python/src/hexorl/model/network.py::GatedResBlock.forward`
- Called from `HexNet.forward`
- About 94-96% of model-layer time.

Current block does three full 3x3 convolutions per residual block: two residual-path convs plus one gate conv. With 16 blocks at 128 channels, this dominates inference.

Scratch benchmark, 128 channels, 16 blocks, batch 64:

| variant | ms/forward | positions/s | params |
| --- | ---: | ---: | ---: |
| current gated block | 21.56 | 2,968 | 7.11M |
| plain 2-conv residual block | 13.32 | 4,804 | 4.74M |
| bottleneck r2 | 6.97 | 9,188 | 0.88M |
| depthwise separable | 7.38 | 8,677 | 0.58M |
| bottleneck r4 | 4.47 | 14,313 | 0.30M |

Best next experiment: add `model.block_type` with `gated`, `plain`, `bottleneck`, and maybe `separable`, then run strength/speed ablations. This is the biggest speed lever, but it changes model capacity and therefore training dynamics.

### 2. Fuse Conv+BatchNorm For Inference

PyTorch provides `torch.nn.utils.fusion.fuse_conv_bn_eval`, which fuses a conv and batchnorm into one eval-mode conv when BN running buffers are available.

This is attractive because it preserves the trained function for inference. It can reduce kernel launches and memory traffic in the residual trunk without changing training. It should be applied only to the inference-server copy of the model, not the train model.

Risk: it must be tested carefully with FP16/autocast, EMA/checkpoint loading, and hot-swapped weights. For a training model in early epochs, BN running stats may also be poor.

### 3. Bucketed Static Inference + Compile/CUDA Graphs

Current server sees dynamic batch sizes. A real self-play run showed:

- Avg server batch: 93.7
- Min batch: 1
- Max batch: 128

Dynamic shape makes `torch.compile` and CUDA graph capture less effective. A bucketed path would pad to a small set of shapes, for example `[1, 32, 64, 96, 128, 192]`, prewarm each shape, and slice outputs back to the real count.

Worth testing after inference model changes. Likely helpful if compile warmup can be amortized. More valuable with persistent inference server.

### 4. Use A Dedicated Inference Model Head Set

Self-play inference only needs:

- `policy`
- `value`

The current model config includes lookahead and axis heads. Profiling says heads are small (<~6% total combined), so this is not the main win. Still, an inference-only forward that computes only policy/value is easy and should be kept as a low-risk cleanup.

## Medium-Impact Options

### 5. Optimize Rust MCTS Tree Selection

Worker-side fake-eval profile:

| phase | share |
| --- | ---: |
| `MCTSEngine::select_leaves` | 60% |
| `MCTSEngine::expand_and_backprop` | 20% |
| root Dirichlet noise | 12% |

The exact hot section is `select_child_puct`, which scans every child and computes Q + PUCT.

Options:

- Store child priors, visits, total values in separate arrays for better cache/vectorization.
- Cache `effective_c_puct * sqrt_parent` per parent visit count.
- Reduce divisions in the score loop.
- Add top-child hints or partial sorting for high-branching nodes.
- Keep legal moves in a flat arena instead of cloning `Vec<Hex>` per pending leaf.
- Reduce PyO3 copies in `select_leaves` and `expand_and_backprop`.

This matters more for small models. For 128x16, GPU inference dominates first.

### 6. Revisit Leaf Batch Autotuning

Fake-eval CPU profile:

| leaf batch | 64-move fake-eval time |
| ---: | ---: |
| 8 | 498 ms |
| 16 | 369 ms |
| 32 | 318 ms |

Larger leaf batches reduce CPU overhead but may worsen GPU/server latency and search quality. This needs a short real self-play sweep per model size.

### 7. Optimize Dirichlet Noise

Root noise is visible in CPU fake-eval profiles because `np.random.dirichlet` runs every move. Options:

- Generate gamma samples in Rust or reuse a faster RNG.
- Apply noise only during temperature/exploration moves.
- Disable or reduce noise after the opening phase.

This is not a primary wall-clock win while GPU inference dominates, but it is easy to ablate.

## Lower-Impact Or Conditional Options

### 8. Sparse Legal-Prior Transfer

Server profile says response download and scatter are tiny:

- CPU download: 0.2%
- scatter: 0.1%

Sparse policy transfer is not worth doing first. It only becomes interesting if MCTS/PyO3 copy costs dominate after model inference is reduced.

### 9. Shared-Memory Copy Tuning

`_build_batch` + `_scatter_results` were under 1% wall time in the real profile. Do not spend time here until bigger bottlenecks move.

### 10. Full Persistent Workers

Useful for amortizing startup, compile, or CUDA graph capture, but not a direct fix for the current 82% model-forward wall time. It becomes more attractive if bucketed compile/CUDA graphs are adopted.

### 11. Subtree Reuse

Current `subtree_reuse = true` was slower in a fake-eval test:

| setting | total 64-move fake-eval time |
| --- | ---: |
| false | 370 ms |
| true | 531 ms |

It appears the current loop still reinitializes/expands roots in a way that does not fully benefit from retained subtrees. Do not enable it without redesigning root expansion semantics.

## Recommended Order

1. Add block-type ablation support and test `plain_2conv`, `bottleneck_r2`, and maybe `separable`.
2. Add inference-only Conv+BN fusion for the server model and compare outputs/speed.
3. Add batch histograms and bucketed static inference behind an opt-in flag.
4. Add a real self-play autotune sweep for workers/leaf batch/wait per model size.
5. Optimize `select_child_puct` after model-forward speed improves enough for CPU MCTS to become the limiter.
