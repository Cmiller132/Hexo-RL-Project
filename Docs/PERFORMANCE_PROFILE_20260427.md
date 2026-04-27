# Hexo-RL Performance Profile - 2026-04-27

## End-To-End Speed

The post-optimization baseline did not show an end-to-end speedup in the first comparable epoch.

| run | epoch | self-play games/min | positions/min | train batches/s | epoch elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 baseline, pre-change | 1 | 12.39 | 1585 | 2.92 | 680.0s |
| v3 post-opt baseline | 1 | 11.51 | 1473 | 3.10 | 725.4s |

Interpretation: the implemented Rust/eval/copy fixes are correct and useful locally, but they did not improve full-pipeline throughput because the current 128x16 run is dominated by GPU model inference.

## Microbench Speedup

The radius-8 legal move path was improved substantially in isolation:

| benchmark | time |
| --- | ---: |
| `legal_moves_near_radius8` using incremental placement candidates | 219 ns |
| brute-force radius-8 scan | 31.2 us |

That is roughly a 140x speedup for that method, but it is too small a slice of total wall time to move the full training pipeline by itself.

## Real Self-Play Server Profile

Command:

```bash
python benches/selfplay_speed.py --config Configs/wsl_speed_probe.toml --games 8 --sims 128 --max-moves 128 --workers 8 --leaf-batch 16 --max-wait-us 200
```

Result:

- End-to-end: 8 games in 96.8s, 4.96 games/min.
- Server batches: 1694.
- Server eval positions: 158,690.
- Avg server batch: 93.7.

Inference server timing:

| section | total ms | share of wall |
| --- | ---: | ---: |
| `InferenceServer._forward()` | 80,660 | 83.3% |
| `self._model(batch_tensor)` inside `_forward()` | 79,906 | 82.5% |
| `_build_batch()` shared-memory gather + H2D staging | 809 | 0.8% |
| policy/value postprocess | 514 | 0.5% |
| CPU download | 239 | 0.2% |
| `_scatter_results()` | 76 | 0.1% |

The slow section is `Python/src/hexorl/inference/server.py::_forward`, specifically the model call. Shared-memory copying is not the primary bottleneck.

## Model Layer Profile

Command:

```bash
python benches/model_layer_profile.py --config Configs/wsl_speed_probe.toml --batch 64 --steps 30
```

Result:

| section | ms/step | share |
| --- | ---: | ---: |
| `HexNet.res_blocks.total` | 20.64 | 94.0% |
| `head.policy` | 0.27 | 1.2% |
| `HexNet.conv_in+relu` | 0.24 | 1.1% |
| `head.value` | 0.21 | 0.9% |
| lookahead heads + axis | <0.20 each | <1% each |

The exact model hotspot is `Python/src/hexorl/model/network.py::GatedResBlock.forward`: three 3x3 convolutions plus batch norms and sigmoid gate per residual block. The policy/value heads are not the bottleneck.

## Worker/MCTS CPU Profile

Command:

```bash
python benches/selfplay_phase_profile.py --config Configs/wsl_speed_probe.toml --moves 64 --sims 128 --leaf-batch 16
```

Fake evaluator result:

| worker phase | total ms | share |
| --- | ---: | ---: |
| `MCTSEngine::select_leaves` | 223.1 | 60.4% |
| `MCTSEngine::expand_and_backprop` | 74.7 | 20.2% |
| root Dirichlet noise | 44.6 | 12.1% |
| policy target construction | 13.5 | 3.7% |
| `re_root` | 5.6 | 1.5% |

Within `select_leaves`, `encoder::encode_board_into` is now about 1us for radius 8, so the remaining cost is mostly tree traversal and PUCT child scans. The hot Rust sections are:

- `crates/hexgame-core/src/mcts.rs::select_leaves`, especially traversal through `select_child_puct`.
- `crates/hexgame-core/src/mcts.rs::select_child_puct`, which scans all children and computes PUCT score for each.
- `crates/hexgame-core/src/mcts.rs::expand_and_backprop`, especially `expand_node`, policy gather/softmax, arena pushes, and path backpropagation.

Leaf batch matters for CPU overhead:

| leaf batch | fake-eval total for 64 moves |
| ---: | ---: |
| 8 | 498 ms |
| 16 | 369 ms |
| 32 | 318 ms |

Larger leaf batches reduce CPU batch overhead, but they can hurt real GPU/server latency and search behavior, so this should be tuned with real self-play, not only fake eval.

## Further Options

Worth doing next:

1. Model architecture/inference optimization. The residual trunk is the bottleneck, so optimize or replace `GatedResBlock`: fewer convs per block, fused/removed BatchNorm in eval, channels-last validation, compile/static batch only after warmup issues are controlled.
2. Real batch-shape instrumentation and bucketed inference experiment. The server average batch was 93.7, max 128 in the real run. Bucketed compile/CUDA graph may be worthwhile if persistent server or compile warmup is solved.
3. MCTS traversal optimization. `select_child_puct` child scans are the CPU hotspot. Consider cached sqrt/cpuct values, child score vectorization, visit/prior arrays, or partial top-child caches before any threaded tree search.
4. Dirichlet noise optimization. `np.random.dirichlet` is visible in CPU fake-eval profile. It is not wall-clock dominant when GPU inference is active, but it is a cheap cleanup candidate.

Lower priority:

- Sparse legal-prior transfer. Server profiling shows download/scatter are <0.4% of wall time, so this will not materially improve current 128x16 throughput.
- Shared-memory copy tuning. `_build_batch` + `_scatter_results` are under 1% of wall time.
- Full persistent workers. It saves startup/shutdown overhead but does not attack the 80s/97s model-forward wall-clock cost in real self-play.
