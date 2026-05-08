# Performance Considerations

## Target Hardware

Primary target machine:

- CPU: AMD Ryzen 9 7950X
- RAM: 32 GB system memory
- GPU: NVIDIA RTX 4070 Ti

The project structure should assume a strong multithreaded CPU, moderate system
memory, and a capable but memory-limited GPU. The design should preserve GPU
batching while pushing suitable work onto CPU/Rust when that reduces VRAM / RAM
pressure or avoids unnecessary tensor materialization.

## Default Work Placement

Prefer CPU/Rust by default for work that is branchy, search-heavy, validation
heavy, or naturally parallel across game states. This is especially important
when the task can be multithreaded on the 7950X and would otherwise increase
GPU memory load.

Good CPU/Rust candidates:

- game state mutation, cloning, and replay,
- MCTS selection, expansion, backup, and widening,
- legal row generation and state identity,
- tactical analysis,
- candidate filtering and admission,
- replay decoding and augmentation,
- training target construction,
- graph or tensor preparation before final batching,
- serialization, compression, and validation.

Reserve VRAM and GPU compute mainly for dense model forward/backward passes
and final batched tensor operations.

## Package Split Implications

`hexo-engine` should expose compact canonical state, legal rows, tactical
payloads, and fast Rust mutation/search support. It should not build model
tensors or graph batches.

`hexo-utils` should include reusable CPU/Rust utilities where performance
matters, including generic MCTS helpers, fast game mutators, replay utilities,
schema helpers, batching helpers, training adapter framework mechanics, and
telemetry support.

`hexo-runner` should own runtime resource coordination: worker counts, Rust
thread pools, inference batching, DataLoader workers, prefetch depth, replay
queues, and memory budgets. This avoids CPU oversubscription and protects GPU
batch quality.

`hexo-model-*` packages should own model-specific input construction and
training data semantics, but should use shared batching, replay, telemetry, and
CPU utility mechanisms where compatible.

Training adapters should keep semantic choices in model packages while using
shared `hexo-utils` mechanics for replay indexing, sampling, CPU transforms,
pinned-memory staging, GPU batch scheduling, and backpressure. This avoids each
model reimplementing the same high-throughput data path.

## Memory Strategy

With 32 GB RAM and a 4070 Ti, avoid storing fully materialized tensors unless
there is a clear reason. Prefer layered storage:

- compact engine replay,
- runner metadata,
- compact model-specific metadata,
- lazy or semi-eager construction of training samples.

V1-style models may need to store search/candidate metadata eagerly, but should
still store compact records rather than large tensors.

## GPU Batching Strategy

The package split should not fragment inference batching. Model packages may own
their adapters, but the runtime should still batch compatible requests across
workers.

Avoid per-node or per-pair GPU calls. Prefer:

- many game/search requests prepared on CPU,
- compact batch collation,
- one batched GPU proposal or inference pass,
- CPU-side candidate/search work,
- one batched final scoring pass when needed.

## Telemetry Requirements

Performance-sensitive paths should report:

- CPU worker and Rust thread counts,
- GPU batch size and fill rate,
- queue depth and backpressure,
- forward latency and decode latency,
- replay memory usage,
- training sample construction time,
- GPU memory allocated/reserved,
- CPU/GPU utilization or proxy timings.

The design goal is not simply to split the project cleanly. It is to split it
without losing multithreading, batching, compact replay, or memory discipline.
