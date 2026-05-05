# Performance Strategy

Date: 2026-04-29

Performance is a required architecture property. The refactor should make the system easier to reason about without reducing CPU utilization, GPU batching, or self-play/training throughput.

## Runtime Ownership

One runtime specification must own host-resource budgeting:

- CPU worker processes
- Rust/Rayon thread pools
- Torch intra-op and inter-op threads
- Python prefetch threads
- DataLoader workers
- inference server processes
- GPU devices
- replay writer/sampler/projector queues
- self-play worker counts

The goal is to avoid oversubscription while keeping the host busy. Runtime components may request capacity, but the active `RuntimeSpec` or equivalent owner allocates it.

## GPU Batching

GPU inference must be centrally batched by request kind and protocol.

Rules:

- self-play workers produce validated requests and leaf batches
- workers do not run model forward directly
- adapters own collation and decode
- batching groups compatible request kinds and schemas
- microbatch wait, max batch size, max in-flight per worker, and fairness are explicit
- saturated queues apply bounded backpressure or structured retry/failure
- no caller waits indefinitely for queue, shared memory, model forward, or decode

Required telemetry:

- batch size and fill rate
- queue depth and high-watermark
- enqueue wait, model wait, forward time, decode time, response wait
- p50/p95/p99 latency where relevant
- GPU utilization or a proxy timing when GPU metrics are unavailable
- timeout and backpressure counts

## CPU And Rust Utilization

CPU-heavy work should remain parallelizable:

- Rust MCTS selection/backprop stays in Rust hot paths
- Python receives contiguous leaf batches and returns contiguous priors/values
- replay projection uses vectorized batch transforms
- training avoids per-sample device transfers
- data loading, self-play, replay writing, and Rust thread pools are budgeted together

Hot loops should avoid per-node Python objects, per-leaf model calls, per-sample tensor copies, and repeated full validation.

## Robustness On Hot Paths

Robustness is not optional, but validation should be tiered:

- full validation: construction, decode, replay, test, debug/probe, and phase artifacts
- hot-path validation: schema id, generation id, row id, length, shape, hash, finite checks, stale-token checks, immutable/mutation-guarded views
- sampled probes: expensive invariant recomputation and debug bundles during configured probe modes

Protocol validation, row identity checks, non-finite rejection, stale-token rejection, and mutation guards must not be disabled for speed.

## Required Baselines

Phase 00 records the baseline for:

- HostProfile and runtime configuration
- inference throughput and latency
- GPU batching behavior where a GPU is available
- self-play move loop timing and games/positions per second
- MCTS selection/backprop timings
- replay write/read/project throughput
- training step throughput and device transfer timing
- dashboard/autotune smoke timings where relevant

Later phases compare against this baseline when they touch the same hot path.

## Phase Performance Gates

- Phase 03: train adapters must batch projection and device transfer; CUDA paths should support pinned transfer, AMP, and compilation only when valid for the model family.
- Phase 04: inference owns batching, bounded queues, fairness, deadlines, and response validation without losing GPU batching.
- Phase 05: search/MCTS exposes split timing for root init, leaf selection, backprop, sampling, token failures, and policy mapping.
- Phase 06: self-play owns process/thread budgeting, inference wait telemetry, replay writer backpressure, and no-progress diagnostics.
- Phase 07: replay owns bounded storage queues, vectorized projection, prefetch policy, memory budgets, and sample throughput.
- Phase 08: autotune can search runtime knobs separately from model semantics and score utilization, throughput, stability, and stalls.
- Phase 09: final CI includes scheduled benchmark comparison with stable runner metadata.

## Required Artifact Format

Performance artifacts should be machine-readable JSON plus a short Markdown interpretation. Every artifact must include:

- git SHA
- command
- config hash
- HostProfile
- runner profile
- seeds when applicable
- workload description
- throughput metrics
- latency metrics
- queue/backpressure metrics
- CPU/GPU utilization or proxy metrics
- comparison baseline
- accepted regressions with owner and reason

Human summaries are useful, but JSON is the source for regression comparison.
