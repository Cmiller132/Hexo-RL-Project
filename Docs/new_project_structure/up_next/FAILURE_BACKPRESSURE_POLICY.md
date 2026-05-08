# Failure And Backpressure Policy

## Purpose

The current system has multiple pressure points: self-play workers, replay
queues, inference batches, DataLoader workers, Rust FFI calls, and GPU memory.
The split should define who owns failure decisions without centralizing all
work into one scheduler.

## Proposal

The runner owns orchestration outcomes:

- player timeout or cancellation;
- unavailable model service;
- illegal action response;
- full replay or diagnostics queue;
- batch worker failure;
- controlled game abort or forfeit policy.

The failing package owns precise structured errors. `hexo-utils` should provide
common error shapes, bounded queues, backpressure counters, and retry/cancel
helpers where useful.

Model packages should expose whether failures are retryable, fatal to a game,
fatal to a batch, or local to a sample.

## Simplification Guardrails

Use bounded queues, explicit timeouts, and clear error results before adding
more advanced scheduling. Most pressure problems should be visible through
queue depths and batch timings.

Do not hide failures with silent fallbacks. A controlled abort is better than a
replay record with unclear semantics.
