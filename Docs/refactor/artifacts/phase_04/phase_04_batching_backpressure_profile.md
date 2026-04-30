# Phase 04 Batching And Backpressure Profile

Runtime batching remains server-owned and bounded by `cfg.inference.max_batch_size` and `cfg.inference.max_wait_us`.

Behavior:

- `_drain_ready_workers(max_total=self.max_batch)` stops accumulating when the configured batch cap is reached.
- Mixed request-kind batches are reduced to one homogeneous request kind before collation.
- Dense/sparse/pair requests still batch across workers.
- Graph requests are dispatched by typed request kind and remain graph-shaped.
- Client waits are bounded and failure moves the transport to `failed`.

Performance evidence:

```text
python -m pytest Python\tests\test_inference_server.py -q
Exit: 0
7 passed in 14.50s
```

This includes adaptive two-client batching and MCTS round-trip. `python -m compileall Python\src\hexorl` also passed.
