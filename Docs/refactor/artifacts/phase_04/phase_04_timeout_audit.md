# Phase 04 Timeout Audit

Transport wait owner: `Python/src/hexorl/inference/client/transport.py`.

Bounded waits:

- Client response wait: `slot.res_ready.wait(timeout=max(timeout_ms, 1.0) / 1000.0)`.
- Server startup wait: `self._ready_event.wait(timeout=timeout_s)`.
- Server joins: `join(timeout=...)` only.
- Shared event wait implementation accepts a timeout and returns `False` on expiry.

Timeout error fields:

- `request_id`
- `trace_id`
- `kind`
- `timeout_ms`
- `queue_depth`
- `heartbeat_age_ms`
- `transport_state`

Evidence:

- `Python/tests/inference/test_shm_transport_timeouts.py`
- `Python/tests/inference/test_inference_no_indefinite_waits.py`
- `git grep -n -E 'Queue\.get\(|Queue\.put\(|\.join\(' -- Python/src/hexorl/inference` returned only bounded joins plus one docstring example.
