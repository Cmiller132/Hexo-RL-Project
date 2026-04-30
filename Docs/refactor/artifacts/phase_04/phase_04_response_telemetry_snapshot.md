# Phase 04 Response Telemetry Snapshot

Sample from `ShmTransport._read_response`:

```json
{
  "request_id": "<uuid>",
  "trace_id": "<uuid>",
  "request_kind": "dense_policy_value",
  "transport_state": "draining",
  "queue_depth": 1,
  "batch_size": 1,
  "wait_ms": 0.0,
  "heartbeat_age_ms": 0.0,
  "adapter_name": "dense",
  "status": "ok",
  "error_code": null
}
```

Response validation rejects non-finite head arrays through `InferenceResponse.require_ok()`. Server model output validation rejects non-finite logits before response scatter.

Evidence: `Python/tests/inference/test_response_telemetry.py` and `Python/tests/test_inference_server.py::TestInferenceServer::test_non_finite_outputs_are_rejected_before_mcts`.
