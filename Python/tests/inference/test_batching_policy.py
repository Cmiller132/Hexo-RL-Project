from hexorl.inference.batching import BatchingPolicy, ReadyRequest


def test_batching_policy_keeps_request_kinds_compatible():
    policy = BatchingPolicy(max_batch_size=4, max_wait_us=100)
    selected = policy.select_batch(
        [
            ReadyRequest(worker_id=0, count=2, request_kind_code=1, enqueued_monotonic_s=1.0),
            ReadyRequest(worker_id=1, count=2, request_kind_code=2, enqueued_monotonic_s=2.0),
            ReadyRequest(worker_id=2, count=1, request_kind_code=1, enqueued_monotonic_s=3.0),
        ]
    )

    assert selected.worker_ids == [0, 2]
    assert selected.request_kind_code == 1
    assert selected.total_positions == 3


def test_batching_policy_reports_retryable_backpressure_at_watermark():
    policy = BatchingPolicy(max_batch_size=3, max_wait_us=100, high_watermark=1.0)
    selected = policy.select_batch(
        [
            ReadyRequest(worker_id=0, count=2, request_kind_code=1, enqueued_monotonic_s=1.0),
            ReadyRequest(worker_id=1, count=2, request_kind_code=1, enqueued_monotonic_s=2.0),
        ]
    )

    assert selected.worker_ids == [0]
    assert selected.retryable_backpressure is True
    assert selected.high_watermark_hit is False
