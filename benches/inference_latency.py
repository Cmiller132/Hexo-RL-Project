"""Measure single-client inference latency.

Reports P50 and P99 latency for a single worker-client at fixed batch size.

Usage:
    python benches/inference_latency.py [--batch-size B] [--iterations N]
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Python", "src"))

from hexorl.config import load_config
from hexorl.inference.server import InferenceServer
from hexorl.inference.shm_queue import connect_inference_queue
from hexorl.inference.client import InferenceClient


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per request")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of iterations")
    args = parser.parse_args()

    cfg = load_config()
    cfg.model.channels = 32
    cfg.model.blocks = 4
    cfg.inference.max_batch_size = args.batch_size * 2
    cfg.inference.fp16 = False

    server = InferenceServer(cfg, num_workers=1)
    server.start()

    client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=args.batch_size * 2, timeout_ms=30000)
    client.connect()

    # Fix event references
    server_q = connect_inference_queue(1, args.batch_size * 2)
    s0 = server_q.get_slot(0)
    client._slot.req_ready = s0.req_ready
    client._slot.res_ready = s0.res_ready

    # Warmup
    for _ in range(10):
        tensor = np.random.randn(args.batch_size, 13, 33, 33).astype(np.float32)
        client.submit(tensor, args.batch_size)

    # Measure
    latencies = []
    for _ in range(args.iterations):
        tensor = np.random.randn(args.batch_size, 13, 33, 33).astype(np.float32)
        t0 = time.monotonic()
        client.submit(tensor, args.batch_size)
        latencies.append((time.monotonic() - t0) * 1000.0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p90 = latencies[int(len(latencies) * 0.9)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P90 latency: {p90:.3f} ms")
    print(f"P99 latency: {p99:.3f} ms")

    client.disconnect()
    server.stop()
    server.join(timeout=5.0)
    server_q.close()


if __name__ == "__main__":
    main()
