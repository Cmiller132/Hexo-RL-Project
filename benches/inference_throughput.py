"""Measure inference server throughput in positions/sec.

Simulates N mock workers submitting random leaf-batch tensors to the
server and measures total positions processed per second.

Usage:
    python benches/inference_throughput.py [--batch-size B] [--duration S] [--workers W]
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Python", "src"))

from hexorl.config import load_config
from hexorl.inference.server import InferenceServer
from hexorl.inference.shm_queue import connect_inference_queue
from hexorl.inference.client import InferenceClient
from hexorl.runtime import autotune_config


def mock_worker(worker_id: int, num_workers: int, max_batch: int,
                batch_size: int, duration_s: float, results_queue: mp.Queue):
    """Simulate a self-play worker submitting batches to the inference server."""
    client = InferenceClient(
        worker_id=worker_id, num_workers=num_workers,
        max_batch_size=max_batch, timeout_ms=30000,
    )
    client.connect()

    # Fix event references — server created them, we need to connect.
    server_q = connect_inference_queue(num_workers, max_batch)
    s = server_q.get_slot(worker_id)
    client._slot.req_ready = s.req_ready
    client._slot.res_ready = s.res_ready

    n_submits = 0
    n_positions = 0
    t_start = time.monotonic()

    while time.monotonic() - t_start < duration_s:
        count = min(batch_size, max_batch)
        tensor = np.random.randn(count, 13, 33, 33).astype(np.float32)
        policies, values = client.submit(tensor, count)
        n_submits += 1
        n_positions += count

    client.disconnect()
    server_q.close()
    results_queue.put((n_submits, n_positions))


def run_benchmark(batch_sizes=None, duration_s=10.0, num_workers=8, config_path=None):
    """Run throughput benchmark at each batch size.

    Args:
        batch_sizes: List of per-worker batch sizes to test.
        duration_s: How long to run each test (seconds).
        num_workers: Number of mock workers to simulate.

    Prints a table of results.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    cfg = load_config(config_path) if config_path else load_config()
    if config_path:
        autotune_config(cfg)
        num_workers = cfg.selfplay.num_workers if num_workers <= 0 else num_workers
    else:
        num_workers = 4 if num_workers <= 0 else num_workers
        cfg.model.channels = 32
        cfg.model.blocks = 4
        cfg.inference.max_batch_size = max(batch_sizes) * num_workers
        cfg.inference.fp16 = False  # CPU/MPS test; set true for CUDA benchmarks

    device_name = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device_name = "cuda"
            cfg.inference.fp16 = True
        elif torch.backends.mps.is_available():
            device_name = "mps"
    except ImportError:
        pass

    print(f"{'Batch':>8}  {'Workers':>8}  {'Total/s':>12}  {'Positions/s':>14}  {'Latency ms':>12}")
    print("-" * 62)

    for batch_size in batch_sizes:
        max_batch = batch_size * num_workers + 64  # headroom
        cfg.inference.max_batch_size = max(cfg.inference.max_batch_size, max_batch)

        server = InferenceServer(cfg, num_workers=num_workers)
        server.start()
        time.sleep(0.5)  # Give server time to initialize MPS/CUDA

        # Warmup
        warmup_q = mp.Queue()
        warmup_worker = mp.Process(
            target=mock_worker,
            args=(0, num_workers, max_batch, batch_size, 1.0, warmup_q),
        )
        warmup_worker.start()
        warmup_worker.join()
        warmup_q.get()  # discard

        # Benchmark
        mp.set_start_method("fork", force=True)
        results_queue = mp.Queue()
        workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=mock_worker,
                args=(i, num_workers, max_batch, batch_size, duration_s, results_queue),
            )
            p.start()
            workers.append(p)

        t0 = time.monotonic()
        for p in workers:
            p.join()
        elapsed = time.monotonic() - t0

        # Collect results
        total_positions = 0
        total_submits = 0
        for _ in range(num_workers):
            submits, positions = results_queue.get()
            total_submits += submits
            total_positions += positions

        positions_per_sec = total_positions / elapsed

        # Compute p50 latency
        total_batches = total_submits
        batch_latency_ms = (elapsed / max(total_batches, 1)) * 1000

        print(f"{batch_size:>8}  {num_workers:>8}  {total_batches:>12}  {positions_per_sec:>14.1f}  {batch_latency_ms:>12.2f}")

        server.stop()
        server.join(timeout=5.0)

    print()
    print(f"Device: {device_name}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference server throughput")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-worker batch size (single test)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Test duration in seconds per batch size")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of mock workers; 0 uses config autotune")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config file; when set, benchmark that model shape")
    args = parser.parse_args()

    batch_sizes = [args.batch_size] if args.batch_size else None
    run_benchmark(batch_sizes=batch_sizes, duration_s=args.duration,
                   num_workers=args.workers, config_path=args.config)


if __name__ == "__main__":
    main()
