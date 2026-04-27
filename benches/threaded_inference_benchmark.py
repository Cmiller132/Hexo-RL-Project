"""Threaded inference benchmark for local Windows/WSL CUDA runs.

This avoids multiprocessing worker quirks while still exercising the real
InferenceServer process, shared-memory slots, GPU batching, and client submit
path.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python" / "src"))

from hexorl.config import load_config
from hexorl.inference.client import InferenceClient
from hexorl.inference.server import InferenceServer
from hexorl.runtime import autotune_config, configure_torch_runtime


def _worker(worker_id: int, num_workers: int, max_batch: int, batch_size: int, duration_s: float, out: list):
    client = InferenceClient(
        worker_id=worker_id,
        num_workers=num_workers,
        max_batch_size=max_batch,
        timeout_ms=30000,
    )
    tensor = np.random.randn(batch_size, 13, 33, 33).astype(np.float32)
    submits = 0
    positions = 0
    waits = []
    try:
        client.connect()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            client.submit(tensor, batch_size)
            waits.append((time.monotonic() - t0) * 1000.0)
            submits += 1
            positions += batch_size
        out[worker_id] = (submits, positions, waits, None)
    except Exception as exc:
        out[worker_id] = (submits, positions, waits, repr(exc))
    finally:
        client.disconnect()


def run(config: Path, workers: int, batch_size: int, duration_s: float) -> None:
    cfg = load_config(config)
    host = autotune_config(cfg)
    runtime = configure_torch_runtime(cfg, host)
    cfg.selfplay.num_workers = workers
    cfg.inference.max_batch_size = max(cfg.inference.max_batch_size, workers * batch_size)

    print(f"runtime={runtime}", flush=True)
    print(
        f"workers={workers} batch_size={batch_size} max_batch={cfg.inference.max_batch_size} "
        f"model={cfg.model.channels}x{cfg.model.blocks} fp16={cfg.inference.fp16}",
        flush=True,
    )

    server = InferenceServer(cfg, num_workers=workers)
    server.start()
    time.sleep(0.5)
    try:
        results = [None] * workers
        threads = [
            threading.Thread(
                target=_worker,
                args=(i, workers, cfg.inference.max_batch_size, batch_size, duration_s, results),
                daemon=True,
            )
            for i in range(workers)
        ]
        t0 = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=duration_s + 35.0)
        elapsed = time.monotonic() - t0

        submits = sum(r[0] for r in results if r is not None)
        positions = sum(r[1] for r in results if r is not None)
        waits = [v for r in results if r is not None for v in r[2]]
        errors = [r[3] for r in results if r is not None and r[3]]
        waits.sort()
        p50 = waits[len(waits) // 2] if waits else 0.0
        p95 = waits[int(len(waits) * 0.95)] if waits else 0.0

        print(
            f"submits={submits} positions={positions} elapsed_s={elapsed:.2f} "
            f"positions_per_s={positions / max(elapsed, 1e-6):.1f} "
            f"submit_latency_p50_ms={p50:.2f} submit_latency_p95_ms={p95:.2f}",
            flush=True,
        )
        if errors:
            print(f"errors={errors[:4]}", flush=True)
    finally:
        server.stop()
        server.join(timeout=10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "Configs" / "production.toml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    run(args.config, args.workers, args.batch_size, args.duration)


if __name__ == "__main__":
    main()
