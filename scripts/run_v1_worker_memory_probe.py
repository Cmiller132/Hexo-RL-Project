from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

from run_v1_selfplay_coherence_smoke import (
    DeterministicGraphClient,
    build_v1_smoke_config,
    validate_record,
)

from hexorl.selfplay import worker as worker_module
from hexorl.selfplay.worker import SelfPlayWorker


def _available_gb() -> float:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0 * 1024.0)
    except Exception:
        return -1.0


def _worker_entry(
    worker_id: int,
    worker_count: int,
    args: dict[str, int],
    queue: mp.Queue,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    started = time.monotonic()
    try:
        cfg = build_v1_smoke_config(
            mcts_simulations=int(args["mcts_simulations"]),
            max_game_moves=int(args["max_game_moves"]),
            pair_budget=int(args["pair_budget"]),
        )
        record_queue = mp.Queue()
        diagnostic_queue = mp.Queue()
        graph_client = DeterministicGraphClient()
        worker = SelfPlayWorker(
            int(worker_id),
            cfg,
            record_queue,
            num_workers=int(worker_count),
            max_batch_size=int(args["inference_batch_size"]),
            diagnostic_queue=diagnostic_queue,
        )
        positions = 0
        games = 0
        max_pair_count = 0
        while positions < int(args["target_states_per_worker"]):
            record = worker._play_one_game(graph_client)
            if record is None:
                raise RuntimeError("V1 worker probe produced no game")
            summary = validate_record(
                record,
                pair_budget=int(args["pair_budget"]),
                max_game_moves=int(args["max_game_moves"]),
            )
            positions += int(summary["positions"])
            games += 1
            max_pair_count = max(max_pair_count, int(summary["max_candidates"]))
            worker._game_counter += int(worker_count)
        record_queue.close()
        record_queue.join_thread()
        diagnostic_queue.close()
        diagnostic_queue.join_thread()
        queue.put(
            {
                "worker_id": int(worker_id),
                "ok": True,
                "positions": int(positions),
                "games": int(games),
                "max_pair_count": int(max_pair_count),
                "graph_calls": int(graph_client.graph_calls),
                "elapsed_s": time.monotonic() - started,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "worker_id": int(worker_id),
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "elapsed_s": time.monotonic() - started,
            }
        )


def run_probe(
    *,
    worker_counts: list[int],
    min_free_gb: float,
    target_states_per_worker: int,
    mcts_simulations: int,
    max_game_moves: int,
    pair_budget: int,
    inference_batch_size: int,
    timeout_s: float,
) -> dict[str, Any]:
    if not worker_module.HAS_ENGINE:
        raise RuntimeError("V1 worker memory probe requires the Rust _engine extension")
    entries: list[dict[str, Any]] = []
    selected = 0
    for worker_count in worker_counts:
        started = time.monotonic()
        queue: mp.Queue = mp.Queue()
        proc_args = {
            "target_states_per_worker": int(target_states_per_worker),
            "mcts_simulations": int(mcts_simulations),
            "max_game_moves": int(max_game_moves),
            "pair_budget": int(pair_budget),
            "inference_batch_size": int(inference_batch_size),
        }
        procs = [
            mp.Process(target=_worker_entry, args=(idx, worker_count, proc_args, queue))
            for idx in range(worker_count)
        ]
        before_free = _available_gb()
        for proc in procs:
            proc.start()
        results: list[dict[str, Any]] = []
        deadline = time.monotonic() + float(timeout_s)
        while len(results) < worker_count and time.monotonic() < deadline:
            try:
                results.append(queue.get(timeout=0.5))
            except Exception:
                pass
        for proc in procs:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [proc.pid for proc in procs if proc.is_alive()]
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        after_free = _available_gb()
        ok = (
            not alive
            and len(results) == worker_count
            and all(bool(row.get("ok")) for row in results)
            and (after_free < 0.0 or after_free >= float(min_free_gb))
        )
        entry = {
            "worker_count": int(worker_count),
            "ok": bool(ok),
            "before_free_gb": round(before_free, 3),
            "after_free_gb": round(after_free, 3),
            "min_free_gb": float(min_free_gb),
            "elapsed_s": time.monotonic() - started,
            "alive_after_timeout": alive,
            "results": sorted(results, key=lambda row: int(row.get("worker_id", 0))),
        }
        entries.append(entry)
        if ok:
            selected = int(worker_count)
    return {
        "schema_version": 1,
        "selected_worker_count": int(selected),
        "worker_counts": worker_counts,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe V1 self-play worker counts under a RAM floor.")
    parser.add_argument("--worker-count", action="append", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--target-states-per-worker", type=int, default=8)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    parser.add_argument("--max-game-moves", type=int, default=64)
    parser.add_argument("--pair-budget", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=48)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    worker_counts = sorted(set(args.worker_count or [2, 4, 6, 8]))
    summary = run_probe(
        worker_counts=worker_counts,
        min_free_gb=args.min_free_gb,
        target_states_per_worker=args.target_states_per_worker,
        mcts_simulations=args.mcts_simulations,
        max_game_moves=args.max_game_moves,
        pair_budget=args.pair_budget,
        inference_batch_size=args.inference_batch_size,
        timeout_s=args.timeout_s,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if int(summary["selected_worker_count"]) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
