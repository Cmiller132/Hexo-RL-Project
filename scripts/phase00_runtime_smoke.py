from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "Python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from hexorl.config import Config  # noqa: E402
from hexorl.epoch import run_tiny_training_smoke  # noqa: E402
from hexorl.inference.client import InferenceClient  # noqa: E402
from hexorl.inference.server import InferenceServer  # noqa: E402
from hexorl.inference.shm_queue import connect_inference_queue  # noqa: E402
from hexorl.runtime import autotune_config, configure_torch_runtime  # noqa: E402
from hexorl.selfplay.orchestrator import run_orchestrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 00 runtime baseline smokes.")
    parser.add_argument("mode", choices=("selfplay", "inference", "training", "autotune"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "selfplay":
        payload = _selfplay_smoke()
    elif args.mode == "inference":
        payload = _inference_smoke()
    elif args.mode == "training":
        payload = _training_smoke(args.output_dir)
    else:
        payload = _autotune_smoke()

    print(json.dumps(payload, sort_keys=True))
    return 0


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.run.seed = 20260429
    cfg.model.channels = 4
    cfg.model.blocks = 1
    cfg.model.heads = ["policy", "value"]
    cfg.model.candidate_budget = 8
    cfg.model.pair_strategy = "none"
    cfg.model.pair_strategy_max_pairs = 0
    cfg.selfplay.num_workers = 1
    cfg.selfplay.games_per_epoch = 1
    cfg.selfplay.states_per_epoch = 1
    cfg.selfplay.max_game_moves = 2
    cfg.selfplay.batch_size_per_worker = 1
    cfg.selfplay.mcts_simulations = 1
    cfg.selfplay.pcr_low_sim_prob = 0.0
    cfg.selfplay.policy_target_top_k = 4
    cfg.inference.max_batch_size = 4
    cfg.inference.max_wait_us = 1000
    cfg.inference.fp16 = False
    cfg.buffer.capacity = 16
    cfg.buffer.lookahead_horizons = []
    cfg.buffer.lookahead_lambdas = []
    cfg.train.batch_size = 2
    cfg.train.batches_per_epoch = 1
    cfg.train.lr_schedule = "constant"
    cfg.runtime.dataloader_workers = 0
    cfg.runtime.cpu_threads = 1
    cfg.runtime.interop_threads = 1
    return cfg


def _selfplay_smoke() -> dict[str, object]:
    cfg = _tiny_cfg()
    orchestrator = run_orchestrator(cfg, buffer_capacity=16)
    stats = orchestrator.stats
    return {
        "mode": "selfplay",
        "games_done": int(stats["games_done"]),
        "positions_done": int(stats["positions_done"]),
        "buffer_size": int(stats["buffer_size"]),
        "pair_strategy": cfg.model.pair_strategy,
        "pair_rows_scored": 0,
        "workers_total": int(stats["workers_total"]),
    }


def _inference_smoke() -> dict[str, object]:
    cfg = _tiny_cfg()
    server = InferenceServer(cfg, num_workers=1)
    server.start()
    client = InferenceClient(worker_id=0, num_workers=1, max_batch_size=4, timeout_ms=10000)
    server_q = None
    try:
        client.connect()
        server_q = connect_inference_queue(1, 4)
        slot = server_q.get_slot(0)
        client._slot.req_ready = slot.req_ready
        client._slot.res_ready = slot.res_ready
        tensor = np.zeros((1, 13, 33, 33), dtype=np.float32)
        policies, values = client.submit(tensor, 1)
        return {
            "mode": "inference",
            "policy_shape": list(policies.shape),
            "value_shape": list(values.shape),
            "policy_finite": bool(np.isfinite(policies).all()),
            "value_finite": bool(np.isfinite(values).all()),
        }
    finally:
        client.disconnect()
        if server_q is not None:
            server_q.close()
        server.stop()
        server.join(timeout=5.0)


def _training_smoke(output_dir: Path | None) -> dict[str, object]:
    cfg = _tiny_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        out = output_dir or Path(tmp) / "phase00-training-smoke"
        results = run_tiny_training_smoke(cfg, epochs=1, output_dir=out)
        latest = results[-1]
        return {
            "mode": "training",
            "epochs": len(results),
            "loss_total": float(latest.train_stats.get("loss_total", 0.0)),
            "buffer_size": int(latest.buffer_stats.get("size", 0)),
            "checkpoint_exists": bool(latest.checkpoint_path and latest.checkpoint_path.exists()),
        }


def _autotune_smoke() -> dict[str, object]:
    cfg = _tiny_cfg()
    host = autotune_config(cfg, selfplay_enabled=False)
    runtime = configure_torch_runtime(cfg, host)
    return {
        "mode": "autotune",
        "host": {
            "logical_cpus": host.logical_cpus,
            "physical_cpus": host.physical_cpus,
            "cuda_available": host.cuda_available,
            "cuda_name": host.cuda_name,
            "system_memory_gb": round(host.system_memory_gb, 3),
        },
        "runtime": runtime,
        "pair_strategy": cfg.model.pair_strategy,
    }


if __name__ == "__main__":
    raise SystemExit(main())
