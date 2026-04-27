"""Profile one worker's self-play loop by phase with a fake evaluator.

This isolates Rust MCTS/game-loop CPU cost from GPU inference-server latency.
Use threaded_inference_benchmark.py alongside this script for the server side.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections import defaultdict

import numpy as np

from hexorl.config import load_config
from hexorl.runtime import autotune_config
from hexorl.selfplay.records import BOARD_AREA, action_to_board_index, sparsify_policy
from hexorl.selfplay.worker import HAS_ENGINE, MockMCTSEngine, RealMCTSEngine, get_temperature

if HAS_ENGINE:
    import _engine


class ReusedZerosClient:
    def __init__(self, max_count: int):
        self.policies = np.zeros(max_count * BOARD_AREA, dtype=np.float32)
        self.values = np.zeros(max_count, dtype=np.float32)
        self.calls = 0
        self.positions = 0

    def submit(self, _tensor, count: int):
        count = int(count)
        self.calls += 1
        self.positions += count
        return self.policies[: count * BOARD_AREA], self.values[:count]


def _add(stats: dict[str, float], key: str, t0: float) -> None:
    stats[key] += time.perf_counter() - t0


def profile_game(cfg, moves: int) -> tuple[dict[str, float], dict[str, int]]:
    stats: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    client = ReusedZerosClient(max(cfg.selfplay.batch_size_per_worker, 1))

    game_seed = cfg.run.seed
    if HAS_ENGINE:
        game = _engine.HexGame()
        engine = RealMCTSEngine(
            game,
            cfg.selfplay.mcts_simulations,
            cfg.selfplay.c_puct,
            cfg.selfplay.near_radius,
            game_seed,
            c_puct_init=cfg.selfplay.c_puct_init,
            constrain_threats=cfg.selfplay.constrain_threats,
            subtree_reuse=getattr(cfg.selfplay, "subtree_reuse", False),
        )
    else:
        engine = MockMCTSEngine(
            cfg.selfplay.mcts_simulations,
            cfg.selfplay.c_puct,
            cfg.selfplay.near_radius,
            game_seed,
        )

    for move_idx in range(moves):
        t0 = time.perf_counter()
        init = engine.init_root()
        _add(stats, "init_root", t0)
        counts["moves"] += 1
        if init is None:
            break

        tensor, offset_q, offset_r, legal_bytes = init
        t0 = time.perf_counter()
        p, v = client.submit(tensor.reshape(1, 13, 33, 33), 1)
        _add(stats, "root_eval_fake", t0)

        t0 = time.perf_counter()
        engine.expand_root(p, v[0], offset_q, offset_r, legal_bytes)
        _add(stats, "expand_root", t0)

        if cfg.selfplay.dirichlet_alpha > 0:
            t0 = time.perf_counter()
            child_priors = engine.root_child_priors()
            n_children = child_priors.shape[0] if hasattr(child_priors, "shape") else len(child_priors)
            noise = np.random.dirichlet([cfg.selfplay.dirichlet_alpha] * max(n_children, 1))
            engine.add_dirichlet_noise(noise.astype(np.float32), cfg.selfplay.dirichlet_fraction)
            _add(stats, "root_noise", t0)

        while not engine.done():
            t0 = time.perf_counter()
            batch_tensor, count = engine.select_leaves(cfg.selfplay.batch_size_per_worker)
            _add(stats, "select_leaves", t0)
            counts["leaf_batches"] += 1

            if count == 0:
                t0 = time.perf_counter()
                engine.expand_and_backprop(
                    np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=np.float32),
                )
                _add(stats, "expand_and_backprop", t0)
                break

            t0 = time.perf_counter()
            p, v = client.submit(batch_tensor, count)
            _add(stats, "leaf_eval_fake", t0)
            counts["leaf_positions"] += int(count)

            t0 = time.perf_counter()
            engine.expand_and_backprop(p, v)
            _add(stats, "expand_and_backprop", t0)

        t0 = time.perf_counter()
        moves_q, moves_r, visits, root_value = engine.get_results()
        _add(stats, "get_results", t0)

        t0 = time.perf_counter()
        q, r = engine.sample_action(get_temperature(move_idx, cfg.selfplay.temperature_schedule))
        _add(stats, "sample_action", t0)
        if q is None:
            break

        t0 = time.perf_counter()
        visit_arr = np.zeros(BOARD_AREA, dtype=np.float32)
        for q_coord, r_coord, visit_count in zip(moves_q, moves_r, visits):
            flat_idx = action_to_board_index(q_coord, r_coord, offset_q, offset_r)
            if flat_idx >= 0:
                visit_arr[flat_idx] = float(visit_count)
        _ = sparsify_policy(visit_arr, top_k=cfg.selfplay.policy_target_top_k)
        _ = root_value
        _add(stats, "policy_target", t0)

        t0 = time.perf_counter()
        engine.re_root(int(q), int(r), cfg.selfplay.mcts_simulations)
        _add(stats, "re_root", t0)
        if engine.is_over:
            break

    counts["eval_calls"] = client.calls
    counts["eval_positions"] = client.positions
    return dict(stats), dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--moves", type=int, default=64)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--leaf-batch", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=False)
    cfg.selfplay.num_workers = 1
    cfg.selfplay.mcts_simulations = args.sims
    cfg.selfplay.batch_size_per_worker = args.leaf_batch
    cfg.selfplay.max_game_moves = args.moves
    cfg.selfplay.pcr_low_sim_prob = 0.0

    total_t0 = time.perf_counter()
    stats, counts = profile_game(cfg, moves=args.moves)
    total = time.perf_counter() - total_t0

    print(
        {
            "engine": "rust" if HAS_ENGINE else "mock",
            "moves": counts.get("moves", 0),
            "sims": cfg.selfplay.mcts_simulations,
            "leaf_batch": cfg.selfplay.batch_size_per_worker,
            "total_s": round(total, 6),
            "eval_calls": counts.get("eval_calls", 0),
            "eval_positions": counts.get("eval_positions", 0),
            "leaf_batches": counts.get("leaf_batches", 0),
        }
    )
    for name, seconds in sorted(stats.items(), key=lambda kv: kv[1], reverse=True):
        print(
            f"{name:24s} {seconds * 1000.0:10.3f} ms "
            f"{seconds / max(total, 1e-9) * 100.0:6.2f}%"
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
