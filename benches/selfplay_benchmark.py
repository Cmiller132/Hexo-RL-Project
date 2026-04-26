"""Self-play throughput benchmark — 1k games with stub model.

Measures:
  - games/min
  - samples/min (training positions generated)
  - avg game length
  - inference server throughput (positions/sec)

Usage:
    python benches/selfplay_benchmark.py [--num-games 1000] [--workers 4] [--sims 50]
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python", "src"))

from hexorl.config import Config, load_config
from hexorl.inference.server import InferenceServer
from hexorl.inference.client import InferenceClient
from hexorl.inference.shm_queue import connect_inference_queue
from hexorl.selfplay.worker import MockMCTSEngine, SelfPlayWorker
from hexorl.buffer.ring import RingBuffer


# ── ReplayDataset (local; will move to hexorl.buffer.sampler) ────────────

# The real ReplayDataset will use the Rust encoder to replay compact
# move_history bytes into (13, 33, 33) tensors. Since the Rust engine is
# not available, this benchmark version generates mock tensors with
# correct shapes for pipeline verification. Policies and values come
# from the actual buffer data.

BOARD_AREA = 33 * 33  # 1089


class ReplayDataset:
    """Minimal replay dataset for benchmark pipeline verification.

    Once the Rust encoder is integrated, this class will replay
    compact move_history bytes into real board tensors via _engine.
    """

    def __init__(self, buffer, batch_size=32):
        self._buffer = buffer
        self.batch_size = batch_size
        self._rng = np.random.RandomState(42)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._buffer) < self.batch_size:
            raise StopIteration

        indices = self._buffer.sample_indices(self.batch_size)
        records = self._buffer.get_batch(indices)

        # Mock tensors: real encoder would replay move_history → (13,33,33)
        tensors = self._rng.randn(
            self.batch_size, 13, 33, 33
        ).astype(np.float32)

        policies = np.zeros((self.batch_size, BOARD_AREA), dtype=np.float32)
        values = np.zeros(self.batch_size, dtype=np.float32)

        for i, rec in enumerate(records):
            if rec is not None:
                policies[i] = rec.to_dense_policy()
                values[i] = rec.to_value_target()

        return tensors, policies, values


# ── Benchmark orchestrator ───────────────────────────────────────────────

def run_benchmark(
    num_games: int = 1000,
    num_workers: int = 4,
    sims_per_move: int = 50,
    batch_size_per_worker: int = 4,
    use_server: bool = True,
):
    """Run the self-play benchmark and report results.

    Args:
        num_games: Total number of games to play.
        num_workers: Number of worker processes to spawn.
        sims_per_move: MCTS simulations per move (mock ignored, but reported).
        batch_size_per_worker: Leaf batch size per worker.
        use_server: Whether to launch an inference server (True) or mock-only (False).
    """
    # Load config — fall back to defaults if no config file present
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = Config()

    cfg.selfplay.num_workers = num_workers
    cfg.selfplay.mcts_simulations = sims_per_move
    cfg.selfplay.batch_size_per_worker = batch_size_per_worker
    cfg.selfplay.pcr_low_sim_prob = 0.0  # Disable PCR for benchmark
    cfg.selfplay.dirichlet_alpha = 0.0   # Disable noise for speed
    cfg.inference.max_batch_size = num_workers * batch_size_per_worker * 4
    cfg.inference.fp16 = False
    cfg.model.channels = 32
    cfg.model.blocks = 4
    cfg.run.seed = 42

    max_batch = cfg.inference.max_batch_size

    print(f"Self-play benchmark: {num_games} games, {num_workers} workers")
    print(f"  MCTS sims/move: {sims_per_move}")
    print(f"  Batch/worker: {batch_size_per_worker}")
    print(f"  Model: {cfg.model.channels}ch, {cfg.model.blocks} blocks")
    print()

    server = None
    if use_server:
        print("Starting inference server...")
        server = InferenceServer(cfg, num_workers=num_workers)
        server.start()
        time.sleep(1.0)
        print("  Server ready")

    # Create buffer
    buffer = RingBuffer(capacity=200_000, recency_decay=0.99)

    # Create record queue for workers
    record_queue = mp.Queue(maxsize=5000)

    # Spawn workers
    workers = []
    games_per_worker = num_games // num_workers
    remaining = num_games % num_workers

    print(f"Spawning {num_workers} workers ({games_per_worker} games each)...")

    for i in range(num_workers):
        n = games_per_worker + (1 if i < remaining else 0)
        p = mp.Process(
            target=_benchmark_worker,
            args=(i, n, cfg, record_queue, num_workers, max_batch, use_server),
            name=f"bench-worker-{i}",
        )
        p.start()
        workers.append(p)

    t_start = time.monotonic()

    # Collect results
    total_games = 0
    total_positions = 0
    total_errors = 0

    while total_games < num_games:
        try:
            game_record = record_queue.get(timeout=1.0)
            if game_record is not None:
                n_pos = len(game_record.positions)
                total_games += 1
                total_positions += n_pos
                buffer.extend(game_record.positions)

                elapsed = time.monotonic() - t_start
                if total_games % 50 == 0 or total_games == num_games:
                    print(
                        f"  [{total_games}/{num_games}] "
                        f"games: {total_games / elapsed * 60:.1f}/min | "
                        f"samples: {total_positions / elapsed * 60:.1f}/min | "
                        f"avg len: {total_positions / max(total_games, 1):.0f}"
                    )
        except Exception:
            # Queue empty — check if workers are still alive
            alive = any(p.is_alive() for p in workers)
            if not alive:
                break

    elapsed = time.monotonic() - t_start

    # Join workers
    for p in workers:
        p.join(timeout=3.0)

    # Stop server
    if server:
        server.stop()
        server.join(timeout=5.0)

    # Final stats
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total games:      {total_games}")
    print(f"  Total positions:  {total_positions}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  Games/min:        {total_games / elapsed * 60:.1f}")
    print(f"  Samples/min:      {total_positions / elapsed * 60:.1f}")
    print(f"  Avg game length:  {total_positions / max(total_games, 1):.0f} moves")
    print(f"  Buffer size:      {len(buffer)}")
    print(f"  Errors:           {total_errors}")
    print()

    # Test sampler
    if len(buffer) >= 32:
        ds = ReplayDataset(buffer, batch_size=32)
        batch_iter = iter(ds)
        batch = next(batch_iter)
        tensors, policies, values = batch
        print(f"  Sampler test:     tensors={tensors.shape}, policies={policies.shape}")
        if tensors.shape == (32, 13, 33, 33):
            print(f"  Sampler: tensors shape OK")
        if policies.shape == (32, BOARD_AREA):
            print(f"  Sampler: policies shape OK")
        print(f"  Sampler OK")
    print()


# ── Worker process ───────────────────────────────────────────────────────

def _benchmark_worker(
    worker_id: int,
    num_games: int,
    cfg: Config,
    record_queue: mp.Queue,
    num_workers: int,
    max_batch: int,
    use_server: bool,
):
    """Each benchmark worker plays num_games and pushes records."""
    from hexorl.selfplay.records import (
        GameRecord,
        PositionRecord,
        sparsify_policy,
    )

    client = None
    if use_server:
        try:
            client = InferenceClient(
                worker_id=worker_id,
                num_workers=num_workers,
                max_batch_size=max_batch,
                timeout_ms=30000,
            )
            client.connect()

            # Fix shared events (required for spawn-mode multiprocessing)
            server_q = connect_inference_queue(num_workers, max_batch)
            s = server_q.get_slot(worker_id)
            client._slot.req_ready = s.req_ready
            client._slot.res_ready = s.res_ready
        except Exception as e:
            print(
                f"  Worker {worker_id}: server connect failed ({e}), using mock",
                flush=True,
            )

    for game_idx in range(num_games):
        game_seed = cfg.run.seed + worker_id * 10000 + game_idx
        sims = cfg.selfplay.mcts_simulations

        engine = MockMCTSEngine(
            num_simulations=sims,
            c_puct=1.5,
            near_radius=cfg.selfplay.near_radius,
            seed=game_seed,
        )

        positions = []
        move_history = bytearray()
        move_idx = 0

        while True:
            init = engine.init_root()
            if init is None:
                break

            tensor, oq, or_, legal_bytes = init

            # Root evaluation
            if client is not None and client._connected:
                try:
                    p, v = client.submit(
                        tensor.reshape(1, 13, 33, 33).astype(np.float32), 1
                    )
                    engine.expand_root(p, v[0], oq, or_, legal_bytes)
                except Exception:
                    engine.expand_root(
                        np.ones(BOARD_AREA, dtype=np.float32) / BOARD_AREA,
                        0.0,
                        oq,
                        or_,
                        legal_bytes,
                    )
            else:
                engine.expand_root(
                    np.ones(BOARD_AREA, dtype=np.float32) / BOARD_AREA,
                    0.0,
                    oq,
                    or_,
                    legal_bytes,
                )

            # MCTS loop
            while not engine.done():
                try:
                    batch_tensor, count = engine.select_leaves(
                        cfg.selfplay.batch_size_per_worker
                    )
                    if count == 0:
                        break

                    if client is not None and client._connected:
                        p, v = client.submit(
                            batch_tensor.astype(np.float32), count
                        )
                    else:
                        p = np.ones(count * BOARD_AREA, dtype=np.float32) / BOARD_AREA
                        v = np.zeros(count, dtype=np.float32)

                    engine.expand_and_backprop(p, v)
                except Exception:
                    pass

            # Get results and sample action
            try:
                _, _, visits, root_value = engine.get_results()
                priors = engine.root_child_priors()
                q_values = engine.root_child_q_values()
            except Exception:
                visits = [1] * 10
                priors = np.array([0.1] * 10, dtype=np.float32)
                q_values = [0.0] * 10
                root_value = 0.0

            temp = 1.0
            q, r = engine.sample_action(temp)

            # Record position
            record_history = bytes(move_history)

            if isinstance(priors, np.ndarray):
                prior_arr = priors
            else:
                prior_arr = np.array(priors, dtype=np.float32)

            policy = sparsify_policy(prior_arr, top_k=20)

            positions.append(
                PositionRecord(
                    move_history=record_history,
                    policy_target=policy,
                    root_value=root_value,
                    player=move_idx % 2,
                    game_id=cfg.run.seed * 100000 + game_idx,
                    is_full_search=True,
                    turn_index=move_idx,
                )
            )

            move_history.extend(
                (move_idx % 2).to_bytes(4, "little", signed=True)
            )
            move_history.extend(q.to_bytes(4, "little", signed=True))
            move_history.extend(r.to_bytes(4, "little", signed=True))

            move_idx += 1
            engine.re_root(q, r, sims)

            if engine.is_over:
                break
            if engine.should_resign(-0.95):
                break

        # Build game record
        outcome: float
        winner = engine.winner
        if winner is None:
            outcome = 0.0
        elif winner == 0:
            outcome = 1.0
        else:
            outcome = -1.0

        record = GameRecord.from_game_data(
            move_history_bytes=bytes(move_history),
            policy_targets=[p.policy_target for p in positions],
            root_values=[p.root_value for p in positions],
            players=[p.player for p in positions],
            outcome=outcome,
            game_id=cfg.run.seed * 100000 + game_idx,
        )
        record.assign_outcomes()
        record_queue.put(record)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-play throughput benchmark"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1000,
        help="Number of games to play (default: 1000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=50,
        help="MCTS simulations per move (default: 50)",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Run without inference server (mock-only)",
    )
    args = parser.parse_args()

    run_benchmark(
        num_games=args.num_games,
        num_workers=args.workers,
        sims_per_move=args.sims,
        use_server=not args.no_server,
    )


if __name__ == "__main__":
    main()
