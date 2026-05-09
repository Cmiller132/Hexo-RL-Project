#!/usr/bin/env python
"""Profile global graph training batch production without mutating run state."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import struct
import time
from pathlib import Path

from torch.utils.data import DataLoader

from hexorl.buffer.ring import ReplaySnapshot, RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.dashboard.replay import encode_tensor_for_history
from hexorl.graph.batch import current_turn_state, legal_moves_for_stones
from hexorl.selfplay.records import PositionRecord
from hexorl.train.trainer import shutdown_dataloader_workers


def _history(moves: list[tuple[int, int, int]]) -> bytes:
    data = bytearray()
    for player, q, r in moves:
        data.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return bytes(data)


def _stones(moves: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    return {(int(q), int(r)): int(player) for player, q, r in moves}


def _valid_nonterminal(moves: list[tuple[int, int, int]]) -> bool:
    try:
        encode_tensor_for_history(_history(moves), near_radius=8, constrain_threats=False)
        return True
    except Exception:
        return False


def _make_synthetic_records(games: int, plies: int) -> list[PositionRecord]:
    rng = random.Random(17)
    records: list[PositionRecord] = []
    for game_id in range(games):
        moves: list[tuple[int, int, int]] = []
        for turn in range(plies):
            hist = _history(moves)
            legal = legal_moves_for_stones(_stones(moves), radius=8)
            if not legal:
                break
            current_player, placements_remaining = current_turn_state(moves)
            ranked = sorted(legal, key=lambda qr: (max(abs(qr[0]), abs(qr[1]), abs(qr[0] + qr[1])), qr[0], qr[1]))
            policy_rows = ranked[: min(16, len(ranked))]
            denom = sum(1.0 / float(i + 1) for i in range(len(policy_rows))) or 1.0
            policy_v2 = [(q, r, (1.0 / float(i + 1)) / denom) for i, (q, r) in enumerate(policy_rows)]
            dense_policy = {}
            for q, r, prob in policy_v2[:8]:
                idx = (int(q) + 16) * 33 + (int(r) + 16)
                if 0 <= idx < 33 * 33:
                    dense_policy[idx] = float(prob)
            pair_policy = []
            if placements_remaining >= 2 and len(ranked) >= 2:
                pair_policy = [(ranked[0], ranked[1], 1.0)]
            elif placements_remaining == 1 and moves and ranked:
                pair_policy = [((moves[-1][1], moves[-1][2]), ranked[0], 1.0)]
            records.append(
                PositionRecord(
                    move_history=hist,
                    policy_target=dense_policy,
                    root_value=0.0,
                    player=current_player,
                    outcome=1.0 if turn % 2 == 0 else -1.0,
                    game_id=game_id,
                    is_full_search=True,
                    turn_index=turn,
                    lookahead_values=[0.0, 0.0, 0.0],
                    policy_target_v2=policy_v2,
                    opp_policy_target_v2=policy_v2[:8],
                    opp_policy_legal_v2=ranked[: min(64, len(ranked))],
                    opp_policy_weight=1.0,
                    pair_policy_target_v2=pair_policy,
                    pair_policy_complete=bool(pair_policy),
                    moves_left=max(0, 500 - turn),
                )
            )
            next_move = None
            candidates = list(ranked[: min(256, len(ranked))])
            rng.shuffle(candidates)
            for q, r in candidates:
                candidate_history = moves + [(current_player, int(q), int(r))]
                if _valid_nonterminal(candidate_history):
                    next_move = (int(q), int(r))
                    break
            if next_move is None:
                break
            moves.append((current_player, next_move[0], next_move[1]))
    return records


def _make_dataset(args: argparse.Namespace, buffer: RingBuffer | ReplaySnapshot, *, batch_size: int) -> ReplayDataset:
    return ReplayDataset(
        buffer,
        batch_size=batch_size,
        include_sparse_policy=True,
        include_pair_policy=True,
        include_graph_policy=True,
        candidate_budget=args.candidate_budget,
        graph_context_tokens=args.graph_context_tokens,
        graph_legal_rows=args.candidate_budget,
        max_game_turns=args.max_game_turns,
        graph_token_set=args.graph_token_set,
        graph_cache_size=args.graph_cache_size,
        lookahead_horizons=[4, 12, 36],
    )


def _worker_init(_worker_id: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def run_synthetic(args: argparse.Namespace) -> None:
    records = _make_synthetic_records(args.games, args.plies)
    replay = RingBuffer(
        capacity=len(records) + 128,
        max_policy_entries=64,
        max_policy_v2_entries=512,
        num_lookahead=3,
        store_opp_policy=True,
        store_pair_policy=True,
        store_sparse_diagnostics=True,
    )
    replay.extend(records)
    print(f"synthetic_records={len(records)} batch_size={args.batch_size}")
    for workers in args.workers:
        kwargs = {}
        if workers > 0:
            kwargs.update(
                persistent_workers=True,
                prefetch_factor=args.prefetch_factor,
                worker_init_fn=_worker_init,
            )
        dataset_buffer = ReplaySnapshot.from_buffer(replay) if workers > 0 else replay
        dataset = _make_dataset(args, dataset_buffer, batch_size=args.batch_size)
        loader = DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=False, **kwargs)
        iterator = None
        try:
            iterator = iter(loader)
            warm_started = time.perf_counter()
            next(iterator)
            warm_s = time.perf_counter() - warm_started
            waits: list[float] = []
            loader_samples: list[float] = []
            component_sums: dict[str, float] = {}
            for _ in range(args.batches):
                started = time.perf_counter()
                batch = next(iterator)
                waits.append(time.perf_counter() - started)
                aux = batch[4]
                timings = aux.get("_loader_timings", {}) if isinstance(aux, dict) else {}
                loader_samples.append(float(timings.get("graph_loader_sample_s", 0.0)))
                for key, value in timings.items():
                    if str(key).startswith("graph_loader_graph_") or str(key) in {
                        "graph_loader_candidate_s",
                        "graph_loader_policy_overlay_s",
                        "graph_loader_collate_s",
                    }:
                        component_sums[str(key)] = component_sums.get(str(key), 0.0) + float(value)
            print(
                "workers={workers} warm_s={warm:.3f} avg_wait_s={avg:.3f} "
                "median_wait_s={median:.3f} p90_wait_s={p90:.3f} avg_loader_sample_s={sample:.3f}".format(
                    workers=workers,
                    warm=warm_s,
                    avg=statistics.mean(waits),
                    median=statistics.median(waits),
                    p90=sorted(waits)[max(0, int(len(waits) * 0.9) - 1)],
                    sample=statistics.mean(loader_samples),
                )
            )
            if component_sums:
                top_components = sorted(
                    ((key, value) for key, value in component_sums.items() if key.endswith("_s")),
                    key=lambda item: item[1],
                    reverse=True,
                )[:12]
                print(
                    "components_per_batch="
                    + " ".join(
                        f"{key}={value / max(1, len(waits)):.3f}s"
                        for key, value in top_components
                    )
                )
                count_components = sorted(
                    (key, value) for key, value in component_sums.items() if not key.endswith("_s")
                )
                if count_components:
                    print(
                        "component_counts_per_batch="
                        + " ".join(
                            f"{key}={value / max(1, len(waits)):.1f}"
                            for key, value in count_components
                        )
                    )
        finally:
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()  # type: ignore[attr-defined]
            shutdown_dataloader_workers(loader)


def run_existing(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    rows = []
    for latest in sorted(run_dir.glob("trials/*/LATEST.json")):
        data = json.loads(latest.read_text())
        metrics = data.get("metrics") or data.get("train") or data
        bps = float(metrics.get("batches_per_sec") or 0.0)
        if bps <= 0.0:
            continue
        timed_step = sum(
            float(metrics.get(key) or 0.0)
            for key in (
                "graph_collate_s",
                "graph_prepare_s",
                "graph_row_table_s",
                "graph_to_device_s",
                "graph_forward_s",
                "graph_loss_s",
                "graph_backward_s",
                "graph_optimizer_s",
            )
        )
        per_batch = 1.0 / bps
        rows.append(
            {
                "trial": latest.parent.name,
                "family": data.get("model_family") or data.get("family") or metrics.get("model_family"),
                "per_batch_s": per_batch,
                "timed_step_s": timed_step,
                "untracked_wait_s": max(0.0, per_batch - timed_step),
                "loader_workers": metrics.get("graph_loader_workers"),
                "loader_sample_s": metrics.get("graph_loader_sample_s"),
                "bottleneck": _bottleneck_label(metrics),
            }
        )
    for row in rows:
        print(json.dumps(row, sort_keys=True))


def _bottleneck_label(metrics: dict) -> str:
    labels = ("cpu_graph_build", "worker_ipc", "gpu_step", "balanced")
    scored = [(float(metrics.get(f"graph_bottleneck_{label}") or 0.0), label) for label in labels]
    score, label = max(scored, key=lambda item: item[0])
    return label if score > 0.0 else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "existing"), default="synthetic")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--workers", default="0,2,4,8,12")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidate-budget", type=int, default=512)
    parser.add_argument("--graph-context-tokens", type=int, default=256)
    parser.add_argument("--graph-token-set", default="graph256_cells")
    parser.add_argument("--graph-cache-size", type=int, default=256)
    parser.add_argument("--max-game-turns", type=int, default=768)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--plies", type=int, default=36)
    args = parser.parse_args()
    args.workers = [int(part) for part in str(args.workers).replace(" ", ",").split(",") if part]
    if args.mode == "existing" and not args.run_dir:
        parser.error("--run-dir is required for --mode existing")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_existing(args)


if __name__ == "__main__":
    main()
