from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from hexorl.config import Config
from hexorl.selfplay import worker as worker_module
from hexorl.selfplay.worker import SelfPlayWorker


V1_OUTPUTS = [
    "cell_marginal_logits",
    "pair_completion_logits",
    "pair_proposal_score",
    "pair_joint_logits",
    "value",
    "terminal_tactical_v1",
]


class DeterministicGraphClient:
    def __init__(self) -> None:
        self.graph_calls = 0
        self.max_pair_count = 0

    def submit_graph(self, graph_batch: Any) -> dict[str, Any]:
        self.graph_calls += 1
        pair_count = int(np.asarray(graph_batch.pair_first_indices).shape[0])
        legal_count = int(np.asarray(graph_batch.legal_qr).shape[0])
        if pair_count <= 0:
            raise RuntimeError("V1 smoke graph call received no admitted pair rows")
        self.max_pair_count = max(self.max_pair_count, pair_count)
        pair_logits = np.linspace(0.0, 1.0, pair_count, dtype=np.float32)
        return {
            "value": np.asarray([0.0], dtype=np.float32),
            "cell_marginal_logits": np.zeros(legal_count, dtype=np.float32),
            "pair_completion_logits": np.zeros(pair_count, dtype=np.float32),
            "pair_proposal_score": pair_logits,
            "pair_joint_logits": pair_logits,
            "terminal_tactical_v1": np.zeros(8, dtype=np.float32),
            "metadata": {
                "legal_qr": np.asarray(graph_batch.legal_qr, dtype=np.int32),
                "outputs": {
                    "value": {
                        "value_decoder": {"perspective": "current_player"},
                    },
                },
            },
        }


def build_v1_smoke_config(
    *,
    mcts_simulations: int,
    max_game_moves: int,
    pair_budget: int,
) -> Config:
    return Config.model_validate(
        {
            "run": {"seed": 17},
            "model": {
                "architecture": "global_pair_biaffine_0",
                "channels": 16,
                "attention_heads": 4,
                "graph_token_set": "graph512_turn_pair_prior",
                "graph_token_budget": 512,
                "graph_layers": 1,
                "heads": V1_OUTPUTS,
                "pair_strategy": "sampled_joint_pair_v1",
                "pair_strategy_max_pairs": int(pair_budget),
                "candidate_budget": 32,
            },
            "selfplay": {
                "mcts_simulations": int(mcts_simulations),
                "pcr_low_sim_prob": 0.0,
                "max_game_moves": int(max_game_moves),
                "policy_target_top_k": 16,
                "dirichlet_alpha": 0.0,
                "legal_row_mode": "full_rust_legal",
                "tactical_mode": "proposal_and_label",
                "constrain_threats": False,
                "train_on_truncated_games": True,
            },
            "inference": {"fp16": False, "max_batch_size": 8},
            "buffer": {"capacity": 2048},
            "train": {"batch_size": 8, "batches_per_epoch": 1},
        }
    )


def decode_history(move_history: bytes) -> list[tuple[int, int, int]]:
    if len(move_history) % 12 != 0:
        raise ValueError("move history byte length is not a multiple of 12")
    return [
        struct.unpack("<iii", move_history[idx : idx + 12])
        for idx in range(0, len(move_history), 12)
    ]


def validate_record(record: Any, *, pair_budget: int, max_game_moves: int) -> dict[str, Any]:
    history = decode_history(bytes(record.final_move_history))
    coords = [(int(q), int(r)) for _player, q, r in history]
    duplicate_coords = len(coords) - len(set(coords))
    if duplicate_coords:
        raise ValueError(f"V1 smoke produced duplicate coordinates: {duplicate_coords}")
    if len(history) != int(record.game_length):
        raise ValueError(
            f"V1 smoke history rows {len(history)} do not match game_length {record.game_length}"
        )
    if int(record.game_length) > int(max_game_moves):
        raise ValueError(
            f"V1 smoke game_length {record.game_length} exceeds max_game_moves {max_game_moves}"
        )
    if not record.positions:
        raise ValueError("V1 smoke produced no training positions")
    max_candidates = 0
    pair_positions = 0
    for position in record.positions:
        metadata = getattr(position, "v1_search_metadata", None)
        if metadata is None:
            raise ValueError("V1 smoke position is missing V1 search metadata")
        count = len(metadata.candidate_pairs)
        max_candidates = max(max_candidates, count)
        if count:
            pair_positions += 1
        if count > pair_budget:
            raise ValueError(f"V1 smoke admitted {count} pairs above budget {pair_budget}")
    return {
        "game_id": int(record.game_id),
        "game_length": int(record.game_length),
        "positions": len(record.positions),
        "terminal_reason": str(getattr(record, "terminal_reason", "")),
        "truncated": bool(getattr(record, "truncated", False)),
        "max_candidates": int(max_candidates),
        "pair_positions": int(pair_positions),
        "first_moves": coords[: min(8, len(coords))],
        "last_moves": coords[-min(8, len(coords)) :],
    }


def run_smoke(
    *,
    target_states: int,
    mcts_simulations: int,
    max_game_moves: int,
    pair_budget: int,
) -> dict[str, Any]:
    if not worker_module.HAS_ENGINE:
        raise RuntimeError("V1 coherence smoke requires the Rust _engine extension")
    cfg = build_v1_smoke_config(
        mcts_simulations=mcts_simulations,
        max_game_moves=max_game_moves,
        pair_budget=pair_budget,
    )
    record_queue = mp.Queue()
    diagnostic_queue = mp.Queue()
    graph_client = DeterministicGraphClient()
    records: list[dict[str, Any]] = []
    positions = 0
    started = time.monotonic()
    try:
        worker = SelfPlayWorker(
            0,
            cfg,
            record_queue,
            num_workers=1,
            max_batch_size=8,
            diagnostic_queue=diagnostic_queue,
        )
        while positions < int(target_states):
            record = worker._play_one_game(graph_client)
            if record is None:
                raise RuntimeError("V1 smoke worker returned no game record")
            summary = validate_record(
                record,
                pair_budget=pair_budget,
                max_game_moves=max_game_moves,
            )
            records.append(summary)
            positions += int(summary["positions"])
            worker._game_counter += 1
    finally:
        record_queue.close()
        record_queue.join_thread()
        diagnostic_queue.close()
        diagnostic_queue.join_thread()
    elapsed = time.monotonic() - started
    return {
        "ok": True,
        "target_states": int(target_states),
        "positions": int(positions),
        "games": len(records),
        "elapsed_s": float(elapsed),
        "positions_per_sec": float(positions / max(elapsed, 1.0e-9)),
        "graph_calls": int(graph_client.graph_calls),
        "max_graph_pair_count": int(graph_client.max_pair_count),
        "mcts_simulations": int(mcts_simulations),
        "max_game_moves": int(max_game_moves),
        "pair_budget": int(pair_budget),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real-Rust V1 self-play coherence smoke.")
    parser.add_argument("--target-states", type=int, default=12)
    parser.add_argument("--mcts-simulations", type=int, default=8)
    parser.add_argument("--max-game-moves", type=int, default=12)
    parser.add_argument("--pair-budget", type=int, default=64)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    summary = run_smoke(
        target_states=args.target_states,
        mcts_simulations=args.mcts_simulations,
        max_game_moves=args.max_game_moves,
        pair_budget=args.pair_budget,
    )
    text = json.dumps(summary, indent=2)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
