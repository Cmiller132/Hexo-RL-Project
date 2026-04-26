"""End-to-end epoch orchestration.

This module wires the existing phase components into a runnable loop:
optional self-play/bootstrap data generation, replay sampling, training,
checkpointing, and optional evaluation. It is intentionally conservative:
the public functions return structured stats and avoid daemon background
state once they finish.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.buffer.targets import process_game_record
from hexorl.config import Config
from hexorl.model.network import HexNet
from hexorl.selfplay.orchestrator import run_orchestrator
from hexorl.selfplay.records import GameRecord, PositionRecord, action_to_board_index
from hexorl.train.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class EpochResult:
    """Structured result from one epoch run."""

    train_stats: Dict[str, float] = field(default_factory=dict)
    buffer_stats: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None
    elapsed_s: float = 0.0


def run_epoch(
    cfg: Config,
    *,
    model: Optional[HexNet] = None,
    buffer: Optional[RingBuffer] = None,
    output_dir: Optional[Path] = None,
    bootstrap_games: int = 0,
    use_selfplay: bool = False,
    train: bool = True,
    device: Optional[torch.device] = None,
) -> EpochResult:
    """Run one complete epoch.

    Args:
        cfg: Validated runtime config.
        model: Optional model to continue training.
        buffer: Optional replay buffer to append to/reuse.
        output_dir: Run directory for checkpoints.
        bootstrap_games: Number of deterministic synthetic games to add before training.
        use_selfplay: If true, run the self-play orchestrator until cfg.selfplay.games_per_epoch.
        train: If true, train cfg.train.batches_per_epoch batches.
        device: Optional torch device override.
    """
    t0 = time.monotonic()
    output_dir = Path(output_dir or cfg.run.output_dir.format(name="default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = buffer or RingBuffer(
        capacity=cfg.buffer.capacity,
        recency_decay=cfg.buffer.recency_decay,
        num_lookahead=len(cfg.buffer.lookahead_horizons),
    )

    if bootstrap_games > 0:
        replay.extend(_make_bootstrap_positions(cfg, bootstrap_games))

    if use_selfplay:
        orchestrator = run_orchestrator(cfg, buffer_capacity=cfg.buffer.capacity)
        replay = orchestrator.buffer

    if model is None:
        model = HexNet(
            channels=cfg.model.channels,
            blocks=cfg.model.blocks,
            heads=cfg.model.heads,
        )

    train_stats: Dict[str, float] = {}
    checkpoint_path: Optional[Path] = None
    if train:
        if len(replay) < cfg.train.batch_size:
            needed = cfg.train.batch_size - len(replay)
            games = max(1, (needed + 5) // 6)
            replay.extend(_make_bootstrap_positions(cfg, games, start_game_id=replay.max_game_id + 1))

        dataset = ReplayDataset(
            replay,
            batch_size=cfg.train.batch_size,
            recency_decay=cfg.buffer.recency_decay,
            pcr_weight=cfg.buffer.pcr_weight,
            use_symmetry=True,
            lookahead_horizons=cfg.buffer.lookahead_horizons,
            regret_fraction=cfg.buffer.regret_fraction,
        )
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        trainer = Trainer(model, cfg, dataloader, device=device)
        train_stats = trainer.train_epoch()

        checkpoint_path = output_dir / f"epoch_{int(train_stats.get('epoch', 1)):04d}.pt"
        trainer.save_checkpoint(checkpoint_path)

    return EpochResult(
        train_stats=train_stats,
        buffer_stats=replay.stats,
        checkpoint_path=checkpoint_path,
        elapsed_s=time.monotonic() - t0,
    )


def run_tiny_training_smoke(
    cfg: Optional[Config] = None,
    *,
    epochs: int = 3,
    output_dir: Optional[Path] = None,
) -> List[EpochResult]:
    """Run a small CPU/MPS-safe multi-epoch training smoke test.

    This intentionally keeps one trainer alive across epochs so optimizer,
    scheduler, EMA, epoch, and global-step state advance exactly as they do in
    a real run.
    """
    if epochs <= 0:
        return []

    cfg = cfg or Config()
    cfg.model.channels = min(cfg.model.channels, 8)
    cfg.model.blocks = min(cfg.model.blocks, 1)
    cfg.model.heads = [
        "policy",
        "value",
        "lookahead_1",
        "regret_rank",
        "regret_value",
        "opp_policy",
        "axis",
        "moves_left",
    ]
    cfg.buffer.capacity = min(cfg.buffer.capacity, 256)
    cfg.buffer.lookahead_horizons = [1]
    cfg.buffer.lookahead_lambdas = [0.5]
    cfg.train.batch_size = min(cfg.train.batch_size, 4)
    cfg.train.batches_per_epoch = min(cfg.train.batches_per_epoch, 3)
    cfg.train.lr_schedule = "constant"
    cfg.train.loss_weights = {
        "policy": 1.0,
        "value": 1.0,
        "lookahead_1": 0.2,
        "regret_rank": 0.1,
        "regret_value": 0.1,
        "opp_policy": 0.1,
        "axis": 0.05,
        "moves_left": 0.01,
        "entropy": 0.001,
    }
    cfg.inference.fp16 = False

    output_dir = Path(output_dir or cfg.run.output_dir.format(name="tiny-smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)

    replay = RingBuffer(
        capacity=cfg.buffer.capacity,
        recency_decay=cfg.buffer.recency_decay,
        num_lookahead=1,
    )
    replay.extend(_make_bootstrap_positions(cfg, 16))

    dataset = ReplayDataset(
        replay,
        batch_size=cfg.train.batch_size,
        recency_decay=cfg.buffer.recency_decay,
        pcr_weight=cfg.buffer.pcr_weight,
        use_symmetry=True,
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        regret_fraction=cfg.buffer.regret_fraction,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    model = HexNet(channels=cfg.model.channels, blocks=cfg.model.blocks, heads=cfg.model.heads)
    trainer = Trainer(model, cfg, dataloader, device=torch.device("cpu"))

    results = []
    for _ in range(epochs):
        t0 = time.monotonic()
        train_stats = trainer.train_epoch()
        checkpoint_path = output_dir / f"epoch_{int(train_stats['epoch']):04d}.pt"
        trainer.save_checkpoint(checkpoint_path)
        results.append(
            EpochResult(
                train_stats=train_stats,
                buffer_stats=replay.stats,
                checkpoint_path=checkpoint_path,
                elapsed_s=time.monotonic() - t0,
            )
        )
    return results


def _make_bootstrap_positions(
    cfg: Config,
    num_games: int,
    *,
    start_game_id: int = 0,
) -> List[PositionRecord]:
    records: List[PositionRecord] = []
    for game_id in range(start_game_id, start_game_id + num_games):
        game = _make_synthetic_game(cfg, game_id)
        process_game_record(
            game,
            lookahead_horizons=cfg.buffer.lookahead_horizons,
            lookahead_lambdas=cfg.buffer.lookahead_lambdas,
        )
        records.extend(game.positions)
    return records


def _make_synthetic_game(cfg: Config, game_id: int) -> GameRecord:
    moves = [
        (0, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, -1, 0),
        (0, 0, -1),
        (1, 2, 0),
    ]
    outcome = 1.0 if game_id % 2 == 0 else -1.0
    positions: List[PositionRecord] = []

    for i, (player, q, r) in enumerate(moves):
        idx = action_to_board_index(q, r)
        alt_idx = action_to_board_index(q + 1, r)
        policy = {idx: 1.0}
        if alt_idx >= 0 and alt_idx != idx:
            policy = {idx: 0.8, alt_idx: 0.2}
        positions.append(
            PositionRecord(
                move_history=_pack_moves(moves[:i]),
                policy_target=policy,
                root_value=float(np.tanh((i - 2) / 3.0)),
                player=player,
                game_id=game_id,
                is_full_search=(game_id % 3 != 0),
                turn_index=i,
            )
        )

    return GameRecord(
        positions=positions,
        outcome=outcome,
        game_id=game_id,
        game_length=len(positions),
        final_move_history=_pack_moves(moves),
    )


def _pack_moves(moves: Iterable[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", player, q, r))
    return bytes(out)
