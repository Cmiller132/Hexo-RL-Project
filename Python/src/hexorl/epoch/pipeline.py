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

from hexorl.config import Config
from hexorl.engine.rust import hex_game_class
from hexorl.models.factory import build_model, model_uses_global_graph
from hexorl.models.crop_network import HexNet
from hexorl.replay.codec import ReplayGameRecord, replay_game_from_selfplay
from hexorl.replay.sampler import ReplayDataset
from hexorl.replay.storage import ReplayStorage
from hexorl.runtime import dataloader_worker_count
from hexorl.selfplay.orchestrator import run_orchestrator
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    dense_policy_from_v2,
)
from hexorl.train.trainer import Trainer
from hexorl.dashboard.recorder import RunRecorder

logger = logging.getLogger(__name__)

GRAPH_PAIR_POLICY_HEADS = {"policy_pair_first", "policy_pair_second", "policy_pair_joint"}


def _uses_pair_policy_targets(cfg: Config) -> bool:
    heads = set(getattr(cfg.model, "heads", []))
    return bool((heads & GRAPH_PAIR_POLICY_HEADS) or "pair_policy" in heads)


@dataclass
class EpochResult:
    """Structured result from one epoch run."""

    train_stats: Dict[str, float] = field(default_factory=dict)
    buffer_stats: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[Path] = None
    elapsed_s: float = 0.0
    trainer: Optional[Trainer] = None


def run_epoch(
    cfg: Config,
    *,
    model: Optional[HexNet] = None,
    trainer: Optional[Trainer] = None,
    buffer: Optional[ReplayStorage] = None,
    output_dir: Optional[Path] = None,
    bootstrap_games: int = 0,
    use_selfplay: bool = False,
    train: bool = True,
    device: Optional[torch.device] = None,
    recorder: Optional[RunRecorder] = None,
) -> EpochResult:
    """Run one complete epoch.

    Args:
        cfg: Validated runtime config.
        model: Optional model to continue training.
        trainer: Optional persistent trainer to preserve optimizer, scheduler,
            EMA, epoch, and global-step state across calls.
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
    recorder = recorder or RunRecorder.for_run_dir(output_dir)
    recorder.event(
        "epoch_start",
        {
            "bootstrap_games": bootstrap_games,
            "use_selfplay": use_selfplay,
            "train": train,
        },
        phase="epoch",
    )

    replay = (
        buffer
        if buffer is not None
        else ReplayStorage(
            capacity=cfg.buffer.capacity,
            prefetch_records=cfg.train.prefetch_batches,
        )
    )

    if bootstrap_games > 0:
        bootstrap_records = _make_bootstrap_game_records(cfg, bootstrap_games)
        for game in bootstrap_records:
            recorder.game(game, source="bootstrap")
            replay.append_game(_to_replay_game(cfg, game, config_identity="bootstrap"))
        recorder.metric(
            {"buffer": replay.stats, "bootstrap_games": bootstrap_games},
            phase="bootstrap",
        )

    if trainer is not None:
        model = trainer.model
    elif model is None:
        model = build_model(cfg, device=device, inference=False)

    if use_selfplay:
        selfplay_epoch = int(getattr(trainer, "epoch", 0)) + 1 if trainer is not None else 1
        inference_state = _model_state_for_inference(model)
        orchestrator = run_orchestrator(
            cfg,
            buffer_capacity=cfg.buffer.capacity,
            initial_model_state=inference_state,
            recorder=recorder,
            epoch=selfplay_epoch,
        )
        if buffer is None:
            replay = orchestrator.replay
        else:
            base_game_id = replay.max_game_id + 1 if len(replay) else 0
            game_id_map: dict[int, int] = {}
            for game in orchestrator.replay.games():
                if game.game_id not in game_id_map:
                    game_id_map[game.game_id] = base_game_id + len(game_id_map)
                replay.append_game(game)
        recorder.metric(orchestrator.stats, phase="selfplay")

    train_stats: Dict[str, float] = {}
    checkpoint_path: Optional[Path] = None
    if train:
        if len(replay) < cfg.train.batch_size:
            needed = cfg.train.batch_size - len(replay)
            games = max(1, (needed + 5) // 6)
            for game in _make_bootstrap_game_records(cfg, games, start_game_id=replay.max_game_id + 1):
                replay.append_game(_to_replay_game(cfg, game, config_identity="bootstrap_fill"))

        dataset = ReplayDataset(
            replay,
            batch_size=cfg.train.batch_size,
            recency_decay=cfg.buffer.recency_decay,
            pcr_weight=cfg.buffer.pcr_weight,
            use_symmetry=True,
            lookahead_horizons=cfg.buffer.lookahead_horizons,
            regret_fraction=cfg.buffer.regret_fraction,
            include_axis_delta_norm="axis_delta_norm" in cfg.model.heads,
            include_sparse_policy=bool(
                getattr(cfg.model, "sparse_policy", False)
                or "sparse_policy" in cfg.model.heads
                or "pair_policy" in cfg.model.heads
            ),
            include_pair_policy=_uses_pair_policy_targets(cfg),
            include_graph_policy=model_uses_global_graph(cfg),
            candidate_budget=int(getattr(cfg.model, "candidate_budget", 256)),
            max_game_turns=int(getattr(cfg.selfplay, "max_game_moves", 256)),
        )
        num_workers = dataloader_worker_count(cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
        if trainer is None:
            trainer = Trainer(model, cfg, dataloader, device=device)
        else:
            trainer.dataloader = dataloader
            trainer.batches_per_epoch = cfg.train.batches_per_epoch
        train_stats = trainer.train_epoch()

        checkpoint_path = output_dir / f"epoch_{int(train_stats.get('epoch', 1)):04d}.pt"
        trainer.save_checkpoint(checkpoint_path)
        recorder.metric(
            {
                "train": train_stats,
                "buffer": replay.stats,
                "checkpoint_path": str(checkpoint_path),
            },
            phase="train",
            epoch=int(train_stats.get("epoch", 0)),
            global_step=int(getattr(trainer, "global_step", 0)),
        )
        recorder.checkpoint(
            checkpoint_path,
            {"buffer": replay.stats},
            epoch=int(train_stats.get("epoch", 0)),
            global_step=int(getattr(trainer, "global_step", 0)),
        )

    result = EpochResult(
        train_stats=train_stats,
        buffer_stats=replay.stats,
        checkpoint_path=checkpoint_path,
        elapsed_s=time.monotonic() - t0,
        trainer=trainer,
    )
    recorder.event(
        "epoch_complete",
        {
            "train_stats": result.train_stats,
            "buffer_stats": result.buffer_stats,
            "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
            "elapsed_s": result.elapsed_s,
        },
        phase="epoch",
        epoch=int(train_stats.get("epoch", 0)) if train_stats else None,
    )
    return result


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

    replay = ReplayStorage(
        capacity=cfg.buffer.capacity,
        prefetch_records=cfg.train.prefetch_batches,
    )
    for game in _make_bootstrap_game_records(cfg, 16):
        replay.append_game(_to_replay_game(cfg, game, config_identity="tiny_smoke"))

    dataset = ReplayDataset(
        replay,
        batch_size=cfg.train.batch_size,
        recency_decay=cfg.buffer.recency_decay,
        pcr_weight=cfg.buffer.pcr_weight,
        use_symmetry=True,
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        regret_fraction=cfg.buffer.regret_fraction,
        include_sparse_policy=bool(
            getattr(cfg.model, "sparse_policy", False)
            or "sparse_policy" in cfg.model.heads
            or "pair_policy" in cfg.model.heads
        ),
        include_pair_policy=_uses_pair_policy_targets(cfg),
        include_graph_policy=model_uses_global_graph(cfg),
        candidate_budget=int(getattr(cfg.model, "candidate_budget", 256)),
    )
    num_workers = dataloader_worker_count(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    model = build_model(cfg, device=torch.device("cpu"), inference=False)
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
    for game in _make_bootstrap_game_records(cfg, num_games, start_game_id=start_game_id):
        records.extend(game.positions)
    return records


def _to_replay_game(cfg: Config, game: GameRecord, *, config_identity: str) -> ReplayGameRecord:
    return replay_game_from_selfplay(
        game,
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        lookahead_lambdas=cfg.buffer.lookahead_lambdas,
        config_identity=config_identity,
        checkpoint_identity="bootstrap" if config_identity.startswith("bootstrap") else "",
    )


def _make_bootstrap_game_records(
    cfg: Config,
    num_games: int,
    *,
    start_game_id: int = 0,
) -> List[GameRecord]:
    games: List[GameRecord] = []
    for game_id in range(start_game_id, start_game_id + num_games):
        game = _make_synthetic_game(cfg, game_id)
        games.append(game)
    return games


def _make_synthetic_game(cfg: Config, game_id: int) -> GameRecord:
    rng = np.random.default_rng(int(cfg.run.seed) ^ (game_id * 0x9E3779B1))
    max_moves = max(6, min(int(cfg.selfplay.max_game_moves), 96))
    moves: List[tuple[int, int, int]] = []
    positions: List[PositionRecord] = []

    try:
        game_cls = hex_game_class(required=True)
        game = game_cls()
        for move_idx in range(max_moves):
            player = int(game.current_player)
            legal = game.threat_constrained_moves(cfg.selfplay.near_radius)
            if legal is None:
                legal = game.legal_moves_near(cfg.selfplay.near_radius)
            legal = [(int(q), int(r)) for q, r in legal]
            if not legal:
                break

            q, r = _sample_bootstrap_move(legal, rng)
            policy_v2 = _bootstrap_policy_v2_for_move(q, r, legal, rng, cfg.selfplay.policy_target_top_k)
            policy, outside_mass = dense_policy_from_v2(
                policy_v2,
                -16,
                -16,
                top_k=cfg.selfplay.policy_target_top_k,
            )
            value_hint = float(np.tanh(float(game.window_eval) / 600.0))
            if player == 1:
                value_hint = -value_hint

            positions.append(
                PositionRecord(
                    move_history=_pack_moves(moves),
                    policy_target=policy,
                    policy_target_v2=policy_v2,
                    target_policy_mass_outside_window=outside_mass,
                    root_value=value_hint,
                    player=player,
                    game_id=game_id,
                    is_full_search=(game_id % 3 != 0),
                    turn_index=move_idx,
                )
            )

            game.place(q, r)
            moves.append((player, q, r))
            if game.is_over:
                break

        winner = game.winner
        if winner == 0:
            outcome = 1.0
        elif winner == 1:
            outcome = -1.0
        else:
            score = float(game.window_eval)
            if abs(score) < 1e-6:
                outcome = 1.0 if game_id % 2 == 0 else -1.0
            else:
                outcome = 1.0 if score > 0.0 else -1.0
        terminal_reason = "win" if game.is_over else "bootstrap_cap"
    except Exception as exc:
        raise RuntimeError(
            f"Rust bootstrap generation failed for game_id={game_id}; "
            "epoch bootstrap requires Rust legal rows"
        ) from exc

    game = GameRecord(
        positions=positions,
        outcome=outcome,
        game_id=game_id,
        game_length=len(positions),
        final_move_history=_pack_moves(moves),
        truncated=(terminal_reason != "win"),
        terminal_reason=terminal_reason,
    )
    return game


def _sample_bootstrap_move(
    legal: List[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    weights = np.array(
        [1.0 / (1.0 + max(abs(q), abs(r), abs(q + r))) for q, r in legal],
        dtype=np.float64,
    )
    weights /= weights.sum()
    idx = int(rng.choice(len(legal), p=weights))
    return legal[idx]


def _bootstrap_policy_v2_for_move(
    q: int,
    r: int,
    legal: List[tuple[int, int]],
    rng: np.random.Generator,
    top_k: int,
) -> List[tuple[int, int, float]]:
    weights: dict[tuple[int, int], float] = {(int(q), int(r)): 1.0}
    if len(legal) > 1:
        alt_count = min(max(1, top_k - 1), len(legal) - 1, 7)
        alt_indices = rng.choice(len(legal), size=alt_count, replace=False)
        for legal_idx in alt_indices:
            aq, ar = legal[int(legal_idx)]
            if (aq, ar) == (q, r):
                continue
            weights[(int(aq), int(ar))] = weights.get((int(aq), int(ar)), 0.0) + float(
                rng.uniform(0.02, 0.12)
            )
    items = sorted(weights.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    items = items[:max(1, int(top_k))]
    total = sum(v for _, v in items)
    return [(qr[0], qr[1], float(v / total)) for qr, v in items]


def _pack_moves(moves: Iterable[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", player, q, r))
    return bytes(out)


def _model_state_for_inference(model: torch.nn.Module) -> dict:
    return model.state_dict()
