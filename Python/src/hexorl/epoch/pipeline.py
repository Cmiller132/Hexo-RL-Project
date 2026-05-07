"""End-to-end epoch orchestration.

This module wires the existing phase components into a runnable loop:
optional self-play/bootstrap data generation, replay sampling, training,
checkpointing, and optional evaluation. It is intentionally conservative:
the public functions return structured stats and avoid daemon background
state once they finish.
"""

from __future__ import annotations

import logging
import os
import struct
import time
from functools import partial
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from hexorl.buffer.ring import RingBuffer, replay_feature_flags
from hexorl.buffer.sampler import ReplayDataset
from hexorl.buffer.targets import process_game_record
from hexorl.config import Config
from hexorl.models.assembly import build_model_from_config
from hexorl.models.registry import is_global_graph_architecture, resolve_model_spec
from hexorl.runtime import dataloader_worker_count
from hexorl.selfplay.orchestrator import run_orchestrator
from hexorl.selfplay.records import (
    GameRecord,
    PositionRecord,
    action_to_board_index,
    dense_policy_from_v2,
)
from hexorl.train.trainer import Trainer
from hexorl.dashboard.recorder import RunRecorder

logger = logging.getLogger(__name__)

GRAPH_PAIR_POLICY_HEADS = {"policy_pair_first", "policy_pair_second", "policy_pair_joint"}


def _training_dataloader_worker_init(worker_id: int, *, torch_threads: int = 1) -> None:
    """Keep graph DataLoader workers from each spawning a full BLAS thread pool."""
    del worker_id
    threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def _is_dataloader_worker_failure(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "dataloader worker",
        "worker exited unexpectedly",
        "multiprocessing",
        "brokenpipe",
        "eoferror",
        "connection reset",
    )
    return any(marker in text for marker in markers)


def _uses_pair_policy_targets(cfg: Config) -> bool:
    heads = set(resolve_model_spec(cfg).outputs)
    return bool((heads & GRAPH_PAIR_POLICY_HEADS) or "pair_policy" in heads)


def _make_training_dataset(
    cfg: Config,
    replay: RingBuffer,
    *,
    resolved_outputs: set[str] | None = None,
    is_global_graph: bool | None = None,
    worker_count: int = 0,
) -> ReplayDataset:
    outputs = resolved_outputs if resolved_outputs is not None else set(resolve_model_spec(cfg).outputs)
    graph_model = (
        bool(is_global_graph)
        if is_global_graph is not None
        else is_global_graph_architecture(getattr(cfg.model, "architecture", ""))
    )
    return ReplayDataset(
        replay,
        batch_size=cfg.train.batch_size,
        recency_decay=cfg.buffer.recency_decay,
        pcr_weight=cfg.buffer.pcr_weight,
        use_symmetry=True,
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        regret_fraction=cfg.buffer.regret_fraction,
        include_axis_delta_norm="axis_delta_norm" in outputs,
        include_sparse_policy=bool(
            getattr(cfg.model, "sparse_policy", False)
            or "sparse_policy" in outputs
            or "pair_policy" in outputs
        ),
        include_pair_policy=_uses_pair_policy_targets(cfg),
        include_graph_policy=graph_model,
        candidate_budget=int(getattr(cfg.model, "candidate_budget", 256)),
        graph_context_tokens=int(getattr(cfg.model, "graph_token_budget", 512)),
        graph_legal_rows=int(getattr(cfg.model, "candidate_budget", 256)),
        max_game_turns=int(getattr(cfg.selfplay, "max_game_moves", 256)),
    )


def _make_training_dataloader(
    cfg: Config,
    replay: RingBuffer,
    *,
    resolved_outputs: set[str] | None = None,
    is_global_graph: bool | None = None,
    worker_count: int | None = None,
    pin_memory: bool | None = None,
) -> DataLoader:
    graph_model = (
        bool(is_global_graph)
        if is_global_graph is not None
        else is_global_graph_architecture(getattr(cfg.model, "architecture", ""))
    )
    workers = (
        dataloader_worker_count(cfg, global_graph_model=graph_model)
        if worker_count is None
        else max(0, int(worker_count))
    )
    kwargs: dict[str, Any] = {
        "batch_size": None,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available() if pin_memory is None else bool(pin_memory),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = max(
            1,
            int(getattr(cfg.runtime, "dataloader_prefetch_factor", 2) or 2),
        )
        kwargs["worker_init_fn"] = partial(
            _training_dataloader_worker_init,
            torch_threads=max(1, int(getattr(cfg.runtime, "graph_worker_torch_threads", 1) or 1)),
        )
    return DataLoader(
        _make_training_dataset(
            cfg,
            replay,
            resolved_outputs=resolved_outputs,
            is_global_graph=graph_model,
            worker_count=workers,
        ),
        **kwargs,
    )


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
    model: Optional[torch.nn.Module] = None,
    trainer: Optional[Trainer] = None,
    buffer: Optional[RingBuffer] = None,
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
        else RingBuffer(
            capacity=cfg.buffer.capacity,
            max_policy_entries=cfg.selfplay.policy_target_top_k,
            max_policy_v2_entries=min(
                max(cfg.selfplay.policy_target_top_k, cfg.model.candidate_budget),
                512,
            ),
            recency_decay=cfg.buffer.recency_decay,
            num_lookahead=len(cfg.buffer.lookahead_horizons),
            **replay_feature_flags(
                resolve_model_spec(cfg).outputs,
                architecture=cfg.model.architecture,
                sparse_policy=cfg.model.sparse_policy,
            ),
        )
    )

    if bootstrap_games > 0:
        bootstrap_records = _make_bootstrap_game_records(cfg, bootstrap_games)
        bootstrap_positions: List[PositionRecord] = []
        for game in bootstrap_records:
            bootstrap_positions.extend(game.positions)
            recorder.game(game, source="bootstrap")
        replay.extend(bootstrap_positions)
        recorder.metric(
            {"buffer": replay.stats, "bootstrap_games": bootstrap_games},
            phase="bootstrap",
        )

    if trainer is not None:
        model = trainer.model
    elif model is None:
        model = build_model_from_config(cfg, device=device, inference=False)

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
            replay = orchestrator.buffer
        else:
            base_game_id = replay.max_game_id + 1 if len(replay) else 0
            game_id_map: dict[int, int] = {}
            appended_positions: List[PositionRecord] = []
            for pos in orchestrator.buffer.records():
                if pos.game_id not in game_id_map:
                    game_id_map[pos.game_id] = base_game_id + len(game_id_map)
                appended_positions.append(replace(pos, game_id=game_id_map[pos.game_id]))
            replay.extend(appended_positions)
        recorder.metric(orchestrator.stats, phase="selfplay")

    train_stats: Dict[str, float] = {}
    checkpoint_path: Optional[Path] = None
    if train:
        if len(replay) < cfg.train.batch_size:
            needed = cfg.train.batch_size - len(replay)
            games = max(1, (needed + 5) // 6)
            replay.extend(_make_bootstrap_positions(cfg, games, start_game_id=replay.max_game_id + 1))

        resolved_outputs = set(resolve_model_spec(cfg).outputs)
        is_global_graph = is_global_graph_architecture(getattr(cfg.model, "architecture", ""))
        num_workers = dataloader_worker_count(cfg, global_graph_model=is_global_graph)

        def make_dataloader(worker_count: int) -> DataLoader:
            return _make_training_dataloader(
                cfg,
                replay,
                resolved_outputs=resolved_outputs,
                is_global_graph=is_global_graph,
                worker_count=worker_count,
            )

        dataloader = make_dataloader(num_workers)
        if trainer is None:
            trainer = Trainer(model, cfg, dataloader, device=device)
        else:
            trainer.dataloader = dataloader
            trainer.batches_per_epoch = cfg.train.batches_per_epoch
        initial_global_step = int(getattr(trainer, "global_step", 0))
        initial_epoch = int(getattr(trainer, "epoch", 0))
        try:
            train_stats = trainer.train_epoch()
        except Exception as exc:
            if (
                num_workers > 0
                and _is_dataloader_worker_failure(exc)
                and int(getattr(trainer, "global_step", 0)) == initial_global_step
            ):
                logger.warning(
                    "DataLoader worker path failed before optimizer progress; "
                    "falling back to single-process loading: %s",
                    exc,
                )
                trainer.epoch = initial_epoch
                trainer.dataloader = make_dataloader(0)
                train_stats = trainer.train_epoch()
                train_stats["dataloader_worker_fallback"] = 1.0
            else:
                raise

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

    replay = RingBuffer(
        capacity=cfg.buffer.capacity,
        recency_decay=cfg.buffer.recency_decay,
        num_lookahead=1,
        max_policy_v2_entries=min(
            max(cfg.selfplay.policy_target_top_k, cfg.model.candidate_budget),
            512,
        ),
        **replay_feature_flags(
            resolve_model_spec(cfg).outputs,
            architecture=cfg.model.architecture,
            sparse_policy=cfg.model.sparse_policy,
        ),
    )
    replay.extend(_make_bootstrap_positions(cfg, 16))

    resolved_outputs = set(resolve_model_spec(cfg).outputs)
    is_global_graph = is_global_graph_architecture(getattr(cfg.model, "architecture", ""))
    dataloader = _make_training_dataloader(
        cfg,
        replay,
        resolved_outputs=resolved_outputs,
        is_global_graph=is_global_graph,
        pin_memory=False,
    )
    model = build_model_from_config(cfg, device=torch.device("cpu"), inference=False)
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


def _make_bootstrap_game_records(
    cfg: Config,
    num_games: int,
    *,
    start_game_id: int = 0,
) -> List[GameRecord]:
    games: List[GameRecord] = []
    for game_id in range(start_game_id, start_game_id + num_games):
        game = _make_synthetic_game(cfg, game_id)
        process_game_record(
            game,
            lookahead_horizons=cfg.buffer.lookahead_horizons,
            lookahead_lambdas=cfg.buffer.lookahead_lambdas,
        )
        games.append(game)
    return games


def _make_synthetic_game(cfg: Config, game_id: int) -> GameRecord:
    rng = np.random.default_rng(int(cfg.run.seed) ^ (game_id * 0x9E3779B1))
    max_moves = max(6, min(int(cfg.selfplay.max_game_moves), 96))
    moves: List[tuple[int, int, int]] = []
    positions: List[PositionRecord] = []

    try:
        import _engine

        game = _engine.HexGame()
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
    except Exception:
        outcome, terminal_reason = _make_fallback_bootstrap_game(
            cfg,
            game_id,
            rng,
            max_moves,
            moves,
            positions,
        )

    game = GameRecord(
        positions=positions,
        outcome=outcome,
        game_id=game_id,
        game_length=len(positions),
        final_move_history=_pack_moves(moves),
        truncated=(terminal_reason != "win"),
        terminal_reason=terminal_reason,
    )
    process_game_record(game)
    return game


def _make_fallback_bootstrap_game(
    cfg: Config,
    game_id: int,
    rng: np.random.Generator,
    max_moves: int,
    moves: List[tuple[int, int, int]],
    positions: List[PositionRecord],
) -> tuple[float, str]:
    occupied: set[tuple[int, int]] = set()
    current_player = 0
    placements_remaining = 1

    for move_idx in range(max_moves):
        legal = _fallback_bootstrap_legal_moves(occupied, cfg.selfplay.near_radius)
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
        positions.append(
            PositionRecord(
                move_history=_pack_moves(moves),
                policy_target=policy,
                policy_target_v2=policy_v2,
                target_policy_mass_outside_window=outside_mass,
                root_value=float(rng.uniform(-0.25, 0.25)),
                player=current_player,
                game_id=game_id,
                is_full_search=(game_id % 3 != 0),
                turn_index=move_idx,
            )
        )
        occupied.add((q, r))
        moves.append((current_player, q, r))
        if placements_remaining > 1:
            placements_remaining -= 1
        else:
            current_player = 1 - current_player
            placements_remaining = 2

    return (1.0 if game_id % 2 == 0 else -1.0), "bootstrap_cap"


def _fallback_bootstrap_legal_moves(
    occupied: set[tuple[int, int]],
    near_radius: int,
) -> List[tuple[int, int]]:
    if not occupied:
        return [(0, 0)]
    radius = max(1, min(int(near_radius), 8))
    legal: set[tuple[int, int]] = set()
    for q, r in occupied:
        for dq in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                if max(abs(dq), abs(dr), abs(dq + dr)) <= radius:
                    candidate = (q + dq, r + dr)
                    if candidate not in occupied and action_to_board_index(*candidate) >= 0:
                        legal.add(candidate)
    return sorted(legal)


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


def _bootstrap_policy_for_move(
    q: int,
    r: int,
    legal: List[tuple[int, int]],
    rng: np.random.Generator,
    top_k: int,
) -> dict[int, float]:
    dense = np.zeros(33 * 33, dtype=np.float32)
    chosen_idx = action_to_board_index(q, r)
    if chosen_idx >= 0:
        dense[chosen_idx] = 1.0
    if len(legal) > 1:
        alt_count = min(max(1, top_k - 1), len(legal) - 1, 7)
        alt_indices = rng.choice(len(legal), size=alt_count, replace=False)
        for legal_idx in alt_indices:
            aq, ar = legal[int(legal_idx)]
            if (aq, ar) == (q, r):
                continue
            flat = action_to_board_index(aq, ar)
            if flat >= 0:
                dense[flat] += float(rng.uniform(0.02, 0.12))
    total = dense.sum()
    if total > 0:
        dense /= total
    nonzero = np.flatnonzero(dense)
    return {int(idx): float(dense[idx]) for idx in nonzero}


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
    original = getattr(model, "_orig_mod", model)
    return original.state_dict()
