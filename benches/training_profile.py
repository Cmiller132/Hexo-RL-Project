"""Profile HexNet training step and replay sampling throughput."""

from __future__ import annotations

import argparse
import time

import torch

from hexorl.config import load_config
from hexorl.epoch.pipeline import _make_bootstrap_game_records
from hexorl.buffer.ring import RingBuffer
from hexorl.buffer.sampler import ReplayDataset
from hexorl.model.network import HexNet
from hexorl.runtime import autotune_config, configure_torch_runtime
from hexorl.train.losses import compute_losses


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_targets(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "policy": torch.nn.functional.one_hot(
            torch.randint(0, 1089, (batch_size,), device=device),
            1089,
        ).float(),
        "value": torch.rand(batch_size, device=device) * 2.0 - 1.0,
        "lookahead_4": torch.rand(batch_size, device=device) * 2.0 - 1.0,
        "lookahead_12": torch.rand(batch_size, device=device) * 2.0 - 1.0,
        "lookahead_36": torch.rand(batch_size, device=device) * 2.0 - 1.0,
        "axis": torch.randint(0, 3, (batch_size,), device=device),
        "policy_weight": torch.ones(batch_size, device=device),
        "value_weight": torch.ones(batch_size, device=device),
    }


def profile_synthetic(cfg, batch_sizes: list[int], steps: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = dict(cfg.train.loss_weights)
    print("synthetic_step")
    for batch_size in batch_sizes:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = HexNet(
            channels=cfg.model.channels,
            blocks=cfg.model.blocks,
            heads=cfg.model.heads,
        ).to(device)
        if device.type == "cuda" and cfg.runtime.channels_last:
            model = model.to(memory_format=torch.channels_last)
        if device.type == "cuda" and cfg.runtime.compile_model:
            model = torch.compile(model, mode=cfg.runtime.compile_mode)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.peak_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.inference.fp16 and device.type == "cuda")
        x = torch.randn(batch_size, 13, 33, 33, device=device)
        if device.type == "cuda" and cfg.runtime.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        targets = _make_targets(batch_size, device)

        for _ in range(3):
            _train_step(model, optimizer, scaler, x, targets, weights, cfg.inference.fp16)
        _sync()
        start = time.monotonic()
        for _ in range(steps):
            _train_step(model, optimizer, scaler, x, targets, weights, cfg.inference.fp16)
        _sync()
        elapsed = (time.monotonic() - start) / steps
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        print(
            {
                "batch": batch_size,
                "step_s": round(elapsed, 4),
                "samples_s": round(batch_size / elapsed, 1),
                "peak_mb": round(peak_mb),
            }
        )


def _train_step(model, optimizer, scaler, x, targets, weights, fp16: bool) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=fp16 and x.device.type == "cuda"):
        predictions = model(x)
        loss, _ = compute_losses(predictions, targets, weights)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def profile_replay(cfg, batch_size: int, batches: int, bootstrap_games: int) -> None:
    replay = RingBuffer(
        capacity=max(cfg.buffer.capacity, bootstrap_games * cfg.selfplay.max_game_moves),
        max_policy_entries=cfg.selfplay.policy_target_top_k,
        recency_decay=cfg.buffer.recency_decay,
        num_lookahead=len(cfg.buffer.lookahead_horizons),
    )
    for game in _make_bootstrap_game_records(cfg, bootstrap_games):
        replay.extend(game.positions)
    dataset = ReplayDataset(
        replay,
        batch_size=batch_size,
        recency_decay=cfg.buffer.recency_decay,
        pcr_weight=cfg.buffer.pcr_weight,
        use_symmetry=True,
        lookahead_horizons=cfg.buffer.lookahead_horizons,
        regret_fraction=cfg.buffer.regret_fraction,
    )
    iterator = iter(dataset)
    next(iterator)
    start = time.monotonic()
    for _ in range(batches):
        next(iterator)
    elapsed = time.monotonic() - start
    print(
        "replay_sampling",
        {
            "batch": batch_size,
            "batches": batches,
            "elapsed_s": round(elapsed, 3),
            "batches_s": round(batches / elapsed, 2),
            "samples_s": round((batches * batch_size) / elapsed, 1),
            "records": len(replay),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--batch-sizes", default="64,128,256,384,512")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--replay-batches", type=int, default=8)
    parser.add_argument("--bootstrap-games", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=False)
    if args.compile:
        cfg.runtime.compile_model = True
    if args.compile_mode:
        cfg.runtime.compile_mode = args.compile_mode
    configure_torch_runtime(cfg)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    print(
        {
            "channels": cfg.model.channels,
            "blocks": cfg.model.blocks,
            "heads": cfg.model.heads,
            "fp16": cfg.inference.fp16,
            "channels_last": cfg.runtime.channels_last,
            "train_batch": cfg.train.batch_size,
        }
    )
    profile_synthetic(cfg, batch_sizes, args.steps)
    profile_replay(cfg, cfg.train.batch_size, args.replay_batches, args.bootstrap_games)


if __name__ == "__main__":
    main()
