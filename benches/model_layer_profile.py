"""Profile HexNet inference by network section.

This synchronizes around each section, so absolute times are conservative, but
the percentages are useful for finding which model methods dominate.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

from hexorl.config import load_config
from hexorl.model.network import HexNet
from hexorl.runtime import autotune_config, configure_torch_runtime


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_section(device: torch.device, stats: dict[str, float], name: str, fn):
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    stats[name] += time.perf_counter() - t0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=True)
    configure_torch_runtime(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexNet(cfg.model.channels, cfg.model.blocks, cfg.model.heads).to(device)
    if device.type == "cuda" and cfg.runtime.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    x = torch.randn(args.batch, 13, 33, 33, device=device)
    if device.type == "cuda" and cfg.runtime.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    stats: dict[str, float] = defaultdict(float)
    autocast_enabled = cfg.inference.fp16 and device.type == "cuda"

    with torch.inference_mode():
        for _ in range(5):
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                model(x)
        _sync(device)

        for _ in range(args.steps):
            with torch.amp.autocast("cuda", enabled=autocast_enabled):
                y = _time_section(
                    device,
                    stats,
                    "HexNet.conv_in+relu",
                    lambda: torch.relu(model.conv_in(x)),
                )
                for i, block in enumerate(model.res_blocks):
                    y = _time_section(
                        device,
                        stats,
                        "HexNet.res_blocks.total",
                        lambda block=block, y=y: block(y),
                    )
                    stats[f"HexNet.res_blocks.{i:02d}"] += 0.0
                for name in model.head_names:
                    _time_section(
                        device,
                        stats,
                        f"head.{name}",
                        lambda name=name, y=y: model.heads[name](y),
                    )

    total = sum(
        value
        for key, value in stats.items()
        if key != "HexNet.res_blocks.total" and not key.startswith("HexNet.res_blocks.")
    ) + stats["HexNet.res_blocks.total"]
    print(
        {
            "device": str(device),
            "batch": args.batch,
            "steps": args.steps,
            "model": f"{cfg.model.channels}x{cfg.model.blocks}",
            "heads": cfg.model.heads,
            "fp16_autocast": autocast_enabled,
            "total_ms_per_step": round(total / args.steps * 1000.0, 3),
            "positions_s": round(args.batch / max(total / args.steps, 1e-9), 1),
        }
    )
    rows = {
        "HexNet.conv_in+relu": stats["HexNet.conv_in+relu"],
        "HexNet.res_blocks.total": stats["HexNet.res_blocks.total"],
    }
    for name in model.head_names:
        rows[f"head.{name}"] = stats[f"head.{name}"]
    for name, seconds in sorted(rows.items(), key=lambda kv: kv[1], reverse=True):
        print(
            f"{name:28s} {seconds / args.steps * 1000.0:10.3f} ms "
            f"{seconds / max(total, 1e-9) * 100.0:6.2f}%"
        )


if __name__ == "__main__":
    main()
