"""Profile raw HexNet inference throughput without the shared-memory server."""

from __future__ import annotations

import argparse
import time

import torch

from hexorl.config import load_config
from hexorl.models.network import HexNet
from hexorl.runtime import autotune_config, configure_torch_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Configs/wsl_speed_probe.toml")
    parser.add_argument("--batches", default="64,128,192,256,384,512")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    autotune_config(cfg, selfplay_enabled=True)
    if args.compile:
        cfg.runtime.compile_model = True
    configure_torch_runtime(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HexNet(cfg.model.channels, cfg.model.blocks, cfg.model.heads).to(device)
    if device.type == "cuda" and cfg.runtime.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if device.type == "cuda" and cfg.runtime.compile_model:
        model = torch.compile(model, mode=cfg.runtime.compile_mode)
    model.eval()

    print(
        {
            "channels": cfg.model.channels,
            "blocks": cfg.model.blocks,
            "heads": cfg.model.heads,
            "fp16": cfg.inference.fp16,
            "compile": bool(cfg.runtime.compile_model),
            "device": str(device),
        }
    )
    for batch in [int(x) for x in args.batches.split(",") if x]:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch, 13, 33, 33, device=device)
        if device.type == "cuda" and cfg.runtime.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            for _ in range(10):
                with torch.amp.autocast("cuda", enabled=cfg.inference.fp16 and device.type == "cuda"):
                    out = model(x)
                _ = out["policy"].float()
                _ = out["value"].float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(args.steps):
                with torch.amp.autocast("cuda", enabled=cfg.inference.fp16 and device.type == "cuda"):
                    out = model(x)
                _ = out["policy"].float()
                _ = out["value"].float()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.monotonic() - start) / args.steps
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
        print(
            {
                "batch": batch,
                "forward_s": round(elapsed, 5),
                "positions_s": round(batch / elapsed, 1),
                "peak_mb": round(peak_mb),
            }
        )


if __name__ == "__main__":
    main()
