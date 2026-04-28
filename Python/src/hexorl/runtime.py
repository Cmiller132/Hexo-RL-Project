"""Runtime autotuning for local training and inference runs."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any

import torch

from hexorl.config import Config


@dataclass(frozen=True)
class HostProfile:
    logical_cpus: int
    physical_cpus: int
    system: str
    cuda_available: bool
    cuda_name: str | None = None
    cuda_memory_gb: float = 0.0


def detect_host() -> HostProfile:
    logical = os.cpu_count() or 1
    physical = _physical_cpu_count(logical)
    cuda_available = torch.cuda.is_available()
    cuda_name = None
    cuda_memory_gb = 0.0
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        cuda_name = props.name
        cuda_memory_gb = props.total_memory / (1024**3)
    return HostProfile(
        logical_cpus=logical,
        physical_cpus=physical,
        system=platform.system().lower(),
        cuda_available=cuda_available,
        cuda_name=cuda_name,
        cuda_memory_gb=cuda_memory_gb,
    )


def autotune_config(
    cfg: Config,
    host: HostProfile | None = None,
    *,
    selfplay_enabled: bool | None = None,
) -> HostProfile:
    """Fill host-derived performance values in-place.

    The tuner deliberately changes only operational throughput knobs. Model
    shape, loss weights, and search semantics remain experiment-defined.
    """
    host = host or detect_host()
    rt = cfg.runtime
    if not rt.autotune:
        return host

    tune_selfplay = True if selfplay_enabled is None else bool(selfplay_enabled)
    reserve = max(1, min(rt.selfplay_cpu_reserve, host.logical_cpus - 1))
    worker_budget = max(1, host.logical_cpus - reserve)
    if tune_selfplay:
        if cfg.selfplay.num_workers <= 0:
            cfg.selfplay.num_workers = _selfplay_worker_target(cfg, host, worker_budget)
        else:
            cfg.selfplay.num_workers = min(cfg.selfplay.num_workers, worker_budget)
    elif cfg.selfplay.num_workers <= 0:
        cfg.selfplay.num_workers = 0

    if host.cuda_available:
        if cfg.inference.max_batch_size <= 0:
            cfg.inference.max_batch_size = _cuda_batch_target(cfg, host)
        if tune_selfplay and cfg.selfplay.batch_size_per_worker <= 0:
            if host.cuda_memory_gb >= 11.0:
                cfg.selfplay.batch_size_per_worker = 16
            elif host.cuda_memory_gb >= 8.0:
                cfg.selfplay.batch_size_per_worker = 8
            else:
                cfg.selfplay.batch_size_per_worker = 4
        if tune_selfplay:
            cfg.inference.max_batch_size = max(
                cfg.inference.max_batch_size,
                cfg.selfplay.num_workers * cfg.selfplay.batch_size_per_worker + 64,
            )
        if cfg.train.batch_size <= 0:
            cfg.train.batch_size = _train_batch_target(cfg, host)
        cfg.inference.fp16 = True
        if rt.compile_model is None:
            rt.compile_model = bool(cfg.train.batches_per_epoch >= 500)
        if rt.compile_inference is None:
            rt.compile_inference = False

    if cfg.train.prefetch_batches <= 0:
        cfg.train.prefetch_batches = 2 if host.cuda_available else 1

    return host


def configure_torch_runtime(cfg: Config, host: HostProfile | None = None) -> dict[str, Any]:
    """Apply Torch and BLAS settings for the current process."""
    host = host or detect_host()
    rt = cfg.runtime
    reserve = max(1, min(rt.selfplay_cpu_reserve, host.logical_cpus - 1))
    cpu_threads = rt.cpu_threads or max(1, host.logical_cpus - reserve)
    interop_threads = rt.interop_threads or max(1, min(4, host.logical_cpus // 8))

    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    if host.cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = bool(rt.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(rt.allow_tf32)
        torch.backends.cudnn.benchmark = bool(rt.cudnn_benchmark and not cfg.run.deterministic)
        torch.set_float32_matmul_precision("high")

    return {
        "logical_cpus": host.logical_cpus,
        "physical_cpus": host.physical_cpus,
        "cpu_threads": cpu_threads,
        "interop_threads": interop_threads,
        "cuda": host.cuda_available,
        "cuda_name": host.cuda_name,
        "cuda_memory_gb": round(host.cuda_memory_gb, 2),
        "channels_last": bool(rt.channels_last and host.cuda_available),
        "compile_model": bool(rt.compile_model and host.cuda_available),
        "compile_inference": bool(rt.compile_inference and host.cuda_available),
    }


def dataloader_worker_count(cfg: Config, host: HostProfile | None = None) -> int:
    """Return a safe process-worker count for PyTorch DataLoader."""
    host = host or detect_host()
    if cfg.runtime.dataloader_workers is not None:
        return max(0, cfg.runtime.dataloader_workers)
    if host.system == "windows":
        return 0
    return 0


def _physical_cpu_count(logical: int) -> int:
    try:
        import psutil  # type: ignore

        return psutil.cpu_count(logical=False) or max(1, logical // 2)
    except Exception:
        return max(1, logical // 2)


def _cuda_batch_target(cfg: Config, host: HostProfile) -> int:
    if host.cuda_memory_gb >= 11.0:
        return 192 if cfg.model.channels >= 96 else 256
    if host.cuda_memory_gb >= 8.0:
        return 128
    return 64


def _selfplay_worker_target(cfg: Config, host: HostProfile, worker_budget: int) -> int:
    if cfg.runtime.selfplay_workers is not None:
        return max(1, min(int(cfg.runtime.selfplay_workers), worker_budget))
    if not host.cuda_available:
        return worker_budget

    model_scale = max(
        0.0625,
        (cfg.model.channels / 128.0) * (cfg.model.blocks / 16.0),
    )
    target = int(round(8.0 / (model_scale ** 0.5)))
    return max(4, min(worker_budget, target))


def _train_batch_target(cfg: Config, host: HostProfile) -> int:
    """Choose a training batch that avoids GPU memory cliffs.

    The old heuristic keyed only off total VRAM and picked 512 on 12GB cards.
    For the production 128x16 network that pushes CUDA allocator pressure past
    the fast region on a 4070 Ti. This estimator scales with model depth/width
    and keeps peak allocated memory below a configurable VRAM fraction.
    """
    if not host.cuda_available:
        return 64

    target_fraction = float(getattr(cfg.runtime, "train_memory_fraction", 0.62))
    target_gb = max(1.0, host.cuda_memory_gb * target_fraction)
    candidates = list(range(64, 1025, 32))
    best = candidates[0]
    for batch in candidates:
        if _estimate_train_peak_gb(cfg, batch) <= target_gb:
            best = batch
        else:
            break
    return best


def _estimate_train_peak_gb(cfg: Config, batch_size: int) -> float:
    channels_scale = max(0.25, cfg.model.channels / 128.0)
    blocks_scale = max(0.25, cfg.model.blocks / 16.0)
    head_scale = max(0.75, len(cfg.model.heads) / 6.0)
    attention_blocks = len(getattr(cfg.model, "attention_positions", []))
    attention_scale = 1.0 + 0.22 * attention_blocks
    if getattr(cfg.model, "architecture", "cnn") == "restnet":
        attention_scale = max(attention_scale, 1.15)
    sparse_scale = 1.0 + (0.04 if getattr(cfg.model, "sparse_policy", False) else 0.0)
    per_sample_gb = 0.0327 * channels_scale * blocks_scale * head_scale * attention_scale * sparse_scale
    model_overhead_gb = 0.35 * (channels_scale ** 2) * blocks_scale * attention_scale
    return model_overhead_gb + batch_size * per_sample_gb
