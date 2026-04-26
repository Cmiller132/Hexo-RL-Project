"""Training loop for Hexo-RL.

Orchestrates one training epoch: data loading, forward pass, loss computation,
backpropagation, gradient clipping, optimizer/scheduler steps, EMA updates,
and periodic logging.
"""

import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path

from hexorl.config import Config
from hexorl.model.network import HexNet
from hexorl.train.losses import compute_losses
from hexorl.train.ema import ModelEMA

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for one epoch."""

    def __init__(
        self,
        model: HexNet,
        cfg: Config,
        dataloader,
        ema: Optional[ModelEMA] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.dataloader = dataloader
        self.train_cfg = cfg.train
        self.batches_per_epoch = self.train_cfg.batches_per_epoch

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        self.model = self.model.to(self.device)
        self.model.train()

        self.ema = ema or ModelEMA(self.model, decay=0.9999)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = (
            cfg.inference.fp16
            and device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self._loss_weights = dict(self.train_cfg.loss_weights)
        self._n_bins = getattr(self.model, 'n_bins', 65)
        # Lookahead horizon names derived from buffer config
        self._lookahead_keys = [
            f"lookahead_{h}" for h in getattr(cfg.buffer, 'lookahead_horizons', [])
        ]

        self.global_step = 0
        self.epoch = 0
        self._epoch_losses: Dict[str, list] = {}
        self._start_time = 0.0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = self.train_cfg.peak_lr
        wd = self.train_cfg.weight_decay
        opt_name = self.train_cfg.optimizer.lower()

        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups, lr=lr)
        elif opt_name == "sgd":
            return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        schedule = self.train_cfg.lr_schedule.lower()

        warmup_steps = int(self.batches_per_epoch * 0.1)
        total_steps = self.batches_per_epoch

        if schedule == "cosine":
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.train_cfg.peak_lr * 0.01,
            )

            def warmup_fn(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps
                return 1.0

            warmup = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=warmup_fn
            )

            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_steps],
            )
        elif schedule == "constant":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {schedule}")

    def train_epoch(self) -> Dict[str, float]:
        self.epoch += 1
        tracked_keys = ["total"] + list(self.train_cfg.loss_weights.keys())
        self._epoch_losses = {k: [] for k in tracked_keys}
        self._start_time = time.monotonic()
        self.model.train()

        batch_iter = iter(self.dataloader)

        for batch_idx in range(self.batches_per_epoch):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(self.dataloader)
                batch = next(batch_iter)

            loss_dict = self._train_step(batch, batch_idx)

            for k, v in loss_dict.items():
                if k not in self._epoch_losses:
                    self._epoch_losses[k] = []
                self._epoch_losses[k].append(v)

            self.global_step += 1

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                self._log_step(batch_idx)

        return self._epoch_stats()

    def _train_step(self, batch, batch_idx: int) -> Dict[str, float]:
        # Batch is (tensors, policies, values[, lookahead_list])
        # lookahead_list is a list of per-horizon arrays when present.
        if len(batch) == 4:
            tensors, policies, values, lookahead_list = batch
        else:
            tensors, policies, values = batch
            lookahead_list = []

        tensors = tensors.to(self.device, non_blocking=True)
        policies = policies.to(self.device, non_blocking=True)
        values = values.to(self.device, non_blocking=True)

        targets = {"policy": policies, "value": values}

        for key, lv_arr in zip(self._lookahead_keys, lookahead_list):
            targets[key] = lv_arr.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(tensors)
            total_loss, per_head = compute_losses(
                predictions, targets,
                loss_weights=self._loss_weights,
                n_bins=self._n_bins,
            )

        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.ema.update()

        result = {"total": float(total_loss.detach().cpu())}
        for k, v in per_head.items():
            if isinstance(v, torch.Tensor):
                result[k] = float(v.detach().cpu())

        return result

    def _log_step(self, batch_idx: int):
        current_lr = self.optimizer.param_groups[0]["lr"]
        elapsed = time.monotonic() - self._start_time
        batches_done = batch_idx + 1
        batches_per_sec = batches_done / max(elapsed, 0.001)
        remaining = (self.batches_per_epoch - batches_done) / max(batches_per_sec, 1)

        total_loss = self._smooth("total")
        policy_loss = self._smooth("policy")
        value_loss = self._smooth("value")

        logger.info(
            f"[Epoch {self.epoch}] "
            f"{batches_done}/{self.batches_per_epoch} "
            f"({batches_per_sec:.1f}/s) | "
            f"loss={total_loss:.4f} | "
            f"p={policy_loss:.4f} v={value_loss:.4f} | "
            f"lr={current_lr:.2e} | "
            f"eta={remaining:.0f}s"
        )

    def _smooth(self, key: str, window: int = 20) -> float:
        values = self._epoch_losses.get(key, [])
        if not values:
            return 0.0
        window = min(window, len(values))
        return sum(values[-window:]) / window

    def _epoch_stats(self) -> Dict[str, float]:
        elapsed = time.monotonic() - self._start_time
        stats = {
            "epoch": self.epoch,
            "batches": self.batches_per_epoch,
            "elapsed_s": elapsed,
            "batches_per_sec": self.batches_per_epoch / max(elapsed, 0.001),
        }
        for key in self._epoch_losses:
            vals = self._epoch_losses[key]
            if vals:
                stats[f"loss_{key}"] = sum(vals) / len(vals)
        return stats

    def save_checkpoint(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "ema_state_dict": self.ema.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "cfg": self.cfg,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("ema_state_dict"):
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(f"Checkpoint loaded from {path} (epoch {self.epoch})")

    def get_ema_model(self) -> nn.Module:
        self.ema.apply_shadow()
        return self.model

    def restore_training_weights(self):
        self.ema.restore()
