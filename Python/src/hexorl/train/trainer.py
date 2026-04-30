"""Training loop for Hexo-RL.

Orchestrates one training epoch: data loading, forward pass, loss computation,
backpropagation, gradient clipping, optimizer/scheduler steps, EMA updates,
and periodic logging.
"""

import time
import logging
import queue
import threading
import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from hexorl.config import Config
from hexorl.models.checkpoint import CheckpointBundle, CheckpointManager
from hexorl.models.factory import train_adapter_for
from hexorl.models.crop_network import HexNet
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
        self._channels_last = (
            bool(getattr(cfg.runtime, "channels_last", True))
            and self.device.type == "cuda"
        )
        if self._channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if bool(getattr(cfg.runtime, "compile_model", False)) and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model,
                    mode=getattr(cfg.runtime, "compile_mode", "reduce-overhead"),
                )
            except Exception as exc:
                logger.warning("torch.compile disabled after failure: %s", exc)
        self.model.train()

        self.ema = ema or ModelEMA(self.model, decay=0.9999)

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = (
            cfg.inference.fp16
            and device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.adapter = train_adapter_for(self.model, cfg, device=self.device)
        self._loss_weights = self.adapter.loss_plan.weights
        self._n_bins = getattr(model, 'n_bins', getattr(self.model, 'n_bins', 65))
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

        batch_iter = _PrefetchIterator(
            iter(self.dataloader),
            max_prefetch=max(0, int(getattr(self.train_cfg, "prefetch_batches", 0))),
        )

        try:
            for batch_idx in range(self.batches_per_epoch):
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter.close()
                    batch_iter = _PrefetchIterator(
                        iter(self.dataloader),
                        max_prefetch=max(0, int(getattr(self.train_cfg, "prefetch_batches", 0))),
                    )
                    batch = next(batch_iter)

                loss_dict = self._train_step(batch, batch_idx)

                for k, v in loss_dict.items():
                    if k not in self._epoch_losses:
                        self._epoch_losses[k] = []
                    self._epoch_losses[k].append(v)

                self.global_step += 1

                if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                    self._log_step(batch_idx)
        finally:
            batch_iter.close()

        return self._epoch_stats()

    def _train_step(self, batch, batch_idx: int) -> Dict[str, float]:
        projected = self.adapter.project_batch(batch, channels_last=self._channels_last)
        targets = projected.targets

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.adapter.forward(projected)
            total_loss, per_head = self.adapter.losses(
                predictions,
                targets,
                n_bins=self._n_bins,
            )
            if not torch.isfinite(total_loss):
                details = {
                    name: float(value.detach().float().cpu())
                    for name, value in per_head.items()
                    if isinstance(value, torch.Tensor)
                }
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={self.epoch} "
                    f"batch={batch_idx}: total={float(total_loss.detach().float().cpu())} "
                    f"per_head={details}"
                )

        stepped = True
        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= scale_before
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if stepped and self.scheduler is not None:
            self.scheduler.step()

        if stepped:
            self.ema.update()

        result = {"total": float(total_loss.detach().cpu())}
        for k, v in per_head.items():
            if isinstance(v, torch.Tensor):
                result[k] = float(v.detach().cpu())
        for weight_key in ("value_weight", "policy_weight", "regret_weight", "opp_policy_weight"):
            weight_tensor = targets.get(weight_key)
            if weight_tensor is not None:
                with torch.no_grad():
                    weight_float = weight_tensor.float()
                    result[f"{weight_key}_mean"] = float(weight_float.mean().detach().cpu())
                    result[f"{weight_key}_zero_frac"] = float((weight_float <= 0).float().mean().detach().cpu())
        if "policy" in predictions and "policy" in targets:
            with torch.no_grad():
                policy_probs = torch.softmax(predictions["policy"], dim=-1)
                target_top = targets["policy"].argmax(dim=-1)
                pred_top = policy_probs.argmax(dim=-1)
                top1_prob = policy_probs.gather(1, target_top.unsqueeze(1)).squeeze(1)
                top1_acc = (pred_top == target_top).float()
                policy_weight = targets.get("policy_weight")
                if policy_weight is not None and torch.any(policy_weight > 0):
                    policy_weight = policy_weight.to(device=top1_prob.device, dtype=top1_prob.dtype)
                    denom = policy_weight.sum().clamp(min=1e-6)
                    result["policy_top1_prob"] = float((top1_prob * policy_weight).sum().div(denom).detach().cpu())
                    result["policy_top1_acc"] = float((top1_acc * policy_weight).sum().div(denom).detach().cpu())
                    result["policy_full_search_frac"] = float((policy_weight > 0).float().mean().detach().cpu())
                else:
                    result["policy_top1_prob"] = float(top1_prob.mean().detach().cpu())
                    result["policy_top1_acc"] = float(top1_acc.mean().detach().cpu())
        if "sparse_policy" in predictions and "sparse_policy_target" in targets:
            with torch.no_grad():
                mask = targets.get("candidate_mask")
                sparse_target = targets["sparse_policy_target"]
                sparse_logits = predictions["sparse_policy"]
                if mask is not None and torch.any(mask):
                    masked_logits = sparse_logits.masked_fill(~mask.to(dtype=torch.bool), -80.0)
                    pred_top = masked_logits.argmax(dim=-1)
                    target_top = sparse_target.argmax(dim=-1)
                    valid = mask.any(dim=-1) & (sparse_target.sum(dim=-1) > 0)
                    if torch.any(valid):
                        result["sparse_policy_top1_acc"] = float(
                            (pred_top[valid] == target_top[valid]).float().mean().detach().cpu()
                        )
                    if "candidate_missing_mass" in targets:
                        result["candidate_missing_mass"] = float(
                            targets["candidate_missing_mass"].float().mean().detach().cpu()
                        )
        if "pair_policy" in predictions and "pair_policy_target" in targets:
            with torch.no_grad():
                mask = targets.get("pair_candidate_mask")
                pair_target = targets["pair_policy_target"]
                pair_logits = predictions["pair_policy"]
                if mask is not None and torch.any(mask):
                    masked_logits = pair_logits.masked_fill(~mask.to(dtype=torch.bool), -80.0)
                    pred_top = masked_logits.argmax(dim=-1)
                    target_top = pair_target.argmax(dim=-1)
                    valid = mask.any(dim=-1) & (pair_target.sum(dim=-1) > 0)
                    if torch.any(valid):
                        result["pair_policy_top1_acc"] = float(
                            (pred_top[valid] == target_top[valid]).float().mean().detach().cpu()
                        )
                    if "pair_candidate_missing_mass" in targets:
                        result["pair_candidate_missing_mass"] = float(
                            targets["pair_candidate_missing_mass"].float().mean().detach().cpu()
                        )

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
        top1_prob = self._smooth("policy_top1_prob")
        top1_acc = self._smooth("policy_top1_acc")

        logger.info(
            f"[Epoch {self.epoch}] "
            f"{batches_done}/{self.batches_per_epoch} "
            f"({batches_per_sec:.1f}/s) | "
            f"loss={total_loss:.4f} | "
            f"p={policy_loss:.4f} v={value_loss:.4f} | "
            f"top1_p={top1_prob:.3f} top1_acc={top1_acc:.3f} | "
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
                if key == "total" or key in self.train_cfg.loss_weights:
                    stats[f"loss_{key}"] = sum(vals) / len(vals)
                else:
                    stats[key] = sum(vals) / len(vals)
        return stats

    def save_checkpoint(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        CheckpointManager().save(
            CheckpointBundle(
                cfg=self.cfg,
                model=self.model,
                optimizer_state_dict=self.optimizer.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
                ema_state_dict=self.ema.state_dict(),
                scaler_state_dict=self.scaler.state_dict() if self.use_amp else None,
                epoch=self.epoch,
                global_step=self.global_step,
                created_by={"command": "Trainer.save_checkpoint"},
            ),
            path,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        loaded = CheckpointManager().load(path, purpose="train", device=self.device)
        checkpoint = loaded.payload
        CheckpointManager().load_state_into_model(self.model, checkpoint["model_state_dict"])
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


class _PrefetchIterator:
    """Small thread-backed prefetcher for a single in-memory DataLoader."""

    _STOP = object()

    def __init__(self, iterator, max_prefetch: int = 0):
        self._iterator = iterator
        self._enabled = max_prefetch > 0
        self._closed = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=max_prefetch) if self._enabled else queue.Queue()
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._thread = threading.Thread(target=self._run, name="train-prefetch", daemon=True)
            self._thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._enabled:
            return next(self._iterator)
        item = self._queue.get()
        if item is self._STOP:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self._closed.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self):
        try:
            while not self._closed.is_set():
                try:
                    item = next(self._iterator)
                except StopIteration:
                    self._queue.put(self._STOP)
                    return
                self._queue.put(item)
        except BaseException as exc:
            self._queue.put(exc)
