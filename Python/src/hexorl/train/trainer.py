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

from hexorl.action_contract.candidates import CANDIDATE_FEATURE_NAMES, CANDIDATE_FEATURE_VERSION
from hexorl.config import Config
from hexorl.models.assembly import is_global_graph_model
from hexorl.models.loading import restore_model_weights
from hexorl.models.registry import resolve_model_spec
from hexorl.models.specs import merge_resolved_loss_weights
from hexorl.replay.training_batch import (
    prepare_dense_training_batch,
    prepare_global_graph_training_batch,
)
from hexorl.train.loss_plan import build_loss_plan
from hexorl.train.losses import compute_losses
from hexorl.train.ema import ModelEMA

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for one epoch."""

    def __init__(
        self,
        model: nn.Module,
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

        self._is_global_graph_model = is_global_graph_model(self.model)
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

        self._resolved_spec = resolve_model_spec(cfg)
        self._loss_weights = merge_resolved_loss_weights(
            self._resolved_spec,
            self.train_cfg.loss_weights,
        )
        self._loss_plan = build_loss_plan(self._resolved_spec, self._loss_weights)
        self._n_bins = getattr(model, 'n_bins', getattr(self.model, 'n_bins', 65))
        # Lookahead horizon names derived from buffer config
        self._lookahead_keys = [
            f"lookahead_{h}" for h in getattr(cfg.buffer, 'lookahead_horizons', [])
            if f"lookahead_{h}" in self._resolved_spec.outputs
        ]

        self.global_step = 0
        self.epoch = 0
        self._epoch_losses: Dict[str, list] = {}
        self._start_time = 0.0
        self._logged_graph_microbatch = False
        self._graph_microbatch_choice: int | None = None
        self._graph_microbatch_heuristic: int | None = None
        self._graph_microbatch_rejections: list[str] = []

    def close_dataloader(self) -> None:
        """Release persistent DataLoader workers owned by the current loader."""
        shutdown_dataloader_workers(self.dataloader)

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

        prefetch_batches = 0 if self._is_global_graph_model else max(
            0,
            int(getattr(self.train_cfg, "prefetch_batches", 0)),
        )
        batch_iter = _PrefetchIterator(iter(self.dataloader), max_prefetch=prefetch_batches)

        try:
            for batch_idx in range(self.batches_per_epoch):
                wait_started = time.perf_counter()
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter.close()
                    batch_iter = _PrefetchIterator(
                        iter(self.dataloader),
                        max_prefetch=prefetch_batches,
                    )
                    batch = next(batch_iter)
                dataloader_wait_s = time.perf_counter() - wait_started

                loss_dict = self._train_step(batch, batch_idx)
                self._attach_loader_runtime_metrics(loss_dict, dataloader_wait_s)

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
        # Batch is (tensors, policies, values[, lookahead_list[, aux_targets]])
        # lookahead_list is a list of per-horizon arrays when present.
        aux_targets = {}
        if len(batch) == 5:
            tensors, policies, values, lookahead_list, aux_targets = batch
        elif len(batch) == 4:
            tensors, policies, values, lookahead_list = batch
        else:
            tensors, policies, values = batch
            lookahead_list = []

        if self._is_global_graph_model:
            return self._train_global_graph_step(
                tensors,
                policies,
                values,
                lookahead_list,
                aux_targets,
                batch_idx,
            )

        prepared = prepare_dense_training_batch(
            tensors=tensors,
            policies=policies,
            values=values,
            lookahead_list=lookahead_list,
            aux_targets=aux_targets,
            lookahead_keys=self._lookahead_keys,
            device=self.device,
            channels_last=self._channels_last,
            train_policy_on_full_search_only=getattr(self.cfg.selfplay, "train_policy_on_full_search_only", True),
        )
        targets = prepared.targets

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(
                prepared.model_inputs["tensors"],
                candidate_features=targets.get("candidate_features"),
                candidate_indices=targets.get("candidate_indices"),
                candidate_mask=targets.get("candidate_mask"),
                pair_candidate_features=targets.get("pair_candidate_features"),
                pair_candidate_row_indices=targets.get("pair_candidate_row_indices"),
                pair_candidate_indices=targets.get("pair_candidate_indices"),
                pair_candidate_mask=targets.get("pair_candidate_mask"),
            )
            total_loss, per_head = compute_losses(
                predictions, targets,
                loss_weights=self._loss_weights,
                n_bins=self._n_bins,
                loss_plan=self._loss_plan,
                row_tables=prepared.row_tables,
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

    def _train_global_graph_step(
        self,
        tensors: torch.Tensor,
        policies: torch.Tensor,
        values: torch.Tensor,
        lookahead_list,
        aux_targets: dict,
        batch_idx: int,
    ) -> Dict[str, float]:
        del tensors
        del policies
        batch_size = int(values.shape[0])
        required = [
            "token_features",
            "token_type",
            "token_qr",
            "token_mask",
            "legal_token_indices",
            "legal_mask",
            "relation_type",
            "relation_bias",
        ]
        missing = [name for name in required if name not in aux_targets]
        if missing:
            raise ValueError(f"global graph training batch is missing graph tensors: {missing}")

        setup_timings: dict[str, float] = {}
        heuristic_size = self._graph_train_microbatch_heuristic(aux_targets, batch_size)
        configured_microbatch = int(getattr(self.train_cfg, "graph_microbatch_size", 0) or 0) > 0
        microbatch_size = self._graph_train_microbatch_size(
            aux_targets,
            batch_size,
            heuristic_size=heuristic_size,
        )
        if not self._logged_graph_microbatch:
            logger.info(
                "global graph training microbatch selected_size=%d heuristic_size=%d "
                "effective_batch_size=%d rejections=%s",
                microbatch_size,
                heuristic_size,
                batch_size,
                "; ".join(self._graph_microbatch_rejections) or "none",
            )
            self._logged_graph_microbatch = True

        oom_retries = 0
        nonfinite_retries = 0
        loader_timings = aux_targets.get("_loader_timings") or {}
        while True:
            try:
                result = self._train_global_graph_step_once(
                    values=values,
                    lookahead_list=lookahead_list,
                    aux_targets=aux_targets,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                    microbatch_size=microbatch_size,
                    setup_timings=setup_timings,
                )
                result["graph_microbatch_heuristic_size"] = float(heuristic_size)
                result["graph_microbatch_autotuned"] = float(
                    (not configured_microbatch) and microbatch_size != heuristic_size
                )
                result["graph_microbatch_configured"] = float(configured_microbatch)
                result["graph_microbatch_rejected_count"] = float(len(self._graph_microbatch_rejections))
                result["graph_microbatch_oom_retries"] = float(oom_retries)
                result["graph_microbatch_nonfinite_retries"] = float(nonfinite_retries)
                for key in (
                    "graph_loader_sample_s",
                    "graph_loader_get_records_s",
                    "graph_loader_tensor_encode_s",
                    "graph_loader_candidate_s",
                    "graph_loader_graph_base_s",
                    "graph_loader_policy_overlay_s",
                    "graph_loader_pair_rows_s",
                    "graph_loader_collate_s",
                ):
                    result[key] = float(loader_timings.get(key, 0.0))
                self._graph_microbatch_choice = microbatch_size
                return result
            except RuntimeError as exc:
                if not self._is_cuda_oom(exc) or microbatch_size <= 1:
                    raise
                next_size = self._lower_graph_microbatch_size(microbatch_size, heuristic_size)
                if next_size >= microbatch_size:
                    raise
                oom_retries += 1
                self._graph_microbatch_rejections.append(
                    f"{microbatch_size}:cuda_oom_retry_{oom_retries}"
                )
                logger.warning(
                    "global graph microbatch_size=%d hit CUDA OOM; retrying current batch with %d",
                    microbatch_size,
                    next_size,
                )
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                microbatch_size = next_size
            except FloatingPointError:
                if microbatch_size <= heuristic_size:
                    raise
                nonfinite_retries += 1
                self._graph_microbatch_rejections.append(
                    f"{microbatch_size}:nonfinite_loss_retry_{nonfinite_retries}"
                )
                logger.warning(
                    "global graph microbatch_size=%d produced non-finite loss; retrying with heuristic_size=%d",
                    microbatch_size,
                    heuristic_size,
                )
                self.optimizer.zero_grad(set_to_none=True)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                microbatch_size = heuristic_size

    def _train_global_graph_step_once(
        self,
        *,
        values: torch.Tensor,
        lookahead_list,
        aux_targets: dict,
        batch_idx: int,
        batch_size: int,
        microbatch_size: int,
        setup_timings: dict[str, float],
    ) -> Dict[str, float]:
        timings = dict(setup_timings)
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass

        self.optimizer.zero_grad(set_to_none=True)
        total_accum = 0.0
        per_head_accum: dict[str, float] = {}
        weight_sums: dict[str, float] = {}
        weight_zero_counts: dict[str, float] = {}
        microbatch_count = 0

        for start in range(0, batch_size, microbatch_size):
            end = min(batch_size, start + microbatch_size)
            micro_count = end - start
            micro_weight = micro_count / max(batch_size, 1)
            micro_aux_targets = self._slice_targets_for_batch(aux_targets, start, end, batch_size)
            micro_lookahead = [
                self._slice_value_for_batch(value, start, end, batch_size)
                for value in lookahead_list
            ]
            prepare_started = time.perf_counter()
            prepared = prepare_global_graph_training_batch(
                values=self._slice_value_for_batch(values, start, end, batch_size),
                lookahead_list=micro_lookahead,
                aux_targets=micro_aux_targets,
                lookahead_keys=self._lookahead_keys,
                device=self.device,
                train_policy_on_full_search_only=getattr(
                    self.cfg.selfplay,
                    "train_policy_on_full_search_only",
                    True,
                ),
                timings=timings,
            )
            timings["graph_prepare_s"] = timings.get("graph_prepare_s", 0.0) + (
                time.perf_counter() - prepare_started
            )
            micro_targets = prepared.targets

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                forward_started = time.perf_counter()
                predictions = self.model(
                    **prepared.model_inputs,
                )
                timings["graph_forward_s"] = timings.get("graph_forward_s", 0.0) + (
                    time.perf_counter() - forward_started
                )
                loss_started = time.perf_counter()
                total_loss, per_head = compute_losses(
                    predictions,
                    micro_targets,
                    loss_weights=self._loss_weights,
                    n_bins=self._n_bins,
                    loss_plan=self._loss_plan,
                    row_tables=prepared.row_tables,
                )
                timings["graph_loss_s"] = timings.get("graph_loss_s", 0.0) + (
                    time.perf_counter() - loss_started
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
                backward_loss = total_loss * micro_weight

            backward_started = time.perf_counter()
            if self.use_amp:
                self.scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            timings["graph_backward_s"] = timings.get("graph_backward_s", 0.0) + (
                time.perf_counter() - backward_started
            )

            total_accum += float(total_loss.detach().float().cpu()) * micro_weight
            for name, value in per_head.items():
                if isinstance(value, torch.Tensor):
                    per_head_accum[name] = (
                        per_head_accum.get(name, 0.0)
                        + float(value.detach().float().cpu()) * micro_weight
                    )
            for weight_key in (
                "value_weight",
                "policy_weight",
                "regret_weight",
                "opp_policy_weight",
                "pair_policy_weight",
            ):
                weight_tensor = micro_targets.get(weight_key)
                if isinstance(weight_tensor, torch.Tensor):
                    weight_float = weight_tensor.detach().float()
                    weight_sums[weight_key] = weight_sums.get(weight_key, 0.0) + float(weight_float.sum().cpu())
                    weight_zero_counts[weight_key] = weight_zero_counts.get(weight_key, 0.0) + float(
                        (weight_float <= 0).float().sum().cpu()
                    )
            microbatch_count += 1

        optimizer_started = time.perf_counter()
        stepped = True
        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            stepped = self.scaler.get_scale() >= scale_before
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        if stepped and self.scheduler is not None:
            self.scheduler.step()
        if stepped:
            self.ema.update()
        timings["graph_optimizer_s"] = timings.get("graph_optimizer_s", 0.0) + (
            time.perf_counter() - optimizer_started
        )

        result = {
            "total": total_accum,
            "graph_microbatch_size": float(microbatch_size),
            "graph_microbatch_count": float(microbatch_count),
        }
        if self.device.type == "cuda":
            try:
                result["graph_peak_cuda_allocated_mb"] = float(
                    torch.cuda.max_memory_allocated(self.device)
                ) / (1024.0 * 1024.0)
                result["graph_peak_cuda_reserved_mb"] = float(
                    torch.cuda.max_memory_reserved(self.device)
                ) / (1024.0 * 1024.0)
            except RuntimeError:
                result["graph_peak_cuda_allocated_mb"] = 0.0
                result["graph_peak_cuda_reserved_mb"] = 0.0
        else:
            result["graph_peak_cuda_allocated_mb"] = 0.0
            result["graph_peak_cuda_reserved_mb"] = 0.0
        for key, value in timings.items():
            result[key] = float(value)
        result["graph_microbatch_efficiency"] = float(batch_size) / max(float(microbatch_count), 1.0)
        result.update(per_head_accum)
        for weight_key, weight_sum in weight_sums.items():
            result[f"{weight_key}_mean"] = weight_sum / max(batch_size, 1)
            result[f"{weight_key}_zero_frac"] = weight_zero_counts.get(weight_key, 0.0) / max(batch_size, 1)

        return result

    def _graph_train_microbatch_size(
        self,
        targets: dict,
        batch_size: int,
        *,
        heuristic_size: int | None = None,
    ) -> int:
        configured = int(getattr(self.train_cfg, "graph_microbatch_size", 0) or 0)
        if configured > 0:
            return max(1, min(configured, batch_size))
        heuristic = heuristic_size or self._graph_train_microbatch_heuristic(targets, batch_size)
        if self.device.type != "cuda":
            self._graph_microbatch_heuristic = heuristic
            self._graph_microbatch_rejections = []
            return heuristic
        if not bool(getattr(self.train_cfg, "graph_microbatch_autotune", True)):
            self._graph_microbatch_heuristic = heuristic
            self._graph_microbatch_rejections = []
            return heuristic

        selected, rejections = self._select_graph_microbatch_candidate(
            targets,
            batch_size,
            heuristic,
        )
        self._graph_microbatch_heuristic = heuristic
        self._graph_microbatch_rejections = rejections
        return selected

    def _graph_train_microbatch_heuristic(self, targets: dict, batch_size: int) -> int:
        if self.device.type != "cuda":
            return max(1, batch_size)

        token_count = int(self._as_tensor(targets["token_features"]).shape[1])
        pair_width = 0
        for key in ("pair_first_indices", "pair_second_indices", "pair_token_indices"):
            value = targets.get(key)
            if value is not None:
                tensor = self._as_tensor(value)
                if tensor.ndim >= 2:
                    pair_width = max(pair_width, int(tensor.shape[1]))
        layers = max(1, int(getattr(self.cfg.model, "graph_layers", 1)))

        if token_count >= 1536:
            limit = 4
        elif token_count >= 640:
            limit = 8
        elif token_count >= 384:
            limit = 16
        else:
            limit = 32
        if layers >= 2:
            limit = max(1, limit // 2)
        if pair_width >= 1024:
            limit = min(limit, 4)
        elif pair_width >= 512:
            limit = min(limit, 8)
        return max(1, min(limit, batch_size))

    def _select_graph_microbatch_candidate(
        self,
        targets: dict,
        batch_size: int,
        heuristic: int,
    ) -> tuple[int, list[str]]:
        configured_max = int(getattr(self.train_cfg, "graph_microbatch_autotune_max_size", 32) or 32)
        configured_max = max(1, configured_max)
        device_max = configured_max
        free_bytes = 0
        total_bytes = 0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            pass
        if total_bytes:
            gib = total_bytes / float(1024**3)
            if gib < 8.0:
                device_max = min(device_max, 8)
            elif gib < 11.0:
                device_max = min(device_max, 16)
        max_candidate = max(1, min(batch_size, device_max))
        candidate_values = {
            heuristic,
            min(batch_size, heuristic * 2),
            min(batch_size, heuristic * 4),
            max_candidate,
            8,
            16,
            32,
        }
        candidates = sorted(
            {
                int(value)
                for value in candidate_values
                if int(value) >= heuristic and int(value) <= max_candidate
            },
            reverse=True,
        )
        if not candidates:
            return heuristic, []

        per_sample_bytes = self._estimated_graph_training_bytes_per_sample(targets, batch_size)
        rejections: list[str] = []
        for candidate in candidates:
            reason = self._graph_microbatch_memory_rejection(
                candidate,
                per_sample_bytes,
                free_bytes,
                total_bytes,
            )
            if reason:
                rejections.append(f"{candidate}:{reason}")
                continue
            return max(1, min(candidate, batch_size)), rejections
        return heuristic, rejections

    def _estimated_graph_training_bytes_per_sample(self, targets: dict, batch_size: int) -> float:
        keys = (
            "token_features",
            "token_type",
            "token_qr",
            "token_mask",
            "legal_token_indices",
            "legal_qr",
            "legal_mask",
            "relation_type",
            "relation_bias",
            "policy_target",
            "opp_legal_qr",
            "opp_legal_mask",
            "opp_policy_target",
            "pair_token_indices",
            "pair_first_indices",
            "pair_second_indices",
            "pair_first_policy_target",
            "pair_policy_target",
            "pair_second_policy_target",
            "tactical_target",
        )
        total_bytes = 0
        for key in keys:
            value = targets.get(key)
            if value is None:
                continue
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                total_bytes += int(nbytes)
                continue
            tensor = self._as_tensor(value)
            if isinstance(tensor, torch.Tensor):
                total_bytes += int(tensor.numel() * tensor.element_size())
        return float(total_bytes) / max(float(batch_size), 1.0)

    def _graph_microbatch_memory_rejection(
        self,
        candidate: int,
        per_sample_bytes: float,
        free_bytes: int,
        total_bytes: int,
    ) -> str | None:
        if per_sample_bytes <= 0.0 or free_bytes <= 0 or total_bytes <= 0:
            return None
        headroom = float(getattr(self.train_cfg, "graph_microbatch_memory_headroom", 0.75) or 0.75)
        headroom = min(max(headroom, 0.1), 0.95)
        reserved_floor = float(total_bytes) * (1.0 - headroom)
        usable_bytes = max(0.0, float(free_bytes) - reserved_floor)
        layers = max(1, int(getattr(self.cfg.model, "graph_layers", 1)))
        activation_multiplier = 4.0 + (1.5 * float(layers))
        estimated_bytes = per_sample_bytes * float(candidate) * activation_multiplier
        if estimated_bytes > usable_bytes:
            return (
                "estimated_memory "
                f"{estimated_bytes / (1024.0**3):.2f}GiB>"
                f"{usable_bytes / (1024.0**3):.2f}GiB"
            )
        return None

    @staticmethod
    def _lower_graph_microbatch_size(current: int, heuristic: int) -> int:
        if current > heuristic:
            return max(heuristic, current // 2)
        return max(1, current // 2)

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        text = str(exc).lower()
        retryable_markers = (
            "out of memory",
            "cublas_status_alloc_failed",
            "cublas_status_internal_error",
            "cublas_status_execution_failed",
            "cublas_status_not_supported",
            "cublasstatusinternalerror",
            "cublasstatusexecutionfailed",
        )
        return ("cuda" in text or "cublas" in text) and any(marker in text for marker in retryable_markers)

    def _slice_targets_for_batch(
        self,
        targets: dict,
        start: int,
        end: int,
        batch_size: int,
    ) -> dict:
        sliced = {}
        for key, value in targets.items():
            if key.startswith("_"):
                sliced[key] = value
                continue
            sliced[key] = self._slice_value_for_batch(value, start, end, batch_size)
        return sliced

    def _slice_value_for_batch(
        self,
        value,
        start: int,
        end: int,
        batch_size: int,
    ):
        tensor = self._as_tensor(value)
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim > 0 and int(tensor.shape[0]) == batch_size:
                tensor = tensor[start:end]
            return tensor
        return value

    @staticmethod
    def _as_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "__array__"):
            return torch.as_tensor(value)
        return value

    def _attach_loader_runtime_metrics(self, loss_dict: Dict[str, float], dataloader_wait_s: float) -> None:
        loss_dict["dataloader_wait_s"] = float(dataloader_wait_s)
        workers = float(getattr(self.dataloader, "num_workers", 0) or 0)
        loss_dict["graph_loader_workers"] = workers
        loss_dict["graph_loader_prefetch_factor"] = float(
            getattr(self.dataloader, "prefetch_factor", 0) or 0
        )
        if not self._is_global_graph_model:
            return

        timed_step = sum(
            float(loss_dict.get(key, 0.0) or 0.0)
            for key in (
                "graph_prepare_s",
                "graph_row_table_s",
                "graph_to_device_s",
                "graph_forward_s",
                "graph_loss_s",
                "graph_backward_s",
                "graph_optimizer_s",
            )
        )
        if dataloader_wait_s > max(0.05, timed_step * 1.25):
            label = "cpu_graph_build" if workers <= 0.0 else "worker_ipc"
        elif timed_step > max(0.05, dataloader_wait_s * 1.25):
            label = "gpu_step"
        else:
            label = "balanced"
        for candidate in ("cpu_graph_build", "worker_ipc", "gpu_step", "balanced"):
            loss_dict[f"graph_bottleneck_{candidate}"] = 1.0 if label == candidate else 0.0

    def _smooth_bottleneck_label(self) -> str:
        labels = ("cpu_graph_build", "worker_ipc", "gpu_step", "balanced")
        scored = [
            (self._smooth(f"graph_bottleneck_{label}"), label)
            for label in labels
        ]
        score, label = max(scored, key=lambda item: item[0])
        return label if score > 0.0 else "unknown"

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
        graph_suffix = ""
        if self._is_global_graph_model:
            graph_suffix = (
                f" | graph_mb={self._smooth('graph_microbatch_size'):.0f}"
                f"/{self._smooth('graph_microbatch_count'):.0f}"
                f" workers={self._smooth('graph_loader_workers'):.0f}"
                f" prefetch={self._smooth('graph_loader_prefetch_factor'):.0f}"
                f" wait={self._smooth('dataloader_wait_s'):.3f}s"
                f" loader={self._smooth('graph_loader_sample_s'):.3f}s"
                f" base={self._smooth('graph_loader_graph_base_s'):.3f}s"
                f" cand={self._smooth('graph_loader_candidate_s'):.3f}s"
                f" collate={self._smooth('graph_loader_collate_s'):.3f}s"
                f" h2d={self._smooth('graph_to_device_s'):.3f}s"
                f" fwd={self._smooth('graph_forward_s'):.3f}s"
                f" loss={self._smooth('graph_loss_s'):.3f}s"
                f" bwd={self._smooth('graph_backward_s'):.3f}s"
                f" opt={self._smooth('graph_optimizer_s'):.3f}s"
                f" peak={self._smooth('graph_peak_cuda_allocated_mb'):.0f}MB"
                f" bottleneck={self._smooth_bottleneck_label()}"
            )

        logger.info(
            f"[Epoch {self.epoch}] "
            f"{batches_done}/{self.batches_per_epoch} "
            f"({batches_per_sec:.1f}/s) | "
            f"loss={total_loss:.4f} | "
            f"p={policy_loss:.4f} v={value_loss:.4f} | "
            f"top1_p={top1_prob:.3f} top1_acc={top1_acc:.3f} | "
            f"lr={current_lr:.2e} | "
            f"eta={remaining:.0f}s"
            f"{graph_suffix}"
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
        candidate_contract = {
            "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
        }
        model_metadata = self.cfg.model.model_dump(mode="json")
        model_metadata.update(candidate_contract)

        checkpoint = {
            "model_state_dict": _uncompiled_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "ema_state_dict": self.ema.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "cfg": self.cfg,
            "cfg_json": self.cfg.model_dump(mode="json"),
            "model_metadata": model_metadata,
            "action_contract_metadata": candidate_contract,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        restore_model_weights(self.model, checkpoint["model_state_dict"], allow_partial=False)
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
        shutdown_dataloader_iterator(self._iterator)
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


def _uncompiled_state_dict(model: nn.Module) -> dict:
    """Return stable checkpoint keys even when torch.compile wraps the model."""
    original = getattr(model, "_orig_mod", model)
    return original.state_dict()


def shutdown_dataloader_iterator(iterator) -> None:
    """Shut down PyTorch multiprocessing workers behind an iterator, if any."""
    if bool(getattr(iterator, "_hexorl_workers_shutdown", False)):
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
            try:
                iterator._hexorl_workers_shutdown = True
            except Exception:
                pass
        except Exception as exc:
            logger.warning("DataLoader worker shutdown failed: %s", exc)


def shutdown_dataloader_workers(dataloader) -> None:
    """Release persistent workers cached on a PyTorch DataLoader."""
    iterator = getattr(dataloader, "_iterator", None)
    if iterator is not None:
        shutdown_dataloader_iterator(iterator)
        try:
            dataloader._iterator = None
        except Exception:
            pass
