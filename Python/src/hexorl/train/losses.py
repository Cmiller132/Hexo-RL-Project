"""Loss functions for Hexo multi-head models.

Includes the exact RGSC ranking loss (Equation 7 from arXiv 2602.20809v1)
and KataGo-style binned value loss with soft cross-entropy.
"""

import torch
import torch.nn.functional as F


def value_to_bins_torch(t: torch.Tensor, n_bins: int = 65) -> torch.Tensor:
    """Convert continuous values in [-1,1] to soft bin targets (PyTorch version).

    Uses linear interpolation between the two nearest bins —
    mirrors KataGo's value head target projection exactly.

    Args:
        t: (B,) tensor of continuous values in [-1, 1].
        n_bins: Number of bins (default 65).

    Returns:
        (B, n_bins) tensor of target probabilities summing to 1.
    """
    bin_width = 2.0 / (n_bins - 1)
    idx = (t + 1.0) / bin_width

    lo = idx.floor().long().clamp(0, n_bins - 1)
    hi = (lo + 1).clamp(0, n_bins - 1)

    w_hi = (idx - lo.float()).clamp(0.0, 1.0)
    w_lo = 1.0 - w_hi

    target = torch.zeros(t.shape[0], n_bins, device=t.device, dtype=torch.float32)
    target.scatter_add_(1, lo.unsqueeze(1), w_lo.unsqueeze(1))
    target.scatter_add_(1, hi.unsqueeze(1), w_hi.unsqueeze(1))
    return target


def scalar_to_bins_torch(
    t: torch.Tensor,
    *,
    n_bins: int = 65,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    """Convert continuous scalar targets to soft bins over a fixed range."""
    values = t.clamp(min_value, max_value)
    bin_width = (max_value - min_value) / (n_bins - 1)
    idx = (values - min_value) / bin_width
    lo = idx.floor().long().clamp(0, n_bins - 1)
    hi = (lo + 1).clamp(0, n_bins - 1)
    w_hi = (idx - lo.float()).clamp(0.0, 1.0)
    w_lo = 1.0 - w_hi
    target = torch.zeros(t.shape[0], n_bins, device=t.device, dtype=torch.float32)
    target.scatter_add_(1, lo.unsqueeze(1), w_lo.unsqueeze(1))
    target.scatter_add_(1, hi.unsqueeze(1), w_hi.unsqueeze(1))
    return target


def binned_value_loss(
    pred_logits: torch.Tensor,
    target_values: torch.Tensor,
    n_bins: int = 65,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """KataGo-style soft cross-entropy on interpolated bin targets.

    Args:
        pred_logits: (B, n_bins) raw logits from value_binned head.
        target_values: (B,) continuous targets in [-1, 1].
        n_bins: Number of value bins.

    Returns:
        Scalar loss.
    """
    logits = pred_logits.float()
    raw_values = target_values.to(device=logits.device, dtype=logits.dtype)
    finite = torch.isfinite(raw_values)
    values = torch.nan_to_num(raw_values, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    target_bins = value_to_bins_torch(values, n_bins=n_bins)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_bins * log_probs).sum(dim=-1)
    if weight is not None:
        weight = weight.to(device=loss.device, dtype=loss.dtype)
        weight = weight * finite.to(dtype=weight.dtype)
        valid = weight > 0
        if not torch.any(valid):
            return pred_logits.sum() * 0.0
        return (loss * weight).sum() / weight.sum().clamp(min=1e-6)
    if not torch.any(finite):
        return pred_logits.sum() * 0.0
    loss = loss[finite]
    return loss.mean()


def regret_rank_loss(
    scores: torch.Tensor,
    regrets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact RGSC ranking loss — Equation 7 from arXiv 2602.20809v1.

    L_rank = -log( Σ_s exp( log_softmax(φ(s)) + R(s) ) )

    The softmax is over the batch dimension, so this is a pairwise ranking
    loss: states with higher regret should receive higher scores φ(s).

    Args:
        scores: (B,) raw scalar scores φ(s) from regret_rank head.
        regrets: (B,) actual computed regret values R(s).

    Returns:
        Scalar loss.
    """
    values = regrets.to(device=scores.device, dtype=scores.dtype)
    if weight is not None:
        row_weight = weight.to(device=scores.device, dtype=scores.dtype)
        valid = row_weight > 0
        if not torch.any(valid):
            return scores.sum() * 0.0
        scores = scores[valid]
        values = values[valid]
    log_softmax_scores = F.log_softmax(scores, dim=0)
    combined = log_softmax_scores + values
    loss = -torch.logsumexp(combined, dim=0)
    return loss


def regret_value_loss(
    pred_logits: torch.Tensor,
    target_regret: torch.Tensor,
    n_bins: int = 65,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binned regret value loss.

    Args:
        pred_logits: (B, n_bins) from regret_value head.
        target_regret: (B,) actual regret R(s).
        n_bins: Number of value bins.

    Returns:
        Scalar loss.
    """
    logits = pred_logits.float()
    raw_regret = target_regret.to(device=logits.device, dtype=logits.dtype)
    finite = torch.isfinite(raw_regret)
    regret = torch.nan_to_num(raw_regret, nan=0.0, posinf=4.0, neginf=0.0)
    target_bins = scalar_to_bins_torch(regret, n_bins=n_bins, min_value=0.0, max_value=4.0)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_bins * log_probs).sum(dim=-1)
    if weight is not None:
        row_weight = weight.to(device=loss.device, dtype=loss.dtype)
        row_weight = row_weight * finite.to(dtype=row_weight.dtype)
        valid = row_weight > 0
        if not torch.any(valid):
            return pred_logits.sum() * 0.0
        row_weight = row_weight * valid.to(dtype=row_weight.dtype)
        return (loss * row_weight).sum() / row_weight.sum().clamp(min=1e-6)
    if not torch.any(finite):
        return pred_logits.sum() * 0.0
    loss = loss[finite]
    return loss.mean()


def policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy between policy logits and MCTS visit distribution (soft target).

    Args:
        pred_logits: (B, 1089) raw policy logits.
        target_probs: (B, 1089) float32 MCTS visit distribution.

    Returns:
        Scalar loss.
    """
    logits = pred_logits.float()
    target = target_probs.to(device=logits.device, dtype=logits.dtype)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target * log_probs).sum(dim=-1)
    if weight is not None:
        weight = weight.to(device=loss.device, dtype=loss.dtype)
        valid = weight > 0
        if not torch.any(valid):
            return pred_logits.sum() * 0.0
        return (loss * weight).sum() / weight.sum().clamp(min=1e-6)
    return loss.mean()


def sparse_policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    candidate_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy for candidate/action-keyed policy logits."""
    logits = pred_logits.float()
    mask = candidate_mask.to(device=logits.device, dtype=torch.bool)
    target = target_probs.to(device=logits.device, dtype=logits.dtype)
    target_mass = (target * mask.to(dtype=target.dtype)).sum(dim=-1)
    valid_rows = mask.any(dim=-1) & (target_mass > 0)
    if not torch.any(valid_rows):
        return pred_logits.sum() * 0.0

    logits = logits.masked_fill(~mask, -80.0)
    norm_target = torch.zeros_like(target)
    norm_target[valid_rows] = target[valid_rows] / target_mass[valid_rows].unsqueeze(-1).clamp(min=1e-6)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(norm_target * log_probs).sum(dim=-1)
    if weight is not None:
        w = weight.to(device=loss.device, dtype=loss.dtype)
        w = w * valid_rows.to(dtype=w.dtype)
        if not torch.any(w > 0):
            return pred_logits.sum() * 0.0
        return (loss * w).sum() / w.sum().clamp(min=1e-6)
    return loss[valid_rows].mean()


def pair_policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    pair_candidate_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy for auxiliary pair-action logits."""
    return sparse_policy_loss(pred_logits, target_probs, pair_candidate_mask, weight)


def graph_policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    row_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked cross-entropy for all-legal graph action rows."""
    return sparse_policy_loss(pred_logits, target_probs, row_mask, weight)


def opp_policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for opponent policy head (same as policy_loss).

    Args:
        pred_logits: (B, 1089) raw opponent policy logits.
        target_probs: (B, 1089) float32 MCTS visit distribution for opponent.

    Returns:
        Scalar loss.
    """
    target = target_probs.to(device=pred_logits.device, dtype=torch.float32)
    valid = target.sum(dim=-1) > 0
    if weight is None:
        row_weight = valid.to(dtype=torch.float32, device=pred_logits.device)
    else:
        row_weight = weight.to(device=pred_logits.device, dtype=torch.float32) * valid.to(dtype=torch.float32)
    if not torch.any(row_weight > 0):
        return pred_logits.sum() * 0.0
    return policy_loss(pred_logits, target, row_weight)


def axis_loss(
    pred_logits: torch.Tensor,
    target_axis: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-entropy on 3-class hex axis classification.

    Args:
        pred_logits: (B, 3) axis classification logits.
        target_axis: (B,) long tensor with class indices {0, 1, 2},
                     or None if axis labels are not available.

    Returns:
        Scalar loss (0.0 if target is None).
    """
    if target_axis is None:
        return torch.tensor(0.0, device=pred_logits.device)
    valid = target_axis >= 0
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred_logits.device)
    return F.cross_entropy(pred_logits[valid], target_axis[valid])


def axis_map_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE for dense six-plane axis-map regression targets."""
    return F.mse_loss(pred, target.to(device=pred.device, dtype=pred.dtype))


def moves_left_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on positive scalar moves-left target.

    Args:
        pred: (B, 1) or (B,) positive scalar.
        target: (B,) positive target.

    Returns:
        Scalar loss.
    """
    loss = (pred.squeeze(-1).float() - target.to(device=pred.device, dtype=torch.float32)).pow(2)
    if weight is not None:
        w = weight.to(device=loss.device, dtype=loss.dtype)
        if not torch.any(w > 0):
            return pred.sum() * 0.0
        return (loss * w).sum() / w.sum().clamp(min=1e-6)
    return loss.mean()


def tactical_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-label tactical state loss for win/block/cover/quiet labels."""
    labels = target.to(device=pred_logits.device, dtype=pred_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(pred_logits.float(), labels.float(), reduction="none").mean(dim=-1)
    if weight is not None:
        w = weight.to(device=loss.device, dtype=loss.dtype)
        if not torch.any(w > 0):
            return pred_logits.sum() * 0.0
        return (loss * w).sum() / w.sum().clamp(min=1e-6)
    return loss.mean()


def entropy_loss(policy_logits: torch.Tensor) -> torch.Tensor:
    """Entropy regularization — encourages higher policy entropy for exploration.

    Args:
        policy_logits: (B, N) policy head logits.

    Returns:
        Scalar loss (negative entropy mean — minimize to maximize entropy).
    """
    logits = policy_logits.float()
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return -entropy.mean()


def compute_losses(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
    n_bins: int = 65,
    loss_plan=None,
    row_tables=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all head losses and return (total_loss, per_head_losses).

    Args:
        predictions: Dict of head_name → tensor from model.forward().
        targets: Dict of target_name → tensor (e.g. 'policy', 'value', 'regret_rank').
        loss_weights: Dict of head_name → weight scalar.
        n_bins: Number of value bins (default 65).

    Returns:
        (total_loss, per_head_losses_dict) where per_head losses are already
        weighted.
    """
    if loss_plan is None:
        from hexorl.train.loss_plan import LossContractError

        raise LossContractError("compute_losses requires an explicit loss_plan")
    return loss_plan.compute(
        predictions,
        targets,
        n_bins=n_bins,
        row_tables=row_tables,
    )
