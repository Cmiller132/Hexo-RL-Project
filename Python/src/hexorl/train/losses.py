"""Loss functions for HexNet multi-head model.

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


def binned_value_loss(
    pred_logits: torch.Tensor,
    target_values: torch.Tensor,
    n_bins: int = 65,
) -> torch.Tensor:
    """KataGo-style soft cross-entropy on interpolated bin targets.

    Args:
        pred_logits: (B, n_bins) raw logits from value_binned head.
        target_values: (B,) continuous targets in [-1, 1].
        n_bins: Number of value bins.

    Returns:
        Scalar loss.
    """
    target_bins = value_to_bins_torch(target_values, n_bins=n_bins)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(target_bins * log_probs).sum(dim=-1)
    return loss.mean()


def regret_rank_loss(
    scores: torch.Tensor,
    regrets: torch.Tensor,
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
    # Normalize regrets to [0, 1] so they are comparable to log-probabilities.
    r_min = regrets.min()
    r_max = regrets.max()
    r_range = (r_max - r_min).clamp(min=1e-6)
    regrets_norm = (regrets - r_min) / r_range

    log_softmax_scores = F.log_softmax(scores, dim=0)
    combined = log_softmax_scores + regrets_norm
    loss = -torch.logsumexp(combined, dim=0)
    return loss


def regret_value_loss(
    pred_logits: torch.Tensor,
    target_regret: torch.Tensor,
    n_bins: int = 65,
) -> torch.Tensor:
    """Binned regret value loss.

    Args:
        pred_logits: (B, n_bins) from regret_value head.
        target_regret: (B,) actual regret R(s).
        n_bins: Number of value bins.

    Returns:
        Scalar loss.
    """
    return binned_value_loss(pred_logits, target_regret, n_bins)


def policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between policy logits and MCTS visit distribution (soft target).

    Args:
        pred_logits: (B, 1089) raw policy logits.
        target_probs: (B, 1089) float32 MCTS visit distribution.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    return loss.mean()


def opp_policy_loss(
    pred_logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for opponent policy head (same as policy_loss).

    Args:
        pred_logits: (B, 1089) raw opponent policy logits.
        target_probs: (B, 1089) float32 MCTS visit distribution for opponent.

    Returns:
        Scalar loss.
    """
    return policy_loss(pred_logits, target_probs)


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
) -> torch.Tensor:
    """MSE on positive scalar moves-left target.

    Args:
        pred: (B, 1) or (B,) positive scalar.
        target: (B,) positive target.

    Returns:
        Scalar loss.
    """
    return F.mse_loss(pred.squeeze(-1), target)


def entropy_loss(policy_logits: torch.Tensor) -> torch.Tensor:
    """Entropy regularization — encourages higher policy entropy for exploration.

    Args:
        policy_logits: (B, N) policy head logits.

    Returns:
        Scalar loss (negative entropy mean — minimize to maximize entropy).
    """
    probs = F.softmax(policy_logits, dim=-1)
    log_probs = F.log_softmax(policy_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return -entropy.mean()


def compute_losses(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_weights: dict[str, float],
    n_bins: int = 65,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute all head losses and return (total_loss, per_head_losses).

    Args:
        predictions: Dict of head_name → tensor from HexNet.forward().
        targets: Dict of target_name → tensor (e.g. 'policy', 'value', 'regret_rank').
        loss_weights: Dict of head_name → weight scalar.
        n_bins: Number of value bins (default 65).

    Returns:
        (total_loss, per_head_losses_dict) where per_head losses are already
        weighted.
    """
    per_head: dict[str, torch.Tensor] = {}

    for head_name, pred in predictions.items():
        if head_name not in loss_weights:
            continue

        weight = loss_weights[head_name]
        required_targets = {
            "policy": "policy",
            "value": "value",
            "regret_rank": "regret_rank",
            "regret_value": "regret_value",
            "moves_left": "moves_left",
        }
        if head_name in required_targets and required_targets[head_name] not in targets:
            continue
        if head_name.startswith("lookahead_") and head_name not in targets:
            continue

        if head_name == "policy":
            loss = policy_loss(pred, targets["policy"])
        elif head_name == "opp_policy":
            target = targets.get("opp_policy", targets.get("policy"))
            if target is None:
                continue
            loss = opp_policy_loss(pred, target)
        elif head_name == "value":
            loss = binned_value_loss(pred, targets["value"], n_bins)
        elif head_name.startswith("lookahead_"):
            loss = binned_value_loss(pred, targets[head_name], n_bins)
        elif head_name == "regret_rank":
            loss = regret_rank_loss(pred.squeeze(-1), targets["regret_rank"])
        elif head_name == "regret_value":
            loss = regret_value_loss(pred, targets["regret_value"], n_bins)
        elif head_name == "axis":
            loss = axis_loss(pred, targets.get("axis"))
        elif head_name == "axis_delta_norm":
            target = targets.get("axis_delta_norm")
            if target is None:
                continue
            loss = axis_map_loss(pred, target)
        elif head_name == "moves_left":
            loss = moves_left_loss(pred, targets["moves_left"])
        else:
            continue

        per_head[head_name] = weight * loss

    if "policy" in predictions and "entropy" in loss_weights:
        ent = entropy_loss(predictions["policy"])
        per_head["entropy"] = loss_weights["entropy"] * ent

    if not per_head:
        raise ValueError(
            "No trainable losses were computed. Check model.heads, train.loss_weights, "
            f"and available targets. Heads={sorted(predictions.keys())}, "
            f"loss_weights={sorted(loss_weights.keys())}, targets={sorted(targets.keys())}"
        )

    total = sum(per_head.values())
    return total, per_head
