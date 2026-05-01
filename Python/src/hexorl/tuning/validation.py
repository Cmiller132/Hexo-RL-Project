"""Recipe and runtime dry-run validation."""

from __future__ import annotations

from typing import Any

from hexorl.contracts.pair_strategy import PAIR_STRATEGY_REGISTRY
from hexorl.models.factory import get_model_registry
from hexorl.models.specs import GLOBAL_MODEL_KINDS
from hexorl.tuning.recipes import ModelRecipe
from hexorl.tuning.runtime_sweep import HostProfile, RuntimeSpec


def dry_run_validate_recipe(recipe: ModelRecipe, runtime: RuntimeSpec, host: HostProfile) -> tuple[dict[str, Any], ...]:
    descriptor = get_model_registry().resolve(recipe.model_family)
    results = [
        _ok("family_capability", "registry", f"{descriptor.name} is registered"),
        _ok("head_loss", "train adapter", "loss plan is descriptor-owned"),
        _ok("input_output_action_contract", "inference adapter", "contracts are descriptor-owned"),
        _ok("inference_protocol", "inference protocol", f"v{recipe.inference_protocol_version}"),
        _ok("checkpoint_manifest", "checkpoint manifest", "manifest provider is registered"),
    ]
    if recipe.model_family in GLOBAL_MODEL_KINDS and recipe.graph_layers <= 0:
        results.append(_fail("family_capability", "model registry", "global graph families require graph_layers"))
    if recipe.model_family not in GLOBAL_MODEL_KINDS and "policy_place" in recipe.heads:
        results.append(_fail("head_loss", "train adapter", "crop families cannot use global policy_place head"))
    try:
        PAIR_STRATEGY_REGISTRY.resolve(recipe.pair_strategy).validate_config(
            max_pairs=recipe.pair_strategy_max_pairs,
            pair_prior_mix=1.0,
        )
    except ValueError as exc:
        results.append(_fail("pair_strategy", "pair strategy", str(exc)))
    else:
        results.append(_ok("pair_strategy", "pair strategy", recipe.pair_strategy))
    for failure in runtime.validate(host):
        results.append({"ok": False, "check": failure["kind"], "owner": failure["owner"], "message": failure["message"]})
    if not runtime.validate(host):
        results.append(_ok("runtime_budget", "runtime_sweep", "runtime spec fits host profile"))
    return tuple(results)


def _ok(check: str, owner: str, message: str) -> dict[str, Any]:
    return {"ok": True, "check": check, "owner": owner, "message": message}


def _fail(check: str, owner: str, message: str) -> dict[str, Any]:
    return {"ok": False, "check": check, "owner": owner, "message": message}
