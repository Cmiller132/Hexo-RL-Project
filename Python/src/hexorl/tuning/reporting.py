"""Autotune reports with actionable failure localization."""

from __future__ import annotations

from typing import Any

ENGINE_FAILURE_CLASSES = (
    "replay/legal generation",
    "tactical status",
    "FFI decode",
    "invariant failure",
    "MCTS token lifecycle",
    "MCTS prior validation",
    "move-application failure",
)


def trial_lifecycle_report(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = {key: [] for key in ("selected", "rejected", "aborted", "retried", "promoted", "stopped")}
    for decision in decisions:
        action = decision.get("action")
        target = {
            "promote": "promoted",
            "reject": "rejected",
            "abort": "aborted",
            "retry": "retried",
            "early_stop": "stopped",
            "select": "selected",
        }.get(str(action), "stopped")
        buckets[target].append(
            {
                "trial_id": decision.get("trial_id"),
                "reason_code": decision.get("reason_code"),
                "score_components": decision.get("score_components", {}),
                "trace_ids": decision.get("trace_ids", []),
                "likely_owner": decision.get("likely_owner"),
            }
        )
    return buckets


def poor_learning_report(*, trace_ids: list[str], debug_bundles: list[str], failure_hints: dict[str, Any]) -> dict[str, Any]:
    rust_detail = failure_hints.get("rust_failure_class")
    if rust_detail is not None and rust_detail not in ENGINE_FAILURE_CLASSES:
        raise ValueError(f"unknown Rust failure class {rust_detail!r}; expected one of {ENGINE_FAILURE_CLASSES}")
    return {
        "trace_ids": trace_ids,
        "debug_bundles": debug_bundles,
        "likely_failure_classes": {
            "model": failure_hints.get("model"),
            "training_targets": failure_hints.get("training_targets"),
            "engine": rust_detail,
            "d6": failure_hints.get("d6"),
            "policy_mapping": failure_hints.get("policy_mapping"),
            "mcts": failure_hints.get("mcts"),
            "replay": failure_hints.get("replay"),
            "runtime_scheduling": failure_hints.get("runtime_scheduling"),
        },
        "next_debugging_action": "open the referenced debug bundles and compare hashes/schema/source fields before changing recipe knobs",
    }
