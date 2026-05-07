"""Candidate quarantine and retest lifecycle primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


QUARANTINE_STATES = (
    "pending",
    "runtime_probe",
    "running",
    "healthy",
    "promoted",
    "champion_candidate",
    "quarantined",
    "ready_for_retest",
    "retesting",
)

RUNTIME_QUARANTINE_CATEGORIES = (
    "graph construction",
    "inference latency",
    "pair row explosion",
    "MCTS simulation cost",
    "CPU queue starvation",
    "GPU underutilization",
    "memory or swap pressure",
    "checkpoint or model-load failure",
    "unknown",
)


@dataclass
class QuarantineEvidence:
    evidence_id: str
    kind: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CandidateQuarantineRecord:
    candidate_id: str
    state: str
    reason: str
    reason_category: str = "unknown"
    evidence: list[QuarantineEvidence] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    retest_attempts: int = 0
    current_config_hash: str = ""
    current_code_hash: str = ""
    notes: list[str] = field(default_factory=list)

    @classmethod
    def quarantined(
        cls,
        *,
        candidate_id: str,
        reason: str,
        reason_category: str,
        evidence: Mapping[str, Any] | None = None,
        config_hash: str = "",
        code_hash: str = "",
    ) -> "CandidateQuarantineRecord":
        if reason_category not in RUNTIME_QUARANTINE_CATEGORIES:
            reason_category = "unknown"
        record = cls(
            candidate_id=candidate_id,
            state="quarantined",
            reason=reason,
            reason_category=reason_category,
            current_config_hash=config_hash,
            current_code_hash=code_hash,
        )
        if evidence is not None:
            record.add_evidence("initial_runtime_failure", evidence)
        return record

    def add_evidence(self, kind: str, payload: Mapping[str, Any]) -> QuarantineEvidence:
        item = QuarantineEvidence(
            evidence_id=f"evidence_{len(self.evidence) + 1:04d}",
            kind=kind,
            payload=dict(payload),
        )
        self.evidence.append(item)
        self._touch()
        return item

    def mark_ready_for_retest(
        self,
        *,
        config_hash: str,
        code_hash: str,
        evidence: Mapping[str, Any] | None = None,
        note: str = "",
    ) -> None:
        self._require_state("quarantined")
        self.state = "ready_for_retest"
        self.current_config_hash = config_hash
        self.current_code_hash = code_hash
        if evidence is not None:
            self.add_evidence("ready_for_retest", evidence)
        if note:
            self.notes.append(note)
        self._touch()

    def begin_retest(self, *, evidence: Mapping[str, Any] | None = None) -> None:
        self._require_state("ready_for_retest")
        self.state = "retesting"
        self.retest_attempts += 1
        if evidence is not None:
            self.add_evidence("retest_started", evidence)
        self._touch()

    def finish_retest(
        self,
        *,
        healthy: bool,
        reason: str = "",
        reason_category: str = "unknown",
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        self._require_state("retesting")
        if evidence is not None:
            self.add_evidence("retest_finished", evidence)
        if healthy:
            self.state = "healthy"
            self.reason = ""
            self.reason_category = "unknown"
        else:
            self.state = "quarantined"
            self.reason = reason or self.reason
            self.reason_category = (
                reason_category if reason_category in RUNTIME_QUARANTINE_CATEGORIES else "unknown"
            )
        self._touch()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateQuarantineRecord":
        raw_evidence = payload.get("evidence", [])
        evidence = [
            item if isinstance(item, QuarantineEvidence) else QuarantineEvidence(**dict(item))
            for item in raw_evidence
        ]
        data = dict(payload)
        data["evidence"] = evidence
        return cls(**data)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CandidateQuarantineRecord":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _require_state(self, expected: str) -> None:
        if self.state != expected:
            raise ValueError(f"candidate {self.candidate_id} is {self.state}, expected {expected}")

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


def classify_runtime_bottleneck(results: list[Mapping[str, Any]]) -> str:
    """Best-effort compute-safety diagnosis from runtime probe rows."""

    if any(bool((row.get("memory") or {}).get("unsafe")) for row in results if isinstance(row.get("memory"), Mapping)):
        return "memory or swap pressure"
    errors = " ".join(str(row.get("error", "")).lower() for row in results)
    if "checkpoint" in errors or "model-load" in errors or "load" in errors:
        return "checkpoint or model-load failure"
    if "pair" in errors:
        return "pair row explosion"
    if "mcts" in errors or "simulation" in errors or "sims" in errors:
        return "MCTS simulation cost"
    if "queue" in errors or "worker" in errors or "starvation" in errors:
        return "CPU queue starvation"
    safe_rows = [
        row for row in results if row.get("ok") and not bool((row.get("memory") or {}).get("unsafe"))
    ]
    if safe_rows:
        gpu_utils = [
            float((row.get("gpu_after") or {}).get("gpu_util_pct", 0.0) or 0.0)
            for row in safe_rows
            if isinstance(row.get("gpu_after"), Mapping)
        ]
        if gpu_utils and max(gpu_utils) < 20.0:
            return "GPU underutilization"
        return "inference latency"
    return "unknown"
