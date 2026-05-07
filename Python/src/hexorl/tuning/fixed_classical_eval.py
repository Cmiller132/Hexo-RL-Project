"""Fixed-classical evaluation evidence for Optuna scout candidates."""

from __future__ import annotations

import base64
import json
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from hexorl.config import Config
from hexorl.eval.arena import MatchResult
from hexorl.eval.scorecard import (
    ScorecardRecord,
    append_scorecard,
    build_classical_scorecard_record,
    load_classical_game_evidence,
    read_scorecards,
)

DEFAULT_FIXED_CLASSICAL_GAMES = 20
DEFAULT_FIXED_CLASSICAL_MAX_MOVES = 200


GameRunner = Callable[[int, int], MatchResult]


@dataclass(frozen=True)
class FixedClassicalEvalSettings:
    games_per_candidate: int = DEFAULT_FIXED_CLASSICAL_GAMES
    seed: int = 20260507
    eval_time_ms: int = 100
    eval_depth: int = 1
    temperature: float = 0.05
    top_p: float = 0.95
    max_moves: int = DEFAULT_FIXED_CLASSICAL_MAX_MOVES
    opponent_id: str = "fixed_strong"
    confidence_method: str = "normal_95"
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.games_per_candidate) <= 0:
            raise ValueError("games_per_candidate must be positive")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.eval_time_ms) <= 0:
            raise ValueError("eval_time_ms must be positive")
        if int(self.eval_depth) <= 0:
            raise ValueError("eval_depth must be positive")
        if int(self.max_moves) <= 0:
            raise ValueError("max_moves must be positive")
        if self.confidence_method not in {"normal_90", "normal_95", "normal_99"}:
            raise ValueError("confidence_method must be normal_90, normal_95, or normal_99")


@dataclass(frozen=True)
class CandidateFixedClassicalEvalResult:
    candidate_id: str
    status: str
    evidence_path: str = ""
    scorecard_path: str = ""
    checkpoint_path: str = ""
    epoch: int = 0
    existing_games: int = 0
    appended_games: int = 0
    appended_scorecards: int = 0
    scalar_score: float | None = None
    hard_pass: bool | None = None
    reason: str = ""


@dataclass(frozen=True)
class FixedClassicalEvalSummary:
    run_dir: str
    settings: dict[str, Any]
    candidates: tuple[CandidateFixedClassicalEvalResult, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def appended_games(self) -> int:
        return sum(item.appended_games for item in self.candidates)

    @property
    def appended_scorecards(self) -> int:
        return sum(item.appended_scorecards for item in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "run_dir": self.run_dir,
            "settings": self.settings,
            "appended_games": self.appended_games,
            "appended_scorecards": self.appended_scorecards,
            "candidates": [asdict(item) for item in self.candidates],
        }


def evaluate_run_fixed_classical(
    run_dir: Path | str,
    *,
    settings: FixedClassicalEvalSettings | None = None,
    candidate_ids: Iterable[str] = (),
    summary_path: Path | str | None = None,
) -> FixedClassicalEvalSummary:
    settings = settings or FixedClassicalEvalSettings()
    root = Path(run_dir)
    requested = {str(candidate_id) for candidate_id in candidate_ids}
    candidates_root = root / "candidates"
    if not candidates_root.exists():
        raise FileNotFoundError(f"run candidates directory not found: {candidates_root}")

    results: list[CandidateFixedClassicalEvalResult] = []
    for candidate_dir in sorted(path for path in candidates_root.iterdir() if path.is_dir()):
        if requested and candidate_dir.name not in requested:
            continue
        results.append(
            evaluate_candidate_fixed_classical(
                candidate_dir,
                run_dir=root,
                settings=settings,
            )
        )

    summary = FixedClassicalEvalSummary(
        run_dir=str(root),
        settings=asdict(settings),
        candidates=tuple(results),
    )
    if summary_path is not None:
        output = Path(summary_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def evaluate_candidate_fixed_classical(
    candidate_dir: Path | str,
    *,
    run_dir: Path | str,
    settings: FixedClassicalEvalSettings,
    game_runner: GameRunner | None = None,
) -> CandidateFixedClassicalEvalResult:
    candidate_path = Path(candidate_dir)
    candidate_id = candidate_path.name
    scorecard_path = candidate_path / "scorecards.jsonl"
    if not scorecard_path.exists() or scorecard_path.stat().st_size == 0:
        return CandidateFixedClassicalEvalResult(
            candidate_id=candidate_id,
            status="skipped",
            scorecard_path=str(scorecard_path),
            reason="missing_scorecards",
        )

    latest = _latest_epoch_floor_scorecard(scorecard_path)
    if latest is None:
        return CandidateFixedClassicalEvalResult(
            candidate_id=candidate_id,
            status="skipped",
            scorecard_path=str(scorecard_path),
            reason="below_epoch_floor_or_not_completed",
        )
    if latest.status in {"quarantined", "failed", "hard_failed"} or not latest.hard_pass:
        return CandidateFixedClassicalEvalResult(
            candidate_id=candidate_id,
            status="skipped",
            scorecard_path=str(scorecard_path),
            epoch=max(latest.completed_epochs, latest.epoch),
            reason="not_hard_gate_eligible",
        )

    epoch = max(latest.completed_epochs, latest.epoch)
    checkpoint_path = _checkpoint_path_for_record(latest, candidate_path, run_dir=Path(run_dir))
    evidence_path = candidate_path / f"fixed_classical_epoch_{epoch:04d}_games.jsonl"
    existing_rows = _read_jsonl(evidence_path)
    existing_games = len(existing_rows)
    appended_games = 0
    if existing_games < settings.games_per_candidate:
        runner = game_runner or _make_arena_game_runner(
            candidate_path=candidate_path,
            checkpoint_path=checkpoint_path,
            settings=settings,
        )
        for game_index in range(existing_games, settings.games_per_candidate):
            seed = int(settings.seed) + game_index
            result = runner(game_index, seed)
            evidence = _evidence_row(
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                result=result,
                game_index=game_index,
                seed=seed,
                settings=settings,
                evidence_path=evidence_path,
            )
            _append_jsonl(evidence_path, evidence)
            appended_games += 1

    if _has_scorecard_for_evidence(scorecard_path, evidence_path, settings.games_per_candidate):
        lcb = _classical_lcb_or_none(evidence_path)
        return CandidateFixedClassicalEvalResult(
            candidate_id=candidate_id,
            status="ready",
            evidence_path=str(evidence_path),
            scorecard_path=str(scorecard_path),
            checkpoint_path=str(checkpoint_path),
            epoch=epoch,
            existing_games=existing_games,
            appended_games=appended_games,
            appended_scorecards=0,
            scalar_score=lcb,
            hard_pass=True,
            reason="existing_scorecard",
        )

    component_metrics = _component_metrics_from_evidence(
        evidence_path,
        latest.component_metrics,
        max_moves=settings.max_moves,
    )
    hard_gates = _hard_gates_from_evidence(latest.hard_gates, component_metrics)
    scorecard = build_classical_scorecard_record(
        candidate_id=candidate_id,
        evidence_path=evidence_path,
        component_metrics=component_metrics,
        hard_gates=hard_gates,
        study_id=latest.study_id,
        trial_id=latest.trial_id,
        config_hash=latest.config_hash,
        checkpoint_lineage={
            **dict(latest.checkpoint_lineage),
            "checkpoint_path": str(checkpoint_path),
        },
        epoch=epoch,
        completed_epochs=epoch,
        status="healthy" if bool(hard_gates.get("hard_pass", False)) else "hard_failed",
        metadata={
            **dict(latest.metadata),
            "fixed_classical_eval": {
                "evidence_path": str(evidence_path),
                "games_requested": int(settings.games_per_candidate),
                "settings": asdict(settings),
                "source_scorecard_epoch": int(latest.epoch),
                "source_scorecard_created_at": latest.created_at,
            },
        },
    )
    append_scorecard(scorecard_path, scorecard)
    return CandidateFixedClassicalEvalResult(
        candidate_id=candidate_id,
        status="ready" if scorecard.hard_pass else "hard_failed",
        evidence_path=str(evidence_path),
        scorecard_path=str(scorecard_path),
        checkpoint_path=str(checkpoint_path),
        epoch=epoch,
        existing_games=existing_games,
        appended_games=appended_games,
        appended_scorecards=1,
        scalar_score=float(scorecard.scalar_score),
        hard_pass=scorecard.hard_pass,
    )


def _latest_epoch_floor_scorecard(scorecard_path: Path, min_epoch: int = 12) -> ScorecardRecord | None:
    eligible = [
        record
        for record in read_scorecards(scorecard_path)
        if max(record.completed_epochs, record.epoch) >= int(min_epoch)
        and record.status not in {"quarantined", "failed", "hard_failed"}
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda record: (max(record.completed_epochs, record.epoch), record.created_at))


def _checkpoint_path_for_record(record: ScorecardRecord, candidate_dir: Path, *, run_dir: Path) -> Path:
    checkpoint_value = record.checkpoint_lineage.get("checkpoint_path")
    if not checkpoint_value:
        checkpoint_value = record.metadata.get("extra_fields", {}).get("checkpoint_path")
    if not checkpoint_value:
        epoch = max(record.completed_epochs, record.epoch)
        checkpoint_value = str(candidate_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")
    checkpoint_path = Path(str(checkpoint_value))
    if checkpoint_path.is_absolute():
        return checkpoint_path
    root_candidate = Path.cwd() / checkpoint_path
    if root_candidate.exists():
        return root_candidate
    return run_dir.parent.parent / checkpoint_path


def _make_arena_game_runner(
    *,
    candidate_path: Path,
    checkpoint_path: Path,
    settings: FixedClassicalEvalSettings,
) -> GameRunner:
    import torch
    from hexorl.eval.arena import load_checkpoint_model, model_move_fn, run_arena
    from hexorl.eval.classical import classical_opponent_fn

    config = Config.model_validate(json.loads((candidate_path / "full_config.json").read_text(encoding="utf-8")))
    if settings.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(settings.device)
    model = load_checkpoint_model(checkpoint_path, config, device=device, allow_partial=False)
    model_player = model_move_fn(
        model,
        device=device,
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=settings.seed,
        near_radius=config.selfplay.near_radius,
        constrain_threats=config.selfplay.constrain_threats,
    )
    classical = classical_opponent_fn(time_ms=settings.eval_time_ms, max_depth=settings.eval_depth)

    def _run_game(_game_index: int, _seed: int) -> MatchResult:
        stats = run_arena(model_player, classical, num_games=1)
        if not stats.results:
            raise RuntimeError("fixed-classical arena returned no game result")
        return stats.results[0]

    return _run_game


def _evidence_row(
    *,
    candidate_id: str,
    checkpoint_path: Path,
    result: MatchResult,
    game_index: int,
    seed: int,
    settings: FixedClassicalEvalSettings,
    evidence_path: Path,
) -> dict[str, Any]:
    outcome = _outcome_from_result(result)
    penalty = _illegal_or_crash_penalty(result)
    row = {
        "candidate_id": candidate_id,
        "checkpoint_id": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "confidence_method": settings.confidence_method,
        "evidence_path": str(evidence_path),
        "game_index": int(game_index),
        "illegal_or_crash_penalty": penalty,
        "max_moves": int(settings.max_moves),
        "moves": int(result.moves),
        "opening_is_black": bool(result.opening_is_black),
        "opponent_id": settings.opponent_id,
        "outcome": outcome,
        "reason": str(result.reason),
        "seed": int(seed),
        "time_ms": float(result.time_ms),
        "winner": int(result.winner),
    }
    if result.move_history:
        row["final_history_b64"] = _move_history_b64(result.move_history)
        row["move_history"] = [
            {"player": int(player), "q": int(q), "r": int(r)}
            for player, q, r in result.move_history
        ]
    return row


def _move_history_b64(moves: Iterable[tuple[int, int, int]]) -> str:
    out = bytearray()
    for player, q, r in moves:
        out.extend(struct.pack("<iii", int(player), int(q), int(r)))
    return base64.b64encode(bytes(out)).decode("ascii")


def _outcome_from_result(result: MatchResult) -> str:
    if result.winner == 0:
        return "model_win"
    if result.winner == 1:
        return "classical_win"
    if result.reason == "max_moves":
        return "survived"
    return "draw"


def _illegal_or_crash_penalty(result: MatchResult) -> float:
    if result.reason.startswith(("crash:", "illegal:", "no_move")) and result.winner != 0:
        return 1.0
    return 0.0


def _component_metrics_from_evidence(
    evidence_path: Path,
    base_metrics: Mapping[str, float],
    *,
    max_moves: int,
) -> dict[str, float]:
    rows = _read_jsonl(evidence_path)
    games = max(len(rows), 1)
    model_wins = sum(1 for row in rows if str(row.get("outcome")) == "model_win")
    draws = sum(1 for row in rows if str(row.get("outcome")) in {"draw", "survived"})
    penalty_total = sum(float(row.get("illegal_or_crash_penalty", 0.0) or 0.0) for row in rows)
    avg_moves = sum(float(row.get("moves", 0.0) or 0.0) for row in rows) / games
    metrics = dict(base_metrics)
    metrics.update(
        {
            "classical_win_rate": float(model_wins) / games,
            "classical_draw_rate": float(draws) / games,
            "classical_avg_moves": avg_moves,
            "classical_survival_mean_moves_frac": avg_moves / max(float(max_moves), 1.0),
            "illegal_or_crash_rate": penalty_total / games,
        }
    )
    return {str(key): float(value) for key, value in metrics.items()}


def _hard_gates_from_evidence(base_gates: Mapping[str, Any], metrics: Mapping[str, float]) -> dict[str, Any]:
    gates = dict(base_gates or {})
    failures = list(gates.get("failures", []) or [])
    if float(metrics.get("illegal_or_crash_rate", 0.0) or 0.0) > 0.0:
        failures.append("illegal_or_crash_rate")
    gates["failures"] = sorted(set(str(item) for item in failures))
    gates["hard_pass"] = not gates["failures"]
    gates["fixed_classical_eval_pass"] = not gates["failures"]
    return gates


def _has_scorecard_for_evidence(scorecard_path: Path, evidence_path: Path, games: int) -> bool:
    target = str(evidence_path)
    for record in read_scorecards(scorecard_path):
        fixed_eval = record.metadata.get("fixed_classical_eval")
        if not isinstance(fixed_eval, Mapping):
            continue
        if str(fixed_eval.get("evidence_path", "")) != target:
            continue
        if int(fixed_eval.get("games_requested", 0) or 0) >= int(games):
            return True
    return False


def _classical_lcb_or_none(evidence_path: Path) -> float | None:
    try:
        from hexorl.eval.scorecard import classical_survival_lcb

        return float(classical_survival_lcb(load_classical_game_evidence(evidence_path)).score)
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(payload)
    return rows


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    for attempt in range(6):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.25 * (attempt + 1))
