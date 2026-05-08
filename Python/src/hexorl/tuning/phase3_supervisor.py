"""Autonomous Phase 3 round supervisor.

The bounded Phase 3 runner remains the authority for individual Optuna trials.
This module only advances the per-study target trial count, refreshes
scorecard-backed artifacts, and records durable supervisor evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from hexorl.tuning.champion import (
    build_champion_selection_report_from_scorecard_files,
    write_champion_selection_report,
)
from hexorl.tuning.fixed_classical_eval import DEFAULT_FIXED_CLASSICAL_MAX_MOVES
from hexorl.tuning.phase3_runner import (
    Phase3StudySpec,
    load_phase3_study_specs,
    phase3_scorecard_paths_for_run,
    rerank_phase3_trials,
)
from hexorl.tuning.review import write_phase2_promotion_report


DEFAULT_MAX_TRIALS_PER_STUDY = 256
DEFAULT_FIXED_CLASSICAL_GAMES = 64
DEFAULT_PHASE3_TRIAL_EPOCHS = 4
STOP_MARKER_NAME = "phase3_autosupervisor.stop"
LOCK_NAME = "phase3_autosupervisor.lock.json"
EVENTS_NAME = "phase3_autosupervisor_events.jsonl"


@dataclass(frozen=True)
class ActiveProcess:
    pid: int
    command_line: str


@dataclass(frozen=True)
class Phase3StudyTerminalCounts:
    study_name: str
    storage: str
    promoted_candidate_id: str
    trials_total: int
    trials_complete: int
    trials_pruned: int
    trials_failed: int
    trials_terminal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase3SupervisorRoundSummary:
    target_trials_per_study: int
    attempts: int
    returncode: int
    status: str
    started_at: str
    completed_at: str
    stdout_log: str
    stderr_log: str
    runner_summary_path: str
    counts: tuple[Phase3StudyTerminalCounts, ...]
    artifacts: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counts"] = [item.to_dict() for item in self.counts]
        return payload


@dataclass(frozen=True)
class Phase3SupervisorSummary:
    run_dir: str
    spec_path: str
    max_trials_per_study: int
    status: str
    stopped_reason: str
    rounds: tuple[Phase3SupervisorRoundSummary, ...]
    latest_counts: tuple[Phase3StudyTerminalCounts, ...]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "run_dir": self.run_dir,
            "spec_path": self.spec_path,
            "max_trials_per_study": self.max_trials_per_study,
            "status": self.status,
            "stopped_reason": self.stopped_reason,
            "rounds": [item.to_dict() for item in self.rounds],
            "latest_counts": [item.to_dict() for item in self.latest_counts],
        }


class Phase3RoundRunner(Protocol):
    def __call__(
        self,
        *,
        target_trials: int,
        summary_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        """Run a bounded Phase 3 target and return a process-style status code."""


class Phase3AutonomousSupervisor:
    def __init__(
        self,
        *,
        run_dir: Path | str,
        spec_path: Path | str | None = None,
        start_target: int | str = "auto",
        max_trials_per_study: int = DEFAULT_MAX_TRIALS_PER_STUDY,
        trial_epochs: int = DEFAULT_PHASE3_TRIAL_EPOCHS,
        fixed_classical_games: int = DEFAULT_FIXED_CLASSICAL_GAMES,
        fixed_classical_seed: int = 20260507,
        eval_time_ms: int = 100,
        eval_depth: int = 1,
        temperature: float = 0.05,
        top_p: float = 0.95,
        max_moves: int = DEFAULT_FIXED_CLASSICAL_MAX_MOVES,
        summary_path: Path | str | None = None,
        dry_run: bool = False,
        max_rounds: int | None = None,
        retry_limit: int = 1,
        mirror_dashboards: bool = True,
        process_finder: Callable[[], list[ActiveProcess]] | None = None,
        round_runner: Phase3RoundRunner | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.spec_path = Path(spec_path) if spec_path is not None else self.run_dir / "phase2_review" / "phase3_study_specs.json"
        self.start_target = start_target
        self.max_trials_per_study = int(max_trials_per_study)
        self.trial_epochs = int(trial_epochs)
        self.fixed_classical_games = int(fixed_classical_games)
        self.fixed_classical_seed = int(fixed_classical_seed)
        self.eval_time_ms = int(eval_time_ms)
        self.eval_depth = int(eval_depth)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_moves = int(max_moves)
        self.summary_path = Path(summary_path) if summary_path is not None else self.run_dir / "phase3_autosupervisor_summary.json"
        self.dry_run = bool(dry_run)
        self.max_rounds = None if max_rounds is None else int(max_rounds)
        self.retry_limit = int(retry_limit)
        self.mirror_dashboards = bool(mirror_dashboards)
        self.process_finder = process_finder or (lambda: find_active_phase3_processes(self.run_dir))
        self.round_runner = round_runner or self._run_round_subprocess
        self.sleep_seconds = float(sleep_seconds)
        if self.max_trials_per_study <= 0:
            raise ValueError("max_trials_per_study must be positive")
        if self.retry_limit < 0:
            raise ValueError("retry_limit cannot be negative")

    def run(self) -> Phase3SupervisorSummary:
        specs = load_phase3_study_specs(self.spec_path)
        if not specs:
            raise ValueError("phase3 study specs are empty")
        self._refuse_if_external_phase3_process()
        rounds: list[Phase3SupervisorRoundSummary] = []
        status = "complete"
        stopped_reason = "max_trials_per_study_reached"
        with _SupervisorLock(self.run_dir, self.process_finder):
            while True:
                if self.stop_marker_path.exists():
                    status = "paused"
                    stopped_reason = "stop_marker_present"
                    self._event("stopped", {"reason": stopped_reason})
                    break
                counts = phase3_terminal_counts(specs)
                target = next_round_target(
                    counts,
                    start_target=self.start_target if not rounds else "auto",
                    max_trials_per_study=self.max_trials_per_study,
                )
                if target is None:
                    status = "complete"
                    stopped_reason = "max_trials_per_study_reached"
                    self._event("stopped", {"reason": stopped_reason})
                    break
                if self.max_rounds is not None and len(rounds) >= self.max_rounds:
                    status = "paused"
                    stopped_reason = "max_rounds_reached"
                    self._event("stopped", {"reason": stopped_reason, "max_rounds": self.max_rounds})
                    break
                round_summary = self._run_target_round(specs, target)
                rounds.append(round_summary)
                self._write_summary(status="running", stopped_reason="", rounds=tuple(rounds), latest_counts=round_summary.counts)
                if round_summary.status != "complete":
                    status = "failed"
                    stopped_reason = round_summary.reason or "round_failed"
                    break
                if self.sleep_seconds > 0:
                    time.sleep(self.sleep_seconds)
        latest_counts = phase3_terminal_counts(specs)
        summary = self._write_summary(
            status=status,
            stopped_reason=stopped_reason,
            rounds=tuple(rounds),
            latest_counts=latest_counts,
        )
        return summary

    @property
    def stop_marker_path(self) -> Path:
        return self.run_dir / STOP_MARKER_NAME

    def _run_target_round(
        self,
        specs: tuple[Phase3StudySpec, ...],
        target: int,
    ) -> Phase3SupervisorRoundSummary:
        stdout_path = self.run_dir / f"phase3_optuna_tpe_auto_round_{target:04d}.stdout.log"
        stderr_path = self.run_dir / f"phase3_optuna_tpe_auto_round_{target:04d}.stderr.log"
        runner_summary_path = self.run_dir / f"phase3_runner_summary_auto_round_{target:04d}.json"
        attempts = 0
        last_returncode = 1
        reason = ""
        started_at = datetime.now(timezone.utc).isoformat()
        before_counts = phase3_terminal_counts(specs)
        self._event("round_start", {"target_trials_per_study": target})
        for attempt in range(self.retry_limit + 1):
            attempts = attempt + 1
            last_returncode = int(
                self.round_runner(
                    target_trials=target,
                    summary_path=runner_summary_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
            counts = phase3_terminal_counts(specs)
            missing = [item for item in counts if item.trials_terminal < target]
            failed_delta = _failed_trial_delta(before_counts, counts)
            if failed_delta:
                reason = "phase3_trial_failures:" + ",".join(
                    f"{candidate_id}+{delta}" for candidate_id, delta in sorted(failed_delta.items())
                )
                self._event(
                    "round_failed",
                    {
                        "target_trials_per_study": target,
                        "attempt": attempts,
                        "returncode": last_returncode,
                        "reason": reason,
                        "counts": [item.to_dict() for item in counts],
                    },
                )
                break
            if last_returncode == 0 and not missing:
                artifacts = refresh_phase3_supervisor_artifacts(
                    self.run_dir,
                    specs=specs,
                    target=target,
                    mirror_dashboards=self.mirror_dashboards,
                    reproduction_command=self.reproduction_command,
                )
                completed_at = datetime.now(timezone.utc).isoformat()
                payload = {
                    "target_trials_per_study": target,
                    "attempts": attempts,
                    "counts": [item.to_dict() for item in counts],
                    "artifacts": artifacts,
                }
                self._event("round_complete", payload)
                return Phase3SupervisorRoundSummary(
                    target_trials_per_study=target,
                    attempts=attempts,
                    returncode=last_returncode,
                    status="complete",
                    started_at=started_at,
                    completed_at=completed_at,
                    stdout_log=str(stdout_path),
                    stderr_log=str(stderr_path),
                    runner_summary_path=str(runner_summary_path),
                    counts=counts,
                    artifacts=artifacts,
                )
            reason = (
                f"runner_returncode_{last_returncode}"
                if last_returncode != 0
                else "terminal_count_below_target:" + ",".join(item.promoted_candidate_id for item in missing)
            )
            self._event(
                "round_retry" if attempt < self.retry_limit else "round_failed",
                {
                    "target_trials_per_study": target,
                    "attempt": attempts,
                    "returncode": last_returncode,
                    "reason": reason,
                    "counts": [item.to_dict() for item in counts],
                },
            )
        return Phase3SupervisorRoundSummary(
            target_trials_per_study=target,
            attempts=attempts,
            returncode=last_returncode,
            status="failed",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            runner_summary_path=str(runner_summary_path),
            counts=phase3_terminal_counts(specs),
            reason=reason,
        )

    @property
    def reproduction_command(self) -> str:
        return (
            "python scripts/run_phase3_autonomous_supervisor.py "
            f"--run-dir {self.run_dir} --start-target auto "
            f"--max-trials-per-study {self.max_trials_per_study}"
        )

    def _run_round_subprocess(
        self,
        *,
        target_trials: int,
        summary_path: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> int:
        args = [
            sys.executable,
            "-u",
            "scripts/run_phase3_optuna_tpe.py",
            "--run-dir",
            str(self.run_dir),
            "--spec-path",
            str(self.spec_path),
            "--n-trials-per-study",
            str(target_trials),
            "--trial-epochs",
            str(self.trial_epochs),
            "--fixed-classical-games",
            str(self.fixed_classical_games),
            "--fixed-classical-seed",
            str(self.fixed_classical_seed),
            "--eval-time-ms",
            str(self.eval_time_ms),
            "--eval-depth",
            str(self.eval_depth),
            "--temperature",
            str(self.temperature),
            "--top-p",
            str(self.top_p),
            "--max-moves",
            str(self.max_moves),
            "--summary",
            str(summary_path),
        ]
        if self.dry_run:
            args.append("--dry-run")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            proc = subprocess.run(args, cwd=Path.cwd(), stdout=stdout, stderr=stderr, check=False)
        return int(proc.returncode)

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        _append_jsonl(
            self.run_dir / EVENTS_NAME,
            {
                "event": event,
                "time": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_dir.name,
                **payload,
            },
        )

    def _write_summary(
        self,
        *,
        status: str,
        stopped_reason: str,
        rounds: tuple[Phase3SupervisorRoundSummary, ...],
        latest_counts: tuple[Phase3StudyTerminalCounts, ...],
    ) -> Phase3SupervisorSummary:
        summary = Phase3SupervisorSummary(
            run_dir=str(self.run_dir),
            spec_path=str(self.spec_path),
            max_trials_per_study=self.max_trials_per_study,
            status=status,
            stopped_reason=stopped_reason,
            rounds=rounds,
            latest_counts=latest_counts,
        )
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    def _refuse_if_external_phase3_process(self) -> None:
        active = [proc for proc in self.process_finder() if proc.pid != os.getpid()]
        if active:
            details = "; ".join(f"{proc.pid}:{proc.command_line}" for proc in active[:3])
            raise RuntimeError(f"phase3_runner_already_active:{details}")


def next_round_target(
    counts: tuple[Phase3StudyTerminalCounts, ...],
    *,
    start_target: int | str = "auto",
    max_trials_per_study: int = DEFAULT_MAX_TRIALS_PER_STUDY,
) -> int | None:
    if not counts:
        raise ValueError("phase3 study counts are empty")
    if start_target != "auto":
        target = int(start_target)
        if target <= 0:
            raise ValueError("start target must be positive")
        return None if min(item.trials_terminal for item in counts) >= max_trials_per_study else min(target, max_trials_per_study)
    current_min = min(item.trials_terminal for item in counts)
    if current_min >= int(max_trials_per_study):
        return None
    return current_min + 1


def phase3_terminal_counts(specs: tuple[Phase3StudySpec, ...]) -> tuple[Phase3StudyTerminalCounts, ...]:
    optuna = _import_optuna()
    rows: list[Phase3StudyTerminalCounts] = []
    for spec in specs:
        try:
            study = optuna.load_study(study_name=spec.study_name, storage=spec.storage)
            trials = study.get_trials(deepcopy=False)
        except Exception:
            trials = []
        complete = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE)
        pruned = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.PRUNED)
        failed = sum(1 for trial in trials if trial.state == optuna.trial.TrialState.FAIL)
        rows.append(
            Phase3StudyTerminalCounts(
                study_name=spec.study_name,
                storage=spec.storage,
                promoted_candidate_id=str(spec.metadata.get("promoted_candidate_id", "")),
                trials_total=len(trials),
                trials_complete=complete,
                trials_pruned=pruned,
                trials_failed=failed,
                trials_terminal=complete + pruned + failed,
            )
        )
    return tuple(rows)


def _failed_trial_delta(
    before: tuple[Phase3StudyTerminalCounts, ...],
    after: tuple[Phase3StudyTerminalCounts, ...],
) -> dict[str, int]:
    previous = {item.promoted_candidate_id: item.trials_failed for item in before}
    delta: dict[str, int] = {}
    for item in after:
        added = item.trials_failed - int(previous.get(item.promoted_candidate_id, 0))
        if added > 0:
            delta[item.promoted_candidate_id] = added
    return delta


def refresh_phase3_supervisor_artifacts(
    run_dir: Path | str,
    *,
    specs: tuple[Phase3StudySpec, ...],
    target: int,
    mirror_dashboards: bool,
    reproduction_command: str,
) -> dict[str, str]:
    run_dir = Path(run_dir)
    artifacts: dict[str, str] = {}
    review_dir = run_dir / "phase3_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    ranking = rerank_phase3_trials(run_dir)
    ranking_path = review_dir / "phase3_trial_ranking_report.json"
    ranking_round_path = review_dir / f"phase3_trial_ranking_report_auto_round_{target:04d}.json"
    write_phase2_promotion_report(ranking_path, ranking)
    write_phase2_promotion_report(ranking_round_path, ranking)
    artifacts["phase3_ranking_report"] = str(ranking_path)
    artifacts["phase3_ranking_report_round"] = str(ranking_round_path)

    scorecard_paths = _champion_scorecard_paths(run_dir, specs)
    if scorecard_paths:
        champion = build_champion_selection_report_from_scorecard_files(
            scorecard_paths,
            reproduction_command=reproduction_command,
            min_completed_epochs=12,
        )
        champion_path = run_dir / "champion_selection_report_phase3.json"
        write_champion_selection_report(champion_path, champion)
        artifacts["champion_selection_report_phase3"] = str(champion_path)

    if mirror_dashboards:
        artifacts.update(_mirror_dashboard_artifacts(run_dir))
    return artifacts


def _champion_scorecard_paths(run_dir: Path, specs: tuple[Phase3StudySpec, ...]) -> list[Path]:
    promoted_ids = {str(spec.metadata.get("promoted_candidate_id", "")) for spec in specs}
    paths: list[Path] = []
    for promoted_id in sorted(promoted_ids):
        path = run_dir / "candidates" / promoted_id / "scorecards.jsonl"
        if path.exists() and path.stat().st_size > 0:
            paths.append(path)
    paths.extend(phase3_scorecard_paths_for_run(run_dir))
    return paths


def _mirror_dashboard_artifacts(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    try:
        from scripts.mirror_phase3_normal_dashboard_suite import mirror_once as mirror_normal

        suite_dir = run_dir / "phase_normal_dashboard_suite"
        summary = run_dir / "phase_normal_dashboard_suite_mirror_summary.json"
        mirror_normal(run_dir, suite_dir, summary)
        artifacts["normal_dashboard_suite_mirror_summary"] = str(summary)
    except Exception as exc:
        artifacts["normal_dashboard_suite_mirror_error"] = f"{type(exc).__name__}:{exc}"
    try:
        from scripts.mirror_phase3_optuna_dashboard import mirror_once as mirror_optuna

        output = run_dir / "phase3_studies" / "phase3_dashboard_combined.sqlite3"
        summary = run_dir / "phase3_optuna_dashboard_mirror_summary.json"
        mirror_optuna(run_dir, output, summary)
        artifacts["optuna_dashboard_mirror_summary"] = str(summary)
    except Exception as exc:
        artifacts["optuna_dashboard_mirror_error"] = f"{type(exc).__name__}:{exc}"
    return artifacts


def find_active_phase3_processes(run_dir: Path | str) -> list[ActiveProcess]:
    needle = str(Path(run_dir)).replace("/", "\\")
    alt_needle = str(Path(run_dir)).replace("\\", "/")
    current_pid = os.getpid()
    processes: list[ActiveProcess] = []
    if os.name == "nt":
        script = (
            "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                check=False,
            )
            payload = json.loads(result.stdout or "[]")
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                cmd = str(row.get("CommandLine") or "")
                pid = int(row.get("ProcessId") or 0)
                if pid == current_pid:
                    continue
                if _is_relevant_phase3_command(cmd, needle, alt_needle):
                    processes.append(ActiveProcess(pid=pid, command_line=cmd))
        except Exception:
            return processes
        return processes
    try:
        result = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False)
    except OSError:
        return processes
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == current_pid:
            continue
        cmd = parts[1]
        if _is_relevant_phase3_command(cmd, needle, alt_needle):
            processes.append(ActiveProcess(pid=pid, command_line=cmd))
    return processes


def _is_relevant_phase3_command(command_line: str, needle: str, alt_needle: str) -> bool:
    if "run_phase3_optuna_tpe.py" not in command_line and "run_phase3_autonomous_supervisor.py" not in command_line:
        return False
    return needle in command_line or alt_needle in command_line or Path(needle).name in command_line


class _SupervisorLock:
    def __init__(self, run_dir: Path, process_finder: Callable[[], list[ActiveProcess]]) -> None:
        self.path = run_dir / LOCK_NAME
        self.process_finder = process_finder

    def __enter__(self) -> "_SupervisorLock":
        if self.path.exists():
            payload = _read_json(self.path)
            pid = int(payload.get("pid", 0) or 0)
            if pid and any(proc.pid == pid for proc in self.process_finder()):
                raise RuntimeError(f"phase3_supervisor_already_active:{pid}")
            self.path.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "command": " ".join(sys.argv),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.path.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _import_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("Optuna is required for Phase 3 supervisor") from exc
    return optuna
