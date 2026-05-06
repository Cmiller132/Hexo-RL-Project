#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="${1:-runs/phase2_phase3_autotune_overnight_20260428}"
HOURS="${2:-8}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
NO_PROGRESS_WARN_CHECKS="${NO_PROGRESS_WARN_CHECKS:-3}"
STATUS_MD="${RUN_ROOT}/overnight_monitor.md"
EVENTS_JSONL="${RUN_ROOT}/overnight_monitor_events.jsonl"

mkdir -p "${RUN_ROOT}"
echo "$$" > "${RUN_ROOT}/monitor.pid"
exec >> "${RUN_ROOT}/monitor.log" 2>&1

append_json() {
    local event="$1"
    local payload="$2"
    python3 - "$EVENTS_JSONL" "$event" "$payload" <<'PY'
import json
import sys
import time
path, event, payload = sys.argv[1:4]
try:
    data = json.loads(payload)
except Exception:
    data = {"message": payload}
data.update({"time": time.time(), "event": event})
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(data, sort_keys=True) + "\n")
PY
}

write_status_header() {
    cat > "${STATUS_MD}" <<EOF
# Overnight Phase 2/3 Autotune Monitor - 2026-04-28

Run root: \`${RUN_ROOT}\`

This watchdog checks the active supervisor every ${INTERVAL_SECONDS}s for ${HOURS}h, records GPU/process/event health, warns after ${NO_PROGRESS_WARN_CHECKS} repeated no-progress checks, and restarts the supervisor up to ${MAX_RESTARTS} times if it exits.

| Time UTC | PID | State | GPU Used MB | GPU % | Last Event | Last Progress | Action |
|---|---:|---|---:|---:|---|---|---|
EOF
}

pid_alive() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    local stat args
    stat="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d ' ' || true)"
    [[ -n "${stat}" && "${stat}" != *Z* ]] || return 1
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${args}" == *"run_phase3_48h_autotune.py"* ]] || return 1
}

read_pid() {
    cat "${RUN_ROOT}/supervisor.pid" 2>/dev/null || true
}

last_event_name() {
    python3 - "$RUN_ROOT/events.jsonl" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    line = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pass
    if not line:
        print("none")
    else:
        print(json.loads(line).get("event", "unknown"))
except Exception as exc:
    print(f"unreadable:{type(exc).__name__}")
PY
}

gpu_snapshot() {
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0,0"
}

progress_snapshot() {
    python3 - "$RUN_ROOT/supervisor.log" <<'PY'
import re
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()[-1500:]
except Exception as exc:
    print(f"unreadable:{type(exc).__name__}")
    raise SystemExit

for line in reversed(lines):
    if "Progress:" in line:
        match = re.search(
            r"Progress:\s*([0-9.]+)%\s*\|\s*Games:\s*(\d+)\s*\(([0-9.]+)/min\)\s*\|\s*Buffer:\s*(\d+)\s*\|\s*Workers:\s*(\d+)/(\d+)",
            line,
        )
        if match:
            pct, games, gpm, buffer_size, alive, total = match.groups()
            print(f"progress={pct}% games={games} gpm={gpm} buffer={buffer_size} workers={alive}/{total}")
            raise SystemExit
    if "Checkpoint saved" in line:
        print("checkpoint_saved")
        raise SystemExit
    if "Epoch complete!" in line:
        print("epoch_complete")
        raise SystemExit
print("no_progress_line")
PY
}

restart_supervisor() {
    MAX_GAME_MOVES="${MAX_GAME_MOVES:-500}" \
    CALIBRATION_THROUGHPUT_GATE="${CALIBRATION_THROUGHPUT_GATE:-0.35}" \
    RUNTIME_SWEEP_STATES="${RUNTIME_SWEEP_STATES:-384}" \
    RUNTIME_SWEEP_WORKERS="${RUNTIME_SWEEP_WORKERS:-2,3}" \
    RUNTIME_SWEEP_MAX_CANDIDATES="${RUNTIME_SWEEP_MAX_CANDIDATES:-2}" \
    ASHA_RESOURCES="${ASHA_RESOURCES:-10,20,30}" \
    PERTURB_INTERVAL="${PERTURB_INTERVAL:-10}" \
    CHAMPION_MIN_EPOCHS="${CHAMPION_MIN_EPOCHS:-20}" \
    STRATEGY_SCORE_MIN_EPOCHS="${STRATEGY_SCORE_MIN_EPOCHS:-10}" \
    CLASSICAL_SCORE_MIN_EPOCHS="${CLASSICAL_SCORE_MIN_EPOCHS:-12}" \
    bash scripts/launch_phase3_48h_autotune.sh "${RUN_ROOT}" background
    sleep 8
}

write_status_header
append_json "monitor_start" "{\"run_root\":\"${RUN_ROOT}\",\"hours\":${HOURS},\"interval_seconds\":${INTERVAL_SECONDS}}"

end_epoch="$(python3 - "$HOURS" <<'PY'
import sys, time
print(time.time() + float(sys.argv[1]) * 3600.0)
PY
)"
restarts=0
last_progress=""
stagnant_checks=0

while python3 - "$end_epoch" <<'PY'
import sys, time
raise SystemExit(0 if time.time() < float(sys.argv[1]) else 1)
PY
do
    now="$(date -u '+%Y-%m-%d %H:%M:%S')"
    pid="$(read_pid)"
    state="dead"
    action="none"
    if pid_alive "${pid}"; then
        state="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d ' ' || echo alive)"
    else
        if (( restarts < MAX_RESTARTS )); then
            action="restart"
            restarts=$((restarts + 1))
            restart_supervisor || action="restart_failed"
            pid="$(read_pid)"
            state="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d ' ' || echo restarted)"
        else
            action="restart_limit_reached"
        fi
    fi
    gpu="$(gpu_snapshot)"
    gpu_mem="${gpu%%,*}"
    gpu_util="${gpu##*,}"
    event="$(last_event_name)"
    progress="$(progress_snapshot)"
    if [[ "${state}" != "dead" && "${progress}" == "${last_progress}" && "${progress}" != "epoch_complete" && "${progress}" != "checkpoint_saved" ]]; then
        stagnant_checks=$((stagnant_checks + 1))
    else
        stagnant_checks=0
    fi
    last_progress="${progress}"
    if [[ "${action}" == "none" && "${stagnant_checks}" -ge "${NO_PROGRESS_WARN_CHECKS}" ]]; then
        action="no_progress_${stagnant_checks}_checks"
    fi
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "${now}" "${pid:-0}" "${state}" "${gpu_mem:-0}" "${gpu_util:-0}" "${event}" "${progress}" "${action}" >> "${STATUS_MD}"
    append_json "monitor_check" "{\"pid\":\"${pid:-}\",\"state\":\"${state}\",\"gpu_mem_mb\":\"${gpu_mem:-0}\",\"gpu_util_pct\":\"${gpu_util:-0}\",\"last_event\":\"${event}\",\"last_progress\":\"${progress}\",\"stagnant_checks\":${stagnant_checks},\"action\":\"${action}\",\"restarts\":${restarts}}"
    sleep "${INTERVAL_SECONDS}"
done

append_json "monitor_complete" "{\"run_root\":\"${RUN_ROOT}\",\"restarts\":${restarts}}"
printf '\nMonitor completed after %sh with %s restart(s).\n' "${HOURS}" "${restarts}" >> "${STATUS_MD}"
