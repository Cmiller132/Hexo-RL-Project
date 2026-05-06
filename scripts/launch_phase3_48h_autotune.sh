#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="${1:-runs/phase3_48h_autotune_20260428}"
RUN_MODE="${2:-background}"
LAUNCH_LOG="${RUN_ROOT}/launcher.log"
DURATION_HOURS="${DURATION_HOURS:-48}"
TARGET_EPOCH_SECONDS="${TARGET_EPOCH_SECONDS:-600}"
CALIBRATION_EPOCH_SECONDS="${CALIBRATION_EPOCH_SECONDS:-240}"
CALIBRATION_STATES="${CALIBRATION_STATES:-1024}"
CALIBRATION_TRAIN_BATCHES="${CALIBRATION_TRAIN_BATCHES:-50}"
CALIBRATION_THROUGHPUT_GATE="${CALIBRATION_THROUGHPUT_GATE:-0.35}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
MAX_GAME_MOVES="${MAX_GAME_MOVES:-500}"
RUNTIME_SWEEP_STATES="${RUNTIME_SWEEP_STATES:-384}"
RUNTIME_SWEEP_WORKERS="${RUNTIME_SWEEP_WORKERS:-2,3}"
RUNTIME_SWEEP_MAX_CANDIDATES="${RUNTIME_SWEEP_MAX_CANDIDATES:-2}"
MAX_ACTIVE_TRIALS="${MAX_ACTIVE_TRIALS:-6}"
ASHA_RESOURCES="${ASHA_RESOURCES:-10,20,30}"
ASHA_PROMOTE_FRACTION="${ASHA_PROMOTE_FRACTION:-0.5}"
PBT_POPULATION="${PBT_POPULATION:-8}"
PBT_GENERATIONS="${PBT_GENERATIONS:-6}"
PERTURB_INTERVAL="${PERTURB_INTERVAL:-10}"
CHAMPION_MIN_EPOCHS="${CHAMPION_MIN_EPOCHS:-20}"
STRATEGY_SCORE_MIN_EPOCHS="${STRATEGY_SCORE_MIN_EPOCHS:-10}"
CLASSICAL_SCORE_MIN_EPOCHS="${CLASSICAL_SCORE_MIN_EPOCHS:-12}"
EVAL_GAMES="${EVAL_GAMES:-4}"
FINAL_EVAL_GAMES="${FINAL_EVAL_GAMES:-12}"
FAMILY_FILTER="${FAMILY_FILTER:-global_xattn_0,global_line_window_0,global_pair_twostage_0,global_graph_full_0}"
USE_DEFAULT_REFERENCE_CHECKPOINT="${USE_DEFAULT_REFERENCE_CHECKPOINT:-0}"

mkdir -p "${RUN_ROOT}"

log_launch() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${LAUNCH_LOG}"
}

log_launch "launch requested mode=${RUN_MODE} cwd=${PWD} shell_pid=$$"
log_launch "settings duration=${DURATION_HOURS} target_epoch_seconds=${TARGET_EPOCH_SECONDS} calibration_epoch_seconds=${CALIBRATION_EPOCH_SECONDS} calibration_states=${CALIBRATION_STATES} train_batches=${TRAIN_BATCHES} max_game_moves=${MAX_GAME_MOVES} runtime_sweep_states=${RUNTIME_SWEEP_STATES} runtime_sweep_workers=${RUNTIME_SWEEP_WORKERS} runtime_sweep_max_candidates=${RUNTIME_SWEEP_MAX_CANDIDATES} max_active_trials=${MAX_ACTIVE_TRIALS} asha_resources=${ASHA_RESOURCES} pbt_population=${PBT_POPULATION} champion_min_epochs=${CHAMPION_MIN_EPOCHS} family_filter=${FAMILY_FILTER} use_default_reference_checkpoint=${USE_DEFAULT_REFERENCE_CHECKPOINT}"
if command -v free >/dev/null 2>&1; then
    log_launch "memory $(free -h | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    log_launch "gpu $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
fi

live_supervisor() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    local stat args
    stat="$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d ' ' || true)"
    [[ -n "${stat}" && "${stat}" != *Z* ]] || return 1
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${args}" == *"run_phase3_48h_autotune.py"* ]] || return 1
}

if [[ -f "${RUN_ROOT}/supervisor.pid" ]]; then
    old_pid="$(cat "${RUN_ROOT}/supervisor.pid" 2>/dev/null || true)"
    if live_supervisor "${old_pid}"; then
        old_args="$(ps -p "${old_pid}" -o args= 2>/dev/null || true)"
        log_launch "existing supervisor still alive pid=${old_pid} args=${old_args}"
        echo "${old_pid}"
        exit 0
    fi
    log_launch "stale supervisor pid file pid=${old_pid:-empty}; replacing"
fi

if [[ -s "${RUN_ROOT}/supervisor.log" ]]; then
    rotated="${RUN_ROOT}/supervisor.log.$(date +%Y%m%d_%H%M%S)"
    mv "${RUN_ROOT}/supervisor.log" "${rotated}"
    log_launch "rotated supervisor log to ${rotated}"
fi

if [[ -f .venv-wsl/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-wsl/bin/activate
fi

args=(
    scripts/run_phase3_48h_autotune.py
    --output-root "${RUN_ROOT}"
    --duration-hours "${DURATION_HOURS}"
    --target-epoch-seconds "${TARGET_EPOCH_SECONDS}"
    --calibration-epoch-seconds "${CALIBRATION_EPOCH_SECONDS}"
    --calibration-states "${CALIBRATION_STATES}"
    --calibration-train-batches "${CALIBRATION_TRAIN_BATCHES}"
    --calibration-throughput-gate "${CALIBRATION_THROUGHPUT_GATE}"
    --train-batches "${TRAIN_BATCHES}"
    --max-game-moves "${MAX_GAME_MOVES}"
    --runtime-sweep-states "${RUNTIME_SWEEP_STATES}"
    --runtime-sweep-workers "${RUNTIME_SWEEP_WORKERS}"
    --runtime-sweep-max-candidates "${RUNTIME_SWEEP_MAX_CANDIDATES}"
    --max-active-trials "${MAX_ACTIVE_TRIALS}"
    --asha-resources "${ASHA_RESOURCES}"
    --asha-promote-fraction "${ASHA_PROMOTE_FRACTION}"
    --pbt-population "${PBT_POPULATION}"
    --pbt-generations "${PBT_GENERATIONS}"
    --perturb-interval "${PERTURB_INTERVAL}"
    --champion-min-epochs "${CHAMPION_MIN_EPOCHS}"
    --strategy-score-min-epochs "${STRATEGY_SCORE_MIN_EPOCHS}"
    --classical-score-min-epochs "${CLASSICAL_SCORE_MIN_EPOCHS}"
    --eval-games "${EVAL_GAMES}"
    --final-eval-games "${FINAL_EVAL_GAMES}"
)

if [[ -n "${FAMILY_FILTER}" ]]; then
    args+=(--family-filter "${FAMILY_FILTER}")
fi

default_reference="runs/restnet_sparse_stage0_epoch10_stable_20260428/epoch_0010.pt"
if [[ "${USE_DEFAULT_REFERENCE_CHECKPOINT}" == "1" && -f "${default_reference}" ]]; then
    args+=(--reference-checkpoint "${default_reference}")
fi

if [[ -n "${REFERENCE_CHECKPOINTS:-}" ]]; then
    read -r -a references <<< "${REFERENCE_CHECKPOINTS}"
    for checkpoint in "${references[@]}"; do
        args+=(--reference-checkpoint "${checkpoint}")
    done
fi

if [[ "${RUN_MODE}" == "--foreground" || "${RUN_MODE}" == "foreground" ]]; then
    echo "$$" > "${RUN_ROOT}/supervisor.pid"
    log_launch "starting foreground supervisor pid=$$ command=python ${args[*]}"
    exec env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1
fi

log_launch "starting background supervisor command=python ${args[*]}"
if command -v setsid >/dev/null 2>&1; then
    setsid nohup env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1 < /dev/null &
else
    nohup env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1 < /dev/null &
fi
pid="$!"
echo "${pid}" > "${RUN_ROOT}/supervisor.pid"
log_launch "background supervisor started pid=${pid}"
echo "${pid}"
