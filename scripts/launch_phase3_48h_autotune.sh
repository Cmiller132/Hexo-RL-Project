#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ROOT="${1:-runs/phase3_48h_autotune_20260428}"
RUN_MODE="${2:-background}"
DURATION_HOURS="${DURATION_HOURS:-48}"
TARGET_EPOCH_SECONDS="${TARGET_EPOCH_SECONDS:-600}"
CALIBRATION_EPOCH_SECONDS="${CALIBRATION_EPOCH_SECONDS:-240}"
CALIBRATION_STATES="${CALIBRATION_STATES:-1024}"
CALIBRATION_TRAIN_BATCHES="${CALIBRATION_TRAIN_BATCHES:-50}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
MAX_ACTIVE_TRIALS="${MAX_ACTIVE_TRIALS:-8}"
PBT_POPULATION="${PBT_POPULATION:-8}"
PBT_GENERATIONS="${PBT_GENERATIONS:-6}"
PERTURB_INTERVAL="${PERTURB_INTERVAL:-2}"
EVAL_GAMES="${EVAL_GAMES:-4}"
FINAL_EVAL_GAMES="${FINAL_EVAL_GAMES:-12}"

mkdir -p "${RUN_ROOT}"

if [[ -f "${RUN_ROOT}/supervisor.pid" ]]; then
    old_pid="$(cat "${RUN_ROOT}/supervisor.pid" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
        echo "${old_pid}"
        exit 0
    fi
fi

if [[ -s "${RUN_ROOT}/supervisor.log" ]]; then
    mv "${RUN_ROOT}/supervisor.log" "${RUN_ROOT}/supervisor.log.$(date +%Y%m%d_%H%M%S)"
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
    --train-batches "${TRAIN_BATCHES}"
    --max-active-trials "${MAX_ACTIVE_TRIALS}"
    --pbt-population "${PBT_POPULATION}"
    --pbt-generations "${PBT_GENERATIONS}"
    --perturb-interval "${PERTURB_INTERVAL}"
    --eval-games "${EVAL_GAMES}"
    --final-eval-games "${FINAL_EVAL_GAMES}"
)

default_reference="runs/restnet_sparse_stage0_epoch10_stable_20260428/epoch_0010.pt"
if [[ -f "${default_reference}" ]]; then
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
    exec env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1
fi

if command -v setsid >/dev/null 2>&1; then
    setsid nohup env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1 < /dev/null &
else
    nohup env PYTHONUNBUFFERED=1 python "${args[@]}" > "${RUN_ROOT}/supervisor.log" 2>&1 < /dev/null &
fi
pid="$!"
echo "${pid}" > "${RUN_ROOT}/supervisor.pid"
echo "${pid}"
