param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$RepoRoot = "/root/Hexo-RL-Project-ext4",
    [string]$RunRoot = "/root/hexo_runs/phase2_phase3_autotune_384moves_simsweep_20260428_ext4",
    [int]$DashboardPort = 8765,
    [string]$RuntimeSweepWorkers = "2,3,4,5,6",
    [int]$RuntimeSweepStates = 768,
    [int]$RuntimeSweepMaxCandidates = 6,
    [int]$MaxGameMoves = 384,
    [int]$MaxActiveTrials = 12,
    [string]$AshaResources = "8,12,14",
    [int]$ChampionMinEpochs = 20,
    [switch]$NoStartProcess
)

$ErrorActionPreference = "Stop"

$wsl = (Get-Command wsl.exe).Source
$repoWin = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($repoWin -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Expected a local Windows drive path, got: $repoWin"
}
$repoWsl = "/mnt/" + $Matches[1].ToLowerInvariant() + "/" + ($Matches[2] -replace '\\', '/')
$localTmp = Join-Path $repoWin ".tmp_wsl_services"
New-Item -ItemType Directory -Force -Path $localTmp | Out-Null
$keepalivePath = Join-Path $localTmp "hexo_phase3_keepalive.sh"
$escapedRepo = $RepoRoot.Replace("'", "'\\''")
$escapedRun = $RunRoot.Replace("'", "'\\''")

$script = @"
#!/usr/bin/env bash
set -u
REPO='$escapedRepo'
RUN='$escapedRun'
DASHBOARD_PORT='$DashboardPort'
LOG="`$RUN/service_keepalive.log"
mkdir -p "`$RUN"
touch "`$LOG"
cd "`$REPO" || exit 1

log() {
    printf '%s %s\n' "`$(date '+%Y-%m-%d %H:%M:%S')" "`$*" >> "`$LOG"
}

alive_match() {
    local pid_file="`$1"
    local pattern="`$2"
    local pid=""
    pid=`$(cat "`$pid_file" 2>/dev/null || true)
    if [[ -z "`$pid" ]]; then
        return 1
    fi
    ps -p "`$pid" -o args= 2>/dev/null | grep -q "`$pattern"
}

snapshot() {
    log "snapshot processes=`$(ps -eo pid,ppid,stat,etime,%cpu,%mem,args | grep -E 'run_phase3_48h|hexorl.cli dashboard|monitor_phase_autotune' | grep -v grep | tr '\n' ';' || true)"
    if command -v free >/dev/null 2>&1; then
        log "snapshot memory=`$(free -h | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "snapshot gpu=`$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || true)"
    fi
    if command -v ss >/dev/null 2>&1; then
        log "snapshot listeners=`$(ss -ltnp 2>/dev/null | grep -E ':8765|:8766' | tr '\n' ';' || true)"
    fi
}

memory_guard() {
    local line=""
    line=`$(awk '
        /^MemAvailable:/ { avail=`$2 }
        /^SwapTotal:/ { swap_total=`$2 }
        /^SwapFree:/ { swap_free=`$2 }
        END {
          if (avail == "") avail = 0
          if (swap_total == "") swap_total = 0
          if (swap_free == "") swap_free = 0
          printf "%.3f %.3f\n", avail / 1048576.0, (swap_total - swap_free) / 1048576.0
        }
      ' /proc/meminfo 2>/dev/null || echo "0 0")
    local available_gb="`$(echo "`$line" | awk '{print `$1}')"
    local swap_used_gb="`$(echo "`$line" | awk '{print `$2}')"
    awk -v avail="`$available_gb" -v swap="`$swap_used_gb" 'BEGIN { exit !((avail > 0 && avail < 3.0) || swap > 2.0) }'
    if [[ `$? -eq 0 ]]; then
        local pid=""
        pid=`$(cat "`$RUN/supervisor.pid" 2>/dev/null || true)
        log "memory_guard stopping supervisor available_gb=`$available_gb swap_used_gb=`$swap_used_gb pid=`${pid:-none}"
        if [[ -n "`$pid" ]]; then
            kill "`$pid" 2>/dev/null || true
            sleep 8
            pkill -P "`$pid" 2>/dev/null || true
        fi
        sleep 10
    fi
}

start_supervisor() {
    log "starting supervisor"
    export MAX_GAME_MOVES='$MaxGameMoves'
    export RUNTIME_SWEEP_STATES='$RuntimeSweepStates'
    export RUNTIME_SWEEP_WORKERS='$RuntimeSweepWorkers'
    export RUNTIME_SWEEP_MAX_CANDIDATES='$RuntimeSweepMaxCandidates'
    export MAX_ACTIVE_TRIALS='$MaxActiveTrials'
    export CALIBRATION_THROUGHPUT_GATE='0.35'
    export ASHA_RESOURCES='$AshaResources'
    export CHAMPION_MIN_EPOCHS='$ChampionMinEpochs'
    export FULL_SIMS_OPTIONS='512,800,1200,1600'
    export MIN_SIMS_OPTIONS='128,192,256,384'
    bash scripts/launch_phase3_48h_autotune.sh "`$RUN" background >> "`$LOG" 2>&1
    sleep 2
}

start_dashboard() {
    log "starting dashboard port=`$DASHBOARD_PORT"
    . .venv-wsl/bin/activate
    python -m hexorl.cli dashboard \
        --db "`$RUN/dashboard_suite.sqlite3" \
        --run-root "`$RUN" \
        --host 0.0.0.0 \
        --port "`$DASHBOARD_PORT" \
        >> "`$RUN/dashboard.log" 2>&1 < /dev/null &
    echo `$! > "`$RUN/dashboard.pid"
    sleep 2
}

start_monitor() {
    log "starting monitor"
    bash scripts/monitor_phase_autotune.sh "`$RUN" 12 >> "`$RUN/monitor.log" 2>&1 < /dev/null &
    echo `$! > "`$RUN/monitor.pid"
    sleep 1
}

log "keepalive boot pid=`$`$ repo=`$REPO run=`$RUN port=`$DASHBOARD_PORT"
echo `$`$ > "`$RUN/service_keepalive.pid"
tick=0
while true; do
    memory_guard
    if ! alive_match "`$RUN/supervisor.pid" "run_phase3_48h_autotune"; then
        log "supervisor missing or stale"
        start_supervisor
    fi
    if ! alive_match "`$RUN/dashboard.pid" "hexorl.cli dashboard"; then
        log "dashboard missing or stale"
        start_dashboard
    fi
    if ! alive_match "`$RUN/monitor.pid" "monitor_phase_autotune"; then
        log "monitor missing or stale"
        start_monitor
    fi
    if (( tick % 6 == 0 )); then
        snapshot
    fi
    tick=`$((tick + 1))
    sleep 10
done
"@

[IO.File]::WriteAllText($keepalivePath, $script.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))

$wslScript = "/tmp/hexo_phase3_services/hexo_phase3_keepalive.sh"
& $wsl -d $Distro -- bash -lc "mkdir -p /tmp/hexo_phase3_services && cp '$repoWsl/.tmp_wsl_services/hexo_phase3_keepalive.sh' $wslScript && chmod +x $wslScript"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install WSL keepalive script into $Distro"
}

if ($NoStartProcess) {
    Write-Host "Installed $wslScript in $Distro"
    exit 0
}

$process = Start-Process -FilePath $wsl -WindowStyle Hidden -PassThru -ArgumentList @(
    "-d", $Distro,
    "--",
    "bash", $wslScript
)
Write-Host "Started WSL Phase 3 keepalive: WindowsPID=$($process.Id) Distro=$Distro RunRoot=$RunRoot DashboardPort=$DashboardPort"
