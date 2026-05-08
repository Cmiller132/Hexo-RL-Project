import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Bot,
  Cpu,
  Database,
  Eye,
  FileSearch,
  Gauge,
  Gamepad2,
  Layers,
  Pause,
  Play,
  RefreshCw,
  Swords,
  Target,
  Trophy
} from "lucide-react";
import "./styles.css";

type AnyRow = Record<string, any>;

const tabs = [
  { id: "suite", label: "Suite", icon: Trophy },
  { id: "charts", label: "Charts", icon: BarChart3 },
  { id: "examples", label: "Examples", icon: Bot },
  { id: "games", label: "Games", icon: FileSearch },
  { id: "replay", label: "Replay", icon: Eye },
  { id: "play", label: "Play", icon: Gamepad2 },
  { id: "arena", label: "Arena", icon: Swords },
  { id: "checkpoints", label: "Checkpoints", icon: Database },
  { id: "axis", label: "Axis Lab", icon: Target }
];

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

function replayQuery(
  runId: string,
  options: { includePositions?: boolean; compact?: boolean } = {}
) {
  const params = new URLSearchParams();
  if (runId) params.set("run_id", runId);
  if (options.includePositions !== undefined) {
    params.set("include_positions", options.includePositions ? "true" : "false");
  }
  if (options.compact !== undefined) params.set("compact", options.compact ? "true" : "false");
  const query = params.toString();
  return query ? `?${query}` : "";
}

function moveWindow(moves: AnyRow[], turn: number, radius = 24) {
  const center = clamp(Math.round(turn), 0, moves.length);
  const start = clamp(center - radius, 0, moves.length);
  const end = clamp(center + radius, 0, moves.length);
  return {
    start,
    end,
    moves: moves.slice(start, end).map((move, offset) => ({ move, index: start + offset }))
  };
}

function shortTrialName(value: string | undefined | null) {
  return String(value || "-")
    .replace(/^optuna_[^_]+_[^_]+_\d+_/, "")
    .replace(/__none__v1$/, "")
    .replace(/__root_pair_mcts__v1$/, " root-pair")
    .replace(/__full_pair_mcts__v1$/, " full-pair")
    .replace(/^global_/, "")
    .replace(/_/g, " ");
}

function modelOptionLabel(run: AnyRow, trial?: AnyRow) {
  const label = trial?.trial_label || run.payload_json?.trial_label || run.name || shortTrialName(run.suite_trial_id || run.trial_id || run.run_id);
  const epoch = trial?.epoch ?? run.payload_json?.epoch;
  const loss = trial?.loss_total;
  const bits = [label];
  if (epoch !== undefined && epoch !== null && epoch !== "") bits.push(`e${epoch}`);
  if (loss !== undefined && loss !== null) bits.push(`loss ${fmt(loss)}`);
  if (trial?.architecture) bits.push(String(trial.architecture));
  return bits.join(" · ");
}

function App() {
  const [active, setActive] = useState("suite");
  const [health, setHealth] = useState<AnyRow | null>(null);
  const [runs, setRuns] = useState<AnyRow[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [metrics, setMetrics] = useState<AnyRow[]>([]);
  const [games, setGames] = useState<AnyRow[]>([]);
  const [selectedGame, setSelectedGame] = useState<number | null>(null);
  const [replay, setReplay] = useState<AnyRow | null>(null);
  const [position, setPosition] = useState<AnyRow | null>(null);
  const [checkpoints, setCheckpoints] = useState<AnyRow[]>([]);
  const [session, setSession] = useState<AnyRow | null>(null);
  const [arena, setArena] = useState<AnyRow[]>([]);
  const [axis, setAxis] = useState<AnyRow[]>([]);
  const [axisResults, setAxisResults] = useState<AnyRow[]>([]);
  const [suiteStatus, setSuiteStatus] = useState<AnyRow | null>(null);
  const [suiteTrials, setSuiteTrials] = useState<AnyRow[]>([]);
  const [phase2, setPhase2] = useState<AnyRow | null>(null);
  const [phase3, setPhase3] = useState<AnyRow | null>(null);
  const [bestCheckpoints, setBestCheckpoints] = useState<AnyRow[]>([]);
  const [suiteEvents, setSuiteEvents] = useState<AnyRow[]>([]);
  const [suiteGames, setSuiteGames] = useState<AnyRow[]>([]);
  const [gameExamples, setGameExamples] = useState<AnyRow[]>([]);
  const [error, setError] = useState<string>("");
  const [refreshNonce, setRefreshNonce] = useState(0);
  const loadInFlight = useRef(false);
  const runDetailInFlight = useRef(false);
  const replayLoadSeq = useRef(0);

  const load = async () => {
    if (loadInFlight.current) return;
    loadInFlight.current = true;
    try {
      setError("");
      const [h, r, a, p, s, t, p2, p3, b, e, sg, gx] = await Promise.all([
        api<AnyRow>("/api/health"),
        api<AnyRow[]>("/api/runs"),
        api<AnyRow[]>("/api/arena/history"),
        api<AnyRow[]>("/api/axis/prototypes"),
        api<AnyRow>("/api/suite/status"),
        api<AnyRow[]>("/api/suite/trials"),
        api<AnyRow>("/api/suite/phase2"),
        api<AnyRow>("/api/suite/phase3"),
        api<AnyRow[]>("/api/suite/best-checkpoints"),
        api<AnyRow[]>("/api/suite/events?limit=64"),
        api<AnyRow[]>("/api/games?limit=32"),
        api<AnyRow[]>("/api/suite/game-examples?limit=96")
      ]);
      setHealth(h);
      setRuns(r);
      setArena(a);
      setAxis(p);
      setSuiteStatus(s);
      setSuiteTrials(t);
      setPhase2(p2);
      setPhase3(p3);
      setBestCheckpoints(b);
      setSuiteEvents(e);
      setSuiteGames(sg);
      setGameExamples(gx);
      if (!selectedRun && r.length) setSelectedRun(r[0].run_id);
    } catch (e: any) {
      setError(e.message);
    } finally {
      loadInFlight.current = false;
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => {
      load();
      setRefreshNonce((n) => n + 1);
    }, 15000);
    return () => window.clearInterval(id);
  }, [selectedRun]);

  const reload = () => {
    load();
    setRefreshNonce((n) => n + 1);
  };

  useEffect(() => {
    if (!selectedRun) return;
    if (runDetailInFlight.current) return;
    runDetailInFlight.current = true;
    const run = encodeURIComponent(selectedRun);
    Promise.all([
      api<AnyRow[]>(`/api/metrics/${run}`),
      api<AnyRow[]>(`/api/games?run_id=${run}&limit=64`),
      api<AnyRow[]>(`/api/checkpoints?run_id=${run}`)
    ])
      .then(([nextMetrics, nextGames, nextCheckpoints]) => {
        setMetrics(nextMetrics);
        setGames(nextGames);
        setCheckpoints(nextCheckpoints);
        setSelectedGame((current) => {
          if (current && nextGames.some((game) => game.game_id === current && game.replay_available !== false)) return current;
          return nextGames.find((game) => game.replay_available !== false)?.game_id ?? nextGames[0]?.game_id ?? null;
        });
      })
      .catch((e) => setError(e.message))
      .finally(() => {
        runDetailInFlight.current = false;
      });
  }, [selectedRun, refreshNonce]);

  useEffect(() => {
    if (!selectedGame) {
      setReplay(null);
      setPosition(null);
      return;
    }
    const requestId = replayLoadSeq.current + 1;
    replayLoadSeq.current = requestId;
    const query = replayQuery(selectedRun, { includePositions: false });
    setReplay(null);
    setPosition(null);
    api<AnyRow>(`/api/games/${selectedGame}/replay${query}`)
      .then((data) => {
        if (replayLoadSeq.current !== requestId) return null;
        setReplay(data);
        return api<AnyRow>(`/api/games/${selectedGame}/position/0${replayQuery(selectedRun, { compact: true })}`);
      })
      .then((data) => {
        if (data && replayLoadSeq.current === requestId) setPosition(data);
      })
      .catch((e) => {
        if (replayLoadSeq.current !== requestId) return;
        setError(e.message);
        setReplay(null);
        setPosition(null);
      });
  }, [selectedGame, selectedRun]);

  const latestMetric = metrics[metrics.length - 1]?.metrics_json || {};
  const selectedGameRun = selectedRun;
  const trialByRun = useMemo(() => {
    const byId = new Map<string, AnyRow>();
    for (const trial of suiteTrials) {
      if (trial.run_id) byId.set(String(trial.run_id), trial);
      if (trial.trial_id) byId.set(String(trial.trial_id), trial);
    }
    return byId;
  }, [suiteTrials]);
  const selectedTrialSummary = trialByRun.get(String(selectedRun));
  const kpis = [
    ["Runs", runs.length],
    ["Suite Games", suiteStatus?.total_games ?? games.length],
    ["Suite Ckpts", suiteStatus?.total_checkpoints ?? checkpoints.length],
    ["Pos/sec", formatRate(suiteStatus?.current_positions_per_sec)],
    ["Current", suiteStatus?.current_model ?? selectedRun ?? "-"],
    ["Epoch", selectedTrialSummary?.epoch ?? latestMetric.train?.epoch ?? latestMetric.epoch ?? "-"],
    ["Loss", fmt(selectedTrialSummary?.loss_total ?? latestMetric.train?.loss_total ?? latestMetric.loss_total)]
  ];

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>Hexo-RL Dashboard</h1>
          <span className="subtle">{health?.db_path || "loading database"}</span>
        </div>
        <div className="toolbar">
          <select value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)}>
            <option value="">No run</option>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>{modelOptionLabel(run, trialByRun.get(String(run.run_id)))}</option>
            ))}
          </select>
          <button title="Refresh" onClick={reload}><RefreshCw size={16} /></button>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="kpis">
        {kpis.map(([label, value]) => (
          <div className="kpi" key={label as string}>
            <span>{label}</span>
            <strong>{String(value)}</strong>
          </div>
        ))}
      </section>

      <nav className="tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              className={active === tab.id ? "active" : ""}
              onClick={() => setActive(tab.id)}
            >
              <Icon size={15} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      {active === "suite" && (
        <SuitePanel
          status={suiteStatus}
          trials={suiteTrials}
          phase2={phase2}
          phase3={phase3}
          bestCheckpoints={bestCheckpoints}
          events={suiteEvents}
          games={suiteGames}
          openTrial={(id) => {
            const trial = suiteTrials.find((row) => row.trial_id === id || row.run_id === id);
            setSelectedRun(trial?.run_id || id);
            setActive("charts");
          }}
          openCheckpointTrial={(id) => {
            const trial = suiteTrials.find((row) => row.trial_id === id || row.run_id === id);
            setSelectedRun(trial?.run_id || id);
            setActive("checkpoints");
          }}
          openGame={(row) => {
            setSelectedRun(row.run_id);
            setSelectedGame(row.game_id);
            setActive("replay");
          }}
        />
      )}
      {active === "charts" && <Charts metrics={metrics} />}
      {active === "examples" && <ExamplesPanel examples={gameExamples} />}
      {active === "games" && (
        <Games
          games={games}
          selectedGame={selectedGame}
          openReplay={(id) => {
            setSelectedGame(id);
            setActive("replay");
          }}
        />
      )}
      {active === "replay" && (
        <Replay
          replay={replay}
          position={position}
          setPosition={setPosition}
          selectedGame={selectedGame}
          runId={selectedGameRun}
        />
      )}
      {active === "play" && <PlayPanel session={session} setSession={setSession} />}
      {active === "arena" && <ArenaPanel arena={arena} reload={reload} />}
      {active === "checkpoints" && <CheckpointPanel checkpoints={checkpoints} reload={reload} selectedRun={selectedRun} />}
      {active === "axis" && (
        <AxisPanel prototypes={axis} results={axisResults} setResults={setAxisResults} />
      )}
    </main>
  );
}

function Charts({ metrics }: { metrics: AnyRow[] }) {
  const lossKeys = collectLossKeys(metrics);
  const policyKeys = collectMetricKeys(metrics, ["policy_top1_prob", "policy_top1_acc"]);
  const lossRows = metrics.map((m) => ({
    ...m,
    ...lossValues(m.metrics_json),
    ...metricValues(m.metrics_json)
  }));
  return (
    <section className="grid two">
      <Panel title="Losses">
        <LossChart metrics={metrics} keys={lossKeys} />
      </Panel>
      <Panel title="Recent Metrics">
        <Table
          rows={lossRows.slice(-12).reverse()}
          columns={["phase", "epoch", "global_step", "created_at", ...lossKeys, ...policyKeys]}
        />
      </Panel>
    </section>
  );
}

function SuitePanel({
  status,
  trials,
  phase2,
  phase3,
  bestCheckpoints,
  events,
  games,
  openTrial,
  openCheckpointTrial,
  openGame
}: {
  status: AnyRow | null;
  trials: AnyRow[];
  phase2: AnyRow | null;
  phase3: AnyRow | null;
  bestCheckpoints: AnyRow[];
  events: AnyRow[];
  games: AnyRow[];
  openTrial: (id: string) => void;
  openCheckpointTrial: (id: string) => void;
  openGame: (row: AnyRow) => void;
}) {
  const lastEvent = status?.last_event || {};
  const activity = status?.current_activity || {};
  const activeTrials = trials.filter((trial) => !trial.pruned && !trial.phase2_excluded);
  const lastEventLabel = status?.last_event_name || lastEvent.event || "-";
  const progress = activity.progress || {};
  const leadingTrial = trials.find((trial) => trial.trial_id === status?.leading_trial_id) || trials[0];
  const phase2Rows = phase2?.rows || status?.phase2?.rows || [];
  const phase2Best = phase2Rows.find((row: AnyRow) => row.phase2_rank === 1) || phase2Rows[0];
  const phase2Ready = phase2Rows.filter((row: AnyRow) => row.phase2_status === "ready");
  const phase2Pending = phase2Rows.filter((row: AnyRow) => row.phase2_status === "pending" || row.phase2_status === "training");
  const phase2Excluded = phase2Rows.filter((row: AnyRow) => row.phase2_excluded);
  const phase3Rows = phase3?.rows || [];
  const phase3Running = phase3Rows.filter((row: AnyRow) => row.state === "RUNNING");
  const phase3Complete = phase3Rows.filter((row: AnyRow) => row.state === "COMPLETE");
  const phase3Best = phase3Rows.find((row: AnyRow) => row.phase3_rank === 1) || phase3Complete[0] || phase3Rows[0];
  const activePhase3 = phase3Running[0] || null;
  const currentStage = activePhase3 ? phase3?.stage : (status?.current_stage || status?.latest_stage || lastEvent.stage || "phase1 scout");
  const currentTrialLabel = activePhase3
    ? activePhase3.phase3_candidate_id
    : shortTrialName(status?.current_trial_id || activity.trial_id || leadingTrial?.trial_id);
  const currentModelLabel = activePhase3
    ? activePhase3.promoted_label
    : (status?.current_model || shortTrialName(leadingTrial?.trial_id));
  const currentEpoch = activePhase3?.completed_epochs ?? activity.epoch ?? status?.leading_epoch ?? leadingTrial?.epoch ?? "-";
  const currentTruncation = activePhase3?.last_event_message?.match(/trunc ([0-9.]+)/)?.[1] ?? fmt(leadingTrial?.truncation_rate);
  const currentAction = activePhase3?.last_event_message || activity.action || "Waiting for trainer activity";
  const [selectedTrialId, setSelectedTrialId] = useState<string>("");
  const [trialDetail, setTrialDetail] = useState<AnyRow | null>(null);
  const [eventFilter, setEventFilter] = useState("all");
  const filteredEvents = events.filter((event) => {
    if (eventFilter === "all") return true;
    if (eventFilter === "warnings") return event.severity === "warning" || event.event === "training_signal_warning";
    return event.trial_id === eventFilter;
  });
  const recentEvents = filteredEvents.slice(-64).reverse();
  const warningEvents = events.filter((event) => event.severity === "warning" || event.event === "training_signal_warning");
  const latestCheckpointEvent = [...events].reverse().find((event) => event.event === "checkpoint");
  const latestSelfplayEvent = [...events].reverse().find((event) => event.event === "metric" && event.phase === "selfplay");
  const latestEpochEvent = [...events].reverse().find((event) => event.event === "epoch_complete");
  useEffect(() => {
    if (selectedTrialId) return;
    const next = activePhase3?.phase3_candidate_id || status?.current_trial_id || bestCheckpoints[0]?.trial_id || trials[0]?.trial_id || "";
    if (next) setSelectedTrialId(String(next));
  }, [selectedTrialId, activePhase3?.phase3_candidate_id, status?.current_trial_id, bestCheckpoints, trials]);
  useEffect(() => {
    if (!selectedTrialId) {
      setTrialDetail(null);
      return;
    }
    api<AnyRow>(`/api/suite/trials/${encodeURIComponent(selectedTrialId)}`)
      .then(setTrialDetail)
      .catch(() => setTrialDetail(null));
  }, [selectedTrialId]);
  return (
    <section className="suiteGrid">
      <Panel
        title="Autotune Suite"
        hint="Live suite totals, current trainer activity, and the model that most recently wrote progress."
      >
        <div className="suiteHero">
          <div>
            <span>Stage</span>
            <strong>{currentStage}</strong>
          </div>
          <div>
            <span>Current Trial</span>
            <strong>{shortTrialName(currentTrialLabel)}</strong>
          </div>
          <div>
            <span>{status?.best_trial_id ? "Best Trial" : "Most Advanced"}</span>
            <strong>{shortTrialName(status?.best_trial_id || status?.leading_trial_id || leadingTrial?.trial_id)}</strong>
          </div>
          <div>
            <span>{status?.best_score !== null && status?.best_score !== undefined ? "Best Score" : "Latest Loss"}</span>
            <strong>{status?.best_score !== null && status?.best_score !== undefined ? fmt(status?.best_score) : fmt(status?.leading_loss_total ?? leadingTrial?.loss_total)}</strong>
          </div>
          <div>
            <span>Current Model</span>
            <strong>{currentModelLabel}</strong>
          </div>
          <div>
            <span>Positions/sec</span>
            <strong>{formatRate(status?.current_positions_per_sec)}</strong>
          </div>
          <div>
            <span>Total Positions</span>
            <strong>{formatCount(status?.total_positions)}</strong>
          </div>
          <div>
            <span>Total Games</span>
            <strong>{formatCount(status?.total_games)}</strong>
          </div>
          <div>
            <span>Epoch</span>
            <strong>{currentEpoch}</strong>
          </div>
          <div>
            <span>Truncation</span>
            <strong>{currentTruncation}</strong>
          </div>
          <div>
            <span>Last Event</span>
            <strong>{lastEventLabel}</strong>
          </div>
          <div>
            <span>Event Time</span>
            <strong>{formatTimestamp(status?.last_event_time || lastEvent.time)}</strong>
          </div>
          <div>
            <span>Live Trials</span>
            <strong>{activeTrials.length}/{status?.trial_count ?? trials.length}</strong>
          </div>
          <div>
            <span>Warnings</span>
            <strong>{status?.warning_count ?? warningEvents.length}</strong>
          </div>
          <div>
            <span>Events</span>
            <strong>{status?.event_count ?? events.length}</strong>
          </div>
        </div>
        <div className="activityStrip">
          <Gauge size={15} />
          <span>{currentAction}</span>
          {activePhase3?.phase3_candidate_id && (
            <button onClick={() => setSelectedTrialId(activePhase3.phase3_candidate_id)}>
              Inspect {shortTrialName(activePhase3.phase3_candidate_id)}
            </button>
          )}
          {!activePhase3?.phase3_candidate_id && activity.trial_id && <button onClick={() => setSelectedTrialId(activity.trial_id)}>Inspect {activity.trial_id}</button>}
          {activity.progress && (
            <span className="activityMeta">
              {activity.progress.workers_alive}/{activity.progress.workers_total} workers,
              {" "}{formatCount(activity.progress.buffer_positions)} buffered positions
            </span>
          )}
        </div>
        <div className="suitePath">{status?.run_root || "No suite run root configured"}</div>
      </Panel>

      <Panel
        title="Phase 2 Evidence"
        hint="Fixed-classical scorecard review for promotion into Phase 3. Rankings use recomputable classical_survival_lcb evidence, not train loss."
      >
        <div className="eventSummary">
          <MetricCard icon={<Trophy size={15} />} label="Leader" value={phase2Best?.trial_label || "-"} />
          <MetricCard icon={<Target size={15} />} label="Leader LCB" value={fmt(phase2Best?.classical_survival_lcb)} />
          <MetricCard icon={<Swords size={15} />} label="Classical Games" value={formatCount(phase2?.total_classical_games ?? status?.phase2?.total_classical_games)} />
          <MetricCard icon={<Activity size={15} />} label="Ready / Pending" value={`${phase2Ready.length}/${phase2Pending.length}`} />
          <MetricCard icon={<Gauge size={15} />} label="Excluded" value={phase2Excluded.length} />
        </div>
        <Table
          rows={phase2Rows}
          columns={[
            "phase2_rank",
            "trial_label",
            "phase2_status",
            "classical_survival_lcb",
            "classical_survival_mean",
            "phase2_gap_to_best",
            "classical_survival_games",
            "classical_win_rate",
            "classical_draw_rate",
            "classical_avg_moves",
            "hard_pass",
            "phase2_exclusion_reason"
          ]}
          onRow={(row) => row.trial_id && setSelectedTrialId(row.trial_id)}
          selected={(row) => row.trial_id === selectedTrialId}
          className="suiteTable"
        />
        <div className="activityStrip phase2Evidence">
          <Database size={15} />
          <span>{phase2?.latest_summary_path || status?.phase2?.latest_summary_path || "No fixed-classical summary artifact yet"}</span>
        </div>
      </Panel>

      <Panel
        title="Phase 3 Tuning"
        hint="Live Optuna TPE child trials from Phase 2 promoted checkpoints. Optuna values are secondary to scorecard-backed Hexo evidence."
      >
        {activePhase3 && (
          <div className="phase3Now">
            <Activity size={16} />
            <div>
              <strong>{shortTrialName(activePhase3.phase3_candidate_id)} is running</strong>
              <span>
                {activePhase3.last_event_message || "Waiting for the next Phase 3 event"} · epoch {activePhase3.completed_epochs ?? "-"}
                /{activePhase3.target_epoch ?? "-"} · sims {activePhase3.mcts_simulations ?? "-"} · loss {fmt(activePhase3.loss_total)}
              </span>
            </div>
          </div>
        )}
        <div className="eventSummary">
          <MetricCard icon={<Activity size={15} />} label="Stage" value={phase3?.stage || "-"} />
          <MetricCard icon={<Gauge size={15} />} label="Running" value={phase3Running.length} />
          <MetricCard icon={<Trophy size={15} />} label="Best Child" value={phase3Best?.phase3_candidate_id || "-"} />
          <MetricCard icon={<Target size={15} />} label="Best LCB" value={fmt(phase3Best?.classical_survival_lcb ?? phase3?.best_value)} />
          <MetricCard icon={<Swords size={15} />} label="Complete" value={`${phase3Complete.length}/${phase3?.trial_count ?? phase3Rows.length}`} />
        </div>
        <Table
          rows={phase3Rows}
          columns={[
            "phase3_rank",
            "phase3_candidate_id",
            "promoted_label",
            "trial_number",
            "state",
            "target_epoch",
            "completed_epochs",
            "classical_survival_lcb",
            "classical_survival_games",
            "classical_win_rate",
            "classical_draw_rate",
            "loss_total",
            "mcts_simulations",
            "pcr_low_sims_ratio",
            "c_puct",
            "lr_multiplier",
            "last_event",
            "started_at",
            "completed_at"
          ]}
          onRow={(row) => row.phase3_candidate_id && setSelectedTrialId(row.phase3_candidate_id)}
          selected={(row) => row.phase3_candidate_id === selectedTrialId}
          className="suiteTable"
        />
        <div className="activityStrip phase2Evidence">
          <Database size={15} />
          <span>{phase3?.latest_summary_path || "Phase 3 summary will appear after the current runner finishes."}</span>
        </div>
      </Panel>

      <Panel
        title="Model Selector"
        hint="Pick a model once, then jump directly to metrics, checkpoints, games, or replay from the same selection."
      >
        <div className="modelPicker">
          {trials.map((trial) => (
            <button
              key={trial.trial_id}
              className={trial.trial_id === selectedTrialId ? "active" : ""}
              onClick={() => setSelectedTrialId(String(trial.trial_id))}
            >
              <strong>{trial.trial_label || shortTrialName(trial.trial_id)}</strong>
              <span>{trial.last_event_message || "Waiting for trainer event"}</span>
              <span>{trial.architecture || "-"} · epoch {trial.epoch ?? "-"} · loss {fmt(trial.loss_total)}</span>
              <span>{formatRate(trial.positions_per_sec)} · {formatCount(trial.games)} games · trunc {fmt(trial.truncation_rate)}</span>
              <span>Phase 2 {trial.phase2_status || "-"} · LCB {fmt(trial.classical_survival_lcb)} · classical games {formatCount(trial.classical_survival_games)}</span>
            </button>
          ))}
        </div>
        <div className="toolbar compact selectorActions">
          <button disabled={!selectedTrialId} onClick={() => selectedTrialId && openTrial(selectedTrialId)}><BarChart3 size={15} /> Metrics</button>
          <button disabled={!selectedTrialId} onClick={() => selectedTrialId && openCheckpointTrial(selectedTrialId)}><Database size={15} /> Checkpoints</button>
          <button
            disabled={!selectedTrialId}
            onClick={() => {
              const game = games.find((row) => row.trial_id === selectedTrialId);
              if (game) openGame(game);
            }}
          >
            <Eye size={15} /> Latest Replay
          </button>
        </div>
      </Panel>

      <Panel
        title="Checkpoint Ranking"
        hint="Before scorecards exist this ranks checkpoints by epoch and recency. After evaluation, score-backed checkpoints sort first."
      >
        <Table
          rows={bestCheckpoints}
          columns={["rank", "trial_label", "rank_basis", "score", "scheduler_score", "epoch", "global_step", "is_loadable", "path"]}
          onRow={(row) => row.trial_id && setSelectedTrialId(row.trial_id)}
          className="mediumTable"
        />
      </Panel>

      <TrialDetail
        detail={trialDetail}
        trialId={selectedTrialId}
        openMetrics={(id) => openTrial(id)}
        openCheckpoints={(id) => openCheckpointTrial(id)}
      />

      <Panel
        title="Trials"
        hint="Every autotune trial with latest completed epoch metrics. Throughput is self-play positions/sec from the most recent epoch."
      >
        <Table
          rows={trials}
          columns={[
            "trial_label",
            "architecture",
            "stage",
            "phase2_status",
            "phase2_rank",
            "classical_survival_lcb",
            "classical_survival_games",
            "epoch",
            "loss_total",
            "loss_value",
            "value_weight_mean",
            "truncation_rate",
            "positions_per_sec",
            "games",
            "positions",
            "checkpoints",
            "mcts_simulations",
            "max_game_moves",
            "recorder_failures",
            "score"
          ]}
          onRow={(row) => row.trial_id && setSelectedTrialId(row.trial_id)}
          selected={(row) => row.trial_id === selectedTrialId}
          className="suiteTable"
        />
      </Panel>

      <Panel title="Recent Saved Games" hint="Recently persisted self-play games. Click one to open its replay.">
        <Table
          rows={games}
          columns={["game_id", "trial_id", "source", "epoch", "move_count", "terminal_reason", "truncated", "created_at"]}
          onRow={openGame}
          className="recentTable"
        />
      </Panel>

      <Panel title="Recent Suite Events" hint="Meaningful suite and trial events. Per-game spam is suppressed; use Games for individual records.">
        <div className="eventSummary">
          <MetricCard icon={<Activity size={15} />} label="Latest epoch" value={latestEpochEvent?.message || "-"} />
          <MetricCard icon={<Gauge size={15} />} label="Latest self-play" value={latestSelfplayEvent?.message || "-"} />
          <MetricCard icon={<Database size={15} />} label="Latest checkpoint" value={latestCheckpointEvent?.message || "-"} />
          <MetricCard icon={<Target size={15} />} label="Warnings" value={warningEvents.length} />
        </div>
        <div className="eventFilters">
          <button className={eventFilter === "all" ? "active" : ""} onClick={() => setEventFilter("all")}>All</button>
          <button className={eventFilter === "warnings" ? "active" : ""} onClick={() => setEventFilter("warnings")}>Warnings</button>
          {trials.map((trial) => (
            <button
              key={trial.trial_id}
              className={eventFilter === trial.trial_id ? "active" : ""}
              onClick={() => setEventFilter(String(trial.trial_id))}
            >
              {trial.trial_label || shortTrialName(trial.trial_id)}
            </button>
          ))}
        </div>
        <Table
          rows={recentEvents}
          columns={[
            "severity",
            "event",
            "trial_label",
            "phase",
            "epoch",
            "message",
            "loss_total",
            "positions_per_sec",
            "truncation_rate",
            "elapsed_s",
            "time"
          ]}
          className="recentTable"
        />
      </Panel>
    </section>
  );
}

function TrialDetail({
  detail,
  trialId,
  openMetrics,
  openCheckpoints
}: {
  detail: AnyRow | null;
  trialId: string;
  openMetrics: (id: string) => void;
  openCheckpoints: (id: string) => void;
}) {
  if (!trialId) {
    return (
      <Panel title="Selected Model" hint="Pick a trial or model row to inspect the exact configuration.">
        <div className="emptyState">No model selected.</div>
      </Panel>
    );
  }
  if (!detail) {
    return (
      <Panel title="Selected Model" hint="Loading model metadata and config from the trial checkpoint.">
        <div className="emptyState">Loading {trialId}...</div>
      </Panel>
    );
  }
  const latest = detail.latest || {};
  const train = latest.train || {};
  const selfplay = latest.selfplay || {};
  const cfg = detail.config || {};
  const model = detail.model_metadata || detail.architecture || {};
  const selfplayCfg = cfg.selfplay || {};
  const inferenceCfg = cfg.inference || {};
  const runtimeCfg = cfg.runtime || {};
  const checkpoint = detail.checkpoint_metadata || {};
  const selected = detail.trial || {};
  const runtimeSweep = detail.state?.runtime_sweep || selected.runtime_sweep || {};
  const phase2 = detail.phase2 || {};
  return (
    <Panel
      title="Selected Model"
      hint="A clickable model inspection view: architecture, search settings, runtime, latest trainer metrics, and raw config."
    >
      <div className="detailHeader">
        <div>
          <h2>{trialId}</h2>
          <p>{detail.architecture_summary || "No architecture metadata yet."}</p>
        </div>
        <div className="toolbar compact">
          <button onClick={() => openMetrics(trialId)}><BarChart3 size={15} /> Metrics</button>
          <button onClick={() => openCheckpoints(trialId)}><Database size={15} /> Checkpoints</button>
        </div>
      </div>
      <div className="detailCards">
        <MetricCard icon={<Layers size={15} />} label="Architecture" value={model.architecture || selected.family?.architecture || "-"} />
        <MetricCard icon={<Cpu size={15} />} label="Workers" value={selfplayCfg.num_workers ?? runtimeSweep.selected?.workers ?? "-"} />
        <MetricCard icon={<Gauge size={15} />} label="Positions/sec" value={formatRate((selfplay.positions_per_min || 0) / 60)} />
        <MetricCard icon={<Activity size={15} />} label="Loss" value={fmt(train.loss_total)} />
        <MetricCard icon={<Trophy size={15} />} label="Phase 2 LCB" value={fmt(phase2.classical_survival_lcb)} />
      </div>
      <div className="detailGrid">
        <KeyValue
          title="Architecture"
          rows={{
            architecture: model.architecture,
            channels: model.channels,
            blocks: model.blocks,
            heads: Array.isArray(model.heads) ? model.heads.join(", ") : model.heads,
            graph_token_set: model.graph_token_set,
            graph_token_budget: model.graph_token_budget,
            graph_layers: model.graph_layers,
            sparse_policy: model.sparse_policy,
            candidate_budget: model.candidate_budget,
            sparse_prior_stage: model.sparse_prior_stage,
            sparse_prior_mix: model.sparse_prior_mix
          }}
        />
        <KeyValue
          title="Self-Play And Search"
          rows={{
            mcts_simulations: selfplayCfg.mcts_simulations,
            pcr_low_sims: selfplayCfg.pcr_low_sims,
            pcr_low_sim_prob: selfplayCfg.pcr_low_sim_prob,
            max_game_moves: selfplayCfg.max_game_moves,
            states_per_epoch: selfplayCfg.states_per_epoch,
            games_per_epoch: selfplayCfg.games_per_epoch,
            c_puct: selfplayCfg.c_puct,
            dirichlet_alpha: selfplayCfg.dirichlet_alpha,
            dirichlet_fraction: selfplayCfg.dirichlet_fraction
          }}
        />
        <KeyValue
          title="Runtime"
          rows={{
            num_workers: selfplayCfg.num_workers,
            batch_size_per_worker: selfplayCfg.batch_size_per_worker,
            max_batch_size: inferenceCfg.max_batch_size,
            max_wait_us: inferenceCfg.max_wait_us,
            fp16: inferenceCfg.fp16,
            cpu_threads: runtimeCfg.cpu_threads,
            compile_model: runtimeCfg.compile_model,
            compile_inference: runtimeCfg.compile_inference,
            runtime_sweep_selected: runtimeSweep.selected
          }}
        />
        <KeyValue
          title="Latest Trainer"
          rows={{
            epoch: latest.epoch || train.epoch,
            loss_total: train.loss_total,
            loss_policy: train.loss_policy,
            loss_value: train.loss_value,
            policy_top1_acc: train.policy_top1_acc,
            sparse_policy_top1_acc: train.sparse_policy_top1_acc,
            pair_policy_top1_acc: train.pair_policy_top1_acc,
            checkpoint_epoch: checkpoint.epoch,
            global_step: checkpoint.global_step,
            checkpoint_path: checkpoint.path
          }}
        />
        <KeyValue
          title="Phase 2 Evidence"
          rows={{
            phase2_status: phase2.phase2_status,
            phase2_rank: phase2.phase2_rank,
            classical_survival_lcb: fmt(phase2.classical_survival_lcb),
            classical_survival_mean: fmt(phase2.classical_survival_mean),
            classical_survival_games: phase2.classical_survival_games,
            classical_win_rate: fmt(phase2.classical_win_rate),
            classical_draw_rate: fmt(phase2.classical_draw_rate),
            classical_avg_moves: fmt(phase2.classical_avg_moves),
            hard_pass: String(phase2.hard_pass ?? "-"),
            evidence_path: phase2.fixed_classical_evidence_path || "-"
          }}
        />
      </div>
      <details className="configDetails">
        <summary>Full checkpoint config</summary>
        <pre>{JSON.stringify(cfg || {}, null, 2)}</pre>
      </details>
    </Panel>
  );
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="metricCard">
      <span>{icon}{label}</span>
      <strong>{value ?? "-"}</strong>
    </div>
  );
}

function KeyValue({ title, rows }: { title: string; rows: AnyRow }) {
  const entries = Object.entries(rows).filter(([, value]) => value !== undefined && value !== null && value !== "");
  return (
    <div className="kvPanel">
      <h3>{title}</h3>
      <dl>
        {entries.map(([key, value]) => (
          <React.Fragment key={key}>
            <dt>{labelFor(key)}</dt>
            <dd>{cell(value, key)}</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
}

function Games({ games, selectedGame, openReplay }: {
  games: AnyRow[];
  selectedGame: number | null;
  openReplay: (id: number) => void;
}) {
  return (
    <Panel title="Game Browser">
      <Table
        rows={games}
        columns={["game_id", "trial_id", "source", "epoch", "move_count", "terminal_reason", "truncated", "outcome", "replay_available", "replay_error", "created_at"]}
        onRow={(row) => row.replay_available !== false && openReplay(row.game_id)}
        selected={(row) => row.game_id === selectedGame}
      />
    </Panel>
  );
}

function ExamplesPanel({ examples }: { examples: AnyRow[] }) {
  const [selectedId, setSelectedId] = useState<string>("");
  const [replay, setReplay] = useState<AnyRow | null>(null);
  const [position, setPosition] = useState<AnyRow | null>(null);
  const [turn, setTurn] = useState(0);
  const [autoplay, setAutoplay] = useState(false);
  const [modelFilter, setModelFilter] = useState("all");
  const [opponentFilter, setOpponentFilter] = useState("all");
  const [sortKey, setSortKey] = useState("newest");
  const [loadingTurn, setLoadingTurn] = useState(false);
  const [positionError, setPositionError] = useState("");
  const replayLoadSeq = useRef(0);
  const positionSeq = useRef(0);
  const opponentValue = (row: AnyRow) => String(row.opponent_type || row.opponent_id || row.kind || "unknown");
  const modelOptions = useMemo(
    () => Array.from(new Set(examples.map((row) => String(row.trial_id || "")).filter(Boolean))).sort(),
    [examples]
  );
  const opponentOptions = useMemo(
    () => Array.from(new Set(examples.map(opponentValue).filter(Boolean))).sort(),
    [examples]
  );
  const filteredExamples = useMemo(() => {
    const rows = examples.filter((row) => {
      const modelOk = modelFilter === "all" || String(row.trial_id || "") === modelFilter;
      const opponentOk = opponentFilter === "all" || opponentValue(row) === opponentFilter;
      return modelOk && opponentOk;
    });
    const sorted = [...rows];
    sorted.sort((a, b) => {
      if (sortKey === "moves_desc") return Number(b.move_count || 0) - Number(a.move_count || 0);
      if (sortKey === "moves_asc") return Number(a.move_count || 0) - Number(b.move_count || 0);
      if (sortKey === "model") {
        const byModel = String(a.model_label || a.trial_id || "").localeCompare(String(b.model_label || b.trial_id || ""));
        return byModel || Number(b.created_at || 0) - Number(a.created_at || 0);
      }
      if (sortKey === "opponent") {
        const byOpponent = opponentValue(a).localeCompare(opponentValue(b));
        return byOpponent || Number(b.created_at || 0) - Number(a.created_at || 0);
      }
      if (sortKey === "outcome") {
        const byOutcome = String(a.outcome || "").localeCompare(String(b.outcome || ""));
        return byOutcome || Number(b.created_at || 0) - Number(a.created_at || 0);
      }
      return Number(b.created_at || 0) - Number(a.created_at || 0);
    });
    return sorted;
  }, [examples, modelFilter, opponentFilter, sortKey]);
  const replayable = useMemo(() => filteredExamples.filter((row) => row.replay_available), [filteredExamples]);
  const selected = replayable.find((row) => row.example_id === selectedId) || replayable[0];
  const moves = replay?.moves || [];
  const visibleMoves = useMemo(() => moveWindow(moves, turn), [moves, turn]);
  const boardMoves = useMemo(() => moves.slice(0, clamp(turn, 0, moves.length)), [moves, turn]);

  useEffect(() => {
    if (!replayable.length) {
      if (selectedId) setSelectedId("");
      return;
    }
    if (!replayable.some((row) => row.example_id === selectedId)) {
      setSelectedId(String(replayable[0].example_id));
    }
  }, [selectedId, replayable]);

  useEffect(() => {
    if (!selectedId) {
      setReplay(null);
      setPosition(null);
      return;
    }
    const requestId = replayLoadSeq.current + 1;
    replayLoadSeq.current = requestId;
    positionSeq.current = requestId;
    setAutoplay(false);
    setTurn(0);
    setLoadingTurn(true);
    setPositionError("");
    setReplay(null);
    setPosition(null);
    api<AnyRow>(`/api/suite/game-examples/${encodeURIComponent(selectedId)}/replay`)
      .then((data) => {
        if (replayLoadSeq.current !== requestId) return null;
        setReplay(data);
        return api<AnyRow>(`/api/suite/game-examples/${encodeURIComponent(selectedId)}/position/0?compact=true`);
      })
      .then((data) => {
        if (data && replayLoadSeq.current === requestId) setPosition(data);
      })
      .catch((e) => {
        if (replayLoadSeq.current !== requestId) return;
        setPositionError(e.message || "Failed to load example replay");
        setReplay(null);
        setPosition(null);
      })
      .finally(() => {
        if (replayLoadSeq.current === requestId) setLoadingTurn(false);
      });
  }, [selectedId]);

  const loadTurn = useCallback((next: number) => {
    if (!selectedId) return;
    const nextTurn = clamp(Math.round(next), 0, moves.length);
    const requestId = positionSeq.current + 1;
    positionSeq.current = requestId;
    setTurn(nextTurn);
    setLoadingTurn(true);
    setPositionError("");
    api<AnyRow>(`/api/suite/game-examples/${encodeURIComponent(selectedId)}/position/${nextTurn}?compact=true`)
      .then((data) => {
        if (positionSeq.current === requestId) setPosition(data);
      })
      .catch((e) => {
        if (positionSeq.current !== requestId) return;
        setPositionError(e.message || "Failed to load example position");
        setPosition(null);
        setAutoplay(false);
      })
      .finally(() => {
        if (positionSeq.current === requestId) setLoadingTurn(false);
      });
  }, [moves.length, selectedId]);

  useEffect(() => {
    if (!autoplay || !selectedId) return;
    const handle = window.setInterval(() => {
      setTurn((current) => {
        const next = current + 1;
        if (next > moves.length) {
          setAutoplay(false);
          return current;
        }
        loadTurn(next);
        return next;
      });
    }, 650);
    return () => window.clearInterval(handle);
  }, [autoplay, selectedId, moves.length, loadTurn]);

  return (
    <section className="examplesGrid">
      <Panel title="Game Examples" hint="Replayable self-play and fixed-classical examples from the current suite.">
        <div className="exampleSummary">
          <MetricCard icon={<Bot size={15} />} label="Shown" value={`${filteredExamples.length}/${examples.length}`} />
          <MetricCard icon={<Gamepad2 size={15} />} label="Self-play" value={filteredExamples.filter((row) => row.kind === "selfplay").length} />
          <MetricCard icon={<Swords size={15} />} label="Classical" value={filteredExamples.filter((row) => row.kind === "fixed_classical").length} />
          <MetricCard icon={<Trophy size={15} />} label="Selected" value={selected?.model_label || selected?.trial_id || "-"} />
        </div>
        <div className="exampleControls">
          <label>
            <span>Model</span>
            <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)}>
              <option value="all">All models</option>
              {modelOptions.map((model) => (
                <option key={model} value={model}>{examples.find((row) => row.trial_id === model)?.model_label || model}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Opponent</span>
            <select value={opponentFilter} onChange={(e) => setOpponentFilter(e.target.value)}>
              <option value="all">All opponents</option>
              {opponentOptions.map((opponent) => (
                <option key={opponent} value={opponent}>{opponent === "selfplay" ? "Self-play" : opponent.replace(/_/g, " ")}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Sort</span>
            <select value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
              <option value="newest">Newest first</option>
              <option value="moves_desc">Longest games</option>
              <option value="moves_asc">Shortest games</option>
              <option value="model">Model</option>
              <option value="opponent">Opponent</option>
              <option value="outcome">Outcome</option>
            </select>
          </label>
          <div className="eventFilters exampleQuickFilters">
            <button className={opponentFilter === "all" ? "active" : ""} onClick={() => setOpponentFilter("all")}>All</button>
            <button className={opponentFilter === "selfplay" ? "active" : ""} onClick={() => setOpponentFilter("selfplay")}>Self-play</button>
            <button
              className={opponentFilter !== "all" && opponentFilter !== "selfplay" ? "active" : ""}
              onClick={() => setOpponentFilter(opponentOptions.find((value) => value !== "selfplay") || "all")}
            >
              Classical
            </button>
          </div>
        </div>
        <Table
          rows={filteredExamples}
          columns={["kind", "model_label", "opponent_label", "trial_id", "game_id", "epoch", "outcome", "move_count", "terminal_reason", "replay_available", "replay_error"]}
          onRow={(row) => row.replay_available && setSelectedId(String(row.example_id))}
          selected={(row) => row.example_id === selectedId}
          className="exampleTable"
        />
      </Panel>
      <Panel title="Example Replay" hint="Use the timeline to inspect how the selected model plays through the game.">
        <div className="toolbar compact">
          <button onClick={() => setAutoplay((v) => !v)} disabled={!moves.length || !selectedId}>
            {autoplay ? <Pause size={15} /> : <Play size={15} />}
            {autoplay ? "Pause" : "Autoplay"}
          </button>
          <button onClick={() => loadTurn(Math.max(0, turn - 1))} disabled={!selectedId || turn <= 0}>Prev</button>
          <button onClick={() => loadTurn(Math.min(moves.length, turn + 1))} disabled={!selectedId || turn >= moves.length}>Next</button>
          <button onClick={() => loadTurn(moves.length)} disabled={!selectedId || !moves.length}>End</button>
          <span className="timelineStatus">Turn {turn}/{moves.length}{loadingTurn ? " loading" : ""}</span>
        </div>
        {positionError && <div className="inlineError">{positionError}</div>}
        <Board position={position} viewKey={selectedId} labelMoves={boardMoves} />
      </Panel>
      <Panel title="Moves">
        <div className="timelineSlider">
          <input
            type="range"
            min={0}
            max={Math.max(0, moves.length)}
            value={clamp(turn, 0, moves.length)}
            onChange={(e) => loadTurn(Number(e.target.value))}
            disabled={!selectedId || !moves.length}
          />
        </div>
        <div className="moveList compactMoves">
          <button className={turn === 0 ? "active" : ""} onClick={() => loadTurn(0)}>Start</button>
          {visibleMoves.start > 0 && <span className="moveGap">...</span>}
          {visibleMoves.moves.map(({ move, index }: { move: AnyRow; index: number }) => (
            <button key={index} className={turn === index + 1 ? "active" : ""} onClick={() => loadTurn(index + 1)}>
              {index + 1}. P{move.player} ({move.q},{move.r})
            </button>
          ))}
          {visibleMoves.end < moves.length && <span className="moveGap">...</span>}
        </div>
      </Panel>
    </section>
  );
}

function Replay({ replay, position, setPosition, selectedGame, runId }: {
  replay: AnyRow | null;
  position: AnyRow | null;
  setPosition: (p: AnyRow | null) => void;
  selectedGame: number | null;
  runId: string;
}) {
  const moves = replay?.moves || [];
  const [turn, setTurn] = useState(0);
  const [autoplay, setAutoplay] = useState(false);
  const [loadingTurn, setLoadingTurn] = useState(false);
  const [positionError, setPositionError] = useState("");
  const positionSeq = useRef(0);
  const visibleMoves = useMemo(() => moveWindow(moves, turn), [moves, turn]);
  const boardMoves = useMemo(() => moves.slice(0, clamp(turn, 0, moves.length)), [moves, turn]);
  const loadTurn = useCallback((next: number) => {
    if (!selectedGame) return;
    const nextTurn = clamp(Math.round(next), 0, moves.length);
    const requestId = positionSeq.current + 1;
    positionSeq.current = requestId;
    setTurn(nextTurn);
    setLoadingTurn(true);
    setPositionError("");
    api<AnyRow>(`/api/games/${selectedGame}/position/${nextTurn}${replayQuery(runId, { compact: true })}`)
      .then((data) => {
        if (positionSeq.current !== requestId) return;
        setPosition(data);
      })
      .catch((e) => {
        if (positionSeq.current !== requestId) return;
        setPositionError(e.message || "Failed to load replay position");
        setPosition(null);
        setAutoplay(false);
      })
      .finally(() => {
        if (positionSeq.current === requestId) setLoadingTurn(false);
      });
  }, [moves.length, runId, selectedGame, setPosition]);
  useEffect(() => {
    setAutoplay(false);
    setTurn(0);
    setPositionError("");
    positionSeq.current += 1;
  }, [selectedGame, runId]);
  useEffect(() => {
    const nextTurn = Number(position?.turn_index ?? 0);
    if (Number.isFinite(nextTurn)) setTurn(nextTurn);
  }, [position?.turn_index]);
  useEffect(() => {
    if (!autoplay || !selectedGame) return;
    const handle = window.setInterval(() => {
      setTurn((current) => {
        const next = current + 1;
        if (next > moves.length) {
          setAutoplay(false);
          return current;
        }
        loadTurn(next);
        return next;
      });
    }, 650);
    return () => window.clearInterval(handle);
  }, [autoplay, selectedGame, moves.length, loadTurn]);
  return (
    <section className="grid replay">
      <Panel title="Board">
        <Board position={position} labelMoves={boardMoves} />
      </Panel>
      <Panel title="Timeline">
        <div className="toolbar compact">
          <button onClick={() => setAutoplay((v) => !v)} disabled={!moves.length || !selectedGame}>
            {autoplay ? <Pause size={15} /> : <Play size={15} />}
            {autoplay ? "Pause" : "Autoplay"}
          </button>
          <button onClick={() => loadTurn(Math.max(0, turn - 1))} disabled={!selectedGame || turn <= 0}>Prev</button>
          <button onClick={() => loadTurn(Math.min(moves.length, turn + 1))} disabled={!selectedGame || turn >= moves.length}>Next</button>
          <button onClick={() => loadTurn(moves.length)} disabled={!selectedGame || !moves.length}>End</button>
          <span className="timelineStatus">Turn {turn}/{moves.length}{loadingTurn ? " loading" : ""}</span>
        </div>
        {positionError && <div className="inlineError">{positionError}</div>}
        <div className="timelineSlider">
          <input
            type="range"
            min={0}
            max={Math.max(0, moves.length)}
            value={clamp(turn, 0, moves.length)}
            onChange={(e) => loadTurn(Number(e.target.value))}
            disabled={!selectedGame || !moves.length}
          />
        </div>
        <div className="moveList compactMoves">
          <button className={turn === 0 ? "active" : ""} onClick={() => loadTurn(0)}>Start</button>
          {visibleMoves.start > 0 && <span className="moveGap">...</span>}
          {visibleMoves.moves.map(({ move, index }: { move: AnyRow; index: number }) => (
            <button key={index} className={turn === index + 1 ? "active" : ""} onClick={() => loadTurn(index + 1)}>
              {index + 1}. P{move.player} ({move.q},{move.r})
            </button>
          ))}
          {visibleMoves.end < moves.length && <span className="moveGap">...</span>}
        </div>
      </Panel>
      <Panel title="Encoding">
        <Table rows={position?.encoding?.channels || []} columns={["index", "name", "sum", "nonzero", "max"]} />
      </Panel>
    </section>
  );
}

function PlayPanel({ session, setSession }: { session: AnyRow | null; setSession: (s: AnyRow) => void }) {
  const create = () => api<AnyRow>("/api/session/create", { method: "POST", body: "{}" }).then(setSession);
  const undo = () => session && api<AnyRow>(`/api/session/${session.session_id}/undo`, { method: "POST", body: "{}" }).then(setSession);
  const reset = () => session && api<AnyRow>(`/api/session/${session.session_id}/reset`, { method: "POST", body: "{}" }).then(setSession);
  const playMove = (m: AnyRow) => session && api<AnyRow>(`/api/session/${session.session_id}/move`, {
    method: "POST",
    body: JSON.stringify({ q: m.q, r: m.r })
  }).then(setSession);
  const clickMove = (q: number, r: number) => playMove({ q, r });
  useEffect(() => {
    if (!session) create();
  }, [session]);
  return (
    <section className="viewerGrid playGrid">
      <Panel title="Interactive Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} /> New</button>
          <button onClick={undo}>Undo</button>
          <button onClick={reset}>Reset</button>
          <span className={`playerBadge p${session?.position?.current_player ?? 0}`}>
            P{session?.position?.current_player ?? 0} to move
          </span>
        </div>
        <Board position={session?.position} interactive onCellClick={clickMove} />
      </Panel>
      <Panel title="Debug">
        <pre>{JSON.stringify(session?.position?.encoding || {}, null, 2)}</pre>
      </Panel>
    </section>
  );
}

function ArenaPanel({ arena, reload }: { arena: AnyRow[]; reload: () => void }) {
  const start = () => api<AnyRow>("/api/arena/start", {
    method: "POST",
    body: JSON.stringify({ side_a: "model", side_b: "classical" })
  }).then(reload);
  return (
    <Panel title="Arena Spectator">
      <div className="toolbar compact">
        <button onClick={start}><Swords size={15} /> Create Match</button>
      </div>
      <Table rows={arena} columns={["match_id", "status", "side_a", "side_b", "updated_at"]} />
    </Panel>
  );
}

function CheckpointPanel({ checkpoints, reload, selectedRun }: { checkpoints: AnyRow[]; reload: () => void; selectedRun: string }) {
  const [path, setPath] = useState("");
  const index = () => api<AnyRow>("/api/import/checkpoints", {
    method: "POST",
    body: JSON.stringify({ path, run_id: selectedRun || undefined })
  }).then(reload);
  return (
    <Panel title="Checkpoint Index">
      <div className="toolbar">
        <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/checkpoints" />
        <button onClick={index}><Database size={15} /> Index</button>
      </div>
      <Table rows={checkpoints} columns={["checkpoint_id", "trial_id", "run_id", "score", "epoch", "global_step", "is_loadable", "path"]} />
    </Panel>
  );
}

function AxisPanel({ prototypes, results, setResults }: {
  prototypes: AnyRow[];
  results: AnyRow[];
  setResults: (rows: AnyRow[]) => void;
}) {
  const [axisSession, setAxisSession] = useState<AnyRow | null>(null);
  const [selectedPrototype, setSelectedPrototype] = useState<string>("");
  const [axisView, setAxisView] = useState<string>("own");
  const [axisScale, setAxisScale] = useState<string>("raw");
  const [params, setParams] = useState<Record<string, number>>({});
  const [fixtures, setFixtures] = useState<AnyRow[]>([]);
  const [fixtureId, setFixtureId] = useState<string>("");
  const [fixtureBusy, setFixtureBusy] = useState(false);
  const selected = results.find((r) => r.prototype_id === selectedPrototype) || results[0];
  const selectedSpec = prototypes.find((p) => p.id === (selectedPrototype || prototypes[0]?.id));
  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    if (!selectedPrototype && prototypes.length) setSelectedPrototype(prototypes[0].id);
  }, [prototypes, selectedPrototype]);
  const create = () => api<AnyRow>("/api/session/create", { method: "POST", body: JSON.stringify({ payload: { mode: "axis_lab" } }) }).then(setAxisSession);
  const refreshFixtures = () => api<AnyRow[]>("/api/axis/fixtures").then(setFixtures);
  useEffect(() => {
    refreshFixtures().catch(() => setFixtures([]));
  }, []);
  useEffect(() => {
    if (!axisSession) create();
  }, [axisSession]);
  const ensureSession = async () => {
    if (axisSession) return axisSession;
    const created = await api<AnyRow>("/api/session/create", { method: "POST", body: JSON.stringify({ payload: { mode: "axis_lab" } }) });
    setAxisSession(created);
    return created;
  };
  const playMove = async (q: number, r: number) => {
    const s = await ensureSession();
    const next = await api<AnyRow>(`/api/session/${s.session_id}/move`, {
      method: "POST",
      body: JSON.stringify({ q, r })
    });
    setAxisSession(next);
    setResults([]);
  };
  const undo = () => axisSession && api<AnyRow>(`/api/session/${axisSession.session_id}/undo`, { method: "POST", body: "{}" }).then((s) => { setAxisSession(s); setResults([]); });
  const reset = () => axisSession && api<AnyRow>(`/api/session/${axisSession.session_id}/reset`, { method: "POST", body: "{}" }).then((s) => { setAxisSession(s); setResults([]); });
  const loadFixture = async (sessionId: string) => {
    setFixtureId(sessionId);
    if (!sessionId) return;
    const fixture = await api<AnyRow>(`/api/session/${sessionId}`);
    setAxisSession(fixture);
    setResults([]);
  };
  const shuffleFixture = async () => {
    const pool = fixtures.filter((fixture) => fixture.session_id !== axisSession?.session_id);
    const next = pool.length ? pool[Math.floor(Math.random() * pool.length)] : fixtures[Math.floor(Math.random() * fixtures.length)];
    if (next?.session_id) await loadFixture(next.session_id);
  };
  const generateFixtures = async () => {
    setFixtureBusy(true);
    try {
      const seed = Math.floor(Date.now() / 1000);
      const data = await api<AnyRow>("/api/axis/fixtures/generate", {
        method: "POST",
        body: JSON.stringify({
          examples_per_move_count: 3,
          move_counts: [8, 16, 24, 32, 40],
          time_ms: 2,
          max_depth: 1,
          near_radius: 6,
          noise_level: 0.08,
          random_move_prob: 0.04,
          opening_random_moves: 2,
          workers: 4,
          seed
        })
      });
      const nextFixtures = await api<AnyRow[]>("/api/axis/fixtures");
      setFixtures(nextFixtures);
      const first = data.fixtures?.[0];
      if (first?.session_id) {
        setFixtureId(first.session_id);
        setAxisSession(first);
        setResults([]);
      }
    } finally {
      setFixtureBusy(false);
    }
  };
  const evaluate = () => {
    const body = axisSession?.session_id
      ? {
          session_id: axisSession.session_id,
          prototype_id: selectedPrototype || undefined,
          parameters: params
        }
      : { history_b64: "", prototype_id: selectedPrototype || undefined, parameters: params };
    api<AnyRow>("/api/axis/evaluate", { method: "POST", body: JSON.stringify(body) })
      .then((data) => setResults(data.results || [data]));
  };
  useEffect(() => {
    if (!axisSession?.session_id || !selectedPrototype) return;
    const handle = window.setTimeout(() => {
      api<AnyRow>("/api/axis/evaluate", {
        method: "POST",
        body: JSON.stringify({
          session_id: axisSession.session_id,
          prototype_id: selectedPrototype,
          parameters: params
        })
      })
        .then((data) => setResults(data.results || [data]))
        .catch(() => setResults([]));
    }, 180);
    return () => window.clearTimeout(handle);
  }, [axisSession?.session_id, axisSession?.position?.turn_index, selectedPrototype, paramsKey]);
  const currentPlayer = Number(selected?.current_player ?? axisSession?.position?.current_player ?? 0);
  const overlayMoves = (selected?.cells || [])
    .map((m: AnyRow) => deriveAxisOverlay(m, axisView, currentPlayer, axisScale))
    .filter((m: AnyRow) => Math.abs(Number(m.score || 0)) > 1e-7);
  return (
    <section className="viewerGrid">
      <Panel title="Axis Target Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} /> New</button>
          <button onClick={undo}>Undo</button>
          <button onClick={reset}>Reset</button>
          <button onClick={evaluate}><Target size={15} /> Evaluate</button>
          <button onClick={generateFixtures} disabled={fixtureBusy}>
            <Bot size={15} /> {fixtureBusy ? "Generating" : "Generate"}
          </button>
          <button onClick={() => refreshFixtures()}><RefreshCw size={14} /> Fixtures</button>
          <button onClick={shuffleFixture} disabled={!fixtures.length}>Shuffle</button>
          <select
            className="toolbarSelect fixtureSelect"
            value={fixtureId}
            onChange={(e) => loadFixture(e.target.value)}
          >
            <option value="">Load fixture</option>
            {fixtures.map((fixture) => (
              <option key={fixture.session_id} value={fixture.session_id}>
                {fixtureLabel(fixture)}
              </option>
            ))}
          </select>
          {["own", "opp", "net", "max", "both"].map((mode) => (
            <button
              key={mode}
              className={axisView === mode ? "active" : ""}
              onClick={() => setAxisView(mode)}
            >
              {mode}
            </button>
          ))}
          <span className="toolbarLabel">scale</span>
          {["raw", "log", "sqrt", "unit"].map((mode) => (
            <button
              key={mode}
              className={axisScale === mode ? "active" : ""}
              onClick={() => setAxisScale(mode)}
            >
              {mode}
            </button>
          ))}
          <span className={`playerBadge p${axisSession?.position?.current_player ?? 0}`}>
            P{axisSession?.position?.current_player ?? 0} to move
          </span>
        </div>
        <Board
          position={axisSession?.position}
          interactive
          onCellClick={playMove}
          overlayMoves={overlayMoves}
          viewKey={axisSession?.session_id}
        />
      </Panel>
      <Panel title="Prototype Controls">
        <select value={selectedPrototype} onChange={(e) => { setSelectedPrototype(e.target.value); setParams({}); }}>
          {prototypes.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
        </select>
        <div className="sliderStack">
          {(selectedSpec?.parameters || []).map((spec: AnyRow) => {
            const value = params[spec.name] ?? spec.default;
            return (
              <label key={spec.name}>
                <span>{spec.name} <b>{Number(value).toFixed(2)}</b></span>
                <input
                  type="range"
                  min={spec.min}
                  max={spec.max}
                  step={spec.step}
                  value={value}
                  onChange={(e) => setParams({ ...params, [spec.name]: Number(e.target.value) })}
                />
              </label>
            );
          })}
        </div>
        <Table rows={prototypes} columns={["id", "label", "description"]} />
      </Panel>
      <Panel title="Results">
        <div className="axisResults">
          {results.map((r) => (
            <div className="result" key={r.prototype_id}>
              <h3>{r.prototype_id}</h3>
              <Table rows={r.axis_summaries || []} columns={["axis", "min", "max", "nonzero"]} />
              <Table rows={(r.cells || []).slice(0, 96)} columns={["q", "r", "score", "owner", "own_axes", "opp_axes", "net_axes"]} />
            </div>
          ))}
        </div>
      </Panel>
    </section>
  );
}

function fixtureLabel(fixture: AnyRow) {
  const payload = fixture.payload || {};
  const label = payload.label || "Classical fixture";
  const moves = fixture.move_count ?? payload.actual_moves ?? 0;
  return `${label} (${moves}m)`;
}

function Board({
  position,
  interactive = false,
  onCellClick,
  overlayMoves = [],
  labelMoves,
  viewKey
}: {
  position: AnyRow | null | undefined;
  interactive?: boolean;
  onCellClick?: (q: number, r: number) => void;
  overlayMoves?: AnyRow[];
  labelMoves?: AnyRow[];
  viewKey?: string | number | null;
}) {
  const [hover, setHover] = useState<AnyRow | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, z: 1 });
  const [panning, setPanning] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    uxPerPx: number;
    uyPerPx: number;
    moved: boolean;
  } | null>(null);
  const stones = position?.stones || [];
  const legal = position?.legal_moves || [];
  const threat = position?.threat_moves || [];
  const threatStones = position?.threat_stones || [];
  const moves = labelMoves || position?.moves || [];
  const geometry = useMemo(() => buildBoardGeometry(position, overlayMoves, moves), [position, overlayMoves, moves]);
  const legalSet = new Set(legal.map((m: AnyRow) => `${m.q},${m.r}`));
  const threatMap = new Map<string, AnyRow>(threat.map((m: AnyRow) => [`${m.q},${m.r}`, m]));
  const threatStoneMap = new Map<string, AnyRow>(threatStones.map((m: AnyRow) => [`${m.q},${m.r}`, m]));
  const overlayMap = new Map<string, AnyRow>(overlayMoves.map((m: AnyRow) => [`${m.q},${m.r}`, m]));
  const moveNum = new Map<string, number>(moves.map((m: AnyRow, i: number) => [`${m.q},${m.r}`, i + 1]));
  const stoneMap = new Map<string, AnyRow>(stones.map((s: AnyRow) => [`${s.q},${s.r}`, s]));
  const winningCells = useMemo(() => winningLineCells(stones, position?.winner, position?.is_over), [stones, position?.winner, position?.is_over]);
  const currentPlayer = position?.current_player ?? 0;
  const last = position?.overlays?.last_move;
  const resetView = () => setView(fitPlayedView(geometry));
  useEffect(() => {
    setView(fitPlayedView(geometry));
    setHover(null);
  }, [viewKey, geometry.focusKey]);
  const zoomBy = (factor: number) => setView((v) => ({ ...v, z: clamp(v.z * factor, 0.45, 2.8) }));
  const zoomAt = (clientX: number, clientY: number, factor: number) => {
    const svg = svgRef.current;
    if (!svg) {
      zoomBy(factor);
      return;
    }
    const rect = svg.getBoundingClientRect();
    const px = (clientX - rect.left) * (geometry.width / rect.width);
    const py = (clientY - rect.top) * (geometry.height / rect.height);
    setView((v) => {
      const nextZ = clamp(v.z * factor, 0.45, 2.8);
      const anchorX = (px - v.x) / v.z;
      const anchorY = (py - v.y) / v.z;
      return {
        x: px - anchorX * nextZ,
        y: py - anchorY * nextZ,
        z: nextZ
      };
    });
  };
  const clickCell = (q: number, r: number) => {
    if (!interactive || !onCellClick || !legalSet.has(`${q},${r}`)) return;
    onCellClick(q, r);
  };
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onNativeWheel = (e: WheelEvent) => {
      e.preventDefault();
      e.stopPropagation();
      zoomAt(e.clientX, e.clientY, clamp(1 - e.deltaY * 0.0007, 0.94, 1.06));
    };
    svg.addEventListener("wheel", onNativeWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onNativeWheel);
  }, [geometry.width, geometry.height]);
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    drag.current = {
      x: e.clientX,
      y: e.clientY,
      vx: view.x,
      vy: view.y,
      uxPerPx: geometry.width / rect.width,
      uyPerPx: geometry.height / rect.height,
      moved: false
    };
    setPanning(true);
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const start = drag.current;
    if (!start) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    if (Math.abs(dx) + Math.abs(dy) > 7) start.moved = true;
    setView((v) => ({ ...v, x: start.vx + dx * start.uxPerPx, y: start.vy + dy * start.uyPerPx }));
  };
  const onPointerUp = () => {
    drag.current = null;
    setPanning(false);
  };
  return (
    <div className="viewerBoardArea">
      <svg
        ref={svgRef}
        className={`board ${interactive ? "interactive" : ""} ${panning ? "panning" : ""}`}
        viewBox={`0 0 ${geometry.width} ${geometry.height}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onMouseLeave={() => setHover(null)}
      >
        <g transform={`matrix(${view.z} 0 0 ${view.z} ${view.x} ${view.y})`}>
          {geometry.cells.map((cell) => {
            const key = `${cell.q},${cell.r}`;
            const stone = stoneMap.get(key);
            const isLegal = legalSet.has(key);
            const threatInfo = threatMap.get(key) || threatStoneMap.get(key);
            const isThreat = Boolean(threatInfo);
            const isWinning = winningCells.has(key);
            const threatPlayer = Number.isFinite(Number(threatInfo?.player)) ? Number(threatInfo?.player) : currentPlayer;
            const overlay = overlayMap.get(key);
            const isBothOverlay = overlay?.kind === "both";
            const overlayOwner = overlayOwnerFor(overlay, currentPlayer);
            const overlayColor = isBothOverlay ? "140, 150, 162" : overlayOwner === 1 ? "221, 51, 51" : "51, 119, 238";
            const overlayStroke = isBothOverlay ? "#6e7681" : overlayOwner === 1 ? "#ff9d93" : "#95b8ff";
            const isLast = last && last.q === cell.q && last.r === cell.r;
            const classes = [
              "hexCell",
              stone ? `stone p${stone.player}` : "empty",
              isLegal ? "legal" : "",
              isThreat ? "threat" : "",
              isThreat ? `p${threatPlayer}` : "",
              isWinning ? "winning" : "",
              overlay ? "overlay" : "",
              isBothOverlay ? "bothOverlay" : "",
              interactive && isLegal ? "clickable" : "",
              isLast ? "last" : ""
            ].filter(Boolean).join(" ");
            const opacity = overlay
              ? Math.min(0.82, 0.16 + Math.min(Math.abs(Number(overlay.score || 0)), 1.5) * 0.36)
              : undefined;
            return (
              <g key={key}>
                <path
                  d={hexPath(cell.x, cell.y, 23)}
                  className={classes}
                  style={overlay ? {
                    "--overlay-alpha": opacity,
                    "--overlay-rgb": overlayColor,
                    "--overlay-stroke": overlayStroke
                  } as React.CSSProperties : undefined}
                  onPointerUp={() => {
                    if (!drag.current?.moved) clickCell(cell.q, cell.r);
                  }}
                  onMouseEnter={() => setHover({ q: cell.q, r: cell.r, legal: isLegal, threat: isThreat, threat_type: threatInfo?.threat_type, threat_player: threatInfo?.player, overlay })}
                />
                {overlay && !stone && isBothOverlay && (
                  <>
                    <defs>
                      <clipPath id={bothClipId(cell.q, cell.r)}>
                        <path d={hexPath(cell.x, cell.y, 16)} />
                      </clipPath>
                    </defs>
                    <g clipPath={`url(#${bothClipId(cell.q, cell.r)})`}>
                      <path
                        className="bothPie p1"
                        d={hexPath(cell.x, cell.y, 17)}
                        style={{ opacity: pieAlpha(overlay.p1_score) }}
                      />
                      {Number(overlay.p0_share || 0) >= 0.999 ? (
                        <path
                          className="bothPie p0"
                          d={hexPath(cell.x, cell.y, 17)}
                          style={{ opacity: pieAlpha(overlay.p0_score) }}
                        />
                      ) : Number(overlay.p0_share || 0) > 0.001 ? (
                        <path
                          className="bothPie p0"
                          d={pieSlicePath(cell.x, cell.y, 23, -Math.PI / 2, -Math.PI / 2 + Number(overlay.p0_share || 0) * Math.PI * 2)}
                          style={{ opacity: pieAlpha(overlay.p0_score) }}
                        />
                      ) : null}
                    </g>
                    <path className="bothPieRing" d={hexPath(cell.x, cell.y, 16)} />
                    <text className="bothPieValue p0" x={cell.x} y={cell.y - 2}>{formatMagnitude(overlay.p0_score)}</text>
                    <text className="bothPieValue p1" x={cell.x} y={cell.y + 9}>{formatMagnitude(overlay.p1_score)}</text>
                  </>
                )}
                {overlay && !stone && !isBothOverlay && (
                  <text className="overlayValue" x={cell.x} y={cell.y + 3}>{formatStrength(overlay.score)}</text>
                )}
                {stone && (
                  <text className="moveNumber" x={cell.x} y={cell.y + 4}>{moveNum.get(key) || ""}</text>
                )}
                {isWinning && (
                  <path className="winningOutline" d={hexPath(cell.x, cell.y, 25)} />
                )}
              </g>
            );
          })}
        </g>
        <g className="boardBadge">
          <rect x="8" y="8" width="128" height="42" rx="5" />
          <circle cx="22" cy="24" r="6" className={`badgeDot p${currentPlayer}`} />
          <text x="34" y="28">P{currentPlayer} to move</text>
          <text x="22" y="43">Move {position?.turn_index ?? 0}</text>
        </g>
      </svg>
      <div className="coordTip">
        {hover ? hoverText(hover) : "Hover a cell"}
      </div>
      <div className="boardControls">
        <button onClick={() => zoomBy(1.1)}>+</button>
        <button onClick={() => zoomBy(0.91)}>-</button>
        <button onClick={resetView}>Fit</button>
      </div>
    </div>
  );
}

const HEX_SIZE = 24;
const NEIGHBORS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];

function buildBoardGeometry(
  position: AnyRow | null | undefined,
  overlayMoves: AnyRow[],
  labelMoves?: AnyRow[]
) {
  const coords = new Set<string>();
  const frameCoords = new Set<string>();
  const stones = position?.stones || [];
  const legal = position?.legal_moves || [];
  const moves = labelMoves || position?.moves || [];
  const addTo = (target: Set<string>, q: number, r: number, withNeighbors = true) => {
    if (!Number.isFinite(q) || !Number.isFinite(r)) return;
    target.add(`${q},${r}`);
    if (withNeighbors) {
      NEIGHBORS.forEach(([dq, dr]) => target.add(`${q + dq},${r + dr}`));
    }
  };
  const add = (q: number, r: number, withNeighbors = true) => addTo(coords, q, r, withNeighbors);
  const addFrame = (q: number, r: number, withNeighbors = true) => addTo(frameCoords, q, r, withNeighbors);
  stones.forEach((s: AnyRow) => {
    add(Number(s.q), Number(s.r));
    addFrame(Number(s.q), Number(s.r));
  });
  moves.forEach((m: AnyRow) => {
    add(Number(m.q), Number(m.r));
    addFrame(Number(m.q), Number(m.r));
  });
  overlayMoves.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  legal.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  if (frameCoords.size === 0 && overlayMoves.length) {
    overlayMoves.forEach((m: AnyRow) => addFrame(Number(m.q), Number(m.r)));
  }
  if (frameCoords.size === 0 && legal.length) {
    legal.forEach((m: AnyRow) => addFrame(Number(m.q), Number(m.r)));
  }
  if (coords.size === 0 || frameCoords.size === 0) {
    for (let q = -3; q <= 3; q++) {
      for (let r = -3; r <= 3; r++) {
        add(q, r, false);
        addFrame(q, r, false);
      }
    }
  }
  const frameParsed = [...frameCoords].map((key) => {
    const [q, r] = key.split(",").map(Number);
    const c = hexCenter(q, r);
    return { q, r, rawX: c.x, rawY: c.y };
  });
  const parsed = [...coords].map((key) => {
    const [q, r] = key.split(",").map(Number);
    const c = hexCenter(q, r);
    return { q, r, rawX: c.x, rawY: c.y };
  });
  const minX = Math.min(...frameParsed.map((c) => c.rawX - HEX_SIZE));
  const maxX = Math.max(...frameParsed.map((c) => c.rawX + HEX_SIZE));
  const minY = Math.min(...frameParsed.map((c) => c.rawY - HEX_SIZE));
  const maxY = Math.max(...frameParsed.map((c) => c.rawY + HEX_SIZE));
  const width = Math.max(360, maxX - minX + 44);
  const height = Math.max(360, maxY - minY + 44);
  return {
    width,
    height,
    focusKey: focusKeyFor(position),
    focus: buildFocusBounds(stones.length ? stones : moves, minX, minY),
    cells: parsed
      .map((c) => ({ q: c.q, r: c.r, x: c.rawX - minX + 22, y: c.rawY - minY + 22 }))
      .sort((a, b) => a.r - b.r || a.q - b.q)
  };
}

function winningLineCells(stones: AnyRow[], winner: any, isOver: any) {
  if (!isOver || !Number.isFinite(Number(winner))) return new Set<string>();
  const byKey = new Map<string, AnyRow>();
  stones.forEach((stone: AnyRow) => byKey.set(`${stone.q},${stone.r}`, stone));
  const winnerPlayer = Number(winner);
  const players = [winnerPlayer];
  const directions = [[1, 0], [0, 1], [1, -1]];
  let best: AnyRow[] = [];
  for (const player of players) {
    for (const stone of stones) {
      if (Number(stone.player) !== player) continue;
      const q = Number(stone.q);
      const r = Number(stone.r);
      for (const [dq, dr] of directions) {
        const prev = byKey.get(`${q - dq},${r - dr}`);
        if (prev && Number(prev.player) === player) continue;
        const line: AnyRow[] = [];
        let cq = q;
        let cr = r;
        while (true) {
          const next = byKey.get(`${cq},${cr}`);
          if (!next || Number(next.player) !== player) break;
          line.push(next);
          cq += dq;
          cr += dr;
        }
        if (line.length > best.length) best = line;
      }
    }
  }
  if (best.length < 6) return new Set<string>();
  return new Set(best.map((stone: AnyRow) => `${stone.q},${stone.r}`));
}

function fitPlayedView(geometry: AnyRow) {
  const focus = geometry.focus;
  if (!focus) return { x: 0, y: 0, z: 1 };
  const focusWidth = Math.max(120, focus.maxX - focus.minX + HEX_SIZE * 3.5);
  const focusHeight = Math.max(120, focus.maxY - focus.minY + HEX_SIZE * 3.5);
  const z = clamp(Math.min((geometry.width - 72) / focusWidth, (geometry.height - 72) / focusHeight), 1, 2.25);
  const cx = (focus.minX + focus.maxX) / 2;
  const cy = (focus.minY + focus.maxY) / 2;
  return {
    x: geometry.width / 2 - cx * z,
    y: geometry.height / 2 - cy * z,
    z
  };
}

function buildFocusBounds(items: AnyRow[], minX: number, minY: number) {
  if (!items.length) return null;
  const pts = items.map((item) => {
    const c = hexCenter(Number(item.q), Number(item.r));
    return { x: c.x - minX + 22, y: c.y - minY + 22 };
  });
  return {
    minX: Math.min(...pts.map((p) => p.x)),
    maxX: Math.max(...pts.map((p) => p.x)),
    minY: Math.min(...pts.map((p) => p.y)),
    maxY: Math.max(...pts.map((p) => p.y))
  };
}

function focusKeyFor(position: AnyRow | null | undefined) {
  const moves = position?.moves || [];
  return moves.map((m: AnyRow) => `${m.q},${m.r}`).join("|");
}

function hexCenter(q: number, r: number) {
  return {
    x: HEX_SIZE * (1.5 * q),
    y: HEX_SIZE * ((Math.sqrt(3) / 2) * q + Math.sqrt(3) * r)
  };
}

function hexPath(cx: number, cy: number, size: number) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i;
    pts.push(`${(cx + size * Math.cos(a)).toFixed(2)},${(cy + size * Math.sin(a)).toFixed(2)}`);
  }
  return `M${pts.join("L")}Z`;
}

function pieSlicePath(cx: number, cy: number, radius: number, startAngle: number, endAngle: number) {
  const span = Math.max(0, Math.min(Math.PI * 2 - 0.0001, endAngle - startAngle));
  if (span <= 0) return "";
  const end = startAngle + span;
  const x1 = cx + radius * Math.cos(startAngle);
  const y1 = cy + radius * Math.sin(startAngle);
  const x2 = cx + radius * Math.cos(end);
  const y2 = cy + radius * Math.sin(end);
  const largeArc = span > Math.PI ? 1 : 0;
  return `M${cx.toFixed(2)},${cy.toFixed(2)} L${x1.toFixed(2)},${y1.toFixed(2)} A${radius},${radius} 0 ${largeArc} 1 ${x2.toFixed(2)},${y2.toFixed(2)} Z`;
}

function bothClipId(q: number, r: number) {
  return `both-pie-${q}-${r}`.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function hoverText(hover: AnyRow) {
  const parts = [`(${hover.q}, ${hover.r})`, hover.legal ? "legal" : "not legal"];
  if (hover.threat) parts.push(`${hover.threat_type || "threat"} P${hover.threat_player ?? "-"}`);
  if (hover.overlay) {
    const axes = Array.isArray(hover.overlay.axes)
      ? hover.overlay.axes.map((v: number) => Number(v).toFixed(2)).join(",")
      : "";
    const own = Array.isArray(hover.overlay.own_axes)
      ? hover.overlay.own_axes.map((v: number) => Number(v).toFixed(2)).join(",")
      : "";
    const opp = Array.isArray(hover.overlay.opp_axes)
      ? hover.overlay.opp_axes.map((v: number) => Number(v).toFixed(2)).join(",")
      : "";
    parts.push(`score ${Number(hover.overlay.score || 0).toFixed(2)}`);
    if (hover.overlay.scale_mode && hover.overlay.scale_mode !== "raw") {
      parts.push(`scaled ${hover.overlay.scale_mode}`);
    }
    if (hover.overlay.kind === "both") {
      parts.push(`P0 max ${Number(hover.overlay.p0_score || 0).toFixed(2)}`);
      parts.push(`P1 max ${Number(hover.overlay.p1_score || 0).toFixed(2)}`);
      parts.push(`raw P0 ${Number(hover.overlay.raw_p0_score || 0).toFixed(2)}`);
      parts.push(`raw P1 ${Number(hover.overlay.raw_p1_score || 0).toFixed(2)}`);
      parts.push(`P0 share ${Math.round(Number(hover.overlay.p0_share || 0) * 100)}%`);
      parts.push(`own max ${Number(hover.overlay.own_score || 0).toFixed(2)}`);
      parts.push(`opp max ${Number(hover.overlay.opp_score || 0).toFixed(2)}`);
      parts.push(`product ${Number(hover.overlay.product_score || 0).toFixed(2)}`);
    } else if (Number.isFinite(Number(hover.overlay.owner))) {
      parts.push(`P${Number(hover.overlay.owner)} strength`);
    }
    if (own) parts.push(`own [${own}]`);
    if (opp) parts.push(`opp [${opp}]`);
    if (axes) parts.push(`axes [${axes}]`);
  }
  return parts.join(" · ");
}

function formatStrength(value: number) {
  const n = Number(value || 0);
  const prefix = n > 0 ? "+" : "";
  return `${prefix}${n.toFixed(2)}`;
}

function formatMagnitude(value: number) {
  return Number(value || 0).toFixed(2);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function pieAlpha(value: number) {
  return clamp(0.28 + Number(value || 0) * 0.52, 0.28, 0.9);
}

function overlayOwnerFor(overlay: AnyRow | undefined, currentPlayer: number) {
  if (!overlay) return currentPlayer;
  if (Number.isFinite(Number(overlay.owner))) return Number(overlay.owner);
  return Number(overlay.score || 0) >= 0 ? currentPlayer : 1 - currentPlayer;
}

function deriveAxisOverlay(cell: AnyRow, mode: string, currentPlayer: number, scaleMode: string) {
  const rawOwnAxes = toNumArray(cell.own_axes);
  const rawOppAxes = toNumArray(cell.opp_axes);
  const rawNetAxes = cell.net_axes ? toNumArray(cell.net_axes) : rawOwnAxes.map((v, i) => v - (rawOppAxes[i] || 0));
  const ownAxes = rawOwnAxes.map((v) => scaleAxisValue(v, scaleMode));
  const oppAxes = rawOppAxes.map((v) => scaleAxisValue(v, scaleMode));
  const netAxes = rawNetAxes.map((v) => scaleAxisValue(v, scaleMode));
  const ownScore = maxValue(ownAxes);
  const oppScore = maxValue(oppAxes);
  const rawOwnScore = maxValue(rawOwnAxes);
  const rawOppScore = maxValue(rawOppAxes);
  const p0Score = currentPlayer === 0 ? ownScore : oppScore;
  const p1Score = currentPlayer === 1 ? ownScore : oppScore;
  const rawP0Score = currentPlayer === 0 ? rawOwnScore : rawOppScore;
  const rawP1Score = currentPlayer === 1 ? rawOwnScore : rawOppScore;
  const shareTotal = Math.max(p0Score + p1Score, 1e-9);
  const p0Share = p0Score / shareTotal;
  const p1Share = p1Score / shareTotal;
  const opponent = 1 - currentPlayer;
  let score = 0;
  let owner = currentPlayer;
  let axes = ownAxes;

  if (mode === "opp") {
    score = oppScore;
    owner = opponent;
    axes = oppAxes;
  } else if (mode === "net") {
    score = maxAbsValue(netAxes);
    owner = score >= 0 ? currentPlayer : opponent;
    axes = netAxes;
  } else if (mode === "max") {
    const ownAbs = Math.max(Math.abs(ownScore), 0);
    const oppAbs = Math.max(Math.abs(oppScore), 0);
    score = ownAbs >= oppAbs ? ownScore : oppScore;
    owner = ownAbs >= oppAbs ? currentPlayer : opponent;
    axes = ownAbs >= oppAbs ? ownAxes : oppAxes;
  } else if (mode === "both") {
    axes = ownAxes.map((v, i) => Math.min(v, oppAxes[i] || 0));
    score = Math.max(p0Score, p1Score);
    owner = p0Score >= p1Score ? 0 : 1;
  } else {
    score = ownScore;
    owner = currentPlayer;
    axes = ownAxes;
  }

  return {
    q: Number(cell.q),
    r: Number(cell.r),
    score,
    owner,
    kind: mode === "both" ? "both" : mode,
    scale_mode: scaleMode,
    own_score: ownScore,
    opp_score: oppScore,
    p0_score: p0Score,
    p1_score: p1Score,
    p0_share: p0Share,
    p1_share: p1Share,
    raw_own_score: rawOwnScore,
    raw_opp_score: rawOppScore,
    raw_p0_score: rawP0Score,
    raw_p1_score: rawP1Score,
    product_score: ownScore * oppScore,
    axes,
    own_axes: ownAxes,
    opp_axes: oppAxes,
    net_axes: netAxes
  };
}

function toNumArray(value: any) {
  return Array.isArray(value) ? value.slice(0, 3).map((v) => Number(v || 0)) : [0, 0, 0];
}

function maxValue(values: number[]) {
  return values.length ? Math.max(...values) : 0;
}

function maxAbsValue(values: number[]) {
  if (!values.length) return 0;
  return values.reduce((best, value) => Math.abs(value) > Math.abs(best) ? value : best, values[0]);
}

function scaleAxisValue(value: number, mode: string) {
  const n = Number(value || 0);
  const sign = n < 0 ? -1 : 1;
  const mag = Math.abs(n);
  if (mode === "log") return sign * Math.log1p(mag) / Math.log1p(3);
  if (mode === "sqrt") return sign * Math.sqrt(Math.min(mag, 3) / 3);
  if (mode === "unit") return clamp(n / 1.5, -1, 1);
  return n;
}

function Panel({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panelTitle"><Activity size={14} /> {title}</div>
      {hint && <p className="panelHint">{hint}</p>}
      {children}
    </section>
  );
}

function Table({ rows, columns, onRow, selected, className = "" }: {
  rows: AnyRow[];
  columns: string[];
  onRow?: (row: AnyRow) => void;
  selected?: (row: AnyRow) => boolean;
  className?: string;
}) {
  return (
    <div className={`tableWrap ${className}`}>
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{labelFor(c)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              onClick={() => onRow?.(row)}
              className={selected?.(row) ? "selected" : onRow ? "clickable" : ""}
            >
              {columns.map((c) => <td key={c}>{cell(row[c], c, row)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LossChart({ metrics, keys }: { metrics: AnyRow[]; keys: string[] }) {
  const width = 560;
  const height = 220;
  const pad = 24;
  const series = keys.map((key, keyIdx) => ({
    key,
    color: lossColor(keyIdx),
    points: metrics.map((m, i) => ({
      x: i,
      y: Number(lossValues(m.metrics_json)[key] ?? NaN)
    })).filter((p) => Number.isFinite(p.y))
  })).filter((s) => s.points.length > 0);
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const min = Math.min(...ys, 0);
  const max = Math.max(...ys, 1);
  const xFor = (i: number) => pad + (metrics.length <= 1 ? 0 : (i / (metrics.length - 1)) * (width - pad * 2));
  const yFor = (value: number) => height - pad - ((value - min) / Math.max(max - min, 1e-6)) * (height - pad * 2);
  return (
    <div className="chartWrap">
      <svg className="chart" viewBox={`0 0 ${width} ${height}`}>
        <line className="chartGrid" x1={pad} y1={pad} x2={pad} y2={height - pad} />
        <line className="chartGrid" x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} />
        <text className="chartTick" x={pad + 3} y={pad + 10}>{fmt(max)}</text>
        <text className="chartTick" x={pad + 3} y={height - pad - 4}>{fmt(min)}</text>
        {series.map((s) => {
          const d = s.points.map((p, i) => {
            const x = xFor(p.x);
            const y = yFor(p.y);
            return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
          }).join(" ");
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.key}>
              <path d={d} style={{ stroke: s.color }} />
              <circle cx={xFor(last.x)} cy={yFor(last.y)} r="2.5" fill={s.color} />
              <text className="chartLabel" x={xFor(last.x) + 5} y={yFor(last.y) + 4} fill={s.color}>
                {shortLossLabel(s.key)} {fmt(last.y)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="chartLegend">
        {series.map((s) => {
          const last = s.points[s.points.length - 1]?.y;
          return (
            <span key={s.key}>
              <i style={{ background: s.color }} />
              {shortLossLabel(s.key)} <b>{fmt(last)}</b>
            </span>
          );
        })}
      </div>
    </div>
  );
}

function collectLossKeys(metrics: AnyRow[]) {
  const found = new Set<string>();
  for (const metric of metrics) {
    for (const key of Object.keys(lossValues(metric.metrics_json))) found.add(key);
  }
  const preferred = ["loss_total", "loss_policy", "loss_value", "loss_axis_delta_norm", "loss_entropy"];
  return Array.from(found).sort((a, b) => {
    const ai = preferred.indexOf(a);
    const bi = preferred.indexOf(b);
    if (ai >= 0 || bi >= 0) return (ai >= 0 ? ai : 999) - (bi >= 0 ? bi : 999);
    return a.localeCompare(b);
  });
}

function collectMetricKeys(metrics: AnyRow[], preferred: string[]) {
  const found = new Set<string>();
  for (const metric of metrics) {
    for (const key of Object.keys(metricValues(metric.metrics_json))) found.add(key);
  }
  return Array.from(found).sort((a, b) => {
    const ai = preferred.indexOf(a);
    const bi = preferred.indexOf(b);
    if (ai >= 0 || bi >= 0) return (ai >= 0 ? ai : 999) - (bi >= 0 ? bi : 999);
    return a.localeCompare(b);
  });
}

function lossValues(metricsJson: AnyRow | undefined) {
  const source = metricsJson?.train || metricsJson || {};
  const result: AnyRow = {};
  for (const [key, value] of Object.entries(source)) {
    if (key.startsWith("loss_") && typeof value === "number") result[key] = value;
  }
  return result;
}

function metricValues(metricsJson: AnyRow | undefined) {
  const source = metricsJson?.train || metricsJson || {};
  const result: AnyRow = {};
  for (const [key, value] of Object.entries(source)) {
    if (!key.startsWith("loss_") && typeof value === "number") result[key] = value;
  }
  return result;
}

function shortLossLabel(key: string) {
  return key.replace(/^loss_/, "").replace(/_/g, " ");
}

function lossColor(index: number) {
  return ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#f2cc60", "#79c0ff", "#ffa657"][index % 7];
}

function Sparkline({ points }: { points: { x: number; y: number }[] }) {
  const width = 560;
  const height = 220;
  const ys = points.map((p) => p.y);
  const min = Math.min(...ys, 0);
  const max = Math.max(...ys, 1);
  const d = points.map((p, i) => {
    const x = points.length <= 1 ? 0 : (i / (points.length - 1)) * width;
    const y = height - ((p.y - min) / Math.max(max - min, 1e-6)) * height;
    return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  return <svg className="chart" viewBox={`0 0 ${width} ${height}`}><path d={d} /></svg>;
}

function cell(value: any, key = "", row?: AnyRow) {
  if (value === null || value === undefined) return "-";
  if (key === "outcome") return outcomeCell(value, row);
  if (isTimestampKey(key)) return formatTimestamp(value);
  if (key.endsWith("_s") || key === "elapsed_s" || key === "epoch_elapsed_s") return formatDuration(value);
  if (key.includes("positions_per_sec")) return formatRate(value);
  if (typeof value === "number") return Number.isInteger(value) ? formatCount(value) : value.toFixed(4);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 140);
  return String(value);
}

function outcomeCell(value: any, row?: AnyRow) {
  const label = String(value || "-");
  const player = outcomePlayer(label, row);
  const normalized = label.replace(/_/g, " ");
  if (player === null) return <span className="outcomeText neutral">{normalized}</span>;
  return <span className={`outcomeText p${player}`}>{normalized}</span>;
}

function outcomePlayer(outcome: string, row?: AnyRow): number | null {
  const value = outcome.toLowerCase();
  if (value.includes("player_0") || value.includes("p0")) return 0;
  if (value.includes("player_1") || value.includes("p1")) return 1;
  if (row?.kind === "fixed_classical" || row?.source === "fixed_classical") {
    const modelPlayer = row.opening_is_black === false ? 1 : 0;
    const classicalPlayer = 1 - modelPlayer;
    if (value.includes("classical_win")) return classicalPlayer;
    if (value.includes("model_win") || value.includes("survived")) return modelPlayer;
  }
  if (value === "win" && Number.isFinite(Number(row?.winner))) return Number(row?.winner);
  return null;
}

function fmt(value: any) {
  return typeof value === "number" ? value.toFixed(4) : value ?? "-";
}

function formatCount(value: any) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return new Intl.NumberFormat().format(number);
}

function formatRate(value: any) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(number >= 10 ? 1 : 2)}/s`;
}

function formatDuration(value: any) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (number < 60) return `${number.toFixed(1)}s`;
  if (number < 3600) return `${Math.floor(number / 60)}m ${Math.round(number % 60)}s`;
  return `${Math.floor(number / 3600)}h ${Math.round((number % 3600) / 60)}m`;
}

function formatTimestamp(value: any) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return value ?? "-";
  const ms = number > 10_000_000_000 ? number : number * 1000;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(ms));
}

function isTimestampKey(key: string) {
  return ["created_at", "updated_at", "indexed_at", "time"].includes(key);
}

function labelFor(key: string) {
  return key.replace(/_/g, " ");
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
