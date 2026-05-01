import { NavLink, Navigate, Route, Routes, useNavigate, useSearchParams } from "react-router-dom";
import { BarChart3, Database, Eye, FileSearch, Gamepad2, RefreshCw, Swords, Target, Trophy } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { ConnectionBanner } from "./components/connection-banner";
import { KpiRow } from "./components/kpi-row";
import { useHealth, useMetrics, useRuns, useSuiteStatusStream } from "./api/hooks";
import { formatRate } from "./components/format";
import SuiteOverview from "./routes/suite";
import TrialRoute from "./routes/suite/trial";
import FamilySpaceRoute from "./routes/suite/family-space";
import SchedulerRoute from "./routes/suite/scheduler";
import RuntimeSweepRoute from "./routes/suite/runtime-sweep";
import ChartsRoute from "./routes/charts";
import GamesRoute from "./routes/games";
import ReplayRoute from "./routes/replay";
import PlayRoute from "./routes/play";
import ArenaRoute from "./routes/arena";
import CheckpointsRoute from "./routes/checkpoints";
import AxisLabRoute from "./routes/axis-lab";

const nav = [
  { to: "/suite", label: "Suite", icon: Trophy },
  { to: "/charts", label: "Charts", icon: BarChart3 },
  { to: "/games", label: "Games", icon: FileSearch },
  { to: "/replay", label: "Replay", icon: Eye },
  { to: "/play", label: "Play", icon: Gamepad2 },
  { to: "/arena", label: "Arena", icon: Swords },
  { to: "/checkpoints", label: "Checkpoints", icon: Database },
  { to: "/axis", label: "Axis Lab", icon: Target }
];

export default function App() {
  const [search, setSearch] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const selectedRun = search.get("run") || "";
  const { data: health } = useHealth();
  const { data: runs = [] } = useRuns();
  const { data: metrics = [] } = useMetrics(selectedRun);
  const { data: suiteStatus, isError } = useSuiteStatusStream();
  const latestMetric = (metrics[metrics.length - 1]?.metrics_json || {}) as Record<string, any>;
  const runId = selectedRun || runs[0]?.run_id || "";

  const setRun = (run: string) => {
    const next = new URLSearchParams(search);
    if (run) next.set("run", run);
    else next.delete("run");
    next.delete("game");
    next.delete("turn");
    setSearch(next, { replace: false });
  };

  const refresh = () => queryClient.invalidateQueries();
  const selectGame = (gameId: number | string, run?: string) => {
    const next = new URLSearchParams(search);
    if (run) next.set("run", run);
    next.set("game", String(gameId));
    next.set("turn", "0");
    setSearch(next);
    navigate(`/replay?${next.toString()}`);
  };

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>Hexo-RL Dashboard</h1>
          <span className="subtle">{String(health?.db_path || "loading database")}</span>
        </div>
        <div className="toolbar">
          <select aria-label="Selected run" value={runId} onChange={(e) => setRun(e.target.value)}>
            <option value="">No run</option>
            {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.name || run.run_id}</option>)}
          </select>
          <button title="Refresh" onClick={refresh}><RefreshCw size={16} /></button>
        </div>
      </header>
      <ConnectionBanner connected={!isError} />
      <KpiRow items={[
        ["Runs", runs.length],
        ["Pos/sec", formatRate(suiteStatus?.current_positions_per_sec)],
        ["Current", suiteStatus?.current_model ?? runId ?? "-"],
        ["Best Trial", suiteStatus?.best_trial_id ?? "-"],
        ["Epoch", latestMetric.train?.epoch ?? latestMetric.epoch ?? "-"],
        ["Loss", latestMetric.train?.loss_total ?? latestMetric.loss_total ?? "-"]
      ]} />
      <nav className="tabs" aria-label="Dashboard routes">
        {nav.map((item) => {
          const Icon = item.icon;
          return <NavLink key={item.to} to={`${item.to}?${search.toString()}`}><Icon size={15} />{item.label}</NavLink>;
        })}
      </nav>
      <Routes>
        <Route path="/" element={<Navigate to={`/suite?${search.toString()}`} replace />} />
        <Route path="/suite" element={<SuiteOverview openGame={selectGame} />} />
        <Route path="/suite/trials/:trialId/*" element={<TrialRoute />} />
        <Route path="/suite/family-space" element={<FamilySpaceRoute />} />
        <Route path="/suite/scheduler" element={<SchedulerRoute />} />
        <Route path="/suite/runtime-sweep" element={<RuntimeSweepRoute />} />
        <Route path="/charts" element={<ChartsRoute runId={runId} />} />
        <Route path="/games" element={<GamesRoute runId={runId} openGame={selectGame} />} />
        <Route path="/replay" element={<ReplayRoute runId={runId} />} />
        <Route path="/play" element={<PlayRoute />} />
        <Route path="/arena" element={<ArenaRoute />} />
        <Route path="/checkpoints" element={<CheckpointsRoute runId={runId} />} />
        <Route path="/axis" element={<AxisLabRoute />} />
      </Routes>
    </main>
  );
}
