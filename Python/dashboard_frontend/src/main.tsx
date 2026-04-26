import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Bot,
  Database,
  Eye,
  FileSearch,
  Gamepad2,
  Play,
  RefreshCw,
  Swords,
  Target
} from "lucide-react";
import "./styles.css";

type AnyRow = Record<string, any>;

const tabs = [
  { id: "charts", label: "Charts", icon: BarChart3 },
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

function App() {
  const [active, setActive] = useState("charts");
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
  const [error, setError] = useState<string>("");

  const load = async () => {
    try {
      setError("");
      const [h, r, c, g, a, p] = await Promise.all([
        api<AnyRow>("/api/health"),
        api<AnyRow[]>("/api/runs"),
        api<AnyRow[]>("/api/checkpoints"),
        api<AnyRow[]>("/api/games"),
        api<AnyRow[]>("/api/arena/history"),
        api<AnyRow[]>("/api/axis/prototypes")
      ]);
      setHealth(h);
      setRuns(r);
      setCheckpoints(c);
      setGames(g);
      setArena(a);
      setAxis(p);
      if (!selectedRun && r.length) setSelectedRun(r[0].run_id);
      if (!selectedGame && g.length) setSelectedGame(g[0].game_id);
    } catch (e: any) {
      setError(e.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    api<AnyRow[]>(`/api/metrics/${encodeURIComponent(selectedRun)}`)
      .then(setMetrics)
      .catch((e) => setError(e.message));
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedGame) return;
    api<AnyRow>(`/api/games/${selectedGame}/replay`)
      .then((data) => {
        setReplay(data);
        return api<AnyRow>(`/api/games/${selectedGame}/position/0`);
      })
      .then(setPosition)
      .catch((e) => setError(e.message));
  }, [selectedGame]);

  const latestMetric = metrics[metrics.length - 1]?.metrics_json || {};
  const kpis = [
    ["Runs", runs.length],
    ["Games", games.length],
    ["Checkpoints", checkpoints.length],
    ["Epoch", latestMetric.train?.epoch ?? latestMetric.epoch ?? "-"],
    ["Buffer", latestMetric.buffer?.size ?? "-"],
    ["Loss", fmt(latestMetric.train?.loss_total ?? latestMetric.loss_total)]
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
              <option key={run.run_id} value={run.run_id}>{run.name || run.run_id}</option>
            ))}
          </select>
          <button title="Refresh" onClick={load}><RefreshCw size={16} /></button>
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

      {active === "charts" && <Charts metrics={metrics} />}
      {active === "games" && (
        <Games games={games} selectedGame={selectedGame} setSelectedGame={setSelectedGame} />
      )}
      {active === "replay" && (
        <Replay replay={replay} position={position} setPosition={setPosition} selectedGame={selectedGame} />
      )}
      {active === "play" && <PlayPanel session={session} setSession={setSession} />}
      {active === "arena" && <ArenaPanel arena={arena} reload={load} />}
      {active === "checkpoints" && <CheckpointPanel checkpoints={checkpoints} reload={load} />}
      {active === "axis" && (
        <AxisPanel prototypes={axis} results={axisResults} setResults={setAxisResults} session={session} />
      )}
    </main>
  );
}

function Charts({ metrics }: { metrics: AnyRow[] }) {
  const points = metrics.map((m, i) => ({
    x: i,
    y: Number(m.metrics_json?.train?.loss_total ?? m.metrics_json?.loss_total ?? 0)
  }));
  return (
    <section className="grid two">
      <Panel title="Loss">
        <Sparkline points={points} />
      </Panel>
      <Panel title="Recent Metrics">
        <Table rows={metrics.slice(-12).reverse()} columns={["phase", "epoch", "global_step", "created_at"]} />
      </Panel>
    </section>
  );
}

function Games({ games, selectedGame, setSelectedGame }: {
  games: AnyRow[];
  selectedGame: number | null;
  setSelectedGame: (id: number) => void;
}) {
  return (
    <Panel title="Game Browser">
      <Table
        rows={games}
        columns={["game_id", "run_id", "source", "epoch", "outcome", "move_count"]}
        onRow={(row) => setSelectedGame(row.game_id)}
        selected={(row) => row.game_id === selectedGame}
      />
    </Panel>
  );
}

function Replay({ replay, position, setPosition, selectedGame }: {
  replay: AnyRow | null;
  position: AnyRow | null;
  setPosition: (p: AnyRow) => void;
  selectedGame: number | null;
}) {
  const moves = replay?.moves || [];
  const loadTurn = (turn: number) => {
    if (!selectedGame) return;
    api<AnyRow>(`/api/games/${selectedGame}/position/${turn}`).then(setPosition);
  };
  return (
    <section className="grid replay">
      <Panel title="Board">
        <Board position={position} />
      </Panel>
      <Panel title="Timeline">
        <div className="moveList">
          <button onClick={() => loadTurn(0)}>Start</button>
          {moves.map((m: AnyRow, i: number) => (
            <button key={i} onClick={() => loadTurn(i + 1)}>
              {i + 1}. P{m.player} ({m.q},{m.r})
            </button>
          ))}
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
  const legal = session?.position?.legal_moves || [];
  const playMove = (m: AnyRow) => session && api<AnyRow>(`/api/session/${session.session_id}/move`, {
    method: "POST",
    body: JSON.stringify({ q: m.q, r: m.r })
  }).then(setSession);
  return (
    <section className="grid replay">
      <Panel title="Interactive Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} /> New</button>
          <button onClick={undo}>Undo</button>
          <button onClick={reset}>Reset</button>
        </div>
        <Board position={session?.position} />
      </Panel>
      <Panel title="Legal Moves">
        <div className="moveList">
          {legal.slice(0, 80).map((m: AnyRow, i: number) => (
            <button key={i} onClick={() => playMove(m)}>({m.q},{m.r})</button>
          ))}
        </div>
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

function CheckpointPanel({ checkpoints, reload }: { checkpoints: AnyRow[]; reload: () => void }) {
  const [path, setPath] = useState("");
  const index = () => api<AnyRow>("/api/import/checkpoints", {
    method: "POST",
    body: JSON.stringify({ path })
  }).then(reload);
  return (
    <Panel title="Checkpoint Index">
      <div className="toolbar">
        <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/checkpoints" />
        <button onClick={index}><Database size={15} /> Index</button>
      </div>
      <Table rows={checkpoints} columns={["checkpoint_id", "run_id", "epoch", "global_step", "is_loadable", "path"]} />
    </Panel>
  );
}

function AxisPanel({ prototypes, results, setResults, session }: {
  prototypes: AnyRow[];
  results: AnyRow[];
  setResults: (rows: AnyRow[]) => void;
  session: AnyRow | null;
}) {
  const evaluate = () => {
    const body = session?.session_id ? { session_id: session.session_id } : { history_b64: "" };
    api<AnyRow>("/api/axis/evaluate", { method: "POST", body: JSON.stringify(body) })
      .then((data) => setResults(data.results || [data]));
  };
  return (
    <section className="grid two">
      <Panel title="Python Prototypes">
        <div className="toolbar compact">
          <button onClick={evaluate}><Target size={15} /> Compare</button>
        </div>
        <Table rows={prototypes} columns={["id", "label", "description"]} />
      </Panel>
      <Panel title="Results">
        <div className="axisResults">
          {results.map((r) => (
            <div className="result" key={r.prototype_id}>
              <h3>{r.prototype_id}</h3>
              <Table rows={r.axis_summaries || []} columns={["axis", "sum", "max", "nonzero"]} />
              <Table rows={r.top || []} columns={["action", "prob"]} />
            </div>
          ))}
        </div>
      </Panel>
    </section>
  );
}

function Board({ position }: { position: AnyRow | null | undefined }) {
  const stones = position?.stones || [];
  const legal = position?.legal_moves || [];
  const view = useMemo(() => ({ w: 620, h: 520, scale: 18 }), []);
  const toXY = (q: number, r: number) => {
    const x = view.w / 2 + view.scale * Math.sqrt(3) * (q + r / 2);
    const y = view.h / 2 + view.scale * 1.5 * r;
    return [x, y];
  };
  return (
    <svg className="board" viewBox={`0 0 ${view.w} ${view.h}`}>
      {legal.slice(0, 240).map((m: AnyRow, i: number) => {
        const [x, y] = toXY(m.q, m.r);
        return <circle key={`l${i}`} cx={x} cy={y} r="5" className="legal" />;
      })}
      {stones.map((s: AnyRow, i: number) => {
        const [x, y] = toXY(s.q, s.r);
        return (
          <g key={i}>
            <circle cx={x} cy={y} r="9" className={s.player === 0 ? "p0" : "p1"} />
            <text x={x} y={y + 3}>{i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panelTitle"><Activity size={14} /> {title}</div>
      {children}
    </section>
  );
}

function Table({ rows, columns, onRow, selected }: {
  rows: AnyRow[];
  columns: string[];
  onRow?: (row: AnyRow) => void;
  selected?: (row: AnyRow) => boolean;
}) {
  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              onClick={() => onRow?.(row)}
              className={selected?.(row) ? "selected" : onRow ? "clickable" : ""}
            >
              {columns.map((c) => <td key={c}>{cell(row[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
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

function cell(value: any) {
  if (value === null || value === undefined) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(4);
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 80);
  return String(value);
}

function fmt(value: any) {
  return typeof value === "number" ? value.toFixed(4) : value ?? "-";
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
