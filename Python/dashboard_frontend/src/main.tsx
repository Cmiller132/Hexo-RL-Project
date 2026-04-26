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
        <AxisPanel prototypes={axis} results={axisResults} setResults={setAxisResults} />
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
  const clickMove = (q: number, r: number) => playMove({ q, r });
  useEffect(() => {
    if (!session) create();
  }, [session]);
  return (
    <section className="viewerGrid">
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
      <Panel title="Legal Moves">
        <div className="viewerInfo">
          <div><span>Legal</span><strong>{legal.length}</strong></div>
          <div><span>Threat filter</span><strong>{(session?.position?.threat_moves || []).length || "off"}</strong></div>
          <div><span>Moves</span><strong>{session?.position?.turn_index ?? 0}</strong></div>
        </div>
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

function AxisPanel({ prototypes, results, setResults }: {
  prototypes: AnyRow[];
  results: AnyRow[];
  setResults: (rows: AnyRow[]) => void;
}) {
  const [axisSession, setAxisSession] = useState<AnyRow | null>(null);
  const [selectedPrototype, setSelectedPrototype] = useState<string>("");
  const [params, setParams] = useState<Record<string, number>>({});
  const selected = results.find((r) => r.prototype_id === selectedPrototype) || results[0];
  const selectedSpec = prototypes.find((p) => p.id === (selectedPrototype || prototypes[0]?.id));
  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    if (!selectedPrototype && prototypes.length) setSelectedPrototype(prototypes[0].id);
  }, [prototypes, selectedPrototype]);
  const create = () => api<AnyRow>("/api/session/create", { method: "POST", body: JSON.stringify({ payload: { mode: "axis_lab" } }) }).then(setAxisSession);
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
  const offsetQ = axisSession?.position?.encoding?.offset_q ?? -16;
  const offsetR = axisSession?.position?.encoding?.offset_r ?? -16;
  const overlayMoves = (selected?.top || []).map((m: AnyRow, idx: number) => {
    const action = Number(m.action);
    return {
      q: Number.isFinite(Number(m.q)) ? Number(m.q) : Math.floor(action / 33) + offsetQ,
      r: Number.isFinite(Number(m.r)) ? Number(m.r) : (action % 33) + offsetR,
      prob: Number(m.prob),
      axes: m.axes,
      action,
      rank: idx + 1
    };
  });
  return (
    <section className="viewerGrid">
      <Panel title="Axis Target Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} /> New</button>
          <button onClick={undo}>Undo</button>
          <button onClick={reset}>Reset</button>
          <button onClick={evaluate}><Target size={15} /> Evaluate</button>
          <span className={`playerBadge p${axisSession?.position?.current_player ?? 0}`}>
            P{axisSession?.position?.current_player ?? 0} to move
          </span>
        </div>
        <Board
          position={axisSession?.position}
          interactive
          onCellClick={playMove}
          overlayMoves={overlayMoves}
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
              <Table rows={r.axis_summaries || []} columns={["axis", "sum", "max", "nonzero"]} />
              <Table rows={(r.top || []).map((m: AnyRow, i: number) => ({ rank: i + 1, ...m }))} columns={["rank", "q", "r", "prob", "axes"]} />
            </div>
          ))}
        </div>
      </Panel>
    </section>
  );
}

function Board({
  position,
  interactive = false,
  onCellClick,
  overlayMoves = []
}: {
  position: AnyRow | null | undefined;
  interactive?: boolean;
  onCellClick?: (q: number, r: number) => void;
  overlayMoves?: AnyRow[];
}) {
  const [hover, setHover] = useState<AnyRow | null>(null);
  const stones = position?.stones || [];
  const legal = position?.legal_moves || [];
  const threat = position?.threat_moves || [];
  const moves = position?.moves || [];
  const geometry = useMemo(() => buildBoardGeometry(position, overlayMoves), [position, overlayMoves]);
  const legalSet = new Set(legal.map((m: AnyRow) => `${m.q},${m.r}`));
  const threatSet = new Set(threat.map((m: AnyRow) => `${m.q},${m.r}`));
  const overlayMap = new Map<string, AnyRow>(overlayMoves.map((m: AnyRow) => [`${m.q},${m.r}`, m]));
  const moveNum = new Map<string, number>(moves.map((m: AnyRow, i: number) => [`${m.q},${m.r}`, i + 1]));
  const stoneMap = new Map<string, AnyRow>(stones.map((s: AnyRow) => [`${s.q},${s.r}`, s]));
  const currentPlayer = position?.current_player ?? 0;
  const last = position?.overlays?.last_move;
  const clickCell = (q: number, r: number) => {
    if (!interactive || !onCellClick || !legalSet.has(`${q},${r}`)) return;
    onCellClick(q, r);
  };
  return (
    <div className="viewerBoardArea">
      <svg
        className={`board ${interactive ? "interactive" : ""}`}
        viewBox={`0 0 ${geometry.width} ${geometry.height}`}
        onMouseLeave={() => setHover(null)}
      >
        {geometry.cells.map((cell) => {
          const key = `${cell.q},${cell.r}`;
          const stone = stoneMap.get(key);
          const isLegal = legalSet.has(key);
          const isThreat = threatSet.has(key);
          const overlay = overlayMap.get(key);
          const isLast = last && last.q === cell.q && last.r === cell.r;
          const classes = [
            "hexCell",
            stone ? `stone p${stone.player}` : "empty",
            isLegal ? "legal" : "",
            isThreat ? "threat" : "",
            overlay ? "overlay" : "",
            interactive && isLegal ? "clickable" : "",
            isLast ? "last" : ""
          ].filter(Boolean).join(" ");
          const opacity = overlay ? Math.min(0.82, 0.18 + Number(overlay.prob || 0) * 3.2) : undefined;
          return (
            <g key={key}>
              <path
                d={hexPath(cell.x, cell.y, 23)}
                className={classes}
                style={overlay ? { "--overlay-alpha": opacity } as React.CSSProperties : undefined}
                onClick={() => clickCell(cell.q, cell.r)}
                onMouseEnter={() => setHover({ q: cell.q, r: cell.r, legal: isLegal, threat: isThreat })}
              />
              {overlay && !stone && (
                <text className="overlayRank" x={cell.x} y={cell.y + 3}>{overlay.rank}</text>
              )}
              {stone && (
                <text className="moveNumber" x={cell.x} y={cell.y + 4}>{moveNum.get(key) || ""}</text>
              )}
            </g>
          );
        })}
        <g className="boardBadge">
          <rect x="8" y="8" width="128" height="42" rx="5" />
          <circle cx="22" cy="24" r="6" className={`badgeDot p${currentPlayer}`} />
          <text x="34" y="28">P{currentPlayer} to move</text>
          <text x="22" y="43">Move {position?.turn_index ?? 0}</text>
        </g>
      </svg>
      <div className="coordTip">
        {hover ? `(${hover.q}, ${hover.r}) ${hover.legal ? "legal" : "not legal"}${hover.threat ? " threat" : ""}` : "Hover a cell"}
      </div>
    </div>
  );
}

const HEX_SIZE = 24;
const NEIGHBORS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]];

function buildBoardGeometry(position: AnyRow | null | undefined, overlayMoves: AnyRow[]) {
  const coords = new Set<string>();
  const stones = position?.stones || [];
  const legal = position?.legal_moves || [];
  const moves = position?.moves || [];
  const add = (q: number, r: number, withNeighbors = true) => {
    coords.add(`${q},${r}`);
    if (withNeighbors) {
      NEIGHBORS.forEach(([dq, dr]) => coords.add(`${q + dq},${r + dr}`));
    }
  };
  stones.forEach((s: AnyRow) => add(Number(s.q), Number(s.r)));
  legal.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  moves.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  overlayMoves.forEach((m: AnyRow) => add(Number(m.q), Number(m.r)));
  if (coords.size === 0) {
    for (let q = -3; q <= 3; q++) {
      for (let r = -3; r <= 3; r++) add(q, r, false);
    }
  }
  const parsed = [...coords].map((key) => {
    const [q, r] = key.split(",").map(Number);
    const c = hexCenter(q, r);
    return { q, r, rawX: c.x, rawY: c.y };
  });
  const minX = Math.min(...parsed.map((c) => c.rawX - HEX_SIZE));
  const maxX = Math.max(...parsed.map((c) => c.rawX + HEX_SIZE));
  const minY = Math.min(...parsed.map((c) => c.rawY - HEX_SIZE));
  const maxY = Math.max(...parsed.map((c) => c.rawY + HEX_SIZE));
  const width = Math.max(360, maxX - minX + 44);
  const height = Math.max(360, maxY - minY + 44);
  return {
    width,
    height,
    cells: parsed
      .map((c) => ({ q: c.q, r: c.r, x: c.rawX - minX + 22, y: c.rawY - minY + 22 }))
      .sort((a, b) => a.r - b.r || a.q - b.q)
  };
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
