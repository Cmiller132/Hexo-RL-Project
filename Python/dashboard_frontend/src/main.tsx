import React, { useEffect, useMemo, useRef, useState } from "react";
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
  viewKey
}: {
  position: AnyRow | null | undefined;
  interactive?: boolean;
  onCellClick?: (q: number, r: number) => void;
  overlayMoves?: AnyRow[];
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
  const moves = position?.moves || [];
  const geometry = useMemo(() => buildBoardGeometry(position, overlayMoves), [position, overlayMoves]);
  const legalSet = new Set(legal.map((m: AnyRow) => `${m.q},${m.r}`));
  const threatSet = new Set(threat.map((m: AnyRow) => `${m.q},${m.r}`));
  const overlayMap = new Map<string, AnyRow>(overlayMoves.map((m: AnyRow) => [`${m.q},${m.r}`, m]));
  const moveNum = new Map<string, number>(moves.map((m: AnyRow, i: number) => [`${m.q},${m.r}`, i + 1]));
  const stoneMap = new Map<string, AnyRow>(stones.map((s: AnyRow) => [`${s.q},${s.r}`, s]));
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
            const isThreat = threatSet.has(key);
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
                  onMouseEnter={() => setHover({ q: cell.q, r: cell.r, legal: isLegal, threat: isThreat, overlay })}
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
    focusKey: focusKeyFor(position),
    focus: buildFocusBounds(stones.length ? stones : moves, minX, minY),
    cells: parsed
      .map((c) => ({ q: c.q, r: c.r, x: c.rawX - minX + 22, y: c.rawY - minY + 22 }))
      .sort((a, b) => a.r - b.r || a.q - b.q)
  };
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
  if (hover.threat) parts.push("threat");
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
