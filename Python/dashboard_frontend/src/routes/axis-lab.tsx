import { Bot, Play, RefreshCw, Target } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { apiPost, type AnyRow } from "../api/client";
import { useAxisFixtures, useAxisPrototypes } from "../api/hooks";
import { Board } from "../components/board";
import { Panel } from "../components/panel";
import { Table } from "../components/table";

export default function AxisLabRoute() {
  const { data: prototypes = [] } = useAxisPrototypes();
  const { data: fixtures = [], refetch: refreshFixtures } = useAxisFixtures();
  const [session, setSession] = useState<AnyRow | null>(null);
  const [prototypeId, setPrototypeId] = useState("");
  const [results, setResults] = useState<AnyRow[]>([]);
  const selected = results.find((row) => row.prototype_id === prototypeId) || results[0];
  const overlays = useMemo(() => (Array.isArray(selected?.cells) ? selected.cells as AnyRow[] : []).map((cell) => ({ ...cell, score: Number(cell.score ?? 0) })), [selected]);
  const create = () => apiPost<AnyRow>("/api/session/create", { payload: { mode: "axis_lab" } }).then(setSession);
  const move = (q: number, r: number) => session && apiPost<AnyRow>(`/api/session/${session.session_id}/move`, { q, r }).then((next) => { setSession(next); setResults([]); });
  const reset = () => session && apiPost<AnyRow>(`/api/session/${session.session_id}/reset`, {}).then((next) => { setSession(next); setResults([]); });
  const evaluate = () => apiPost<AnyRow>("/api/axis/evaluate", { session_id: session?.session_id, prototype_id: prototypeId || undefined }).then((data) => setResults(Array.isArray(data.results) ? data.results as AnyRow[] : [data]));
  const generate = () => apiPost<AnyRow>("/api/axis/fixtures/generate", {}).then(() => refreshFixtures());
  useEffect(() => { if (!session) create(); }, [session]);
  useEffect(() => { if (!prototypeId && prototypes[0]?.id) setPrototypeId(String(prototypes[0].id)); }, [prototypeId, prototypes]);
  return (
    <section className="viewerGrid">
      <Panel title="Axis Target Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} />New</button>
          <button onClick={reset}>Reset</button>
          <button onClick={evaluate}><Target size={15} />Evaluate</button>
          <button onClick={generate}><Bot size={15} />Generate</button>
          <button onClick={() => refreshFixtures()}><RefreshCw size={14} />Fixtures</button>
          <select value={prototypeId} onChange={(e) => setPrototypeId(e.target.value)}>{prototypes.map((p) => <option key={String(p.id)} value={String(p.id)}>{String(p.label || p.id)}</option>)}</select>
        </div>
        <Board position={session?.position as AnyRow | undefined} interactive onCellClick={move} overlayMoves={overlays} />
      </Panel>
      <Panel title="Prototypes"><Table rows={prototypes} columns={["id", "label", "description"]} /></Panel>
      <Panel title="Results"><Table rows={Array.isArray(selected?.cells) ? selected.cells as AnyRow[] : []} columns={["q", "r", "score", "owner", "own_axes", "opp_axes", "net_axes"]} /></Panel>
      <Panel title="Fixtures"><Table rows={fixtures} columns={["session_id", "move_count", "created_at", "payload"]} /></Panel>
    </section>
  );
}
