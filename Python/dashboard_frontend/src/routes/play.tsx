import { Database, Play } from "lucide-react";
import { useEffect, useState } from "react";
import { apiPost, type AnyRow } from "../api/client";
import { Board } from "../components/board";
import { Panel } from "../components/panel";

export default function PlayRoute() {
  const [session, setSession] = useState<AnyRow | null>(null);
  const create = () => apiPost<AnyRow>("/api/session/create", {}).then(setSession);
  const undo = () => session && apiPost<AnyRow>(`/api/session/${session.session_id}/undo`, {}).then(setSession);
  const reset = () => session && apiPost<AnyRow>(`/api/session/${session.session_id}/reset`, {}).then(setSession);
  const move = (q: number, r: number) => session && apiPost<AnyRow>(`/api/session/${session.session_id}/move`, { q, r }).then(setSession);
  useEffect(() => { if (!session) create(); }, [session]);
  return (
    <section className="viewerGrid playGrid">
      <Panel title="Interactive Board">
        <div className="toolbar compact">
          <button onClick={create}><Play size={15} />New</button>
          <button onClick={undo}>Undo</button>
          <button onClick={reset}>Reset</button>
          <span className={`playerBadge p${String((session?.position as AnyRow)?.current_player ?? 0)}`}>P{String((session?.position as AnyRow)?.current_player ?? 0)} to move</span>
        </div>
        <Board position={session?.position as AnyRow | undefined} interactive onCellClick={move} />
      </Panel>
      <Panel title="Debug"><pre>{JSON.stringify((session?.position as AnyRow)?.encoding || {}, null, 2)}</pre><Database size={16} /></Panel>
    </section>
  );
}
