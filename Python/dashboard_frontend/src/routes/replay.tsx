import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { enc, type AnyRow } from "../api/client";
import { usePosition, useReplay } from "../api/hooks";
import { Board } from "../components/board";
import { EmptyState, Panel } from "../components/panel";
import { Table } from "../components/table";

export default function ReplayRoute({ runId }: { runId: string }) {
  const [search, setSearch] = useSearchParams();
  const gameId = search.get("game") || "";
  const turn = Math.max(0, Number(search.get("turn") || 0));
  const { data: replay } = useReplay(gameId, runId);
  const { data: position, refetch } = usePosition(gameId, turn, runId);
  const [autoplay, setAutoplay] = useState(false);
  const moves = useMemo(() => Array.isArray(replay?.moves) ? replay.moves as AnyRow[] : [], [replay]);
  const encoding = (position?.encoding || {}) as AnyRow;
  const setTurn = (nextTurn: number) => {
    const next = new URLSearchParams(search);
    next.set("turn", String(Math.max(0, Math.min(moves.length, nextTurn))));
    setSearch(next, { replace: false });
  };
  useEffect(() => {
    if (!autoplay || !gameId) return;
    const handle = window.setInterval(() => {
      const next = turn + 1;
      if (next > moves.length) setAutoplay(false);
      else setTurn(next);
    }, 650);
    return () => window.clearInterval(handle);
  }, [autoplay, gameId, turn, moves.length]);
  useEffect(() => { refetch(); }, [turn, refetch]);
  if (!gameId) return <EmptyState>Select a game to replay.</EmptyState>;
  return (
    <section className="grid replay">
      <Panel title="Board"><Board position={position} /></Panel>
      <Panel title="Timeline">
        <div className="toolbar compact">
          <button onClick={() => setAutoplay((v) => !v)} disabled={!moves.length}>{autoplay ? <Pause size={15} /> : <Play size={15} />}{autoplay ? "Pause" : "Autoplay"}</button>
          <button onClick={() => setTurn(turn - 1)} disabled={turn <= 0}>Prev</button>
          <button onClick={() => setTurn(turn + 1)} disabled={turn >= moves.length}>Next</button>
          <a href={`/api/games/${enc(gameId)}/snapshot.png${runId ? `?run_id=${enc(runId)}&turn_index=${turn}` : `?turn_index=${turn}`}`}>Snapshot</a>
        </div>
        <div className="moveList">
          <button className={turn === 0 ? "active" : ""} onClick={() => setTurn(0)}>Start</button>
          {moves.map((move, i) => <button key={i} className={turn === i + 1 ? "active" : ""} onClick={() => setTurn(i + 1)}>{i + 1}. P{String(move.player)} ({String(move.q)},{String(move.r)})</button>)}
        </div>
      </Panel>
      <Panel title="Encoding"><Table rows={Array.isArray(encoding.channels) ? encoding.channels as AnyRow[] : []} columns={["index", "name", "sum", "nonzero", "max"]} /></Panel>
    </section>
  );
}
