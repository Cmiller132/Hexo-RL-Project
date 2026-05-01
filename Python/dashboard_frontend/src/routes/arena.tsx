import { Swords } from "lucide-react";
import { apiPost } from "../api/client";
import { useArenaStream } from "../api/hooks";
import { Panel } from "../components/panel";
import { Table } from "../components/table";

export default function ArenaRoute() {
  const { data = [], refetch } = useArenaStream();
  const start = () => apiPost("/api/arena/start", { side_a: "model", side_b: "classical" }).then(() => refetch());
  return (
    <Panel title="Arena Spectator">
      <div className="toolbar compact"><button onClick={start}><Swords size={15} />Create Match</button></div>
      <Table rows={data} columns={["match_id", "status", "side_a", "side_b", "updated_at"]} />
    </Panel>
  );
}
