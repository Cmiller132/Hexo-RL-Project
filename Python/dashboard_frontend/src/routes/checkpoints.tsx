import { Database } from "lucide-react";
import { useState } from "react";
import { apiPost } from "../api/client";
import { useCheckpoints } from "../api/hooks";
import { Panel } from "../components/panel";
import { Table } from "../components/table";

export default function CheckpointsRoute({ runId }: { runId: string }) {
  const [path, setPath] = useState("");
  const { data = [], refetch } = useCheckpoints(runId);
  const index = () => apiPost("/api/import/checkpoints", { path, run_id: runId || undefined }).then(() => refetch());
  return (
    <Panel title="Checkpoint Index">
      <div className="toolbar"><input value={path} onChange={(e) => setPath(e.target.value)} placeholder="/path/to/checkpoints" /><button onClick={index}><Database size={15} />Index</button></div>
      <Table rows={data} columns={["checkpoint_id", "trial_id", "run_id", "score", "epoch", "global_step", "is_loadable", "path"]} />
    </Panel>
  );
}
