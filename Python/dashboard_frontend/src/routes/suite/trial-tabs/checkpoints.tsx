import type { AnyRow } from "../../../api/client";
import { useCheckpoints } from "../../../api/hooks";
import { Panel } from "../../../components/panel";
import { Table } from "../../../components/table";

export default function CheckpointsTab({ detail, trialId }: { detail: AnyRow; trialId: string }) {
  const runId = String(detail.run_id || detail.suite_run_id || "");
  const { data = [] } = useCheckpoints(runId);
  const scoped = data.filter((row) => !trialId || row.trial_id === trialId);
  return (
    <Panel title="Trial Checkpoints">
      <Table rows={scoped} columns={["checkpoint_id", "trial_id", "run_id", "score", "epoch", "global_step", "is_loadable", "path"]} />
    </Panel>
  );
}
