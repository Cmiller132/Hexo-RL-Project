import { useTrialEvents } from "../../../api/hooks";
import { Panel } from "../../../components/panel";
import { Table } from "../../../components/table";

export default function EventsTab({ trialId }: { trialId: string }) {
  const { data = [] } = useTrialEvents(trialId);
  return (
    <Panel title="Trial Events">
      <Table rows={data} columns={["event", "stage", "severity", "reason", "score", "selected_positions_per_min", "elapsed_s", "time"]} />
    </Panel>
  );
}
