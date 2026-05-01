import { useScheduler } from "../../api/hooks";
import { Panel } from "../../components/panel";
import { Table } from "../../components/table";
import type { AnyRow } from "../../api/client";

export default function SchedulerRoute() {
  const { data = {} } = useScheduler();
  const stages = Array.isArray(data.planned_stages) ? data.planned_stages as AnyRow[] : [];
  const decisions = Array.isArray(data.decisions) ? data.decisions as AnyRow[] : [];
  return (
    <section className="suiteGrid">
      <Panel title="Scheduler State">
        <div className="detailCards">
          {["current_stage", "budget_remaining_positions", "budget_remaining_wall_time_s", "queue_depth"].map((key) => <div className="metricCard" key={key}><span>{key.replace(/_/g, " ")}</span><strong>{String(data[key] ?? "-")}</strong></div>)}
        </div>
      </Panel>
      <Panel title="Planned Stages"><Table rows={stages} columns={["stage", "positions", "wall_time_s", "promotion_threshold", "prune_threshold"]} /></Panel>
      <Panel title="Decisions"><Table rows={decisions} columns={["time", "event", "trial_id", "reason", "score", "threshold"]} /></Panel>
    </section>
  );
}
